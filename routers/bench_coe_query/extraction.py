"""Deterministic independent-rank feature extraction driver."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentContract, load_full_run_approval, sha256_file
from .data import load_vlrb_strict
from .distributed_sharding import context_from_environment, shard_rows
from .extraction_gate import build_full_extraction_gate
from .feature_cache import FeatureShardWriter, feature_signature, merge_feature_shards
from .full_policy import partition_feature_rows
from .input_content import verify_pinned_input_content
from .manifest import build_manifest, write_manifest
from .selection import select_rows


def _extractor(name: str, contract: ExperimentContract, device: torch.device):
    if name == "tinyllava":
        from .backbones.tinyllava import TinyLLaVAFeatureExtractor

        return TinyLLaVAFeatureExtractor(contract, device=device)
    if name == "qwen3vl":
        from .backbones.qwen3vl import Qwen3VLFeatureExtractor

        return Qwen3VLFeatureExtractor(contract, device=device)
    raise ValueError(name)


def run_extraction(args, contract: ExperimentContract) -> int:
    data = load_vlrb_strict(args.dataset_dir, contract=contract)
    split = getattr(args, "split", "train")
    selected_rows = select_rows(data, split=split, max_samples=args.max_samples)
    is_full = os.environ.get("RUN_FULL", "0") == "1" and args.max_samples is None
    approval_sha256: str | None = None
    exclusions: tuple[dict[str, Any], ...] = ()
    if is_full:
        _approval, approval_sha256 = load_full_run_approval(contract)
        rows, exclusions = partition_feature_rows(
            data,
            selected_rows,
            contract=contract,
            require_release_counts=split == "all",
        )
    else:
        rows = selected_rows
    context = context_from_environment(getattr(args, "world_size", None))
    local_rows = shard_rows(rows, rank=context.rank, world_size=context.world_size)
    local_exclusions = tuple(
        record for record in exclusions if int(record["row_index"]) % context.world_size == context.rank
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(context.local_rank)
        device = torch.device("cuda", context.local_rank)
    else:
        device = torch.device("cpu")
    input_content_verification = None
    if is_full:
        # Verify bytes before from_pretrained touches them.  Each rank performs
        # the check so no worker can silently use a different local mount.
        input_content_verification = verify_pinned_input_content(
            contract,
            model_names=(args.backbone,),
            include_visual_assets=True,
        )
    extractor = _extractor(args.backbone, contract, device)
    signature_payload = dict(extractor.feature_signature_payload())
    if is_full:
        implementation_root = Path(__file__).resolve().parent
        signature_payload.update(
            {
                "run_scope": "full_official_release",
                "approval_sha256": approval_sha256,
                "full_multi_image_policy": "exclude_with_frozen_denominator_report",
                "full_multi_image_fallback": "train_strongest_global",
                "asset_resolver_code_sha256": sha256_file(implementation_root / "assets.py"),
                "extraction_driver_code_sha256": sha256_file(Path(__file__)),
                "feature_cache_code_sha256": sha256_file(implementation_root / "feature_cache.py"),
                "full_policy_code_sha256": sha256_file(implementation_root / "full_policy.py"),
            }
        )
    signature = feature_signature(signature_payload)
    output_override = getattr(args, "output_dir", None)
    cache_root = (
        Path(output_override)
        if output_override
        else contract.dataset_root / str(contract.values["features"]["root"]) / args.backbone / signature
    )
    writer = FeatureShardWriter(
        cache_root,
        rank=context.rank,
        world_size=context.world_size,
        signature=signature,
        feature_dim=extractor.feature_dim,
        dtype="float16",
        resume=args.resume,
    )
    pending = [row for row in local_rows if row not in writer.completed_rows]
    batch_size = int(getattr(args, "batch_size", None) or contract.values["scheduling"]["oom_batch_ladder"][args.backbone][0])
    before_checksum = extractor.parameter_checksum()
    started = time.perf_counter()
    preprocess_forward_seconds = 0.0
    preprocess_seconds = 0.0
    forward_seconds = 0.0
    write_seconds = 0.0
    text_tokens = 0
    visual_tokens = 0
    for batch_number, start in enumerate(range(0, len(pending), batch_size), 1):
        batch_rows = pending[start : start + batch_size]
        samples = [data.samples[row] for row in batch_rows]
        batch_started = time.perf_counter()
        result = extractor.encode_batch(samples)
        preprocess_forward_seconds += time.perf_counter() - batch_started
        timing = getattr(extractor, "last_batch_timing", {})
        preprocess_seconds += float(timing.get("preprocess_seconds", 0.0))
        forward_seconds += float(timing.get("forward_seconds", 0.0))
        text_tokens += int(timing.get("text_tokens", 0))
        visual_tokens += int(timing.get("visual_tokens", 0))
        write_started = time.perf_counter()
        writer.append(
            batch_rows,
            [sample.sample_id for sample in samples],
            result.features.numpy(),
            metadata=result.metadata,
            flush=batch_number % int(contract.values["features"]["save_every_batches"]) == 0,
        )
        write_seconds += time.perf_counter() - write_started
    finalize_started = time.perf_counter()
    writer.finalize()
    write_seconds += time.perf_counter() - finalize_started
    after_checksum = extractor.parameter_checksum()
    if before_checksum != after_checksum:
        raise RuntimeError("frozen backbone parameter checksum changed during extraction")
    elapsed = time.perf_counter() - started
    resource = {
        "schema_version": 1,
        "backbone": args.backbone,
        "rank": context.rank,
        "world_size": context.world_size,
        "rows_assigned": len(local_rows),
        "rows_assigned_before_policy": len(local_rows) + len(local_exclusions),
        "structured_exclusions": len(local_exclusions),
        "rows_processed_this_attempt": len(pending),
        "elapsed_seconds": elapsed,
        "processor_plus_forward_seconds": preprocess_forward_seconds,
        "preprocess_seconds": preprocess_seconds,
        "forward_seconds": forward_seconds,
        "write_seconds": write_seconds,
        "text_tokens": text_tokens,
        "visual_tokens": visual_tokens,
        "tokens_per_forward_second": (text_tokens + visual_tokens) / forward_seconds
        if forward_seconds
        else None,
        "samples_per_second": len(pending) / elapsed if elapsed else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "backbone_checksum_before": before_checksum,
        "backbone_checksum_after": after_checksum,
        "backbone_unchanged": True,
        "optimizer_parameters": 0,
        "feature_signature": signature,
        "feature_signature_payload": signature_payload,
        "input_content": signature_payload.get("input_content"),
        "input_content_verification": input_content_verification,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / f"resource_rank_{context.rank:03d}.json").write_text(
        json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(
        cache_root / f"run_manifest_rank_{context.rank:03d}.json",
        build_manifest(
            contract,
            seed=args.seed,
            run_id=args.run_id,
            input_content_verification=input_content_verification,
        ),
    )
    write_manifest(
        cache_root / f"exclusions_rank_{context.rank:03d}.json",
        {
            "schema_version": 1,
            "backbone": args.backbone,
            "rank": context.rank,
            "world_size": context.world_size,
            "policy": "exclude_with_frozen_denominator_report" if is_full else "none",
            "records": list(local_exclusions),
        },
    )
    if context.world_size > 1:
        if not torch.distributed.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            init_kwargs: dict[str, Any] = {"backend": backend}
            if device.type == "cuda":
                # Binding the process group to this rank's device avoids NCCL's
                # ambiguous-device fallback (and its associated hang warning).
                init_kwargs["device_id"] = device
            torch.distributed.init_process_group(**init_kwargs)
        barrier_kwargs = {"device_ids": [context.local_rank]} if device.type == "cuda" else {}
        torch.distributed.barrier(**barrier_kwargs)
    if context.rank == 0:
        manifest = merge_feature_shards(
            cache_root,
            expected_rows=rows,
            world_size=context.world_size,
            signature=signature,
        )
        (cache_root / "feature_signature.json").write_text(
            json.dumps(signature_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        gate = None
        if is_full and split == "all" and context.world_size == 8:
            gate = build_full_extraction_gate(
                contract,
                backbone=args.backbone,
                cache_root=cache_root,
                world_size=context.world_size,
            )
            if gate.get("passed") is not True:
                raise RuntimeError(f"{args.backbone} full extraction gate failed")
        print(
            json.dumps(
                {"cache_root": str(cache_root), "manifest": manifest, "extraction_gate": gate},
                indent=2,
                sort_keys=True,
            )
        )
    if torch.distributed.is_initialized():
        barrier_kwargs = {"device_ids": [context.local_rank]} if device.type == "cuda" else {}
        torch.distributed.barrier(**barrier_kwargs)
        torch.distributed.destroy_process_group()
    return 0
