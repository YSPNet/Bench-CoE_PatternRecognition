"""Target-free prediction entry point for frozen best checkpoints.

This process never loads Y/C and never evaluates a prediction. It freezes
JSONL plus a hash-bound receipt, then exits before a separate evaluator runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .backbones.utils import force_offline_environment, require_model_identity
from .checkpoints import checkpoint_sha256, load_checkpoint
from .config import ExperimentContract, effective_model_path, sha256_file
from .feature_cache import load_full_feature_cache, load_merged_features
from .full_policy import frozen_train_strongest_column, materialize_frozen_denominator_logits
from .heads import AuditedBenchCoEHead
from .input_content import feature_input_content_fields, verify_pinned_input_content
from .predictor import write_logits_predictions
from .reproducibility import seed_everything
from .selection import select_rows
from .target_free import (
    TargetFreeVLRouterData,
    deny_target_matrix_access,
    load_target_free_vlrb,
    prediction_receipt_path,
    write_target_free_receipt,
)


class _BertPredictionDataset(Dataset):
    def __init__(self, samples: Sequence[Any], tokenizer, max_length: int):
        self.samples = tuple(samples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.samples[index].text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def _predict_bert_logits(
    model,
    tokenizer,
    samples: Sequence[Any],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        _BertPredictionDataset(samples, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    values = [
        model(**{name: tensor.to(device) for name, tensor in batch.items()})
        .logits.float()
        .cpu()
        .numpy()
        for batch in loader
    ]
    return np.concatenate(values, axis=0)


@torch.no_grad()
def _predict_head_logits(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    model.eval()
    return np.concatenate(
        [model(batch[0].to(device)).float().cpu().numpy() for batch in loader], axis=0
    )


def _producer_sources(contract: ExperimentContract, backbone: str) -> tuple[Path, ...]:
    common = (
        Path(__file__),
        contract.dataset_root / "routers/bench_coe_query/target_free.py",
        contract.dataset_root / "routers/bench_coe_query/predictor.py",
        contract.dataset_root / "routers/bench_coe_query/checkpoints.py",
        contract.dataset_root / "routers/bench_coe_query/reproducibility.py",
    )
    if backbone == "bert":
        return common + (
            contract.dataset_root / "routers/bench_coe_query/backbones/utils.py",
        )
    return common + (
        contract.dataset_root / "routers/bench_coe_query/feature_cache.py",
        contract.dataset_root / "routers/bench_coe_query/full_policy.py",
        contract.dataset_root / "routers/bench_coe_query/heads.py",
    )


def _output_paths(args, splits: Iterable[str]) -> dict[str, Path]:
    names = tuple(splits)
    if getattr(args, "output_dir", None):
        root = Path(args.output_dir)
        return {split: root / f"predictions_{split}.jsonl" for split in names}
    if len(names) != 1 or not getattr(args, "output", None):
        raise ValueError("single-split prediction requires --output")
    return {names[0]: Path(args.output)}


def _freeze_predictions(
    args,
    contract: ExperimentContract,
    *,
    splits: Sequence[str],
) -> dict[str, Any]:
    if any(split not in {"train", "dev", "test"} for split in splits):
        raise ValueError(f"invalid prediction split set: {splits}")
    force_offline_environment()
    seed_info = seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_full = os.environ.get("RUN_FULL", "0") == "1" and args.max_samples is None
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_id = checkpoint_sha256(checkpoint_path)
    paths = _output_paths(args, splits)

    with deny_target_matrix_access(contract) as target_guard:
        data: TargetFreeVLRouterData = load_target_free_vlrb(
            args.dataset_dir, contract=contract
        )
        rows_by_split = {
            split: select_rows(data, split=split, max_samples=args.max_samples)
            for split in splits
        }
        input_binding: dict[str, Any]
        annotations_by_split: dict[str, Sequence[dict[str, Any]] | None] = {
            split: None for split in splits
        }
        logits_by_split: dict[str, np.ndarray] = {}

        if args.backbone == "bert":
            verification = (
                verify_pinned_input_content(
                    contract,
                    model_names=("bert",),
                    include_visual_assets=False,
                )
                if is_full
                else {"passed": True, "scope": "smoke_local_model"}
            )
            model_path = effective_model_path(contract, "bert")
            require_model_identity(
                model_path,
                expected_model_type="bert",
                expected_architecture="BertForMaskedLM",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), use_fast=True, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                num_labels=len(data.model_order),
                local_files_only=True,
                ignore_mismatched_sizes=True,
            ).to(device)
            load_checkpoint(
                checkpoint_path,
                model=model,
                expected_config_sha256=contract.sha256,
                map_location=device,
            )
            max_length = int(contract.values["models"]["bert"]["max_length"])
            batch_size = int(
                contract.values["profiles"]["vlrb_adapted_v1"]["bert"][
                    "batch_size_per_gpu"
                ]
            )
            for split, rows in rows_by_split.items():
                logits_by_split[split] = _predict_bert_logits(
                    model,
                    tokenizer,
                    [data.samples[row] for row in rows],
                    max_length=max_length,
                    batch_size=batch_size,
                    device=device,
                )
            input_binding = {
                "kind": "verified_local_bert_payload",
                "verification": verification,
            }
        else:
            if not args.feature_cache:
                raise ValueError("VLM head prediction requires --feature-cache")
            cache_root = Path(args.feature_cache)
            manifest_path = cache_root / "merged/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if is_full:
                features, records, _, cache_integrity = load_full_feature_cache(
                    cache_root,
                    dataset_root=contract.dataset_root,
                    expected_contract_sha256=contract.sha256,
                    expected_input_content=feature_input_content_fields(
                        contract, args.backbone
                    ),
                )
            else:
                features, records, loaded_manifest = load_merged_features(
                    cache_root,
                    expected_signature=str(manifest["signature"]),
                    expected_manifest_sha256=str(manifest["manifest_sha256"]),
                )
                cache_integrity = {
                    "passed": True,
                    "cache_root": str(cache_root),
                    "feature_signature": loaded_manifest["signature"],
                    "manifest_internal_sha256": loaded_manifest["manifest_sha256"],
                }
            position_by_row = {
                int(record["row_index"]): position for position, record in enumerate(records)
            }
            if len(position_by_row) != len(records):
                raise RuntimeError("feature cache contains duplicate rows")
            if any(
                data.samples[int(record["row_index"])].sample_id
                != str(record["sample_id"])
                for record in records
            ):
                raise RuntimeError("feature cache sample IDs differ from target-free registry")
            model = AuditedBenchCoEHead(
                hidden_dim=int(features.shape[1]),
                num_experts=len(data.model_order),
                dropout=float(
                    contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"][
                        "dropout"
                    ]
                ),
            ).to(device)
            load_checkpoint(
                checkpoint_path,
                model=model,
                expected_config_sha256=contract.sha256,
                expected_feature_manifest_sha256=str(manifest["manifest_sha256"]),
                map_location=device,
            )
            batch_size = int(
                contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"][
                    "batch_size_per_gpu"
                ]
            )
            strongest = frozen_train_strongest_column(contract, data.model_order)
            label_meta_path = (
                contract.dataset_root
                / "outputs/bench_coe_query/labels/label_manifest.meta.json"
            )
            for split, rows in rows_by_split.items():
                feature_rows = tuple(row for row in rows if row in position_by_row)
                positions = np.asarray(
                    [position_by_row[row] for row in feature_rows], dtype=np.int64
                )
                feature_logits = _predict_head_logits(
                    model,
                    features[positions],
                    batch_size=batch_size,
                    device=device,
                )
                if is_full:
                    logits, annotations, _ = materialize_frozen_denominator_logits(
                        data=data,
                        official_rows=rows,
                        feature_rows=feature_rows,
                        feature_logits=feature_logits,
                        strongest_train_column=strongest,
                    )
                    logits_by_split[split] = logits
                    annotations_by_split[split] = annotations
                else:
                    missing = set(rows).difference(feature_rows)
                    if missing:
                        raise RuntimeError(
                            f"smoke feature cache lacks {len(missing)} requested rows"
                        )
                    logits_by_split[split] = feature_logits
            input_binding = {
                "kind": "verified_frozen_feature_cache",
                "cache_root": cache_integrity.get("cache_root"),
                "feature_signature": cache_integrity.get("feature_signature"),
                "manifest_internal_sha256": cache_integrity.get(
                    "manifest_internal_sha256"
                ),
                "features_sha256": cache_integrity.get("features_sha256"),
                "index_sha256": cache_integrity.get("index_sha256"),
                "pointer_sha256": cache_integrity.get("pointer_sha256"),
                "gate_sha256": cache_integrity.get("gate_sha256"),
                "input_content": cache_integrity.get("input_content"),
                "train_only_fallback_policy": {
                    "label_manifest_meta_path": label_meta_path.relative_to(
                        contract.dataset_root
                    ).as_posix(),
                    "label_manifest_meta_sha256": sha256_file(label_meta_path),
                    "strongest_expert_column": strongest,
                    "strongest_expert_id": data.model_order[strongest],
                },
                "passed": cache_integrity.get("passed") is True,
            }

        summaries: dict[str, Any] = {}
        for split, rows in rows_by_split.items():
            samples = [data.samples[row] for row in rows]
            summary = write_logits_predictions(
                paths[split],
                sample_ids=[sample.sample_id for sample in samples],
                row_indices=rows,
                logits=logits_by_split[split],
                model_order=data.model_order,
                split=split,
                seed=args.seed,
                checkpoint_id=checkpoint_id,
                annotations=annotations_by_split[split],
            )
            receipt_path = prediction_receipt_path(paths[split])
            receipt = write_target_free_receipt(
                receipt_path,
                contract=contract,
                data=data,
                backbone=args.backbone,
                split=split,
                seed=args.seed,
                checkpoint_path=checkpoint_path,
                prediction_path=paths[split],
                prediction_sha256=str(summary["sha256"]),
                prediction_records=int(summary["count"]),
                guard_audit=target_guard,
                producer_sources=_producer_sources(contract, args.backbone),
                input_binding=input_binding,
            )
            summaries[split] = {
                **summary,
                "receipt": str(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
            }

    if target_guard.get("passed") is not True:
        raise RuntimeError("target matrix guard did not pass")
    result = {
        "schema_version": 1,
        "kind": "target_free_prediction_set",
        "backbone": args.backbone,
        "seed": args.seed,
        "checkpoint_sha256": checkpoint_id,
        "target_free_input_manifest_sha256": data.input_manifest_sha256,
        "target_matrix_guard": target_guard,
        "seed_settings": seed_info,
        "device": str(device),
        "predictions": summaries,
        "evaluation_performed": False,
        "passed": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def run_prediction(args, contract: ExperimentContract) -> int:
    _freeze_predictions(args, contract, splits=(args.split,))
    return 0


def run_prediction_set(args, contract: ExperimentContract) -> int:
    _freeze_predictions(args, contract, splits=("train", "dev", "test"))
    return 0
