#!/usr/bin/env python3
"""Evaluate StateRoute-MM after supplying native multi-image Qwen3-VL features.

This is a post-hoc, single-seed extension of the frozen held-out 11 -> 3
StateRoute experiment.  It preserves the original audited single-image cache
and models.  The only prediction-time change is that the 47 MMMU rows excluded
by the old single-image contract receive features from all of their images, in
the original order, using the same frozen Qwen3-VL model, processor, prompt
template, dtype, and attention-mask mean pooling.

Run the stages in order so routes are frozen before test Y/C are opened:

  python scripts/run_stateroute_multi_image_20260918.py --stage extract --device cuda:0
  python scripts/run_stateroute_multi_image_20260918.py --stage predict
  python scripts/run_stateroute_multi_image_20260918.py --stage evaluate
"""
from __future__ import annotations

import argparse
import ast
import base64
import binascii
import io
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RUN_API_REGEN", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, UnidentifiedImageError

from routers.bench_coe_query.assets import TSVRowIndex
from routers.bench_coe_query.backbones.qwen3vl import Qwen3VLFeatureExtractor
from routers.bench_coe_query.config import load_contract
from routers.bench_coe_query.schemas import AssetSpec, StrictSample
from routers.utils.rank_score import rank_score
from scripts import run_stateroute_k_sweep_20260916 as base
from src.next_round_x.data import FeatureCache, manifests, role_uids
from src.next_round_x.models import normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json


OUT = ROOT / "outputs/stateroute_multi_image_20260918"
REPORT = ROOT / "reports/stateroute_multi_image_20260918"
CONTRACT_PATH = ROOT / "configs/bench_coe_query/experiment_contract.yaml"
FEATURE_POINTER = ROOT / "outputs/bench_coe_query/features/qwen3vl/LATEST_FULL.json"
KS = (1, 2, 4, 6, 8, 10, 12)
SEED = 0
FEATURE_DIM = 2048
EXPECTED_SPLIT_COUNTS = {"train": 32, "dev": 5, "test": 10}
METRICS = ("accuracy_percent", "cost_usd_per_10k", "rank_score")


def say(**value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _verify_records(records: Iterable[dict[str, Any]]) -> None:
    for item in records:
        path = ROOT / item["path"]
        if item.get("exists") is not True or not path.is_file() or digest(path) != item["sha256"]:
            raise RuntimeError(f"frozen artifact changed: {path}")


def _qwen_cache_root() -> Path:
    pointer = read_json(FEATURE_POINTER)
    return ROOT / pointer["cache_root"]


def load_multi_image_exclusions() -> list[dict[str, Any]]:
    cache_root = _qwen_cache_root()
    records: list[dict[str, Any]] = []
    paths = sorted(cache_root.glob("exclusions_rank_*.json"))
    if len(paths) != 8:
        raise RuntimeError(f"expected eight exclusion manifests, found {len(paths)}")
    for path in paths:
        payload = read_json(path)
        if payload.get("backbone") != "qwen3vl":
            raise RuntimeError(f"unexpected exclusion backbone: {path}")
        records.extend(payload["records"])
    records.sort(key=lambda row: int(row["row_index"]))
    split_counts = Counter(str(row["split"]) for row in records)
    if len(records) != 47 or dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"unexpected multi-image exclusion population: {split_counts}")
    if any(row.get("error_type") != "multi_image" for row in records):
        raise RuntimeError("exclusion set contains a non-multi-image row")
    if len({str(row["sample_id"]) for row in records}) != len(records):
        raise RuntimeError("duplicate multi-image sample ID")
    return records


def load_mmmu_benchmark_records() -> tuple[dict[str, dict[str, Any]], Path]:
    path = ROOT / "BENCHMARKS/vqa_mc/mmmu_dev_val_samples.jsonl"
    values: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            uid = str(value["sample_id"])
            if uid in values:
                raise RuntimeError(f"duplicate benchmark UID at line {line_number}: {uid}")
            values[uid] = value
    return values, path


def sample_from_record(exclusion: dict[str, Any], benchmark: dict[str, Any]) -> StrictSample:
    uid = str(exclusion["sample_id"])
    if str(benchmark.get("sample_id")) != uid or str(benchmark.get("dataset")) != "MMMU_DEV_VAL":
        raise RuntimeError(f"benchmark/exclusion mismatch for {uid}")
    assets = tuple(AssetSpec.from_mapping(value) for value in benchmark["assets"])
    if len(assets) != int(exclusion["image_count"]) or len(assets) <= 1:
        raise RuntimeError(f"asset count mismatch for {uid}")
    return StrictSample(
        sample_id=uid,
        row_index=int(exclusion["row_index"]),
        dataset="MMMU_DEV_VAL",
        task_type=str(benchmark["task_type"]),
        text=str(benchmark["prompt"]),
        assets=assets,
        split=str(exclusion["split"]),
        y_row=(),
        c_row=(),
    )


class MultiImageTSVResolver:
    """Resolve every image in an official TSV list without resizing."""

    def __init__(self, root: Path):
        self.index = TSVRowIndex(root / "vlm_router_data/TSV_images/MMMU_DEV_VAL.tsv", "id")

    @staticmethod
    def _tail(uid: str) -> str:
        return uid.split("/", 1)[1]

    def _dereference(self, value: str, *, uid: str) -> str:
        current = value.strip()
        seen: set[str] = set()
        for _ in range(9):
            if len(current) > 64 or not self.index.contains(current):
                return current
            if current in seen:
                raise RuntimeError(f"cyclic TSV image reference for {uid}: {current}")
            seen.add(current)
            _, row = self.index.row_by_key(current)
            current = str(row.get("image", "")).strip()
        raise RuntimeError(f"TSV image reference depth exceeded for {uid}")

    @staticmethod
    def _decode(value: str, *, uid: str, image_index: int) -> tuple[Image.Image, dict[str, Any]]:
        encoded = value.strip()
        if encoded.startswith("data:"):
            if "," not in encoded:
                raise RuntimeError(f"malformed data URI for {uid} image {image_index}")
            encoded = encoded.split(",", 1)[1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(f"invalid base64 for {uid} image {image_index}: {exc}") from exc
        try:
            with Image.open(io.BytesIO(raw)) as loaded:
                loaded.load()
                size = tuple(map(int, loaded.size))
                mode = str(loaded.mode)
                image = loaded.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"invalid image for {uid} image {image_index}: {exc}") from exc
        return image, {"image_index": image_index, "original_size": list(size), "original_mode": mode}

    def resolve(self, sample: StrictSample) -> tuple[list[Image.Image], list[dict[str, Any]]]:
        _, row = self.index.row_by_key(self._tail(sample.sample_id))
        raw = str(row.get("image", "")).strip()
        try:
            values = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(f"invalid TSV image list for {sample.sample_id}: {exc}") from exc
        if not isinstance(values, (list, tuple)) or len(values) != len(sample.assets):
            raise RuntimeError(f"TSV/BENCHMARK image-count mismatch for {sample.sample_id}")
        images: list[Image.Image] = []
        metadata: list[dict[str, Any]] = []
        for image_index, value in enumerate(values, 1):
            resolved = self._dereference(str(value), uid=sample.sample_id)
            image, item = self._decode(resolved, uid=sample.sample_id, image_index=image_index)
            images.append(image)
            metadata.append(item)
        return images, metadata


def encode_native_multi_image(
    adapter: Qwen3VLFeatureExtractor,
    resolver: MultiImageTSVResolver,
    sample: StrictSample,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    images, image_metadata = resolver.resolve(sample)
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": sample.text})
    preprocess_started = time.perf_counter()
    inputs = adapter.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(adapter.device)
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
    preprocess_seconds = time.perf_counter() - preprocess_started
    forward_started = time.perf_counter()
    with torch.inference_mode():
        output = adapter.model.model(**inputs, return_dict=True)
        attention_mask = inputs["attention_mask"]
        weights = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        pooled = (output.last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
    forward_seconds = time.perf_counter() - forward_started
    if tuple(pooled.shape) != (1, FEATURE_DIM) or not torch.isfinite(pooled).all():
        raise RuntimeError(f"invalid pooled feature for {sample.sample_id}: {tuple(pooled.shape)}")
    grid = inputs.get("image_grid_thw")
    if grid is None or int(grid.shape[0]) != len(images):
        raise RuntimeError(f"processor did not preserve every image for {sample.sample_id}")
    visual_tokens = int(grid.prod(dim=-1).sum().item() // 4)
    metadata = {
        "sample_id": sample.sample_id,
        "row_index": sample.row_index,
        "split": sample.split,
        "image_count": len(images),
        "images": image_metadata,
        "image_grid_thw": grid.detach().cpu().tolist(),
        "sequence_tokens": int(attention_mask.sum().item()),
        "visual_tokens": visual_tokens,
        "pooling_positions": int(attention_mask.sum().item()),
        "preprocess_seconds": preprocess_seconds,
        "forward_seconds": forward_seconds,
    }
    return pooled.float().cpu().numpy()[0], metadata


def extract(device: str) -> None:
    if os.environ.get("RUN_API_REGEN") != "0":
        raise RuntimeError("RUN_API_REGEN must remain 0")
    feature_path = OUT / "multi_image_features.npz"
    manifest_path = OUT / "FEATURE_MANIFEST.json"
    if feature_path.exists() or manifest_path.exists():
        raise RuntimeError("multi-image feature artifacts already exist")
    exclusions = load_multi_image_exclusions()
    benchmark, benchmark_path = load_mmmu_benchmark_records()
    samples = [sample_from_record(row, benchmark[str(row["sample_id"])]) for row in exclusions]
    contract = load_contract(CONTRACT_PATH)
    adapter = Qwen3VLFeatureExtractor(contract, device=device, dataset_root=ROOT)
    if adapter.feature_dim != FEATURE_DIM:
        raise RuntimeError(f"unexpected Qwen3-VL feature dimension: {adapter.feature_dim}")
    resolver = MultiImageTSVResolver(ROOT)
    features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, sample in enumerate(samples, 1):
        feature, item = encode_native_multi_image(adapter, resolver, sample)
        features.append(feature)
        metadata.append(item)
        say(stage="extract", position=position, total=len(samples), uid=sample.sample_id,
            images=len(sample.assets), tokens=item["sequence_tokens"], seconds=item["forward_seconds"])
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape != (47, FEATURE_DIM) or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid multi-image feature matrix: {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1)
    if (norms <= 1e-12).any():
        raise RuntimeError("zero multi-image feature vector")
    _atomic_npz(
        feature_path,
        uids=np.asarray([sample.sample_id for sample in samples]),
        row_indices=np.asarray([sample.row_index for sample in samples], dtype=np.int64),
        splits=np.asarray([sample.split for sample in samples]),
        image_counts=np.asarray([len(sample.assets) for sample in samples], dtype=np.int64),
        features=matrix,
    )
    pointer = read_json(FEATURE_POINTER)
    manifest = {
        "kind": "stateroute_qwen3vl_native_multi_image_feature_overlay",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": len(samples),
        "feature_dim": FEATURE_DIM,
        "dtype": str(matrix.dtype),
        "finite": True,
        "minimum_l2_norm": float(norms.min()),
        "maximum_l2_norm": float(norms.max()),
        "split_counts": dict(Counter(sample.split for sample in samples)),
        "image_count_histogram": {str(k): v for k, v in sorted(Counter(len(sample.assets) for sample in samples).items())},
        "model": "Qwen3-VL-2B-Instruct",
        "device": device,
        "offline": True,
        "expert_or_api_calls": 0,
        "input": "all RGB images in original TSV order, followed by the unchanged sample text",
        "processor": "frozen native AutoProcessor chat template with add_generation_prompt=True",
        "pooling": "attention-mask mean of Qwen3-VL multimodal final hidden states",
        "dtype_forward": "bfloat16",
        "old_single_image_cache_untouched": True,
        "old_cache_root": pointer["cache_root"],
        "contract_sha256": contract.sha256,
        "seconds": time.perf_counter() - started,
        "per_sample": metadata,
        "sources": [record(FEATURE_POINTER), record(benchmark_path), record(CONTRACT_PATH)],
        "feature_file": record(feature_path),
        "implementation": record(Path(__file__)),
    }
    write_json(manifest_path, manifest)
    say(stage="extract-complete", rows=len(samples), seconds=manifest["seconds"], output=str(feature_path))


def model_path(k: int) -> Path:
    if k == 1:
        root = ROOT / "outputs/stateroute_k1_20260918"
    elif k in (2, 4, 6, 8):
        root = ROOT / "outputs/stateroute_k_sweep_20260916"
    elif k in (10, 12):
        root = ROOT / "outputs/stateroute_k10_k12_20260916"
    else:
        raise KeyError(k)
    return root / f"models/MM/k{k}/seed-{SEED}/model.npz"


def baseline_prediction_path(k: int) -> Path:
    if k == 1:
        return ROOT / "outputs/stateroute_k1_20260918/predictions.npz"
    if k in (2, 4, 6, 8):
        return ROOT / "outputs/stateroute_k_sweep_20260916/predictions.npz"
    return ROOT / "outputs/stateroute_k10_k12_20260916/predictions.npz"


def baseline_route(k: int, uids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(baseline_prediction_path(k), allow_pickle=False)
    if not np.array_equal(archive["uids"], uids):
        raise RuntimeError(f"baseline UID mismatch at K={k}")
    target = f"StateRoute-MM-K{k}"
    matches = np.flatnonzero((archive["methods"] == target) & (archive["seeds"] == SEED))
    if len(matches) != 1:
        raise RuntimeError(f"baseline route missing or duplicated at K={k}")
    index = int(matches[0])
    return np.asarray(archive["actions"][index], dtype=np.int64), np.asarray(archive["states"][index], dtype=np.int64)


def predict() -> None:
    from src.next_round_x.read_fence import install

    if (OUT / "PREDICTION_RECEIPT.json").exists():
        raise RuntimeError("multi-image predictions are already frozen")
    feature_manifest = read_json(OUT / "FEATURE_MANIFEST.json")
    _verify_records(feature_manifest["sources"] + [feature_manifest["feature_file"], feature_manifest["implementation"]])
    denied = install()
    uids = np.asarray(role_uids("heldout11", "id") + role_uids("heldout11", "ood"))
    if uids.shape != (6108,) or len(set(uids.tolist())) != len(uids):
        raise RuntimeError("unexpected held-out test universe")
    base_features, original_available = FeatureCache().rows(uids.tolist())
    if (
        int((~original_available).sum()) != 10
        or int((~original_available[:5341]).sum()) != 0
        or int((~original_available[5341:]).sum()) != 10
    ):
        raise RuntimeError("original missing-feature pattern changed")
    overlay = np.load(OUT / "multi_image_features.npz", allow_pickle=False)
    overlay_position = {str(uid): i for i, uid in enumerate(overlay["uids"])}
    missing_uids = uids[~original_available].tolist()
    if any(uid not in overlay_position for uid in missing_uids):
        raise RuntimeError("multi-image overlay does not cover every test exclusion")
    complete = np.asarray(base_features, dtype=np.float32).copy()
    for row in np.flatnonzero(~original_available):
        complete[row] = overlay["features"][overlay_position[str(uids[row])]]
    if not np.isfinite(complete).all() or (np.linalg.norm(complete, axis=1) <= 1e-12).any():
        raise RuntimeError("completed test feature matrix is invalid")
    nx = normalized(complete)
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    baseline_actions: list[np.ndarray] = []
    baseline_states: list[np.ndarray] = []
    methods: list[str] = []
    model_records: list[dict[str, Any]] = []
    for k in KS:
        path = model_path(k)
        artifact = np.load(path, allow_pickle=False)
        centers = np.asarray(artifact["centers"], dtype=np.float32)
        distances = (
            np.square(nx).sum(axis=1)[:, None]
            + np.square(centers).sum(axis=1)[None, :]
            - 2.0 * nx @ centers.T
        )
        assigned = distances.argmin(axis=1).astype(np.int64)
        selected = np.asarray(artifact["table"], dtype=np.int64)[assigned]
        old_actions, old_states = baseline_route(k, uids)
        if not np.array_equal(selected[original_available], old_actions[original_available]):
            raise RuntimeError(f"non-excluded actions changed at K={k}")
        if not np.array_equal(assigned[original_available], old_states[original_available]):
            raise RuntimeError(f"non-excluded states changed at K={k}")
        actions.append(selected)
        states.append(assigned)
        baseline_actions.append(old_actions)
        baseline_states.append(old_states)
        methods.append(f"StateRoute-MM-MultiImage-K{k}")
        model_records.append(record(path))
        say(stage="predict-route", k=k,
            changed_missing=int((selected[~original_available] != old_actions[~original_available]).sum()),
            experts=int(len(set(selected.tolist()))))
    plan = {
        "kind": "stateroute_native_multi_image_posthoc_single_seed_validation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "heldout11; same frozen source-fitted router for 11-ID and 3-OOD",
        "k_values": list(KS),
        "seed": SEED,
        "runs_per_k": 1,
        "test_selection": False,
        "model_refit": False,
        "reason_model_refit_not_needed": "heldout11 source train/dev contain no missing Qwen3-VL rows; only ten OOD MMMU test rows were excluded",
        "changed_component": "prediction-time Qwen3-VL features for the original multi-image exclusions",
        "unchanged": ["source states", "profiles", "policy tables", "experts", "test denominator", "metrics"],
        "cost_scope": "selected expert USD proxy; Qwen3-VL feature extraction overhead excluded as in the original experiment",
        "historical_exposure": "post-hoc exploratory correction after original K=1..12 outcomes were exposed",
        "user_request": "改成可以处理多张图像，然后给出各种K的指标结果；此前指定只跑一次",
        "feature_manifest": record(OUT / "FEATURE_MANIFEST.json"),
        "models": model_records,
        "baseline_predictions": [record(path) for path in sorted(set(map(baseline_prediction_path, KS)))],
        "implementation": record(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    write_json(OUT / "EXPERIMENT_PLAN.json", plan)
    lock = {
        "kind": "stateroute_multi_image_model_lock",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "k_values": list(KS),
        "models_reused_without_refit": True,
        "files": [plan["feature_manifest"], plan["implementation"], *model_records, *plan["baseline_predictions"]],
    }
    write_json(OUT / "MODEL_LOCK.json", lock)
    prediction_path = OUT / "predictions.npz"
    _atomic_npz(
        prediction_path,
        uids=uids,
        methods=np.asarray(methods),
        seeds=np.full(len(KS), SEED, dtype=np.int64),
        k_values=np.asarray(KS, dtype=np.int64),
        views=np.asarray(["11-ID"] * 5341 + ["3-OOD"] * 767),
        actions=np.stack(actions),
        states=np.stack(states),
        baseline_actions=np.stack(baseline_actions),
        baseline_states=np.stack(baseline_states),
        original_available=original_available,
        multi_image_resolved=~original_available,
    )
    receipt = {
        "kind": "subject_routing_final_evaluation_unlock",
        "target_split": "test",
        "predictions_frozen": True,
        "method_selection_complete": True,
        "approved": True,
        "approval_source": "user requested native multi-image StateRoute K=1..12 single-run validation",
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": digest(prediction_path),
        "methods": methods,
        "seeds": [SEED] * len(KS),
        "k_values": list(KS),
        "test_outcomes_opened": 0,
        "labels_or_actual_cost_used": False,
        "read_fence_denied_attempts": denied,
        "model_lock": record(OUT / "MODEL_LOCK.json"),
        "test_fit_performed": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(OUT / "PREDICTION_RECEIPT.json", receipt)
    say(stage="predict-complete", routes=len(KS), rows=len(uids), missing_before=10, missing_after=0)


def scores(y: np.ndarray, c: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    rows = np.arange(len(actions))
    quality = np.asarray(y[rows, actions], dtype=np.float64)
    costs = np.asarray(c[rows, actions], dtype=np.float64)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    return {
        "n": len(actions),
        "accuracy_percent": float(quality.mean() * 100.0),
        "cost_usd_per_10k": float(costs.mean() * 10_000.0),
        "rank_score": float(100.0 * rank_score(float(quality.mean()), float(costs.mean()), bounds["cmin"], bounds["cmax"])),
        "correct_count": int(quality.sum()),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def evaluate() -> None:
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock

    if (REPORT / "RESULTS.json").exists():
        raise RuntimeError("multi-image evaluation already exists")
    lock = read_json(OUT / "MODEL_LOCK.json")
    _verify_records(lock["files"])
    prediction_path = OUT / "predictions.npz"
    receipt_path = OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(prediction_path) != receipt["prediction_sha256"]:
        raise RuntimeError("frozen prediction hash mismatch")
    prediction = np.load(prediction_path, allow_pickle=False)
    uids = prediction["uids"].tolist()
    if uids != role_uids("heldout11", "id") + role_uids("heldout11", "ood"):
        raise RuntimeError("prediction UID order mismatch")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids,
            allowed_test_uids=uids,
            prediction_sha256=digest(prediction_path),
            prediction_receipt_sha256=digest(receipt_path),
            audit=audit,
            unlock_capability=capability,
            **base.label_args(),
        )
    write_json(OUT / "OUTCOME_PROVENANCE.json", {"provenance": provenance, "events": audit.events})
    y = np.asarray(labels.matrix.y)
    c = np.asarray(labels.matrix.c)
    views = prediction["views"]
    datasets = np.asarray([uid.split("/", 1)[0] for uid in uids])
    experts = manifests()[1]["canonical_order"]
    summary: list[dict[str, Any]] = []
    per_dataset: list[dict[str, Any]] = []
    affected: list[dict[str, Any]] = []
    quality_rows: list[np.ndarray] = []
    cost_rows: list[np.ndarray] = []
    missing = ~prediction["original_available"]
    for index, k_value in enumerate(prediction["k_values"]):
        k = int(k_value)
        actions = np.asarray(prediction["actions"][index], dtype=np.int64)
        baseline = np.asarray(prediction["baseline_actions"][index], dtype=np.int64)
        states = np.asarray(prediction["states"][index], dtype=np.int64)
        quality_rows.append(y[np.arange(len(uids)), actions])
        cost_rows.append(c[np.arange(len(uids)), actions])
        for view in ("11-ID", "3-OOD"):
            mask = views == view
            new_score = scores(y[mask], c[mask], actions[mask])
            old_score = scores(y[mask], c[mask], baseline[mask])
            row: dict[str, Any] = {
                "method": f"StateRoute-MM-MultiImage-K{k}",
                "family": "MM",
                "k": k,
                "seed": SEED,
                "runs": 1,
                "view": view,
                **new_score,
                **{f"baseline_{name}": old_score[name] for name in (*METRICS, "correct_count")},
                **{f"delta_{name}": new_score[name] - old_score[name] for name in METRICS},
                "changed_actions": int((actions[mask] != baseline[mask]).sum()),
                "resolved_multi_image_rows": int(missing[mask].sum()),
            }
            summary.append(row)
        for dataset in sorted(set(datasets.tolist())):
            mask = datasets == dataset
            new_score = scores(y[mask], c[mask], actions[mask])
            old_score = scores(y[mask], c[mask], baseline[mask])
            per_dataset.append({
                "method": f"StateRoute-MM-MultiImage-K{k}",
                "k": k,
                "seed": SEED,
                "dataset": str(dataset),
                **new_score,
                **{f"baseline_{name}": old_score[name] for name in (*METRICS, "correct_count")},
                **{f"delta_{name}": new_score[name] - old_score[name] for name in METRICS},
                "changed_actions": int((actions[mask] != baseline[mask]).sum()),
            })
        for row in np.flatnonzero(missing):
            old_action = int(baseline[row])
            new_action = int(actions[row])
            affected.append({
                "k": k,
                "seed": SEED,
                "uid": str(uids[row]),
                "state": int(states[row]),
                "baseline_expert": experts[old_action],
                "multi_image_expert": experts[new_action],
                "action_changed": new_action != old_action,
                "baseline_correct": int(y[row, old_action]),
                "multi_image_correct": int(y[row, new_action]),
                "baseline_cost_usd_per_10k": float(c[row, old_action] * 10_000.0),
                "multi_image_cost_usd_per_10k": float(c[row, new_action] * 10_000.0),
            })
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "summary.csv", summary)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    write_csv(REPORT / "affected_samples.csv", affected)
    _atomic_npz(
        OUT / "evaluation_traces.npz",
        uids=prediction["uids"],
        methods=prediction["methods"],
        seeds=prediction["seeds"],
        k_values=prediction["k_values"],
        views=views,
        actions=prediction["actions"],
        states=prediction["states"],
        correct=np.stack(quality_rows),
        cost=np.stack(cost_rows),
    )
    result = {
        "kind": "stateroute_native_multi_image_single_seed_results",
        "summary": summary,
        "multi_image_rows_total": 47,
        "multi_image_test_rows": 10,
        "seed": SEED,
        "runs_per_k": 1,
        "model_refit": False,
        "posthoc": True,
        "encoder_cost_included": False,
        "prediction": record(prediction_path),
        "receipt": record(receipt_path),
        "plan": record(OUT / "EXPERIMENT_PLAN.json"),
        "feature_manifest": record(OUT / "FEATURE_MANIFEST.json"),
        "outcome_provenance": record(OUT / "OUTCOME_PROVENANCE.json"),
    }
    write_json(REPORT / "RESULTS.json", result)
    checks = {
        "feature_rows": 47,
        "feature_split_counts": read_json(OUT / "FEATURE_MANIFEST.json")["split_counts"],
        "original_missing_test_rows": int(missing.sum()),
        "remaining_missing_test_rows": 0,
        "id_missing_rows": int(missing[:5341].sum()),
        "ood_resolved_rows": int(missing[5341:].sum()),
        "routes": int(len(prediction["k_values"])),
        "all_states_assigned": bool((prediction["states"] >= 0).all()),
        "id_actions_unchanged": bool(np.array_equal(prediction["actions"][:, :5341], prediction["baseline_actions"][:, :5341])),
        "all_nonexcluded_actions_unchanged": bool(np.array_equal(prediction["actions"][:, ~missing], prediction["baseline_actions"][:, ~missing])),
        "prediction_frozen_before_label_access": True,
        "source_models_reused_without_refit": True,
        "single_seed": True,
        "finite_metrics": bool(all(math.isfinite(float(row[name])) for row in summary for name in METRICS)),
    }
    status = "PASS" if (
        checks["feature_rows"] == 47
        and checks["feature_split_counts"] == EXPECTED_SPLIT_COUNTS
        and checks["original_missing_test_rows"] == 10
        and checks["remaining_missing_test_rows"] == 0
        and checks["id_missing_rows"] == 0
        and checks["ood_resolved_rows"] == 10
        and checks["routes"] == len(KS)
        and all(checks[name] is True for name in (
            "all_states_assigned", "id_actions_unchanged", "all_nonexcluded_actions_unchanged",
            "prediction_frozen_before_label_access", "source_models_reused_without_refit",
            "single_seed", "finite_metrics"))
    ) else "FAIL"
    validation = {
        "status": status,
        **checks,
        "files": [record(path) for path in (
            Path(__file__), OUT / "FEATURE_MANIFEST.json", OUT / "multi_image_features.npz",
            OUT / "EXPERIMENT_PLAN.json", OUT / "MODEL_LOCK.json", prediction_path,
            receipt_path, OUT / "OUTCOME_PROVENANCE.json", OUT / "evaluation_traces.npz",
            REPORT / "RESULTS.json", REPORT / "summary.csv", REPORT / "per_dataset.csv",
            REPORT / "affected_samples.csv")],
    }
    write_json(REPORT / "VALIDATION.json", validation)
    if status != "PASS":
        raise RuntimeError(f"validation failed: {checks}")
    id_rows = [row for row in summary if row["view"] == "11-ID"]
    ood_rows = [row for row in summary if row["view"] == "3-OOD"]
    lines = [
        "# StateRoute-MM 原生多图特征实验（2026-09-18）",
        "",
        "本实验把旧单图合同排除的 47 条 MMMU 多图样本全部送入同一个冻结的 Qwen3-VL-2B，按原 TSV 顺序同时输入全部图片，并沿用原聊天模板、BF16 前向和注意力掩码均值池化。held-out 11→3 测试中涉及 10 条 OOD MMMU 样本；补齐后 6,108 条测试样本均有图文特征。",
        "",
        "按用户要求，每个 K 只运行一次（MM seed=0）。聚类中心、画像和状态到专家映射均复用原源域模型，没有用测试结果重新训练或选参。以下 C 为所选专家的 USD/10k 请求代理，不包含 Qwen3-VL 路由特征抽取开销。",
        "",
        "## 完整结果",
        "",
        "| K | ID A (%) | ID C | ID Rank | OOD A (%) | OOD C | OOD Rank | OOD ΔA | OOD ΔC | OOD ΔRank | 10 条中改路由数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for id_row, ood_row in zip(id_rows, ood_rows):
        lines.append(
            f"| {id_row['k']} | {_fmt(id_row['accuracy_percent'])} | {_fmt(id_row['cost_usd_per_10k'], 6)} | {_fmt(id_row['rank_score'])} "
            f"| {_fmt(ood_row['accuracy_percent'])} | {_fmt(ood_row['cost_usd_per_10k'], 6)} | {_fmt(ood_row['rank_score'])} "
            f"| {ood_row['delta_accuracy_percent']:+.4f} | {ood_row['delta_cost_usd_per_10k']:+.6f} | {ood_row['delta_rank_score']:+.4f} | {ood_row['changed_actions']} |"
        )
    lines += [
        "",
        "OOD 的 Δ 列均以同一 K、同一 seed=0 的旧回退结果为基准。ID 没有多图特征缺口，所以所有 ID 指标及逐题动作与原结果完全一致。",
        "",
        "## 解释边界",
        "",
        "这是已观察原结果之后进行的单次修正实验，不能据此给出 seed 方差或置信区间。多图样本只占 OOD 的 10/767，整体变化只来自这 10 条；其余 6,098 条测试样本的状态和专家动作逐项不变。",
        "",
        "[汇总 CSV](summary.csv) · [逐数据集结果](per_dataset.csv) · [10 条样本逐 K 路由](affected_samples.csv) · [完整结果 JSON](RESULTS.json) · [校验](VALIDATION.json)",
        "",
        "实现与冻结产物：",
        "",
        f"- 脚本：`{Path(__file__).relative_to(ROOT)}`",
        f"- 多图特征：`{(OUT / 'multi_image_features.npz').relative_to(ROOT)}`",
        f"- 冻结预测：`{prediction_path.relative_to(ROOT)}`",
    ]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Include README in a final, self-consistent validation manifest.
    validation = read_json(REPORT / "VALIDATION.json")
    validation["files"].append(record(REPORT / "README.md"))
    write_json(REPORT / "VALIDATION.json", validation)
    for row in summary:
        say(stage="result", k=row["k"], view=row["view"],
            accuracy=row["accuracy_percent"], cost=row["cost_usd_per_10k"], rank=row["rank_score"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("extract", "predict", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.device)
    elif args.stage == "predict":
        predict()
    else:
        evaluate()


if __name__ == "__main__":
    main()
