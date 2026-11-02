"""Independent offline evaluator for already-frozen prediction JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from routers.utils.rank_score import rank_score

from .config import load_contract
from .data import load_vlrb_strict
from .errors import PredictionIntegrityError
from .labels import IGNORE_INDEX, build_diagnostic_labels_for_split, compute_train_expert_accuracy
from .schemas import StrictVLRouterData
from .target_free import (
    deny_target_matrix_access,
    load_and_validate_target_free_receipt,
    load_target_free_vlrb,
)


REQUIRED_PREDICTION_FIELDS = {
    "sample_id",
    "row_index",
    "predicted_expert_index",
    "predicted_expert_id",
    "split",
    "seed",
    "checkpoint_id",
    "logits",
}
FALLBACK_METADATA_FIELDS = {
    "asset_error_type",
    "image_count",
    "fallback_policy",
    "evaluation_denominator",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_frozen_predictions(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise PredictionIntegrityError(f"prediction file does not exist: {prediction_path}")
    digest_before = _sha256(prediction_path)
    records: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise PredictionIntegrityError(f"blank prediction line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictionIntegrityError(f"invalid prediction JSON at line {line_number}: {exc}") from exc
            missing = REQUIRED_PREDICTION_FIELDS.difference(record)
            if missing:
                raise PredictionIntegrityError(
                    f"prediction line {line_number} missing fields: {sorted(missing)}"
                )
            records.append(record)
    if not records:
        raise PredictionIntegrityError("prediction file is empty")
    if _sha256(prediction_path) != digest_before:
        raise PredictionIntegrityError("prediction file changed while being read")
    return records, digest_before


def validate_predictions(
    records: list[dict[str, Any]],
    data: StrictVLRouterData,
    *,
    expected_split: str | None = None,
    expected_sample_ids: Iterable[str] | None = None,
    expected_seed: int | None = None,
    expected_checkpoint_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [sample for sample, count in Counter(sample_ids).items() if count > 1]
        raise PredictionIntegrityError(f"duplicate prediction sample IDs: {duplicates[:3]}")
    observed_splits = {str(record["split"]) for record in records}
    if len(observed_splits) != 1:
        raise PredictionIntegrityError(f"prediction file mixes splits: {sorted(observed_splits)}")
    split = next(iter(observed_splits))
    if split not in data.split_to_rows:
        raise PredictionIntegrityError(f"unknown prediction split: {split}")
    if expected_split is not None and split != expected_split:
        raise PredictionIntegrityError(f"expected split {expected_split}, observed {split}")

    expected_ids = (
        {str(item) for item in expected_sample_ids}
        if expected_sample_ids is not None
        else {data.samples[row].sample_id for row in data.split_to_rows[split]}
    )
    if set(sample_ids) != expected_ids:
        raise PredictionIntegrityError(
            f"prediction coverage mismatch for {split}: missing={len(expected_ids.difference(sample_ids))}, extra={len(set(sample_ids).difference(expected_ids))}"
        )

    rows = np.empty(len(records), dtype=np.int64)
    predictions = np.empty(len(records), dtype=np.int64)
    observed_seeds: set[int] = set()
    observed_checkpoint_ids: set[str] = set()
    for position, record in enumerate(records):
        sample_id = str(record["sample_id"])
        canonical_row = data.sample_to_row.get(sample_id)
        if canonical_row is None:
            raise PredictionIntegrityError(f"unknown sample ID: {sample_id}")
        row = int(record["row_index"])
        if row != canonical_row:
            raise PredictionIntegrityError(
                f"row mismatch for {sample_id}: prediction={row}, registry={canonical_row}"
            )
        column = int(record["predicted_expert_index"])
        if column < 0 or column >= len(data.model_order):
            raise PredictionIntegrityError(f"expert column out of range for {sample_id}: {column}")
        if str(record["predicted_expert_id"]) != data.model_order[column]:
            raise PredictionIntegrityError(f"expert ID/column mismatch for {sample_id}")
        if data.samples[row].split != split:
            raise PredictionIntegrityError(f"sample {sample_id} is not in claimed split {split}")
        if not str(record["checkpoint_id"]):
            raise PredictionIntegrityError(f"empty checkpoint_id for {sample_id}")
        seed_value = record["seed"]
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            raise PredictionIntegrityError(f"invalid seed for {sample_id}: {seed_value!r}")
        checkpoint_id = str(record["checkpoint_id"])
        try:
            logits = np.asarray(record["logits"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PredictionIntegrityError(f"invalid logits for {sample_id}") from exc
        if logits.shape != (len(data.model_order),) or not np.isfinite(logits).all():
            raise PredictionIntegrityError(
                f"logits must be finite and {len(data.model_order)}-wide for {sample_id}"
            )
        if int(np.argmax(logits)) != column:
            raise PredictionIntegrityError(f"predicted expert is not logits argmax for {sample_id}")
        observed_seeds.add(seed_value)
        observed_checkpoint_ids.add(checkpoint_id)
        rows[position] = row
        predictions[position] = column
    if len(observed_seeds) != 1:
        raise PredictionIntegrityError(f"prediction file mixes seeds: {sorted(observed_seeds)}")
    if expected_seed is not None and observed_seeds != {int(expected_seed)}:
        raise PredictionIntegrityError(
            f"prediction seed mismatch: expected={expected_seed}, observed={sorted(observed_seeds)}"
        )
    if len(observed_checkpoint_ids) != 1:
        raise PredictionIntegrityError("prediction file mixes checkpoint IDs")
    if expected_checkpoint_id is not None and observed_checkpoint_ids != {str(expected_checkpoint_id)}:
        raise PredictionIntegrityError("prediction checkpoint ID does not match expected checkpoint bytes")
    return rows, predictions, split


def _usage(predictions: np.ndarray, model_order: tuple[str, ...]) -> dict[str, Any]:
    counts = np.bincount(predictions, minlength=len(model_order)).astype(np.int64)
    shares = counts / len(predictions)
    positive = shares[shares > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    normalized = entropy / math.log(len(model_order)) if len(model_order) > 1 else 0.0
    ordered = np.sort(shares)
    gini = float((2 * np.arange(1, len(ordered) + 1) - len(ordered) - 1) @ ordered / len(ordered))
    return {
        "counts": {model: int(counts[index]) for index, model in enumerate(model_order)},
        "shares": {model: float(shares[index]) for index, model in enumerate(model_order)},
        "max_share": float(shares.max()),
        "entropy": entropy,
        "normalized_entropy": normalized,
        "gini": gini,
    }


def _cost_bounds(data: StrictVLRouterData) -> tuple[float, float, float]:
    path = data.dataset_root / "data/matrices/cost_bounds.json"
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return float(value["cmin"]), float(value["cmax"]), float(value.get("beta", 0.1))


def expected_multi_image_fallback_sample_ids(
    data: StrictVLRouterData,
    sample_ids: Iterable[str],
) -> tuple[str, ...]:
    """Derive the expected VLM fallback set independently of prediction annotations."""

    expected = []
    for sample_id in sample_ids:
        value = str(sample_id)
        row = data.sample_to_row.get(value)
        if row is None:
            raise PredictionIntegrityError(f"unknown expected fallback candidate: {value}")
        if len(data.samples[row].assets) != 1:
            expected.append(value)
    return tuple(sorted(expected))


def validate_structured_fallback(
    records: list[dict[str, Any]],
    data: StrictVLRouterData,
    rows: np.ndarray,
    predictions: np.ndarray,
    *,
    expected_fallback_sample_ids: Iterable[str],
) -> tuple[np.ndarray, int]:
    """Require an exact expected fallback set, including the empty-set case."""

    expected_values = tuple(str(item) for item in expected_fallback_sample_ids)
    if len(expected_values) != len(set(expected_values)):
        raise PredictionIntegrityError("expected fallback sample IDs contain duplicates")
    expected = set(expected_values)
    evaluated = {str(record["sample_id"]) for record in records}
    outside = expected.difference(evaluated)
    if outside:
        raise PredictionIntegrityError(
            f"expected fallback IDs are outside prediction coverage: {sorted(outside)[:3]}"
        )
    fallback_mask = np.asarray(
        [record.get("routing_input_status") == "structured_asset_error_fallback" for record in records],
        dtype=bool,
    )
    observed = {
        str(records[position]["sample_id"]) for position in np.flatnonzero(fallback_mask)
    }
    if observed != expected:
        raise PredictionIntegrityError(
            "structured fallback sample-ID set mismatch: "
            f"missing={len(expected.difference(observed))}, extra={len(observed.difference(expected))}"
        )

    train_accuracy = compute_train_expert_accuracy(
        data.y,
        data.split_to_rows["train"],
        data.model_order,
        expected_model_order=data.model_order,
    )
    strongest_column = int(np.argmax(train_accuracy))
    expected_logits = np.full(len(data.model_order), -1.0, dtype=np.float64)
    expected_logits[strongest_column] = 1.0
    for position, record in enumerate(records):
        if not fallback_mask[position]:
            unexpected = FALLBACK_METADATA_FIELDS.intersection(record)
            if unexpected:
                raise PredictionIntegrityError(
                    f"non-fallback row carries fallback metadata for {record['sample_id']}: {sorted(unexpected)}"
                )
            continue
        row = int(rows[position])
        sample = data.samples[row]
        if len(sample.assets) == 1:
            raise PredictionIntegrityError(
                f"single-image row is incorrectly marked as fallback: {sample.sample_id}"
            )
        required_metadata = {
            "asset_error_type": "multi_image",
            "image_count": len(sample.assets),
            "fallback_policy": "train_strongest_global",
            "evaluation_denominator": "official_split_frozen",
        }
        mismatched = {
            key: {"expected": value, "observed": record.get(key)}
            for key, value in required_metadata.items()
            if record.get(key) != value
        }
        if mismatched:
            raise PredictionIntegrityError(
                f"structured fallback metadata is invalid for {sample.sample_id}: {mismatched}"
            )
        if int(predictions[position]) != strongest_column:
            raise PredictionIntegrityError(
                "structured fallback is not the train-only StrongestGlobal expert"
            )
        logits = np.asarray(record["logits"], dtype=np.float64)
        if not np.array_equal(logits, expected_logits):
            raise PredictionIntegrityError(
                f"structured fallback logits are not the approved deterministic logits for {sample.sample_id}"
            )
    return fallback_mask, strongest_column


def evaluate_records(
    records: list[dict[str, Any]],
    data: StrictVLRouterData,
    *,
    expected_fallback_sample_ids: Iterable[str],
    expected_split: str | None = None,
    expected_sample_ids: Iterable[str] | None = None,
    expected_seed: int | None = None,
    expected_checkpoint_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, predictions, split = validate_predictions(
        records,
        data,
        expected_split=expected_split,
        expected_sample_ids=expected_sample_ids,
        expected_seed=expected_seed,
        expected_checkpoint_id=expected_checkpoint_id,
    )
    positions = np.arange(len(rows), dtype=np.int64)
    selected_correct = data.y[rows, predictions].astype(np.int8)
    selected_cost = data.c[rows, predictions].astype(np.float64)
    routed_accuracy = float(selected_correct.mean())
    average_cost = float(selected_cost.mean())
    cmin, cmax, beta = _cost_bounds(data)
    official_rank_score = float(rank_score(routed_accuracy, average_cost, cmin, cmax, beta=beta))

    fallback_mask, strongest_column = validate_structured_fallback(
        records,
        data,
        rows,
        predictions,
        expected_fallback_sample_ids=expected_fallback_sample_ids,
    )
    train_accuracy = compute_train_expert_accuracy(
        data.y, data.split_to_rows["train"], data.model_order, expected_model_order=data.model_order
    )
    labels = build_diagnostic_labels_for_split(
        data.y,
        rows,
        train_accuracy,
        data.model_order,
        expected_model_order=data.model_order,
    )
    trainable = labels.is_trainable
    hard_accuracy = (
        float((predictions[trainable] == labels.labels[trainable]).mean()) if trainable.any() else None
    )

    per_dataset: dict[str, dict[str, Any]] = {}
    dataset_values = np.asarray([data.samples[row].dataset for row in rows], dtype=object)
    for dataset in sorted(set(dataset_values.tolist())):
        mask = dataset_values == dataset
        accuracy = float(selected_correct[mask].mean())
        cost = float(selected_cost[mask].mean())
        per_dataset[dataset] = {
            "samples": int(mask.sum()),
            "routed_accuracy": accuracy,
            "average_cost": cost,
            "rank_score": float(rank_score(accuracy, cost, cmin, cmax, beta=beta)),
        }
    macro_accuracy = float(np.mean([value["routed_accuracy"] for value in per_dataset.values()]))
    macro_cost = float(np.mean([value["average_cost"] for value in per_dataset.values()]))

    correct_counts = labels.correct_expert_count
    strata_masks = {
        "all_wrong": correct_counts == 0,
        "single_correct": correct_counts == 1,
        "multi_correct": correct_counts > 1,
    }
    strata = {
        name: {
            "samples": int(mask.sum()),
            "routed_accuracy": float(selected_correct[mask].mean()) if mask.any() else None,
        }
        for name, mask in strata_masks.items()
    }

    strongest_correct = data.y[rows, strongest_column].astype(np.int8)
    strongest_accuracy = float(strongest_correct.mean())
    oracle_correct = (data.y[rows].sum(axis=1) > 0).astype(np.int8)
    oracle_accuracy = float(oracle_correct.mean())
    oracle_gap = oracle_accuracy - strongest_accuracy
    gap_closure = (
        (routed_accuracy - strongest_accuracy) / oracle_gap if abs(oracle_gap) > 1e-15 else None
    )

    selected_records = []
    for position, record in enumerate(records):
        selected_records.append(
            {
                "sample_id": str(record["sample_id"]),
                "row_index": int(rows[position]),
                "split": split,
                "dataset": data.samples[int(rows[position])].dataset,
                "predicted_expert_index": int(predictions[position]),
                "predicted_expert_id": data.model_order[int(predictions[position])],
                "selected_correct": int(selected_correct[position]),
                "selected_cost": float(selected_cost[position]),
                "hard_label_index": int(labels.labels[position]),
                "correct_expert_count": int(correct_counts[position]),
                "routing_input_status": str(record.get("routing_input_status", "native_input")),
            }
        )

    metrics = {
        "schema_version": 1,
        "split": split,
        "num_samples": len(records),
        "routed_accuracy": routed_accuracy,
        "average_cost": average_cost,
        "rank_score": official_rank_score,
        "rank_score_beta": beta,
        "hard_label_accuracy_non_all_wrong": hard_accuracy,
        "hard_label_samples": int(trainable.sum()),
        "per_dataset": per_dataset,
        "dataset_macro": {"routed_accuracy": macro_accuracy, "average_cost": macro_cost},
        "expert_usage": _usage(predictions, data.model_order),
        "input_coverage": {
            "native_or_feature_rows": int((~fallback_mask).sum()),
            "structured_fallback_rows": int(fallback_mask.sum()),
            "structured_fallback_fraction": float(fallback_mask.mean()),
            "fallback_policy": "train_strongest_global" if fallback_mask.any() else None,
            "denominator": "official_split_frozen",
            "fallback_routed_accuracy": (
                float(selected_correct[fallback_mask].mean()) if fallback_mask.any() else None
            ),
            "fallback_average_cost": (
                float(selected_cost[fallback_mask].mean()) if fallback_mask.any() else None
            ),
        },
        "difficulty_strata": strata,
        "strongest_global": {
            "selected_from": "train_only",
            "expert_index": strongest_column,
            "expert_id": data.model_order[strongest_column],
            "split_accuracy": strongest_accuracy,
            "delta_routed_accuracy": routed_accuracy - strongest_accuracy,
        },
        "oracle": {
            "split_accuracy": oracle_accuracy,
            "gap_from_strongest": oracle_gap,
            "router_gap_closure": gap_closure,
        },
    }
    return metrics, selected_records


def evaluate_prediction_file(
    prediction_path: str | Path,
    dataset_root: str | Path,
    *,
    expected_fallback_sample_ids: Iterable[str] | None = None,
    expected_fallback_policy: str | None = None,
    contract_path: str | Path = "configs/bench_coe_query/experiment_contract.yaml",
    output_path: str | Path | None = None,
    selected_output_path: str | Path | None = None,
    expected_split: str | None = None,
    expected_sample_ids: Iterable[str] | None = None,
    expected_seed: int | None = None,
    expected_checkpoint_id: str | None = None,
    expected_checkpoint_path: str | Path | None = None,
    expected_prediction_sha256: str | None = None,
    expected_producer_receipt_path: str | Path | None = None,
    expected_backbone: str | None = None,
) -> dict[str, Any]:
    if (expected_fallback_sample_ids is None) == (expected_fallback_policy is None):
        raise PredictionIntegrityError(
            "exactly one of expected_fallback_sample_ids or expected_fallback_policy is required"
        )
    if expected_sample_ids is not None:
        expected_sample_ids = tuple(str(item) for item in expected_sample_ids)
    if expected_fallback_sample_ids is not None:
        expected_fallback_sample_ids = tuple(
            str(item) for item in expected_fallback_sample_ids
        )
    # Intentional ordering: freeze/read prediction bytes before the evaluator loads Y/C.
    records, prediction_sha256 = read_frozen_predictions(prediction_path)
    if expected_prediction_sha256 is not None and prediction_sha256 != expected_prediction_sha256:
        raise PredictionIntegrityError("prediction SHA-256 differs from the previously frozen digest")
    checkpoint_bytes_sha256 = None
    if expected_checkpoint_path is not None:
        checkpoint_path = Path(expected_checkpoint_path)
        if not checkpoint_path.is_file():
            raise PredictionIntegrityError(f"expected checkpoint does not exist: {checkpoint_path}")
        checkpoint_bytes_sha256 = _sha256(checkpoint_path)
        if expected_checkpoint_id is not None and checkpoint_bytes_sha256 != expected_checkpoint_id:
            raise PredictionIntegrityError("expected checkpoint ID differs from checkpoint bytes")
        expected_checkpoint_id = checkpoint_bytes_sha256
    contract = load_contract(contract_path)
    prediction_resolved = Path(prediction_path).resolve()
    try:
        prediction_report_path = prediction_resolved.relative_to(
            contract.dataset_root
        ).as_posix()
    except ValueError:
        prediction_report_path = prediction_resolved.as_posix()
    producer_receipt = None
    receipt_guard = None
    if expected_producer_receipt_path is not None:
        if expected_checkpoint_path is None:
            raise PredictionIntegrityError(
                "target-free producer receipt validation requires the checkpoint path"
            )
        # Validate the producer's metadata-only view, receipt, prediction bytes,
        # checkpoint bytes and source hashes before this evaluator opens Y/C.
        with deny_target_matrix_access(contract) as receipt_guard:
            target_free_data = load_target_free_vlrb(dataset_root, contract=contract)
            producer_receipt = load_and_validate_target_free_receipt(
                expected_producer_receipt_path,
                contract=contract,
                data=target_free_data,
                prediction_path=prediction_path,
                checkpoint_path=expected_checkpoint_path,
                expected_backbone=expected_backbone,
                expected_split=expected_split,
                expected_seed=expected_seed,
            )
        if receipt_guard.get("passed") is not True:
            raise PredictionIntegrityError(
                "target-free receipt validation attempted to open Y/C before evaluation"
            )
        receipt_prediction_sha256 = str(producer_receipt["prediction"]["sha256"])
        if expected_prediction_sha256 is None:
            expected_prediction_sha256 = receipt_prediction_sha256
        elif expected_prediction_sha256 != receipt_prediction_sha256:
            raise PredictionIntegrityError(
                "expected prediction digest differs from target-free producer receipt"
            )
        if prediction_sha256 != receipt_prediction_sha256:
            raise PredictionIntegrityError(
                "frozen prediction bytes differ from target-free producer receipt"
            )
    data = load_vlrb_strict(dataset_root, contract=contract)
    if expected_fallback_policy is not None:
        if expected_fallback_policy not in {"none", "multi_image"}:
            raise PredictionIntegrityError(
                f"unknown expected fallback policy: {expected_fallback_policy}"
            )
        if expected_sample_ids is not None:
            fallback_candidates = tuple(str(item) for item in expected_sample_ids)
        else:
            observed_split = str(records[0]["split"])
            selected_split = expected_split or observed_split
            if selected_split not in data.split_to_rows:
                raise PredictionIntegrityError(f"unknown expected split: {selected_split}")
            fallback_candidates = tuple(
                data.samples[row].sample_id for row in data.split_to_rows[selected_split]
            )
        expected_fallback_sample_ids = (
            ()
            if expected_fallback_policy == "none"
            else expected_multi_image_fallback_sample_ids(data, fallback_candidates)
        )
    assert expected_fallback_sample_ids is not None
    expected_fallback_values = tuple(str(item) for item in expected_fallback_sample_ids)
    metrics, selected = evaluate_records(
        records,
        data,
        expected_fallback_sample_ids=expected_fallback_values,
        expected_split=expected_split,
        expected_sample_ids=expected_sample_ids,
        expected_seed=expected_seed,
        expected_checkpoint_id=expected_checkpoint_id,
    )
    after = _sha256(prediction_resolved)
    if after != prediction_sha256:
        raise PredictionIntegrityError("prediction file changed during evaluation")
    # Use a canonical root-relative representation for experiment artifacts
    # (and a canonical absolute path for supported external test artifacts) so
    # independent documents do not depend on how callers spell the input path.
    metrics["prediction_file"] = prediction_report_path
    metrics["prediction_sha256"] = prediction_sha256
    metrics["prediction_unchanged"] = True
    metrics["contract_sha256"] = contract.sha256
    metrics["prediction_provenance"] = {
        "expected_prediction_sha256": expected_prediction_sha256,
        "prediction_sha256_verified": expected_prediction_sha256 is not None,
        "expected_seed": expected_seed,
        "seed_verified": expected_seed is not None,
        "expected_checkpoint_id": expected_checkpoint_id,
        "checkpoint_bytes_sha256": checkpoint_bytes_sha256,
        "checkpoint_bytes_verified": expected_checkpoint_path is not None,
        "logits_width_finiteness_and_argmax_verified": True,
        "expected_fallback_policy": expected_fallback_policy or "explicit_sample_id_set",
        "expected_fallback_rows": len(expected_fallback_values),
        "expected_fallback_sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(expected_fallback_values)).encode("utf-8")
        ).hexdigest(),
        "fallback_policy_verified": True,
        "target_free_producer_receipt_required": expected_producer_receipt_path is not None,
        "target_free_producer_receipt_verified": producer_receipt is not None,
        "target_free_producer_receipt_path": (
            None
            if expected_producer_receipt_path is None
            else Path(expected_producer_receipt_path)
            .resolve()
            .relative_to(contract.dataset_root)
            .as_posix()
        ),
        "target_free_producer_receipt_file_sha256": (
            None if producer_receipt is None else producer_receipt["receipt_file_sha256"]
        ),
        "target_free_input_manifest_sha256": (
            None
            if producer_receipt is None
            else producer_receipt["target_free_input_view"]["manifest_sha256"]
        ),
        "producer_process_loaded_y_or_c": (
            None
            if producer_receipt is None
            else producer_receipt["process_boundary"][
                "prediction_process_loaded_y_or_c"
            ]
        ),
        "producer_process_evaluated_predictions": (
            None
            if producer_receipt is None
            else producer_receipt["process_boundary"][
                "evaluation_performed_in_this_process"
            ]
        ),
        "receipt_validated_before_evaluator_loaded_y_c": producer_receipt is not None
        and receipt_guard is not None
        and receipt_guard.get("passed") is True,
    }
    if output_path is not None:
        _atomic_text(Path(output_path), json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if selected_output_path is not None:
        payload = "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in selected
        )
        _atomic_text(Path(selected_output_path), payload)
    return metrics


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-output")
    parser.add_argument("--split")
    parser.add_argument("--fallback-policy", choices=("none", "multi_image"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prediction-sha256")
    parser.add_argument("--producer-receipt")
    parser.add_argument("--backbone", choices=("bert", "tinyllava", "qwen3vl"))
    args = parser.parse_args()
    metrics = evaluate_prediction_file(
        args.predictions,
        args.dataset_dir,
        contract_path=args.config,
        output_path=args.output,
        selected_output_path=args.selected_output,
        expected_split=args.split,
        expected_fallback_policy=args.fallback_policy,
        expected_seed=args.seed,
        expected_checkpoint_path=args.checkpoint,
        expected_prediction_sha256=args.prediction_sha256,
        expected_producer_receipt_path=args.producer_receipt,
        expected_backbone=args.backbone,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
