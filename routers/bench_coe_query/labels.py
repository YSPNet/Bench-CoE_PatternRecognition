"""Train-only Bench-CoE hard labels with deterministic canonical tie breaking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import load_contract
from .data import load_vlrb_strict
from .errors import DataIntegrityError
from .registry import require_exact_order
from .schemas import StrictVLRouterData


IGNORE_INDEX = -100


@dataclass(frozen=True)
class HardLabelResult:
    row_indices: np.ndarray
    labels: np.ndarray
    is_trainable: np.ndarray
    correct_expert_count: np.ndarray
    train_expert_accuracy: np.ndarray
    model_order: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        row_indices: np.ndarray,
        labels: np.ndarray,
        is_trainable: np.ndarray,
        correct_expert_count: np.ndarray,
        train_expert_accuracy: np.ndarray,
        model_order: Sequence[str],
    ) -> "HardLabelResult":
        for value in (row_indices, labels, is_trainable, correct_expert_count, train_expert_accuracy):
            value.flags.writeable = False
        return cls(
            row_indices=row_indices,
            labels=labels,
            is_trainable=is_trainable,
            correct_expert_count=correct_expert_count,
            train_expert_accuracy=train_expert_accuracy,
            model_order=tuple(model_order),
        )


def _validated_rows(rows: Iterable[int], num_rows: int, *, name: str) -> np.ndarray:
    result = np.asarray(tuple(int(row) for row in rows), dtype=np.int64)
    if result.ndim != 1 or len(result) == 0:
        raise DataIntegrityError(f"{name} rows must be a non-empty one-dimensional sequence")
    if len(result) != len(np.unique(result)):
        raise DataIntegrityError(f"{name} rows contain duplicates")
    if (result < 0).any() or (result >= num_rows).any():
        raise DataIntegrityError(f"{name} rows contain an out-of-range matrix index")
    return result


def _validate_y_and_order(
    y: np.ndarray,
    model_order: Sequence[str],
    expected_model_order: Sequence[str] | None = None,
) -> tuple[str, ...]:
    matrix = np.asarray(y)
    if matrix.ndim != 2:
        raise DataIntegrityError(f"Y must be 2-D, got shape {matrix.shape}")
    if not np.isin(matrix, (0, 1)).all():
        raise DataIntegrityError("Y must contain only 0/1 values")
    order = tuple(str(model) for model in model_order)
    if len(order) != matrix.shape[1] or len(set(order)) != len(order):
        raise DataIntegrityError("model order length/uniqueness does not match Y columns")
    if expected_model_order is not None:
        require_exact_order(order, expected_model_order, name="label model registry")
    return order


def compute_train_expert_accuracy(
    y: np.ndarray,
    train_rows: Iterable[int],
    model_order: Sequence[str],
    *,
    expected_model_order: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute one accuracy per expert using only the supplied frozen train rows."""

    matrix = np.asarray(y)
    _validate_y_and_order(matrix, model_order, expected_model_order)
    rows = _validated_rows(train_rows, matrix.shape[0], name="train")
    accuracy = matrix[rows].astype(np.float64, copy=False).mean(axis=0, dtype=np.float64)
    accuracy.flags.writeable = False
    return accuracy


def build_benchcoe_hard_labels(
    y: np.ndarray,
    row_indices: Iterable[int],
    train_expert_accuracy: np.ndarray,
    model_order: Sequence[str],
    *,
    expected_model_order: Sequence[str] | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> HardLabelResult:
    """Choose the train-global-best member of each row's correct expert set."""

    matrix = np.asarray(y)
    order = _validate_y_and_order(matrix, model_order, expected_model_order)
    rows = _validated_rows(row_indices, matrix.shape[0], name="label")
    accuracy = np.asarray(train_expert_accuracy, dtype=np.float64)
    if accuracy.shape != (matrix.shape[1],) or not np.isfinite(accuracy).all():
        raise DataIntegrityError("train expert accuracy shape/values do not match Y columns")

    # lexsort uses the final key as primary: descending accuracy, then ascending
    # canonical column index. This order is independent of filesystem/model names.
    canonical = np.arange(matrix.shape[1], dtype=np.int64)
    priority = np.lexsort((canonical, -accuracy))
    selected = np.full(len(rows), int(ignore_index), dtype=np.int64)
    correct_counts = matrix[rows].sum(axis=1, dtype=np.int64).astype(np.int16)
    for output_index, row in enumerate(rows):
        correct = matrix[row].astype(bool, copy=False)
        if correct_counts[output_index] == 0:
            continue
        selected[output_index] = int(next(column for column in priority if correct[column]))
    trainable = selected != int(ignore_index)
    return HardLabelResult.create(
        row_indices=rows,
        labels=selected,
        is_trainable=trainable,
        correct_expert_count=correct_counts,
        train_expert_accuracy=accuracy.copy(),
        model_order=order,
    )


def build_diagnostic_labels_for_split(
    y: np.ndarray,
    split_rows: Iterable[int],
    train_expert_accuracy: np.ndarray,
    model_order: Sequence[str],
    *,
    expected_model_order: Sequence[str] | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> HardLabelResult:
    """Build diagnostics for any split without recomputing expert accuracy."""

    return build_benchcoe_hard_labels(
        y,
        split_rows,
        train_expert_accuracy,
        model_order,
        expected_model_order=expected_model_order,
        ignore_index=ignore_index,
    )


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


def _distribution(labels: np.ndarray, num_experts: int, ignore_index: int) -> dict[str, Any]:
    valid = labels[labels != ignore_index]
    counts = np.bincount(valid, minlength=num_experts).astype(np.int64)
    total = int(counts.sum())
    shares = counts / total if total else np.zeros(num_experts, dtype=np.float64)
    positive = shares[shares > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if len(positive) else 0.0
    normalized_entropy = entropy / math.log(num_experts) if num_experts > 1 else 0.0
    sorted_shares = np.sort(shares)
    gini = (
        float((2 * np.arange(1, num_experts + 1) - num_experts - 1) @ sorted_shares / num_experts)
        if total
        else 0.0
    )
    return {
        "counts": counts,
        "shares": shares,
        "total": total,
        "max_share": float(shares.max(initial=0.0)),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "gini": gini,
    }


def label_manifest_bytes(
    data: StrictVLRouterData,
    results: dict[str, HardLabelResult],
) -> bytes:
    by_row: dict[int, tuple[str, int, bool, int]] = {}
    for split, result in results.items():
        for position, row in enumerate(result.row_indices):
            by_row[int(row)] = (
                split,
                int(result.labels[position]),
                bool(result.is_trainable[position]),
                int(result.correct_expert_count[position]),
            )
    records = []
    for row, sample in enumerate(data.samples):
        split, label, trainable, correct_count = by_row[row]
        records.append(
            {
                "correct_expert_count": correct_count,
                "hard_label_expert_id": None if label == IGNORE_INDEX else data.model_order[label],
                "hard_label_index": label,
                "is_all_wrong": not trainable,
                "is_trainable": trainable,
                "row_index": row,
                "sample_id": sample.sample_id,
                "split": split,
            }
        )
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in records
    )
    return payload.encode("utf-8")


def build_and_write_label_artifacts(data: StrictVLRouterData, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = data.split_to_rows["train"]
    accuracy = compute_train_expert_accuracy(
        data.y,
        train_rows,
        data.model_order,
        expected_model_order=data.model_order,
    )
    results = {
        split: build_diagnostic_labels_for_split(
            data.y,
            data.split_to_rows[split],
            accuracy,
            data.model_order,
            expected_model_order=data.model_order,
        )
        for split in ("train", "dev", "test")
    }

    accuracy_frame = pd.DataFrame(
        {
            "canonical_index": np.arange(len(data.model_order), dtype=np.int64),
            "expert_id": data.model_order,
            "train_accuracy": accuracy,
            "train_correct_count": data.y[np.asarray(train_rows)].sum(axis=0, dtype=np.int64),
            "train_sample_count": len(train_rows),
        }
    )
    accuracy_frame.to_csv(output / "train_expert_accuracy.csv", index=False, float_format="%.17g")

    manifest = label_manifest_bytes(data, results)
    _atomic_text(output / "label_manifest.jsonl", manifest.decode("utf-8"))
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    rows_frame = pd.DataFrame([json.loads(line) for line in manifest.splitlines()])
    rows_frame.to_parquet(output / "labels_all_splits.parquet", index=False)
    rows_frame[rows_frame["split"] == "train"].to_parquet(output / "train_labels.parquet", index=False)

    train_result = results["train"]
    distribution = _distribution(train_result.labels, len(data.model_order), IGNORE_INDEX)
    distribution_frame = pd.DataFrame(
        {
            "canonical_index": np.arange(len(data.model_order), dtype=np.int64),
            "expert_id": data.model_order,
            "selected_count": distribution["counts"],
            "selected_share": distribution["shares"],
        }
    )
    distribution_frame.to_csv(output / "label_distribution.csv", index=False, float_format="%.17g")

    correct_counts = train_result.correct_expert_count
    strata = {
        "all_wrong": int((correct_counts == 0).sum()),
        "single_correct": int((correct_counts == 1).sum()),
        "multi_correct": int((correct_counts > 1).sum()),
    }
    collapse_warning = distribution["max_share"] >= 0.5 or distribution["normalized_entropy"] < 0.5
    meta = {
        "schema_version": 1,
        "definition": "argmax train expert accuracy within each sample correct set; canonical-index tie break",
        "statistics_source": "train_only",
        "ignore_index": IGNORE_INDEX,
        "model_order": list(data.model_order),
        "train_expert_accuracy": accuracy.tolist(),
        "split_rows_sha256": {
            split: hashlib.sha256(np.asarray(data.split_to_rows[split], dtype="<i8").tobytes()).hexdigest()
            for split in ("train", "dev", "test")
        },
        "manifest_sha256": manifest_sha256,
        "strata": strata,
        "distribution": {
            key: value
            for key, value in distribution.items()
            if key not in {"counts", "shares"}
        },
        "collapse_warning": bool(collapse_warning),
    }
    _atomic_text(output / "label_manifest.meta.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")

    selected_lines = "\n".join(
        f"- `{row.expert_id}`: {int(row.selected_count):,} ({row.selected_share:.4%})"
        for row in distribution_frame.itertuples()
    )
    report = f"""# Label audit

Definition: train-global-best-among-correct, with ties broken only by ascending canonical model index. Expert accuracy was computed from the frozen train rows and reused unchanged for dev/test diagnostics.

## Train strata

- All wrong / ignored by CE: {strata['all_wrong']:,}
- Exactly one correct expert: {strata['single_correct']:,}
- Multiple correct experts: {strata['multi_correct']:,}

## Selected hard labels

{selected_lines}

## Collapse diagnostics

- Maximum label share: {distribution['max_share']:.6f}
- Entropy: {distribution['entropy']:.6f}
- Normalized entropy: {distribution['normalized_entropy']:.6f}
- Gini: {distribution['gini']:.6f}
- Collapse-risk warning: {str(bool(collapse_warning)).lower()}

All-wrong rows have label `-100`; no pseudo-expert was fabricated. Class weights, oversampling, focal/cost/load-balancing losses are absent from the main configuration. The deterministic JSONL manifest SHA-256 is `{manifest_sha256}`.
"""
    _atomic_text(output / "LABEL_AUDIT.md", report)
    return meta


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", nargs="?")
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--output", default="outputs/bench_coe_query/labels")
    args = parser.parse_args()
    contract = load_contract(args.config)
    data = load_vlrb_strict(args.dataset_dir, contract=contract)
    result = build_and_write_label_artifacts(data, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
