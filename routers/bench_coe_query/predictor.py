"""Prediction serialization deliberately decoupled from Y/C."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def write_logits_predictions(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    row_indices: Sequence[int],
    logits: np.ndarray,
    model_order: Sequence[str],
    split: str,
    seed: int,
    checkpoint_id: str,
    annotations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str | int]:
    values = np.asarray(logits)
    if values.shape != (len(sample_ids), len(model_order)) or len(row_indices) != len(sample_ids):
        raise ValueError("sample/row/logit dimensions do not match")
    if not np.isfinite(values).all():
        raise ValueError("logits contain non-finite values")
    if len(set(str(item) for item in sample_ids)) != len(sample_ids):
        raise ValueError("prediction sample IDs contain duplicates")
    if annotations is not None and len(annotations) != len(sample_ids):
        raise ValueError("prediction annotations do not match sample count")
    predictions = values.argmax(axis=1)
    records = []
    for position, column in enumerate(predictions):
        record: dict[str, Any] = {
            "sample_id": str(sample_ids[position]),
            "row_index": int(row_indices[position]),
            "predicted_expert_index": int(column),
            "predicted_expert_id": str(model_order[int(column)]),
            "split": str(split),
            "seed": int(seed),
            "checkpoint_id": str(checkpoint_id),
            "logits": [float(value) for value in values[position]],
        }
        if annotations is not None:
            overlap = set(record).intersection(annotations[position])
            if overlap:
                raise ValueError(f"prediction annotation overwrites required fields: {sorted(overlap)}")
            record.update(dict(annotations[position]))
        records.append(record)
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "path": str(output),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "count": len(records),
    }
