"""Frozen row selection for bounded smoke or explicitly gated full runs."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import DataIntegrityError
from .schemas import StrictVLRouterData


def read_manifest_rows(path: str | Path, data: StrictVLRouterData) -> tuple[int, ...]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise DataIntegrityError(f"smoke manifest is missing: {manifest_path}")
    rows: list[int] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            row = int(record["row_index"])
            if data.sample_to_row.get(sample_id) != row:
                raise DataIntegrityError(f"smoke manifest row mismatch at {manifest_path}:{line_number}")
            if len(data.samples[row].assets) != 1:
                raise DataIntegrityError(f"smoke manifest contains non-single-image sample {sample_id}")
            rows.append(row)
    if len(rows) != len(set(rows)):
        raise DataIntegrityError(f"smoke manifest has duplicate rows: {manifest_path}")
    if rows != sorted(rows):
        raise DataIntegrityError(f"smoke manifest is not in ascending matrix-row order: {manifest_path}")
    return tuple(rows)


def select_rows(
    data: StrictVLRouterData,
    *,
    split: str,
    max_samples: int | None,
    manifest_root: str | Path = "outputs/bench_coe_query/manifests",
) -> tuple[int, ...]:
    if split not in {"train", "dev", "test", "all"}:
        raise ValueError(f"unknown split selection: {split}")
    if max_samples is None:
        if split == "all":
            return tuple(range(len(data.samples)))
        return data.split_to_rows[split]
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    root = Path(manifest_root)
    names = ("train", "dev", "test") if split == "all" else (split,)
    rows = []
    for name in names:
        available = read_manifest_rows(root / f"smoke_{name}.jsonl", data)
        rows.extend(available[:max_samples])
    return tuple(sorted(rows))
