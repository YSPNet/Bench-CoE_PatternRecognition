"""Fail-closed VL-RouterBench data adapter and deterministic smoke manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import ExperimentContract, load_contract, sha256_file
from .errors import DataIntegrityError
from .registry import build_index, load_registry_list, require_exact_order
from .schemas import AssetSpec, StrictSample, StrictVLRouterData


SPLIT_NAMES = ("train", "dev", "test")


def benchmark_manifest_sha256(dataset_root: Path, benchmarks_dir: Path) -> tuple[str, int, int]:
    """Hash the frozen sorted path/size/content-hash BENCHMARKS manifest."""

    files = sorted(benchmarks_dir.glob("*/*_samples.jsonl"), key=lambda item: item.as_posix())
    payload = "".join(
        f"{path.relative_to(dataset_root).as_posix()}\t{path.stat().st_size}\t{sha256_file(path)}\n"
        for path in files
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(files), sum(path.stat().st_size for path in files)


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


def _read_jsonl(path: Path, *, required_key: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataIntegrityError(f"required JSONL file is missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise DataIntegrityError(f"blank JSONL record in {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise DataIntegrityError(f"JSONL value is not an object in {path}:{line_number}")
            if required_key is not None and required_key not in value:
                raise DataIntegrityError(f"missing {required_key!r} in {path}:{line_number}")
            records.append(value)
    return records


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise DataIntegrityError(f"missing frozen {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise DataIntegrityError(
            f"frozen {label} hash mismatch: expected {expected}, observed {observed}, path={path}"
        )


def _load_y(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != ("Y",):
                raise DataIntegrityError(f"Y archive must contain exactly key 'Y', got {archive.files}")
            return np.asarray(archive["Y"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, DataIntegrityError):
            raise
        raise DataIntegrityError(f"failed to load Y from {path}: {exc}") from exc


def _load_pickle_after_hash(path: Path, *, name: str) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        raise DataIntegrityError(f"failed to load frozen {name} {path}: {exc}") from exc


def _load_benchmark_map(benchmarks_dir: Path, *, expected_files: int) -> dict[str, dict[str, Any]]:
    if not benchmarks_dir.is_dir():
        raise DataIntegrityError(f"BENCHMARKS directory is missing: {benchmarks_dir}")
    files = sorted(benchmarks_dir.glob("*/*_samples.jsonl"), key=lambda item: item.as_posix())
    if len(files) != expected_files:
        raise DataIntegrityError(f"expected {expected_files} BENCHMARKS files, found {len(files)}")
    samples: dict[str, dict[str, Any]] = {}
    for path in files:
        for record in _read_jsonl(path, required_key="sample_id"):
            sample_id = str(record["sample_id"])
            if sample_id in samples:
                raise DataIntegrityError(f"duplicate BENCHMARKS sample_id: {sample_id}")
            text = record.get("prompt")
            if not isinstance(text, str) or not text.strip():
                raise DataIntegrityError(f"empty prompt for BENCHMARKS sample {sample_id}")
            assets = record.get("assets")
            if not isinstance(assets, list) or not assets:
                raise DataIntegrityError(f"empty assets for BENCHMARKS sample {sample_id}")
            if any(not isinstance(asset, dict) for asset in assets):
                raise DataIntegrityError(f"malformed asset list for BENCHMARKS sample {sample_id}")
            samples[sample_id] = record
    return samples


def load_vlrb_strict(
    dataset_root: str | Path,
    *,
    contract_path: str | Path | None = None,
    contract: ExperimentContract | None = None,
    verify_hashes: bool = True,
) -> StrictVLRouterData:
    root = Path(dataset_root).resolve()
    frozen = contract or load_contract(contract_path)
    if root != frozen.dataset_root:
        raise DataIntegrityError(
            f"dataset root differs from frozen contract: observed={root}, expected={frozen.dataset_root}"
        )
    data_cfg = frozen.values["data"]

    if verify_hashes:
        for split_name, entry in data_cfg["splits"].items():
            _verify_hash(frozen.resolve(str(entry["path"])), str(entry["sha256"]), label=f"{split_name} split")
        for group_name in ("matrices", "registry"):
            for artifact_name, entry in data_cfg[group_name].items():
                _verify_hash(
                    frozen.resolve(str(entry["path"])),
                    str(entry["sha256"]),
                    label=f"{group_name}.{artifact_name}",
                )

    y_path = frozen.resolve(str(data_cfg["matrices"]["y"]["path"]))
    c_path = frozen.resolve(str(data_cfg["matrices"]["c"]["path"]))
    y = _load_y(y_path)
    try:
        c = np.load(c_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"failed to load C from {c_path}: {exc}") from exc

    expected_shape = tuple(int(item) for item in data_cfg["matrices"]["y"]["shape"])
    expected_c_shape = tuple(int(item) for item in data_cfg["matrices"]["c"]["shape"])
    if y.shape != expected_shape or c.shape != expected_c_shape or y.shape != c.shape:
        raise DataIntegrityError(
            f"Y/C shape mismatch: Y={y.shape}, C={c.shape}, expected={expected_shape}/{expected_c_shape}"
        )
    if str(y.dtype) != str(data_cfg["matrices"]["y"]["dtype"]):
        raise DataIntegrityError(f"Y dtype mismatch: {y.dtype}")
    if str(c.dtype) != str(data_cfg["matrices"]["c"]["dtype"]):
        raise DataIntegrityError(f"C dtype mismatch: {c.dtype}")
    if not np.isfinite(y).all() or not np.isfinite(c).all():
        raise DataIntegrityError("Y/C contain non-finite values")
    if not np.isin(y, (0, 1)).all():
        raise DataIntegrityError("Y must contain only binary correctness values")
    if (c < 0).any():
        raise DataIntegrityError("C contains negative costs")

    model_registry_path = frozen.resolve(str(data_cfg["registry"]["models"]["path"]))
    sample_registry_path = frozen.resolve(str(data_cfg["registry"]["samples"]["path"]))
    models = load_registry_list(model_registry_path, name="model")
    models = require_exact_order(models, frozen.model_order, name="model registry")
    sample_ids = load_registry_list(sample_registry_path, name="sample")
    if len(sample_ids) != y.shape[0]:
        raise DataIntegrityError(f"sample registry length {len(sample_ids)} does not match Y rows {y.shape[0]}")
    if len(models) != y.shape[1]:
        raise DataIntegrityError(f"model registry length {len(models)} does not match Y columns {y.shape[1]}")

    duplicate_matrix_models = root / "data/matrices/model_names.pkl"
    if duplicate_matrix_models.is_file():
        require_exact_order(
            _load_pickle_after_hash(duplicate_matrix_models, name="matrix model names"),
            models,
            name="matrix model names",
        )
    duplicate_matrix_samples = root / "data/matrices/sample_ids.pkl"
    if duplicate_matrix_samples.is_file():
        require_exact_order(
            _load_pickle_after_hash(duplicate_matrix_samples, name="matrix sample IDs"),
            sample_ids,
            name="matrix sample IDs",
        )

    meta_path = frozen.resolve(str(data_cfg["registry"]["meta"]["path"]))
    try:
        meta = pd.read_parquet(meta_path)
    except Exception as exc:
        raise DataIntegrityError(f"failed to load meta parquet {meta_path}: {exc}") from exc
    required_meta = {"sample_id", "dataset", "split", "task"}
    if not required_meta.issubset(meta.columns):
        raise DataIntegrityError(f"meta is missing columns: {sorted(required_meta.difference(meta.columns))}")
    meta_ids = tuple(meta["sample_id"].astype(str).tolist())
    require_exact_order(meta_ids, sample_ids, name="meta/sample registry")

    split_ids: dict[str, tuple[str, ...]] = {}
    split_sets: dict[str, set[str]] = {}
    for split_name in SPLIT_NAMES:
        entry = data_cfg["splits"].get(split_name)
        if entry is None:
            raise DataIntegrityError(f"contract has no {split_name} split")
        records = _read_jsonl(frozen.resolve(str(entry["path"])), required_key="sample_id")
        ids = tuple(str(record["sample_id"]) for record in records)
        if len(ids) != int(entry["count"]):
            raise DataIntegrityError(f"{split_name} count mismatch: {len(ids)} != {entry['count']}")
        if len(ids) != len(set(ids)):
            raise DataIntegrityError(f"{split_name} contains duplicate sample IDs")
        split_ids[split_name] = ids
        split_sets[split_name] = set(ids)

    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_sets[left].intersection(split_sets[right])
            if overlap:
                raise DataIntegrityError(f"split overlap {left}/{right}: {sorted(overlap)[:3]}")
    union = set().union(*(split_sets[name] for name in SPLIT_NAMES))
    if union != set(sample_ids):
        missing = set(sample_ids).difference(union)
        extra = union.difference(sample_ids)
        raise DataIntegrityError(f"split union mismatch: missing={len(missing)}, extra={len(extra)}")

    sample_to_row = build_index(sample_ids, name="sample")
    model_to_column = build_index(models, name="model")
    split_to_rows = {
        name: tuple(sorted((sample_to_row[sample_id] for sample_id in split_ids[name])))
        for name in SPLIT_NAMES
    }
    observed_meta_splits = tuple(str(value) for value in meta["split"].tolist())
    for split_name, rows in split_to_rows.items():
        wrong = [row for row in rows if observed_meta_splits[row] != split_name]
        if wrong:
            raise DataIntegrityError(f"meta split disagrees with {split_name} manifest at row {wrong[0]}")

    benchmark_map = _load_benchmark_map(
        frozen.resolve(str(data_cfg["benchmarks"]["path"])),
        expected_files=int(data_cfg["benchmarks"]["files"]),
    )
    benchmark_root = frozen.resolve(str(data_cfg["benchmarks"]["path"]))
    observed_benchmark_manifest, _benchmark_files, _benchmark_bytes = benchmark_manifest_sha256(
        root, benchmark_root
    )
    if verify_hashes and observed_benchmark_manifest != str(data_cfg["benchmarks"]["manifest_sha256"]):
        raise DataIntegrityError(
            "frozen BENCHMARKS manifest hash mismatch: "
            f"expected {data_cfg['benchmarks']['manifest_sha256']}, observed {observed_benchmark_manifest}"
        )
    if len(benchmark_map) != int(data_cfg["benchmarks"]["records"]):
        raise DataIntegrityError(
            f"BENCHMARKS record count mismatch: {len(benchmark_map)} != {data_cfg['benchmarks']['records']}"
        )
    if set(benchmark_map) != set(sample_ids):
        missing = set(sample_ids).difference(benchmark_map)
        extra = set(benchmark_map).difference(sample_ids)
        raise DataIntegrityError(f"BENCHMARKS ID mismatch: missing={len(missing)}, extra={len(extra)}")

    samples: list[StrictSample] = []
    for row_index, sample_id in enumerate(sample_ids):
        meta_row = meta.iloc[row_index]
        record = benchmark_map[sample_id]
        dataset = str(meta_row["dataset"])
        if str(record.get("dataset")) != dataset:
            raise DataIntegrityError(f"dataset mismatch for {sample_id}")
        task_type = str(record.get("task_type", meta_row["task"]))
        if task_type != str(meta_row["task"]):
            raise DataIntegrityError(f"task mismatch for {sample_id}")
        assets = tuple(AssetSpec.from_mapping(asset) for asset in record["assets"])
        if any(asset.kind not in {"image", "image_tsv"} for asset in assets):
            raise DataIntegrityError(f"unsupported asset type for {sample_id}: {[asset.kind for asset in assets]}")
        samples.append(
            StrictSample(
                sample_id=sample_id,
                row_index=row_index,
                dataset=dataset,
                task_type=task_type,
                text=str(record["prompt"]),
                assets=assets,
                split=observed_meta_splits[row_index],
                y_row=tuple(int(value) for value in y[row_index]),
                c_row=tuple(float(value) for value in c[row_index]),
            )
        )
    return StrictVLRouterData.create(
        dataset_root=root,
        samples=tuple(samples),
        model_order=models,
        sample_to_row=sample_to_row,
        model_to_column=model_to_column,
        split_to_rows=split_to_rows,
        y=y,
        c=c,
        contract_sha256=frozen.sha256,
    )


def data_audit(data: StrictVLRouterData) -> dict[str, Any]:
    asset_histogram = Counter(len(sample.assets) for sample in data.samples)
    task_histogram = Counter(sample.task_type for sample in data.samples)
    dataset_histogram = Counter(sample.dataset for sample in data.samples)
    split_counts = {name: len(rows) for name, rows in data.split_to_rows.items()}
    return {
        "schema_version": 1,
        "status": "strict_validation_passed",
        "dataset_root": str(data.dataset_root),
        "contract_sha256": data.contract_sha256,
        "samples": len(data.samples),
        "experts": len(data.model_order),
        "splits": split_counts,
        "datasets": dict(sorted(dataset_histogram.items())),
        "tasks": dict(sorted(task_histogram.items())),
        "asset_count_histogram": {str(key): value for key, value in sorted(asset_histogram.items())},
        "single_image_rows": int(asset_histogram.get(1, 0)),
        "multi_image_rows": int(sum(count for size, count in asset_histogram.items() if size > 1)),
        "y_shape": list(data.y.shape),
        "y_dtype": str(data.y.dtype),
        "c_shape": list(data.c.shape),
        "c_dtype": str(data.c.dtype),
        "model_order": list(data.model_order),
    }


def _manifest_record(sample: StrictSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "row_index": sample.row_index,
        "dataset": sample.dataset,
        "task_type": sample.task_type,
        "split": sample.split,
        "text_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest(),
        "asset_count": len(sample.assets),
        "asset_status": "audited_single_image",
    }


def make_smoke_manifests(data: StrictVLRouterData, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    limits = {"train": 64, "dev": 32, "test": 64}
    files: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        eligible = [
            data.samples[row]
            for row in data.split_to_rows[split_name]
            if len(data.samples[row].assets) == 1
        ]
        selected = eligible[: limits[split_name]]
        if len(selected) != limits[split_name]:
            raise DataIntegrityError(
                f"not enough valid {split_name} samples for smoke: {len(selected)} < {limits[split_name]}"
            )
        payload = "".join(
            json.dumps(_manifest_record(sample), sort_keys=True, separators=(",", ":")) + "\n"
            for sample in selected
        )
        path = output / f"smoke_{split_name}.jsonl"
        _atomic_text(path, payload)
        files[split_name] = {
            "path": str(path),
            "count": len(selected),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "sample_ids": [sample.sample_id for sample in selected],
        }

    stratified: list[StrictSample] = []
    datasets = sorted({sample.dataset for sample in data.samples})
    for split_name in SPLIT_NAMES:
        for dataset in datasets:
            candidate = next(
                (
                    data.samples[row]
                    for row in data.split_to_rows[split_name]
                    if data.samples[row].dataset == dataset and len(data.samples[row].assets) == 1
                ),
                None,
            )
            if candidate is None:
                raise DataIntegrityError(f"no valid stratified smoke sample for {split_name}/{dataset}")
            stratified.append(candidate)
    stratified_payload = "".join(
        json.dumps(_manifest_record(sample), sort_keys=True, separators=(",", ":")) + "\n"
        for sample in stratified
    )
    stratified_path = output / "smoke_stratified.jsonl"
    _atomic_text(stratified_path, stratified_payload)
    files["stratified"] = {
        "path": str(stratified_path),
        "count": len(stratified),
        "sha256": hashlib.sha256(stratified_payload.encode("utf-8")).hexdigest(),
        "datasets": datasets,
    }
    summary = {"schema_version": 1, "ordering": "ascending_matrix_row_index", "files": files}
    _atomic_text(output / "SMOKE_MANIFEST.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_data_audit(data: StrictVLRouterData, output_root: str | Path) -> None:
    root = Path(output_root)
    audit = data_audit(data)
    _atomic_text(root / "data/data_audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    preview_payload = "".join(
        json.dumps(_manifest_record(sample), sort_keys=True, separators=(",", ":")) + "\n"
        for sample in data.samples[:32]
    )
    _atomic_text(root / "data/sample_registry_preview.jsonl", preview_payload)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "make-smoke-manifest"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--dataset-dir", default=".")
        subparser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
        subparser.add_argument("--output-root", default="outputs/bench_coe_query")
    args = parser.parse_args()
    data = load_vlrb_strict(args.dataset_dir, contract_path=args.config)
    if args.command == "audit":
        write_data_audit(data, args.output_root)
        print(json.dumps(data_audit(data), indent=2, sort_keys=True))
    else:
        summary = make_smoke_manifests(data, Path(args.output_root) / "manifests")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
