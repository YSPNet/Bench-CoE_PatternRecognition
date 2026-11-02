"""Target-free metadata view and receipts for isolated prediction processes.

This module is deliberately unable to represent Y/C.  Prediction entry points
must construct this view while :func:`deny_target_matrix_access` is active,
freeze logits to disk, and exit before an evaluator process opens Y/C.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentContract, sha256_file
from .data import SPLIT_NAMES, _load_benchmark_map, _read_jsonl, benchmark_manifest_sha256
from .errors import ContractError, DataIntegrityError, PredictionIntegrityError
from .input_content import canonical_sha256
from .registry import build_index, load_registry_list, require_exact_order
from .schemas import AssetSpec


RECEIPT_SUFFIX = ".target_free_receipt.json"


@dataclass(frozen=True)
class PredictionSample:
    """Only fields that a router may consume while producing logits."""

    sample_id: str
    row_index: int
    dataset: str
    task_type: str
    text: str
    assets: tuple[AssetSpec, ...]
    split: str


@dataclass(frozen=True)
class TargetFreeVLRouterData:
    """Registry/split/input view with no correctness or cost attributes."""

    dataset_root: Path
    samples: tuple[PredictionSample, ...]
    model_order: tuple[str, ...]
    sample_to_row: Mapping[str, int]
    model_to_column: Mapping[str, int]
    split_to_rows: Mapping[str, tuple[int, ...]]
    contract_sha256: str
    input_manifest_sha256: str
    input_artifacts: tuple[Mapping[str, Any], ...]

    def split_samples(self, split: str) -> tuple[PredictionSample, ...]:
        return tuple(self.samples[row] for row in self.split_to_rows[split])


class TargetMatrixAccessError(RuntimeError):
    """Raised before a prediction process can open Y or C."""


def target_matrix_paths(contract: ExperimentContract) -> tuple[Path, Path]:
    matrices = contract.values["data"]["matrices"]
    return tuple(
        contract.resolve(str(matrices[name]["path"])).resolve() for name in ("y", "c")
    )  # type: ignore[return-value]


def _resolved_path(value: Any) -> Path | None:
    if isinstance(value, int) or value is None:
        return None
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, (str, os.PathLike)):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


@contextmanager
def deny_target_matrix_access(contract: ExperimentContract) -> Iterator[dict[str, Any]]:
    """Fail closed on Python-level attempts to open the frozen Y/C files.

    The remediation additionally records an OS-level filtered ``strace`` for
    each full prediction process.  This in-process guard makes the boundary
    testable and keeps future callers fail-closed even without ptrace support.
    """

    forbidden = frozenset(target_matrix_paths(contract))
    attempts: list[dict[str, str]] = []

    def reject(value: Any, interface: str) -> None:
        path = _resolved_path(value)
        if path in forbidden:
            attempts.append({"interface": interface, "path": str(path)})
            raise TargetMatrixAccessError(
                f"target-free prediction process refused to open target matrix: {path}"
            )

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_numpy_load = np.load
    original_path_open = Path.open

    def guarded_builtin_open(file, *args, **kwargs):
        reject(file, "builtins.open")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):
        reject(file, "io.open")
        return original_io_open(file, *args, **kwargs)

    def guarded_os_open(path, *args, **kwargs):
        reject(path, "os.open")
        return original_os_open(path, *args, **kwargs)

    def guarded_numpy_load(file, *args, **kwargs):
        reject(file, "numpy.load")
        return original_numpy_load(file, *args, **kwargs)

    def guarded_path_open(path_self, *args, **kwargs):
        reject(path_self, "pathlib.Path.open")
        return original_path_open(path_self, *args, **kwargs)

    audit: dict[str, Any] = {
        "active": True,
        "forbidden_paths": sorted(str(path) for path in forbidden),
        "guarded_interfaces": [
            "builtins.open",
            "io.open",
            "os.open",
            "numpy.load",
            "pathlib.Path.open",
        ],
        "attempts": attempts,
        "attempt_count": 0,
        "passed": False,
    }
    builtins.open = guarded_builtin_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    np.load = guarded_numpy_load
    Path.open = guarded_path_open
    try:
        yield audit
    finally:
        Path.open = original_path_open
        np.load = original_numpy_load
        os.open = original_os_open
        io.open = original_io_open
        builtins.open = original_builtin_open
        audit["attempt_count"] = len(attempts)
        audit["passed"] = len(attempts) == 0


def _verified_artifact(contract: ExperimentContract, entry: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = contract.resolve(str(entry["path"])).resolve()
    if not path.is_file():
        raise DataIntegrityError(f"missing target-free {label}: {path}")
    observed = sha256_file(path)
    expected = str(entry["sha256"])
    if observed != expected:
        raise DataIntegrityError(
            f"target-free {label} hash mismatch: expected={expected}, observed={observed}"
        )
    return {
        "label": label,
        "path": path.relative_to(contract.dataset_root).as_posix(),
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def load_target_free_vlrb(
    dataset_root: str | Path,
    *,
    contract: ExperimentContract,
) -> TargetFreeVLRouterData:
    """Load prompts/assets/splits/registries without opening or representing Y/C."""

    root = Path(dataset_root).resolve()
    if root != contract.dataset_root:
        raise DataIntegrityError(
            f"dataset root differs from frozen contract: observed={root}, expected={contract.dataset_root}"
        )
    data_cfg = contract.values["data"]
    artifacts: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        artifacts.append(
            _verified_artifact(contract, data_cfg["splits"][split], f"split.{split}")
        )
    for name in ("meta", "models", "samples"):
        artifacts.append(
            _verified_artifact(contract, data_cfg["registry"][name], f"registry.{name}")
        )

    models = require_exact_order(
        load_registry_list(contract.resolve(str(data_cfg["registry"]["models"]["path"])), name="model"),
        contract.model_order,
        name="target-free model registry",
    )
    sample_ids = load_registry_list(
        contract.resolve(str(data_cfg["registry"]["samples"]["path"])), name="sample"
    )
    if len(models) != int(data_cfg["expected_experts"]):
        raise DataIntegrityError("target-free model registry count mismatch")
    if len(sample_ids) != int(data_cfg["expected_samples"]):
        raise DataIntegrityError("target-free sample registry count mismatch")
    sample_to_row = build_index(sample_ids, name="target-free sample")
    model_to_column = build_index(models, name="target-free model")

    meta_path = contract.resolve(str(data_cfg["registry"]["meta"]["path"]))
    try:
        meta = pd.read_parquet(meta_path)
    except Exception as exc:
        raise DataIntegrityError(f"failed to load target-free metadata {meta_path}: {exc}") from exc
    required_meta = {"sample_id", "dataset", "split", "task"}
    if not required_meta.issubset(meta.columns):
        raise DataIntegrityError(
            f"target-free metadata is missing columns: {sorted(required_meta.difference(meta.columns))}"
        )
    meta_ids = tuple(meta["sample_id"].astype(str).tolist())
    require_exact_order(meta_ids, sample_ids, name="target-free metadata/sample registry")

    split_ids: dict[str, tuple[str, ...]] = {}
    split_sets: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        entry = data_cfg["splits"][split]
        records = _read_jsonl(contract.resolve(str(entry["path"])), required_key="sample_id")
        ids = tuple(str(record["sample_id"]) for record in records)
        if len(ids) != int(entry["count"]) or len(ids) != len(set(ids)):
            raise DataIntegrityError(f"target-free {split} split count/uniqueness mismatch")
        split_ids[split] = ids
        split_sets[split] = set(ids)
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if split_sets[left].intersection(split_sets[right]):
                raise DataIntegrityError(f"target-free split overlap: {left}/{right}")
    if set().union(*(split_sets[name] for name in SPLIT_NAMES)) != set(sample_ids):
        raise DataIntegrityError("target-free split union differs from sample registry")
    split_to_rows = {
        split: tuple(sorted(sample_to_row[sample_id] for sample_id in split_ids[split]))
        for split in SPLIT_NAMES
    }

    benchmark_root = contract.resolve(str(data_cfg["benchmarks"]["path"])).resolve()
    benchmark_digest, benchmark_files, benchmark_bytes = benchmark_manifest_sha256(
        root, benchmark_root
    )
    if benchmark_digest != str(data_cfg["benchmarks"]["manifest_sha256"]):
        raise DataIntegrityError("target-free BENCHMARKS manifest hash mismatch")
    if benchmark_files != int(data_cfg["benchmarks"]["files"]):
        raise DataIntegrityError("target-free BENCHMARKS file count mismatch")
    benchmark_map = _load_benchmark_map(
        benchmark_root, expected_files=int(data_cfg["benchmarks"]["files"])
    )
    if set(benchmark_map) != set(sample_ids):
        raise DataIntegrityError("target-free BENCHMARKS/sample registry mismatch")
    artifacts.append(
        {
            "label": "benchmarks.sorted_content_manifest",
            "path": str(data_cfg["benchmarks"]["path"]),
            "sha256": benchmark_digest,
            "files": benchmark_files,
            "bytes": benchmark_bytes,
        }
    )

    observed_splits = tuple(str(value) for value in meta["split"].tolist())
    samples: list[PredictionSample] = []
    for row, sample_id in enumerate(sample_ids):
        meta_row = meta.iloc[row]
        record = benchmark_map[sample_id]
        dataset = str(meta_row["dataset"])
        if str(record.get("dataset")) != dataset:
            raise DataIntegrityError(f"target-free dataset mismatch for {sample_id}")
        task_type = str(record.get("task_type", meta_row["task"]))
        if task_type != str(meta_row["task"]):
            raise DataIntegrityError(f"target-free task mismatch for {sample_id}")
        split = observed_splits[row]
        if split not in SPLIT_NAMES or row not in split_to_rows[split]:
            raise DataIntegrityError(f"target-free split mismatch for {sample_id}")
        assets = tuple(AssetSpec.from_mapping(asset) for asset in record["assets"])
        if not assets or any(asset.kind not in {"image", "image_tsv"} for asset in assets):
            raise DataIntegrityError(f"target-free asset mismatch for {sample_id}")
        samples.append(
            PredictionSample(
                sample_id=sample_id,
                row_index=row,
                dataset=dataset,
                task_type=task_type,
                text=str(record["prompt"]),
                assets=assets,
                split=split,
            )
        )
    if len({sample.dataset for sample in samples}) != int(data_cfg["expected_datasets"]):
        raise DataIntegrityError("target-free dataset count mismatch")

    fingerprint_payload = {
        "schema_version": 1,
        "kind": "target_free_prediction_input_view",
        "contract_sha256": contract.sha256,
        "sample_order_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "model_order": list(models),
        "split_rows_sha256": {
            split: hashlib.sha256(
                "\n".join(str(row) for row in split_to_rows[split]).encode("utf-8")
            ).hexdigest()
            for split in SPLIT_NAMES
        },
        "input_artifacts": artifacts,
        "explicitly_absent_fields": ["y", "c", "y_row", "c_row"],
        "forbidden_target_paths": [str(path) for path in target_matrix_paths(contract)],
    }
    fingerprint = canonical_sha256(fingerprint_payload)
    return TargetFreeVLRouterData(
        dataset_root=root,
        samples=tuple(samples),
        model_order=models,
        sample_to_row=MappingProxyType(sample_to_row),
        model_to_column=MappingProxyType(model_to_column),
        split_to_rows=MappingProxyType(split_to_rows),
        contract_sha256=contract.sha256,
        input_manifest_sha256=fingerprint,
        input_artifacts=tuple(MappingProxyType(dict(item)) for item in artifacts),
    )


def prediction_receipt_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    if path.suffix != ".jsonl":
        raise ValueError("prediction receipt requires a .jsonl prediction path")
    return path.with_suffix(RECEIPT_SUFFIX)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_target_free_receipt(
    path: str | Path,
    *,
    contract: ExperimentContract,
    data: TargetFreeVLRouterData,
    backbone: str,
    split: str,
    seed: int,
    checkpoint_path: str | Path,
    prediction_path: str | Path,
    prediction_sha256: str,
    prediction_records: int,
    guard_audit: Mapping[str, Any],
    producer_sources: Sequence[str | Path],
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if guard_audit.get("active") is not True or guard_audit.get("attempts"):
        raise TargetMatrixAccessError("cannot attest a prediction process with target access attempts")
    checkpoint = Path(checkpoint_path).resolve()
    prediction = Path(prediction_path).resolve()
    root = contract.dataset_root
    sources = {}
    for value in producer_sources:
        source = Path(value).resolve()
        relative = source.relative_to(root).as_posix()
        sources[relative] = sha256_file(source)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "target_free_prediction_receipt",
        "contract_sha256": contract.sha256,
        "backbone": str(backbone),
        "split": str(split),
        "seed": int(seed),
        "producer_pid": os.getpid(),
        "process_boundary": {
            "prediction_process_loaded_y_or_c": False,
            "evaluation_performed_in_this_process": False,
            "must_exit_before_evaluator": True,
            "target_matrix_guard": {
                "active": True,
                "forbidden_paths": list(guard_audit["forbidden_paths"]),
                "guarded_interfaces": list(guard_audit["guarded_interfaces"]),
                "attempt_count": len(guard_audit.get("attempts", ())),
            },
        },
        "target_free_input_view": {
            "manifest_sha256": data.input_manifest_sha256,
            "samples": len(data.samples),
            "experts": len(data.model_order),
            "represented_target_fields": [],
            "explicitly_absent_fields": ["y", "c", "y_row", "c_row"],
        },
        "prediction": {
            "path": prediction.relative_to(root).as_posix(),
            "sha256": str(prediction_sha256),
            "records": int(prediction_records),
            "frozen_before_evaluation": True,
        },
        "checkpoint": {
            "path": checkpoint.relative_to(root).as_posix(),
            "sha256": sha256_file(checkpoint),
        },
        "input_binding": dict(input_binding),
        "producer_sources": sources,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_json(Path(path), payload)
    return payload


def load_and_validate_target_free_receipt(
    receipt_path: str | Path,
    *,
    contract: ExperimentContract,
    data: TargetFreeVLRouterData,
    prediction_path: str | Path,
    checkpoint_path: str | Path,
    expected_backbone: str | None = None,
    expected_split: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    path = Path(receipt_path).resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionIntegrityError(f"invalid target-free prediction receipt: {path}") from exc
    if not isinstance(receipt, dict):
        raise PredictionIntegrityError("target-free prediction receipt must be an object")
    recorded_internal = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if recorded_internal != canonical_sha256(payload):
        raise PredictionIntegrityError("target-free prediction receipt internal digest mismatch")
    if receipt.get("kind") != "target_free_prediction_receipt":
        raise PredictionIntegrityError("unexpected prediction receipt kind")
    if receipt.get("contract_sha256") != contract.sha256:
        raise PredictionIntegrityError("prediction receipt contract mismatch")
    if receipt.get("target_free_input_view", {}).get("manifest_sha256") != data.input_manifest_sha256:
        raise PredictionIntegrityError("prediction receipt target-free input view mismatch")
    boundary = receipt.get("process_boundary", {})
    guard = boundary.get("target_matrix_guard", {})
    if (
        boundary.get("prediction_process_loaded_y_or_c") is not False
        or boundary.get("evaluation_performed_in_this_process") is not False
        or boundary.get("must_exit_before_evaluator") is not True
        or guard.get("active") is not True
        or int(guard.get("attempt_count", -1)) != 0
        or sorted(guard.get("forbidden_paths", ()))
        != sorted(str(item) for item in target_matrix_paths(contract))
    ):
        raise PredictionIntegrityError("prediction receipt does not attest a target-free process")
    prediction = Path(prediction_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    prediction_record = receipt.get("prediction", {})
    checkpoint_record = receipt.get("checkpoint", {})
    if prediction_record.get("path") != prediction.relative_to(contract.dataset_root).as_posix():
        raise PredictionIntegrityError("prediction receipt path mismatch")
    if prediction_record.get("sha256") != sha256_file(prediction):
        raise PredictionIntegrityError("prediction receipt digest mismatch")
    if prediction_record.get("frozen_before_evaluation") is not True:
        raise PredictionIntegrityError("prediction receipt lacks freeze-before-evaluation attestation")
    if checkpoint_record.get("path") != checkpoint.relative_to(contract.dataset_root).as_posix():
        raise PredictionIntegrityError("prediction receipt checkpoint path mismatch")
    if checkpoint_record.get("sha256") != sha256_file(checkpoint):
        raise PredictionIntegrityError("prediction receipt checkpoint digest mismatch")
    if expected_backbone is not None and receipt.get("backbone") != expected_backbone:
        raise PredictionIntegrityError("prediction receipt backbone mismatch")
    if expected_split is not None and receipt.get("split") != expected_split:
        raise PredictionIntegrityError("prediction receipt split mismatch")
    if expected_seed is not None and receipt.get("seed") != int(expected_seed):
        raise PredictionIntegrityError("prediction receipt seed mismatch")
    for relative, expected in receipt.get("producer_sources", {}).items():
        source = contract.dataset_root / str(relative)
        if not source.is_file() or sha256_file(source) != expected:
            raise PredictionIntegrityError(f"prediction producer source changed: {relative}")
    receipt["receipt_file_sha256"] = sha256_file(path)
    receipt["validated"] = True
    return receipt


def target_free_loader_audit(contract: ExperimentContract) -> dict[str, Any]:
    """Run the production metadata loader under the Y/C denial guard."""

    with deny_target_matrix_access(contract) as guard:
        data = load_target_free_vlrb(contract.dataset_root, contract=contract)
        evidence = {
            "samples": len(data.samples),
            "experts": len(data.model_order),
            "split_counts": {name: len(rows) for name, rows in data.split_to_rows.items()},
            "input_manifest_sha256": data.input_manifest_sha256,
            "data_has_y": hasattr(data, "y"),
            "data_has_c": hasattr(data, "c"),
            "sample_has_y_row": hasattr(data.samples[0], "y_row"),
            "sample_has_c_row": hasattr(data.samples[0], "c_row"),
        }
    evidence["guard"] = guard
    evidence["passed"] = bool(
        guard.get("passed") is True
        and not evidence["data_has_y"]
        and not evidence["data_has_c"]
        and not evidence["sample_has_y_row"]
        and not evidence["sample_has_c_row"]
    )
    if evidence["passed"] is not True:
        raise ContractError("target-free prediction input audit failed")
    return evidence
