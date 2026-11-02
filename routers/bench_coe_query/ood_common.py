"""Shared, hash-bound helpers for the reconstructed held-out experiment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import ExperimentContract, sha256_file
from .errors import ContractError, DataIntegrityError
from .input_content import canonical_sha256
from .labels import build_diagnostic_labels_for_split, compute_train_expert_accuracy
from .ood_protocol import OODProtocolView


OOD_RUN_ID = "reconstructed-heldout-v1"
OOD_OUTPUT_ROOT = Path("outputs/bench_coe_query/ood")
OOD_GATE_PATH = OOD_OUTPUT_ROOT / "OOD_EXECUTION_GATE.json"
OOD_PROTOCOL_PATH = OOD_OUTPUT_ROOT / "protocol/HELD_OUT_PROTOCOL.json"
OOD_CONFIG_PATH = Path("configs/bench_coe_query/ood_reconstructed.yaml")
OOD_BACKBONES = ("bert", "tinyllava", "qwen3vl")
OOD_EVAL_ROLES = ("id_test", "ood_test")
OOD_SEEDS = (0, 1, 2, 3, 4)


def resolve_under_root(contract: ExperimentContract, value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else contract.dataset_root / path).resolve()
    try:
        resolved.relative_to(contract.dataset_root)
    except ValueError as exc:
        raise ContractError(f"path escapes dataset root: {resolved}") from exc
    return resolved


def atomic_text(path: str | Path, payload: str) -> None:
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


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {source}")

    try:
        value = json.loads(source.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid strict JSON object: {source}") from exc
    if not isinstance(value, dict):
        raise DataIntegrityError(f"JSON artifact is not an object: {source}")
    return value


def ood_run_dir(contract: ExperimentContract, backbone: str, seed: int) -> Path:
    if backbone not in OOD_BACKBONES or int(seed) not in OOD_SEEDS:
        raise ContractError(f"invalid OOD backbone/seed: {backbone}/{seed}")
    return contract.dataset_root / OOD_OUTPUT_ROOT / "runs" / backbone / OOD_RUN_ID / f"seed-{seed}"


def protocol_role_sample_ids(data: Any, view: OODProtocolView, role: str) -> tuple[str, ...]:
    if role not in view.rows:
        raise DataIntegrityError(f"unknown OOD protocol role: {role}")
    return tuple(str(data.samples[row].sample_id) for row in view.rows[role])


def build_ood_training_labels(
    data: Any,
    view: OODProtocolView,
) -> tuple[np.ndarray, Any, Any]:
    """Build labels from train_seen statistics; never index held-out or test targets."""

    train_accuracy = compute_train_expert_accuracy(
        data.y,
        view.rows["train_seen"],
        data.model_order,
        expected_model_order=data.model_order,
    )
    train_labels = build_diagnostic_labels_for_split(
        data.y,
        view.rows["train_seen"],
        train_accuracy,
        data.model_order,
        expected_model_order=data.model_order,
    )
    dev_labels = build_diagnostic_labels_for_split(
        data.y,
        view.rows["dev_seen"],
        train_accuracy,
        data.model_order,
        expected_model_order=data.model_order,
    )
    return train_accuracy, train_labels, dev_labels


def ood_training_binding(
    contract: ExperimentContract,
    protocol: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Bind checkpoints to the protocol, profile, and training implementation bytes."""

    source_relatives = (
        "routers/bench_coe_query/ood_common.py",
        "routers/bench_coe_query/ood_protocol.py",
        "routers/bench_coe_query/ood_training.py",
        "routers/bench_coe_query/bert_router.py",
        "routers/bench_coe_query/head_trainer.py",
        "routers/bench_coe_query/heads.py",
        "routers/bench_coe_query/labels.py",
        "routers/bench_coe_query/checkpoints.py",
        "routers/bench_coe_query/trainer.py",
    )
    sources: dict[str, str] = {}
    for relative in source_relatives:
        path = contract.dataset_root / relative
        if not path.is_file():
            raise ContractError(f"OOD training source is missing: {relative}")
        sources[relative] = sha256_file(path)
    role_hashes = {
        role: str(record["sha256"])
        for role, record in protocol["views"].items()
        if role in {"train_seen", "dev_seen"}
    }
    payload = {
        "kind": "ood_training_binding_v1",
        "base_contract_sha256": contract.sha256,
        "ood_config_sha256": protocol["ood_config_sha256"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "training_role_manifest_sha256": role_hashes,
        "profile": "vlrb_adapted_v1",
        "checkpoint_monitor": "dev_hard_label_loss",
        "implementation_sources": sources,
    }
    return canonical_sha256(payload), payload


def full_feature_cache_root(contract: ExperimentContract, backbone: str) -> Path:
    if backbone not in {"tinyllava", "qwen3vl"}:
        raise ContractError(f"no VLM feature cache for backbone {backbone}")
    pointer_path = contract.dataset_root / f"outputs/bench_coe_query/features/{backbone}/LATEST_FULL.json"
    pointer = load_strict_json(pointer_path)
    if pointer.get("backbone") != backbone or pointer.get("contract_sha256") != contract.sha256:
        raise DataIntegrityError(f"invalid full feature pointer for {backbone}")
    return resolve_under_root(contract, str(pointer.get("cache_root", "")))


def file_record(contract: ExperimentContract, path: str | Path) -> dict[str, Any]:
    resolved = resolve_under_root(contract, path)
    return {
        "path": resolved.relative_to(contract.dataset_root).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def row_indices_sha256(rows: Any) -> str:
    return hashlib.sha256("\n".join(str(int(row)) for row in rows).encode("utf-8")).hexdigest()
