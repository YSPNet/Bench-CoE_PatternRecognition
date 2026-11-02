"""Deterministic held-out-dataset views over the immutable official splits.

These are protocol views, not replacement data splits: membership is the exact
intersection of an official split with a frozen dataset-name partition.  The
official ``SPLITS/*.jsonl`` files are never written by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import ExperimentContract, sha256_file
from .errors import ContractError, DataIntegrityError
from .input_content import canonical_sha256
from .labels import build_diagnostic_labels_for_split, compute_train_expert_accuracy


DEFAULT_OOD_CONFIG = Path("configs/bench_coe_query/ood_reconstructed.yaml")
DEFAULT_PROTOCOL_ROOT = Path("outputs/bench_coe_query/ood/protocol")
PROTOCOL_FILENAME = "HELD_OUT_PROTOCOL.json"
ROLE_NAMES = ("train_seen", "dev_seen", "id_test", "ood_test")
EXPECTED_HELD_OUT = MappingProxyType(
    {
        "general": "MMMU_DEV_VAL",
        "stem": "MathVision_MINI",
        "charts_ocr": "ChartQA_TEST",
    }
)


@dataclass(frozen=True)
class OODProtocolView:
    rows: Mapping[str, tuple[int, ...]]
    held_out_datasets: tuple[str, ...]
    seen_datasets: tuple[str, ...]
    ignored_held_out_train_rows: tuple[int, ...]
    ignored_held_out_dev_rows: tuple[int, ...]


def _strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def load_ood_config(
    contract: ExperimentContract,
    path: str | Path = DEFAULT_OOD_CONFIG,
) -> tuple[dict[str, Any], Path, str]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = contract.dataset_root / config_path
    if not config_path.is_file():
        raise ContractError(f"OOD configuration is missing: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ContractError("OOD configuration root must be a mapping")
    if value.get("kind") != "reconstructed_held_out_protocol":
        raise ContractError("OOD configuration must be explicitly reconstructed")
    if value.get("base_contract_sha256") != contract.sha256:
        raise ContractError("OOD configuration/base contract hash mismatch")
    protocol = value.get("protocol", {})
    held = protocol.get("held_out_datasets", {})
    if held != dict(EXPECTED_HELD_OUT):
        raise ContractError(f"unexpected held-out datasets: {held}")
    if protocol.get("official_splits_overwritten") is not False:
        raise ContractError("OOD configuration may not replace official splits")
    if protocol.get("exact_table_a2_reproduction_claimed") is not False:
        raise ContractError("reconstructed OOD protocol cannot claim exact Table A2 reproduction")
    safety = value.get("safety", {})
    if safety.get("physical_gpu_indices") != [0, 1, 2, 3]:
        raise ContractError("OOD execution must be restricted to physical GPUs 0-3")
    if safety.get("feature_extraction_allowed") is not False:
        raise ContractError("OOD protocol must reuse features and forbid extraction")
    return value, config_path, sha256_file(config_path)


def derive_protocol_view(data: Any, config: Mapping[str, Any]) -> OODProtocolView:
    held = tuple(str(value) for value in config["protocol"]["held_out_datasets"].values())
    seen = tuple(str(value) for value in config["protocol"]["seen_datasets"])
    if len(held) != 3 or len(set(held)) != 3:
        raise DataIntegrityError("OOD held-out dataset list must contain three unique names")
    if len(seen) != 11 or len(set(seen)) != 11:
        raise DataIntegrityError("OOD seen dataset list must contain eleven unique names")
    observed = {str(sample.dataset) for sample in data.samples}
    if set(held).intersection(seen) or set(held).union(seen) != observed:
        raise DataIntegrityError(
            "seen/held-out dataset partition does not exactly cover the 14 frozen datasets"
        )

    def select(official_split: str, datasets: set[str]) -> tuple[int, ...]:
        return tuple(
            int(row)
            for row in data.split_to_rows[official_split]
            if str(data.samples[int(row)].dataset) in datasets
        )

    held_set = set(held)
    seen_set = set(seen)
    rows = {
        "train_seen": select("train", seen_set),
        "dev_seen": select("dev", seen_set),
        "id_test": select("test", seen_set),
        "ood_test": select("test", held_set),
    }
    ignored_train = select("train", held_set)
    ignored_dev = select("dev", held_set)
    if any(not values for values in rows.values()):
        raise DataIntegrityError("one or more OOD protocol views are empty")
    if set(rows["train_seen"]).intersection(rows["dev_seen"]):
        raise DataIntegrityError("train_seen/dev_seen overlap")
    if set(rows["id_test"]).intersection(rows["ood_test"]):
        raise DataIntegrityError("ID/OOD test views overlap")
    if tuple(sorted(rows["id_test"] + rows["ood_test"])) != tuple(
        sorted(int(row) for row in data.split_to_rows["test"])
    ):
        raise DataIntegrityError("ID/OOD views do not exactly partition official test")
    return OODProtocolView(
        rows=MappingProxyType(rows),
        held_out_datasets=held,
        seen_datasets=seen,
        ignored_held_out_train_rows=ignored_train,
        ignored_held_out_dev_rows=ignored_dev,
    )


def _role_payload(data: Any, role: str, rows: Sequence[int]) -> str:
    official_split = "train" if role == "train_seen" else "dev" if role == "dev_seen" else "test"
    return "".join(
        json.dumps(
            {
                "dataset": str(data.samples[int(row)].dataset),
                "official_split": official_split,
                "protocol_role": role,
                "row_index": int(row),
                "sample_id": str(data.samples[int(row)].sample_id),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )


def _author_manifest_search(contract: ExperimentContract, config: Mapping[str, Any]) -> dict[str, Any]:
    search = config["author_manifest_search"]
    ignored = tuple(str(item).rstrip("/") + "/" for item in search["ignored_extension_roots"])
    candidates: list[str] = []
    text_matches: list[dict[str, Any]] = []
    pattern = re.compile(
        r"held.?out|\bOOD\b|test_ood(?:\.jsonl|\b)|test_id(?:\.jsonl|\b)",
        re.IGNORECASE,
    )
    scanned_files: list[Path] = []
    for root_name in search["official_search_roots"]:
        root = contract.dataset_root / str(root_name)
        values = [root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []
        for path in values:
            if not path.is_file():
                continue
            relative = path.relative_to(contract.dataset_root).as_posix()
            if relative.startswith(ignored):
                continue
            scanned_files.append(path)
            lower_name = path.name.lower()
            if "ood" in lower_name or "held" in lower_name or lower_name in {
                "test_id.jsonl",
                "test_ood.jsonl",
            }:
                candidates.append(relative)
            if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".sh", ".json"}:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, 1):
                if pattern.search(line):
                    text_matches.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                        }
                    )
    expected_candidates = [
        contract.dataset_root / str(value) for value in search["expected_manifest_candidates"]
    ]
    expected_candidate_status = {
        path.relative_to(contract.dataset_root).as_posix(): path.is_file()
        for path in expected_candidates
    }
    manifest_like_candidates = sorted(
        value for value in set(candidates) if value.startswith("SPLITS/")
    )
    found = bool(manifest_like_candidates or any(expected_candidate_status.values()))
    return {
        "upstream_commit": search["upstream_commit"],
        "upstream_archive_sha256": search["upstream_archive_sha256"],
        "upstream_files_scanned_in_prior_byte_audit": int(search["upstream_files_scanned"]),
        "current_official_scope_files_scanned": len(set(scanned_files)),
        "candidate_manifest_files": manifest_like_candidates,
        "expected_candidate_presence": expected_candidate_status,
        "text_matches": text_matches,
        "author_original_ood_manifest_found": found,
        "interpretation": (
            "No author OOD membership manifest exists in the pinned repository. "
            "The only official OOD-like source reference is legacy cleanup code "
            "for absent test_id/test_ood files; it defines no membership."
        ),
    }


def build_protocol(
    contract: ExperimentContract,
    data: Any,
    *,
    config_path: str | Path = DEFAULT_OOD_CONFIG,
    output_root: str | Path = DEFAULT_PROTOCOL_ROOT,
) -> dict[str, Any]:
    config, resolved_config, config_sha256 = load_ood_config(contract, config_path)
    view = derive_protocol_view(data, config)
    output = Path(output_root)
    if not output.is_absolute():
        output = contract.dataset_root / output
    output.mkdir(parents=True, exist_ok=True)

    role_files: dict[str, Any] = {}
    for role in ROLE_NAMES:
        role_path = output / f"{role}.jsonl"
        payload = _role_payload(data, role, view.rows[role])
        _atomic_text(role_path, payload)
        role_files[role] = {
            "path": role_path.relative_to(contract.dataset_root).as_posix(),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "count": len(view.rows[role]),
            "official_split": (
                "train" if role == "train_seen" else "dev" if role == "dev_seen" else "test"
            ),
            "datasets": sorted({data.samples[row].dataset for row in view.rows[role]}),
            "row_indices_sha256": hashlib.sha256(
                "\n".join(str(row) for row in view.rows[role]).encode("utf-8")
            ).hexdigest(),
        }

    train_rows = view.rows["train_seen"]
    train_accuracy = compute_train_expert_accuracy(
        data.y,
        train_rows,
        data.model_order,
        expected_model_order=data.model_order,
    )
    train_labels = build_diagnostic_labels_for_split(
        data.y,
        train_rows,
        train_accuracy,
        data.model_order,
        expected_model_order=data.model_order,
    )
    strongest = int(np.argmax(train_accuracy))
    source_search = _author_manifest_search(contract, config)
    if source_search["author_original_ood_manifest_found"]:
        raise ContractError(
            "an author OOD manifest candidate exists; reconstructed protocol creation refused"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "reconstructed_held_out_protocol",
        "claim": "reconstructed held-out protocol; not an exact per-sample reproduction of Table A2",
        "base_contract": contract.path.relative_to(contract.dataset_root).as_posix(),
        "base_contract_sha256": contract.sha256,
        "ood_config": resolved_config.relative_to(contract.dataset_root).as_posix(),
        "ood_config_sha256": config_sha256,
        "official_splits_overwritten": False,
        "selection_rule": "official_split_intersection_frozen_dataset_partition",
        "held_out_datasets_by_domain": dict(config["protocol"]["held_out_datasets"]),
        "seen_datasets": list(view.seen_datasets),
        "views": role_files,
        "ignored_not_used_for_training_or_tuning": {
            "held_out_official_train_rows": len(view.ignored_held_out_train_rows),
            "held_out_official_dev_rows": len(view.ignored_held_out_dev_rows),
        },
        "training_statistics": {
            "source_view": "train_seen",
            "held_out_y_used": False,
            "model_order": list(data.model_order),
            "expert_accuracy": [float(value) for value in train_accuracy],
            "strongest_expert_index": strongest,
            "strongest_expert_id": data.model_order[strongest],
            "trainable_label_rows": int(train_labels.is_trainable.sum()),
            "all_wrong_ignored_rows": int((~train_labels.is_trainable).sum()),
        },
        "author_manifest_search": source_search,
        "feature_reuse_boundary": {
            "reuse_existing_unlabeled_query_features": True,
            "new_feature_extraction": False,
            "pretraining_caveat": "The pretrained backbones may have seen these benchmark datasets.",
        },
    }
    payload["protocol_manifest_sha256"] = canonical_sha256(payload)
    _atomic_json(output / PROTOCOL_FILENAME, payload)
    return payload


def load_and_validate_protocol(
    contract: ExperimentContract,
    data: Any,
    *,
    config_path: str | Path = DEFAULT_OOD_CONFIG,
    protocol_path: str | Path = DEFAULT_PROTOCOL_ROOT / PROTOCOL_FILENAME,
    validate_training_statistics: bool = False,
) -> tuple[dict[str, Any], OODProtocolView]:
    config, resolved_config, config_sha256 = load_ood_config(contract, config_path)
    path = Path(protocol_path)
    if not path.is_absolute():
        path = contract.dataset_root / path
    protocol = _strict_json(path)
    if not isinstance(protocol, dict):
        raise DataIntegrityError("OOD protocol manifest must be an object")
    recorded = str(protocol.get("protocol_manifest_sha256", ""))
    unsigned = dict(protocol)
    unsigned.pop("protocol_manifest_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise DataIntegrityError("OOD protocol internal hash mismatch")
    if protocol.get("base_contract_sha256") != contract.sha256:
        raise DataIntegrityError("OOD protocol/base contract mismatch")
    if protocol.get("ood_config_sha256") != config_sha256:
        raise DataIntegrityError("OOD protocol/config hash mismatch")
    if protocol.get("ood_config") != resolved_config.relative_to(contract.dataset_root).as_posix():
        raise DataIntegrityError("OOD protocol/config path mismatch")
    if protocol.get("official_splits_overwritten") is not False:
        raise DataIntegrityError("OOD protocol claims official split mutation")
    if protocol.get("author_manifest_search", {}).get("author_original_ood_manifest_found") is not False:
        raise DataIntegrityError("reconstructed protocol cannot supersede an author manifest")

    view = derive_protocol_view(data, config)
    for role in ROLE_NAMES:
        record = protocol.get("views", {}).get(role, {})
        role_path = contract.dataset_root / str(record.get("path", ""))
        expected_payload = _role_payload(data, role, view.rows[role])
        expected_sha = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
        if (
            not role_path.is_file()
            or sha256_file(role_path) != expected_sha
            or record.get("sha256") != expected_sha
            or int(record.get("count", -1)) != len(view.rows[role])
            or role_path.read_text(encoding="utf-8") != expected_payload
        ):
            raise DataIntegrityError(f"OOD protocol role manifest mismatch: {role}")
    if validate_training_statistics:
        if not hasattr(data, "y"):
            raise DataIntegrityError("training-statistics validation requires strict Y data")
        accuracy = compute_train_expert_accuracy(
            data.y,
            view.rows["train_seen"],
            data.model_order,
            expected_model_order=data.model_order,
        )
        stored = np.asarray(protocol["training_statistics"]["expert_accuracy"], dtype=np.float64)
        if not np.array_equal(accuracy, stored):
            raise DataIntegrityError("OOD train_seen expert accuracy mismatch")
        strongest = int(np.argmax(accuracy))
        if protocol["training_statistics"].get("strongest_expert_index") != strongest:
            raise DataIntegrityError("OOD train_seen strongest expert mismatch")
    return protocol, view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument(
        "--contract", default="configs/bench_coe_query/experiment_contract.yaml"
    )
    parser.add_argument("--config", default=str(DEFAULT_OOD_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_PROTOCOL_ROOT))
    args = parser.parse_args()

    from .config import load_contract
    from .data import load_vlrb_strict

    contract = load_contract(args.contract)
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    protocol_path = Path(args.output_root) / PROTOCOL_FILENAME
    if args.command == "build":
        protocol = build_protocol(
            contract,
            data,
            config_path=args.config,
            output_root=args.output_root,
        )
        view = derive_protocol_view(data, load_ood_config(contract, args.config)[0])
    else:
        protocol, view = load_and_validate_protocol(
            contract,
            data,
            config_path=args.config,
            protocol_path=protocol_path,
            validate_training_statistics=True,
        )
    print(
        json.dumps(
            {
                "passed": True,
                "protocol": str(protocol_path),
                "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
                "author_original_ood_manifest_found": protocol[
                    "author_manifest_search"
                ]["author_original_ood_manifest_found"],
                "views": {name: len(view.rows[name]) for name in ROLE_NAMES},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
