"""Loading and validation for the frozen Bench-CoE query experiment contract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .errors import ContractError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = REPOSITORY_ROOT / "configs/bench_coe_query/experiment_contract.yaml"
FULL_RUN_APPROVAL_RELATIVE_PATH = Path("outputs/bench_coe_query/audit/FULL_RUN_APPROVAL.json")
FULL_MULTI_IMAGE_POLICY = "exclude_with_frozen_denominator_report"
FULL_MULTI_IMAGE_FALLBACK = "train_strongest_global"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _as_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _as_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_as_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentContract:
    path: Path
    sha256: str
    values: Mapping[str, Any]

    @property
    def dataset_root(self) -> Path:
        return Path(str(self.values["data"]["root"])).resolve()

    @property
    def model_order(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.values["experts"]["canonical_order"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.values["reproducibility"]["seeds"])

    def resolve(self, relative_or_absolute: str | Path) -> Path:
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else self.dataset_root / path

    def plain(self) -> dict[str, Any]:
        return _as_plain(self.values)


def validate_contract_values(values: Mapping[str, Any]) -> None:
    required_sections = {
        "method",
        "data",
        "experts",
        "labels",
        "evaluation",
        "reproducibility",
        "models",
        "profiles",
        "features",
        "scheduling",
        "safety",
    }
    missing = sorted(required_sections.difference(values))
    if missing:
        raise ContractError(f"contract missing required sections: {missing}")

    status = values["method"].get("implementation_status")
    if status not in {"faithful", "adapted"}:
        raise ContractError(f"invalid implementation_status: {status!r}")

    order = list(values["experts"].get("canonical_order", []))
    expected_experts = int(values["data"]["expected_experts"])
    if len(order) != expected_experts or len(set(order)) != len(order):
        raise ContractError("canonical expert order must be unique and match expected_experts")
    if order[0:2] != ["GPT4o", "GeminiFlash2-5"]:
        raise ContractError("canonical API expert IDs/order do not match the frozen release")

    seeds = list(values["reproducibility"].get("seeds", []))
    if seeds != [0, 1, 2, 3, 4]:
        raise ContractError(f"seeds must be exactly [0, 1, 2, 3, 4], got {seeds}")
    if values["labels"].get("all_wrong_policy") != "exclude_from_ce":
        raise ContractError("main all-wrong policy must be exclude_from_ce")
    if int(values["labels"].get("ignore_index", 0)) != -100:
        raise ContractError("main ignore_index must be -100")
    checkpoint = values["evaluation"].get("checkpoint_selection", {})
    if checkpoint.get("split") != "dev" or checkpoint.get("monitor") != "dev_hard_label_loss":
        raise ContractError("checkpoint selection must use dev_hard_label_loss")
    if int(values["safety"].get("RUN_API_REGEN", -1)) != 0:
        raise ContractError("RUN_API_REGEN must be frozen to 0")
    if int(values["safety"].get("RUN_FULL", -1)) != 0:
        raise ContractError("contract default RUN_FULL must be 0")

    for name in ("bert", "tinyllava", "qwen3vl"):
        model = values["models"].get(name, {})
        if not model.get("path") or not model.get("config_sha256"):
            raise ContractError(f"model {name!r} lacks a local path or config hash")
    for name in ("tinyllava", "qwen3vl"):
        if values["models"][name].get("backbone_frozen") is not True:
            raise ContractError(f"{name} backbone must be frozen")


def load_contract(path: str | Path | None = None) -> ExperimentContract:
    contract_path = Path(path or DEFAULT_CONTRACT_PATH).resolve()
    if not contract_path.is_file():
        raise ContractError(f"contract not found: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ContractError("contract root must be a mapping")
    validate_contract_values(values)
    return ExperimentContract(contract_path, sha256_file(contract_path), _freeze(values))


def load_full_run_approval(
    contract: ExperimentContract,
    *,
    required: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load the immutable-contract approval overlay and validate every decision.

    The experiment contract remains byte-frozen. Human decisions made after the
    Prompt-10 review live in a separate, hash-bound audit artifact so a full run
    cannot silently reinterpret an older approval after the quality gate.
    """

    path = contract.dataset_root / FULL_RUN_APPROVAL_RELATIVE_PATH
    if not path.is_file():
        if required:
            raise ContractError(f"full-run approval is missing: {path}")
        return None, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            approval = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid full-run approval artifact: {path}") from exc
    if not isinstance(approval, dict):
        raise ContractError("full-run approval artifact must be a JSON object")
    decisions = approval.get("decisions")
    expected = {
        "source_provenance": "approve_hash_pinned_candidate_as_adapted",
        "tinyllava_pooling": "approve_hash_pinned_adapted_contract",
        "qwen3vl_pooling": "approve_hash_pinned_adapted_contract",
        "multi_image_policy": FULL_MULTI_IMAGE_POLICY,
        "multi_image_fallback": FULL_MULTI_IMAGE_FALLBACK,
        "full_run_authorized": True,
    }
    if approval.get("contract_sha256") != contract.sha256:
        raise ContractError("full-run approval is bound to a different experiment contract")
    if not isinstance(decisions, dict):
        raise ContractError("full-run approval lacks a decisions object")
    mismatched = {
        key: {"expected": value, "observed": decisions.get(key)}
        for key, value in expected.items()
        if decisions.get(key) != value
    }
    if mismatched:
        raise ContractError(f"full-run approval decisions are incomplete or mismatched: {mismatched}")
    if not str(approval.get("authorized_at", "")).strip():
        raise ContractError("full-run approval lacks authorized_at")
    if not str(approval.get("authorization_basis", "")).strip():
        raise ContractError("full-run approval lacks authorization_basis")
    # The immutable YAML contract is extended only through this separately
    # authorized, hash-pinned overlay.  Validate the overlay before any full
    # run can use local model or image bytes.
    from .input_content import validate_input_content_approval_binding

    validate_input_content_approval_binding(contract, approval)
    return approval, sha256_file(path)


def effective_model_path(contract: ExperimentContract, model_name: str) -> Path:
    try:
        model = contract.values["models"][model_name]
    except KeyError as exc:
        raise ContractError(f"unknown router model {model_name!r}") from exc
    env_name = str(model["path_env"])
    configured = Path(str(model["path"])).resolve()
    path = Path(os.environ.get(env_name, str(configured))).resolve()
    if path != configured:
        raise ContractError(
            f"{env_name} differs from the frozen user-confirmed path: observed={path}, expected={configured}"
        )
    return path


def assert_run_safety(
    contract: ExperimentContract,
    *,
    requested_samples: int | None = None,
    requested_epochs: int | None = None,
    full_operation: bool = False,
    artifact_maintenance: bool = False,
) -> None:
    api_regen = os.environ.get("RUN_API_REGEN", str(contract.values["safety"]["RUN_API_REGEN"]))
    if api_regen != "0":
        raise ContractError("RUN_API_REGEN must equal 0")

    run_full = os.environ.get("RUN_FULL", str(contract.values["safety"]["RUN_FULL"]))
    if full_operation and run_full != "1":
        raise ContractError("full operation refused because RUN_FULL != 1")

    if full_operation and run_full == "1":
        gate_path = contract.dataset_root / str(contract.values["safety"]["output_root"]) / "smoke/QUALITY_GATE.json"
        if not gate_path.is_file():
            raise ContractError(f"full operation refused because quality gate is missing: {gate_path}")
        with gate_path.open("r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if gate.get("contract_sha256") != contract.sha256:
            raise ContractError("full operation refused because quality gate/contract hash differs")
        required_gate = (
            "ready_for_artifact_maintenance"
            if artifact_maintenance
            else "ready_for_full_run"
        )
        if gate.get(required_gate) is not True:
            raise ContractError(
                f"full operation refused because QUALITY_GATE.{required_gate} is not true"
            )
        recorded_hashes = gate.get("source_safety_audit", {}).get("scanned_file_hashes", {})
        for relative, recorded in recorded_hashes.items():
            source = contract.dataset_root / str(relative)
            if not source.is_file() or sha256_file(source) != recorded:
                raise ContractError(f"full operation refused because gated source changed: {relative}")
        for relative, recorded in gate.get("frozen_runtime_inputs", {}).items():
            artifact = contract.dataset_root / str(relative)
            if not artifact.is_file() or sha256_file(artifact) != recorded:
                raise ContractError(f"full operation refused because gated runtime input changed: {relative}")
        _approval, approval_sha256 = load_full_run_approval(contract)
        recorded_approval = gate.get("full_run_approval", {})
        if recorded_approval.get("sha256") != approval_sha256:
            raise ContractError("full operation refused because the gated approval artifact changed")
        disk = shutil.disk_usage(contract.dataset_root)
        free_percent = 100.0 * disk.free / disk.total
        required = float(contract.values["safety"]["minimum_full_run_free_disk_percent"])
        if free_percent < required:
            raise ContractError(
                f"full operation refused because free disk is {free_percent:.2f}% < {required:.2f}%"
            )

    limits = contract.values["safety"]["smoke_limits"]
    if run_full != "1":
        maximum_samples = max(int(limits[name]) for name in ("train", "dev", "test"))
        if requested_samples is not None and requested_samples > maximum_samples:
            raise ContractError(
                f"requested {requested_samples} samples exceeds RUN_FULL=0 smoke maximum {maximum_samples}"
            )
        if requested_epochs is not None and requested_epochs > int(limits["epochs"]):
            raise ContractError(
                f"requested {requested_epochs} epochs exceeds RUN_FULL=0 maximum {limits['epochs']}"
            )


def assert_full_extraction_gates(contract: ExperimentContract) -> None:
    """Require both hash-bound Prompt-11 extraction gates before Prompt 12."""

    from .errors import FeatureCacheError
    from .feature_cache import load_full_feature_cache
    from .input_content import feature_input_content_fields

    for backbone in ("tinyllava", "qwen3vl"):
        pointer_path = (
            contract.dataset_root
            / str(contract.values["features"]["root"])
            / backbone
            / "LATEST_FULL.json"
        )
        if not pointer_path.is_file():
            raise ContractError(f"full training refused because {backbone} extraction pointer is missing")
        with pointer_path.open("r", encoding="utf-8") as handle:
            pointer = json.load(handle)
        gate_path = Path(str(pointer.get("extraction_gate", "")))
        if not gate_path.is_absolute():
            gate_path = contract.dataset_root / gate_path
        if not gate_path.is_file():
            raise ContractError(f"full training refused because {backbone} extraction gate is missing")
        with gate_path.open("r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if gate.get("passed") is not True or gate.get("contract_sha256") != contract.sha256:
            raise ContractError(f"full training refused because {backbone} extraction gate did not pass")
        if pointer.get("gate_sha256") != sha256_file(gate_path):
            raise ContractError(f"full training refused because {backbone} extraction gate hash changed")
        manifest_path = Path(str(pointer.get("manifest", "")))
        if not manifest_path.is_absolute():
            manifest_path = contract.dataset_root / manifest_path
        if not manifest_path.is_file() or gate.get("feature_manifest_sha256") != sha256_file(manifest_path):
            raise ContractError(f"full training refused because {backbone} feature manifest changed")
        try:
            _features, _records, _manifest, integrity = load_full_feature_cache(
                pointer.get("cache_root", ""),
                dataset_root=contract.dataset_root,
                expected_contract_sha256=contract.sha256,
                expected_input_content=feature_input_content_fields(contract, backbone),
            )
        except (FeatureCacheError, OSError, ValueError, KeyError, TypeError) as exc:
            raise ContractError(
                f"full training refused because {backbone} cache integrity failed: {exc}"
            ) from exc
        if integrity.get("passed") is not True:
            raise ContractError(f"full training refused because {backbone} cache integrity did not pass")
