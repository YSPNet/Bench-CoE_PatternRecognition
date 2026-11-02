"""Fail-closed execution gate for the post-audit OOD extension."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import ExperimentContract, load_contract, sha256_file
from .data import load_vlrb_strict
from .errors import ContractError, DataIntegrityError
from .feature_cache import load_full_feature_cache
from .input_content import canonical_sha256, feature_input_content_fields
from .ood_common import (
    OOD_CONFIG_PATH,
    OOD_GATE_PATH,
    OOD_PROTOCOL_PATH,
    atomic_json,
    full_feature_cache_root,
    load_strict_json,
    resolve_under_root,
)
from .ood_protocol import load_and_validate_protocol


EXECUTION_SOURCES = (
    "routers/bench_coe_query/ood_common.py",
    "routers/bench_coe_query/ood_protocol.py",
    "routers/bench_coe_query/ood_gate.py",
    "routers/bench_coe_query/ood_training.py",
    "routers/bench_coe_query/ood_prediction.py",
    "routers/bench_coe_query/ood_evaluator.py",
    "routers/bench_coe_query/ood_aggregate.py",
    "routers/bench_coe_query/bert_router.py",
    "routers/bench_coe_query/head_trainer.py",
    "routers/bench_coe_query/heads.py",
    "routers/bench_coe_query/checkpoints.py",
    "routers/bench_coe_query/predictor.py",
    "routers/bench_coe_query/target_free.py",
    "routers/bench_coe_query/full_policy.py",
    "routers/bench_coe_query/feature_cache.py",
    "routers/bench_coe_query/labels.py",
    "routers/bench_coe_query/trainer.py",
    "routers/bench_coe_query/evaluator.py",
    "routers/utils/rank_score.py",
    "scripts/bench_coe_query/run_ood_experiment.sh",
)


def _validate_internal_digest(value: Mapping[str, Any], field: str) -> None:
    recorded = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not recorded or recorded != canonical_sha256(unsigned):
        raise DataIntegrityError(f"invalid internal digest: {field}")


def _prompt14_evidence(contract: ExperimentContract) -> dict[str, Any]:
    final_root = contract.dataset_root / "outputs/bench_coe_query/audit_final"
    archive_root = contract.dataset_root / "outputs/bench_coe_query/audit_history/prompt14_round5"
    required = ("REPRODUCIBILITY_AUDIT.md", "ISSUES.json", "VERDICT.json")
    if sorted(path.name for path in final_root.iterdir() if path.is_file()) != sorted(required):
        raise ContractError("audit_final must contain exactly the three Prompt 14 artifacts")
    final_hashes: dict[str, str] = {}
    for name in required:
        current = final_root / name
        archived = archive_root / name
        if not current.is_file() or not archived.is_file():
            raise ContractError(f"Prompt 14 artifact is missing: {name}")
        current_hash = sha256_file(current)
        if current_hash != sha256_file(archived):
            raise ContractError(f"Prompt 14 final/archive mismatch: {name}")
        final_hashes[name] = current_hash
    verdict = load_strict_json(final_root / "VERDICT.json")
    issues = load_strict_json(final_root / "ISSUES.json")
    if (
        verdict.get("verdict") != "pass"
        or int(verdict.get("issue_count", -1)) != 0
        or int(verdict.get("validation_failed", -1)) != 0
        or issues.get("issues") not in ([], None)
    ):
        raise ContractError("Prompt 14 did not finish with an exact pass and zero issues")
    return {
        "round": 5,
        "verdict": "pass",
        "issue_count": 0,
        "final_and_archive_byte_identical": True,
        "artifacts": final_hashes,
    }


def _main_final_evidence(contract: ExperimentContract) -> dict[str, Any]:
    path = contract.dataset_root / "outputs/bench_coe_query/final/FINAL_GATE.json"
    gate = load_strict_json(path)
    if gate.get("passed") is not True or gate.get("contract_sha256") != contract.sha256:
        raise ContractError("main 7:1:2 FINAL_GATE is not passing")
    final_root = path.parent
    artifacts: dict[str, str] = {}
    for name, record in gate.get("artifacts", {}).items():
        artifact = final_root / str(record.get("path", name))
        digest = sha256_file(artifact)
        if digest != record.get("sha256"):
            raise ContractError(f"main final artifact changed: {artifact}")
        artifacts[artifact.relative_to(contract.dataset_root).as_posix()] = digest
    artifacts[path.relative_to(contract.dataset_root).as_posix()] = sha256_file(path)
    return {"passed": True, "artifacts": artifacts}


def _source_evidence(contract: ExperimentContract) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in EXECUTION_SOURCES:
        path = contract.dataset_root / relative
        if not path.is_file():
            raise ContractError(f"OOD execution source is missing: {relative}")
        result[relative] = sha256_file(path)
    scan_paths = [contract.dataset_root / value for value in EXECUTION_SOURCES]
    forbidden = re.compile(
        r"(?:openai\.|google\.generativeai|anthropic\.|requests\.|urllib\.request|\.generate\(|train_test_split\()"
    )
    violations = []
    for path in scan_paths:
        if path.name == "ood_gate.py":
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if forbidden.search(line):
                violations.append(f"{path.name}:{line_number}")
    if violations:
        raise ContractError(f"forbidden network/generation/resplit token in OOD sources: {violations}")
    prediction_text = (contract.dataset_root / "routers/bench_coe_query/ood_prediction.py").read_text(
        encoding="utf-8"
    )
    if "load_vlrb_strict" in prediction_text:
        raise ContractError("OOD prediction source imports the target-bearing strict loader")
    return result


def _gpu_evidence() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "-i",
        "0,1,2,3",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("cannot inspect physical GPUs 0-3") from exc
    observed: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ContractError(f"unexpected nvidia-smi row: {line}")
        index = int(fields[0])
        observed[index] = {
            "physical_index": index,
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "utilization_percent": int(fields[4]),
        }
    if not all(index in observed for index in range(4)):
        raise ContractError("one or more approved physical GPUs 0-3 are missing")
    selected = [observed[index] for index in range(4)]
    if any(item["memory_total_mib"] < 48000 for item in selected):
        raise ContractError("approved OOD GPUs are not 48-GiB-class devices")
    if any(item["memory_used_mib"] > 1024 or item["utilization_percent"] > 10 for item in selected):
        raise ContractError("one or more approved physical GPUs 0-3 are not idle")
    return selected


def _cache_evidence(contract: ExperimentContract) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for backbone in ("tinyllava", "qwen3vl"):
        root = full_feature_cache_root(contract, backbone)
        features, records, manifest, integrity = load_full_feature_cache(
            root,
            dataset_root=contract.dataset_root,
            expected_contract_sha256=contract.sha256,
            expected_input_content=feature_input_content_fields(contract, backbone),
        )
        if len(records) != 30493 or features.shape[0] != 30493 or integrity.get("passed") is not True:
            raise ContractError(f"full feature cache is incomplete: {backbone}")
        result[backbone] = {
            "cache_root": root.relative_to(contract.dataset_root).as_posix(),
            "rows": len(records),
            "feature_dim": int(features.shape[1]),
            "manifest_internal_sha256": manifest["manifest_sha256"],
            "features_sha256": integrity["features_sha256"],
            "index_sha256": integrity["index_sha256"],
            "pointer_sha256": integrity["pointer_sha256"],
            "gate_sha256": integrity["gate_sha256"],
            "passed": True,
        }
    return result


def build_ood_execution_gate(
    contract: ExperimentContract,
    *,
    output_path: str | Path = OOD_GATE_PATH,
) -> dict[str, Any]:
    required_env = {"RUN_FULL": "1", "RUN_API_REGEN": "0", "BENCHCOE_OOD_RUN": "1"}
    if any(os.environ.get(key) != value for key, value in required_env.items()):
        raise ContractError(f"OOD gate requires environment {required_env}")
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    protocol, view = load_and_validate_protocol(
        contract,
        data,
        config_path=OOD_CONFIG_PATH,
        protocol_path=OOD_PROTOCOL_PATH,
        validate_training_statistics=True,
    )
    if protocol["author_manifest_search"]["author_original_ood_manifest_found"] is not False:
        raise ContractError("reconstructed protocol is invalid after finding an author manifest")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "prompt15_ood_execution_gate",
        "status": "ready",
        "passed": True,
        "contract_sha256": contract.sha256,
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "protocol_file_sha256": sha256_file(contract.dataset_root / OOD_PROTOCOL_PATH),
        "ood_config_sha256": sha256_file(contract.dataset_root / OOD_CONFIG_PATH),
        "view_counts": {name: len(rows) for name, rows in view.rows.items()},
        "prompt14": _prompt14_evidence(contract),
        "main_final": _main_final_evidence(contract),
        "feature_caches": _cache_evidence(contract),
        "physical_gpu_policy": {
            "allowed_indices": [0, 1, 2, 3],
            "forbidden_indices": [4, 5, 6, 7],
            "checked_before_launch": _gpu_evidence(),
        },
        "environment": required_env,
        "execution_sources": _source_evidence(contract),
        "safety": {
            "expert_or_api_regeneration": False,
            "feature_extraction": False,
            "official_split_files_written": False,
            "main_artifacts_mutable": False,
            "held_out_targets_available_to_training_or_tuning": False,
        },
    }
    payload["gate_manifest_sha256"] = canonical_sha256(payload)
    output = resolve_under_root(contract, output_path)
    atomic_json(output, payload)
    return payload


def assert_ood_execution_gate(
    contract: ExperimentContract,
    *,
    phase: str,
    gate_path: str | Path = OOD_GATE_PATH,
) -> dict[str, Any]:
    if phase not in {"train", "predict"}:
        raise ContractError(f"unsupported gated OOD phase: {phase}")
    required = {"RUN_FULL": "1", "RUN_API_REGEN": "0", "BENCHCOE_OOD_RUN": "1"}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ContractError(f"OOD {phase} requires environment {required}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3"}:
        raise ContractError(
            "OOD GPU process must expose exactly one approved physical GPU from 0-3"
        )
    path = resolve_under_root(contract, gate_path)
    gate = load_strict_json(path)
    _validate_internal_digest(gate, "gate_manifest_sha256")
    if (
        gate.get("passed") is not True
        or gate.get("status") != "ready"
        or gate.get("contract_sha256") != contract.sha256
    ):
        raise ContractError("OOD execution gate is not ready for this contract")
    for relative, digest in gate.get("execution_sources", {}).items():
        source = contract.dataset_root / str(relative)
        if not source.is_file() or sha256_file(source) != digest:
            raise ContractError(f"OOD execution source changed after gate: {relative}")
    if sha256_file(contract.dataset_root / OOD_PROTOCOL_PATH) != gate.get(
        "protocol_file_sha256"
    ):
        raise ContractError("OOD protocol changed after execution gate")
    if sha256_file(contract.dataset_root / OOD_CONFIG_PATH) != gate.get("ood_config_sha256"):
        raise ContractError("OOD config changed after execution gate")
    for relative, digest in gate.get("main_final", {}).get("artifacts", {}).items():
        artifact = contract.dataset_root / str(relative)
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ContractError(f"audited main artifact changed after OOD gate: {relative}")
    current_audit = _prompt14_evidence(contract)
    if current_audit != gate.get("prompt14"):
        raise ContractError("Prompt 14 pass evidence changed after OOD gate")
    gate["gate_file_sha256"] = sha256_file(path)
    gate["validated_for_phase"] = phase
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--contract", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--output", default=str(OOD_GATE_PATH))
    parser.add_argument("--phase", choices=("train", "predict"), default="train")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    gate = (
        build_ood_execution_gate(contract, output_path=args.output)
        if args.command == "build"
        else assert_ood_execution_gate(contract, phase=args.phase, gate_path=args.output)
    )
    print(
        json.dumps(
            {
                "passed": True,
                "status": gate["status"],
                "gate_manifest_sha256": gate["gate_manifest_sha256"],
                "physical_gpu_indices": [0, 1, 2, 3],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
