"""Read-only Prompt-10 smoke audit and full-run quality gate."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import (
    FULL_MULTI_IMAGE_FALLBACK,
    FULL_MULTI_IMAGE_POLICY,
    FULL_RUN_APPROVAL_RELATIVE_PATH,
    ExperimentContract,
    load_contract,
    load_full_run_approval,
    sha256_file,
)
from .data import load_vlrb_strict
from .evaluator import evaluate_prediction_file, read_frozen_predictions, validate_predictions
from .feature_cache import feature_signature, load_full_feature_cache
from .input_content import (
    INPUT_CONTENT_MANIFEST_RELATIVE_PATH,
    feature_input_content_fields,
    verify_pinned_input_content,
)
from .full_policy import partition_feature_rows
from .head_trainer import build_head_training_labels
from .labels import build_benchcoe_hard_labels, compute_train_expert_accuracy
from .errors import ContractError, FeatureCacheError
from .target_free import target_free_loader_audit
from .provenance_remediation import DEFAULT_REPORT as PROVENANCE_REMEDIATION_REPORT
from .provenance_remediation import verify_remediation_report


SINGLE_MULTI_MIN_COSINE = 0.9999
SINGLE_MULTI_MAX_RELATIVE_L2 = 0.01
EXPECTED_HEAD_PARAMETERS = (
    "layer_norm.weight",
    "layer_norm.bias",
    "classifier.weight",
    "classifier.bias",
)


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    severity: str = "critical",
    scope: str = "technical_smoke",
    evidence: Any = None,
    required_action: str | None = None,
) -> None:
    item = {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "scope": scope,
        "evidence": evidence,
    }
    if required_action and not passed:
        item["required_action"] = required_action
    checks.append(item)


def _expected_smoke_ids(root: Path) -> tuple[str, ...]:
    return tuple(str(record["sample_id"]) for record in _jsonl(root / "outputs/bench_coe_query/manifests/smoke_test.jsonl"))


def _source_safety_audit(root: Path) -> dict[str, Any]:
    python_root = root / "routers/bench_coe_query"
    script_root = root / "scripts/bench_coe_query"
    banned_import_roots = {"aiohttp", "httpx", "openai", "requests", "urllib3"}
    forbidden_imports: list[dict[str, str]] = []
    generate_calls: list[str] = []
    split_calls: list[str] = []
    nonlocal_loads: list[str] = []
    hashes: dict[str, str] = {}
    for path in sorted(python_root.rglob("*.py")):
        relative = str(path.relative_to(root))
        hashes[relative] = sha256_file(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in banned_import_roots or alias.name.startswith("google.generativeai"):
                        forbidden_imports.append({"path": relative, "module": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".")[0] in banned_import_roots or module.startswith("google.generativeai"):
                    forbidden_imports.append({"path": relative, "module": module})
            elif isinstance(node, ast.Call):
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else ""
                if name == "generate":
                    generate_calls.append(f"{relative}:{node.lineno}")
                if name == "train_test_split":
                    split_calls.append(f"{relative}:{node.lineno}")
                if name == "from_pretrained" and path.name in {
                    "bert_router.py",
                    "runtime_predict.py",
                    "tinyllava.py",
                    "qwen3vl.py",
                }:
                    local_only = any(
                        keyword.arg == "local_files_only"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in node.keywords
                    )
                    if not local_only:
                        nonlocal_loads.append(f"{relative}:{node.lineno}")
    unsafe_shell: list[str] = []
    shell_pattern = re.compile(r"(^|[ ;|&])(curl|wget)([ ;|&]|$)|\bhf\s+download\b|run_step[1-6]", re.I)
    for path in sorted(script_root.glob("*.sh")):
        relative = str(path.relative_to(root))
        hashes[relative] = sha256_file(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if shell_pattern.search(line):
                unsafe_shell.append(f"{relative}:{line_number}")
    return {
        "passed": not (forbidden_imports or generate_calls or split_calls or nonlocal_loads or unsafe_shell),
        "forbidden_network_or_api_imports": forbidden_imports,
        "model_generate_calls": generate_calls,
        "train_test_split_calls": split_calls,
        "from_pretrained_without_local_files_only": nonlocal_loads,
        "unsafe_shell_commands": unsafe_shell,
        "scanned_file_hashes": hashes,
    }


def _feature_comparison(left_root: Path, right_root: Path) -> dict[str, Any]:
    left = np.load(left_root / "merged/features.npy", allow_pickle=False).astype(np.float32)
    right = np.load(right_root / "merged/features.npy", allow_pickle=False).astype(np.float32)
    left_index = _jsonl(left_root / "merged/index.jsonl")
    right_index = _jsonl(right_root / "merged/index.jsonl")
    ids_equal = [item["sample_id"] for item in left_index] == [item["sample_id"] for item in right_index]
    rows_equal = [item["row_index"] for item in left_index] == [item["row_index"] for item in right_index]
    shapes_equal = left.shape == right.shape
    if not shapes_equal:
        return {
            "passed": False,
            "ids_equal": ids_equal,
            "rows_equal": rows_equal,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    difference = np.abs(left - right)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, np.finfo(np.float32).tiny)
    cosine = (left * right).sum(axis=1) / denominator
    relative_l2 = np.linalg.norm(left - right, axis=1) / np.maximum(left_norm, np.finfo(np.float32).tiny)
    max_abs = float(difference.max(initial=0.0))
    result = {
        "ids_equal": ids_equal,
        "rows_equal": rows_equal,
        "shapes_equal": shapes_equal,
        "shape": list(left.shape),
        "bitwise_equal": bool(np.array_equal(left, right)),
        "exact_rows": int(np.all(left == right, axis=1).sum()),
        "max_absolute_difference": max_abs,
        "mean_absolute_difference": float(difference.mean()),
        "p99_absolute_difference": float(np.quantile(difference, 0.99)),
        "minimum_cosine_similarity": float(cosine.min()),
        "mean_cosine_similarity": float(cosine.mean()),
        "maximum_relative_l2": float(relative_l2.max(initial=0.0)),
        "acceptance": {
            "minimum_cosine_similarity": SINGLE_MULTI_MIN_COSINE,
            "maximum_relative_l2": SINGLE_MULTI_MAX_RELATIVE_L2,
            "max_absolute_difference": "report_only_for_fp16_features",
        },
    }
    result["passed"] = bool(
        ids_equal
        and rows_equal
        and float(cosine.min()) >= SINGLE_MULTI_MIN_COSINE
        and float(relative_l2.max(initial=0.0)) <= SINGLE_MULTI_MAX_RELATIVE_L2
    )
    return result


def _repeat_comparison(reference_root: Path, repeat_root: Path, count: int = 8) -> dict[str, Any]:
    reference = np.load(reference_root / "merged/features.npy", allow_pickle=False)[:count]
    repeat = np.load(repeat_root / "merged/features.npy", allow_pickle=False)
    reference_index = _jsonl(reference_root / "merged/index.jsonl")[:count]
    repeat_index = _jsonl(repeat_root / "merged/index.jsonl")
    difference = np.abs(reference.astype(np.float32) - repeat.astype(np.float32))
    exact = bool(
        [item["sample_id"] for item in reference_index] == [item["sample_id"] for item in repeat_index]
        and [item["row_index"] for item in reference_index] == [item["row_index"] for item in repeat_index]
        and np.array_equal(reference, repeat)
    )
    return {
        "passed": exact,
        "count": len(repeat),
        "bitwise_equal": bool(np.array_equal(reference, repeat)),
        "max_absolute_difference": float(difference.max(initial=0.0)),
    }


def _resource_evidence(cache_root: Path) -> dict[str, Any]:
    resources = [_json(path) for path in sorted(cache_root.glob("resource_rank_*.json"))]
    passed = bool(resources) and all(
        item.get("backbone_unchanged") is True
        and item.get("backbone_checksum_before") == item.get("backbone_checksum_after")
        and int(item.get("optimizer_parameters", -1)) == 0
        for item in resources
    )
    return {
        "passed": passed,
        "ranks": len(resources),
        "rows": sum(int(item.get("rows_assigned", 0)) for item in resources),
        "checksums": sorted({item.get("backbone_checksum_before") for item in resources}),
        "optimizer_parameters": [item.get("optimizer_parameters") for item in resources],
    }


def _feature_cache_evidence(
    root: Path,
    cache_root: Path,
    *,
    expected_rows: int,
    expected_contract_sha256: str,
    expected_split_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    manifest = _json(cache_root / "merged/manifest.json")
    payload = _json(cache_root / "feature_signature.json")
    features = np.load(cache_root / "merged/features.npy", allow_pickle=False)
    records = _jsonl(cache_root / "merged/index.jsonl")
    current_adapter_hash = sha256_file(root / f"routers/bench_coe_query/backbones/{payload['backbone']}.py")
    split_counts: dict[str, int] = {}
    if expected_split_counts is not None:
        data = load_vlrb_strict(root)
        for record in records:
            split = data.samples[int(record["row_index"])].split
            split_counts[split] = split_counts.get(split, 0) + 1
    signature_valid = feature_signature(payload) == manifest.get("signature")
    passed = bool(
        int(manifest.get("rows", -1)) == expected_rows
        and len(records) == expected_rows
        and features.shape[0] == expected_rows
        and np.isfinite(features).all()
        and signature_valid
        and payload.get("adapter_code_sha256") == current_adapter_hash
        and payload.get("contract_sha256") == expected_contract_sha256
        and (expected_split_counts is None or dict(expected_split_counts) == split_counts)
    )
    return {
        "passed": passed,
        "rows": len(records),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "finite": bool(np.isfinite(features).all()),
        "signature": manifest.get("signature"),
        "signature_valid": signature_valid,
        "adapter_code_sha256": payload.get("adapter_code_sha256"),
        "current_adapter_code_sha256": current_adapter_hash,
        "payload_contract_sha256": payload.get("contract_sha256"),
        "expected_contract_sha256": expected_contract_sha256,
        "split_counts": split_counts,
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


def _performance_evidence(cache_root: Path, *, full_rows: int, feature_dim: int) -> dict[str, Any]:
    resources = [_json(path) for path in sorted(cache_root.glob("resource_rank_*.json"))]
    rows = sum(int(item["rows_assigned"]) for item in resources)
    wall_seconds = max(float(item["elapsed_seconds"]) for item in resources)
    observed_throughput = rows / wall_seconds
    ideal_eight_gpu_throughput = observed_throughput * (8 / len(resources))
    conservative_efficiency = 0.85
    conservative_throughput = ideal_eight_gpu_throughput * conservative_efficiency
    return {
        "measured_samples": rows,
        "measured_gpus": len(resources),
        "wall_seconds_excluding_model_load": wall_seconds,
        "aggregate_samples_per_second": observed_throughput,
        "preprocess_gpu_seconds_sum": sum(float(item["preprocess_seconds"]) for item in resources),
        "forward_gpu_seconds_sum": sum(float(item["forward_seconds"]) for item in resources),
        "write_gpu_seconds_sum": sum(float(item["write_seconds"]) for item in resources),
        "tokens_per_forward_second_per_rank": [item["tokens_per_forward_second"] for item in resources],
        "peak_cuda_memory_bytes": max(int(item["peak_cuda_memory_bytes"]) for item in resources),
        "full_rows": full_rows,
        "estimated_full_cache_bytes_features_only": full_rows * feature_dim * 2,
        "estimated_eight_gpu_ideal_seconds": full_rows / ideal_eight_gpu_throughput,
        "estimated_eight_gpu_seconds_at_assumed_85pct_scaling": full_rows / conservative_throughput,
        "eight_gpu_efficiency": {
            "measured": False,
            "assumption": conservative_efficiency,
            "basis": "linear extrapolation from the two-GPU 64-sample smoke; must be replaced by a full-run measurement",
        },
    }


def _head_evidence(
    path: Path, *, expected_signature: str, expected_contract_sha256: str
) -> dict[str, Any]:
    resource = _json(path / "resource_usage.json")
    history = _jsonl(path / "training_history.jsonl")
    run_manifest = _json(path / "manifest.json")
    expected_names = list(EXPECTED_HEAD_PARAMETERS)
    passed = bool(
        len(history) == 1
        and history[0].get("epoch") == 1
        and resource.get("feature_signature") == expected_signature
        and resource.get("optimizer_parameter_set_exact") is True
        and resource.get("backbone_parameters_present_in_optimizer") == 0
        and resource.get("checkpoint_monitor") == "dev_hard_label_loss"
        and resource.get("checkpoint_round_trip_prediction_equal") is True
        and run_manifest.get("contract", {}).get("sha256") == expected_contract_sha256
        and resource.get("trainable_parameter_names") == expected_names
        and all(resource.get("head_state_changed", {}).values())
        and (path / "checkpoints/best/checkpoint.pt").is_file()
    )
    return {
        "passed": passed,
        "history": history,
        "resource": resource,
        "run_contract_sha256": run_manifest.get("contract", {}).get("sha256"),
    }


def _bert_evidence(path: Path, *, expected_contract_sha256: str) -> dict[str, Any]:
    resource = _json(path / "resource_usage.json")
    history = _jsonl(path / "training_history.jsonl")
    run_manifest = _json(path / "manifest.json")
    passed = bool(
        len(history) == 1
        and history[0].get("epoch") == 1
        and float(history[0].get("gradient_norm_mean", 0.0)) > 0
        and resource.get("backbone_updated") is True
        and resource.get("all_parameters_trainable") is True
        and resource.get("optimizer_covers_all_trainable_parameters") is True
        and resource.get("checkpoint_monitor") == "dev_hard_label_loss"
        and resource.get("checkpoint_round_trip_prediction_equal") is True
        and run_manifest.get("contract", {}).get("sha256") == expected_contract_sha256
        and (path / "checkpoints/best/checkpoint.pt").is_file()
    )
    return {
        "passed": passed,
        "history": history,
        "resource": resource,
        "run_contract_sha256": run_manifest.get("contract", {}).get("sha256"),
    }


def _gpu_inventory() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"passed": False, "error": str(exc), "gpus": []}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    parsed = []
    for line in rows:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 6:
            parsed.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_total_mib": int(fields[3]),
                    "memory_used_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                }
            )
    target = [item for item in parsed if "A6000" in item["name"]]
    idle_thresholds = {"memory_used_mib_max": 1024, "utilization_percent_max": 10}

    def _idle(item: Mapping[str, Any]) -> bool:
        return bool(
            int(item["memory_used_mib"]) <= idle_thresholds["memory_used_mib_max"]
            and int(item["utilization_percent"])
            <= idle_thresholds["utilization_percent_max"]
        )

    idle = len(target) == 8 and all(_idle(item) for item in target)
    first_four = [item for item in target if int(item["index"]) in {0, 1, 2, 3}]
    first_four_idle = bool(
        result.returncode == 0
        and {int(item["index"]) for item in first_four} == {0, 1, 2, 3}
        and all(_idle(item) for item in first_four)
    )
    return {
        "passed": result.returncode == 0 and len(parsed) == 8 and len(target) == 8 and idle,
        "returncode": result.returncode,
        "gpus": parsed,
        "observed": len(parsed),
        "observed_a6000": len(target),
        "idle_for_full_run": idle,
        "artifact_maintenance_gpu_indices": [0, 1, 2, 3],
        "artifact_maintenance_first_four_idle": first_four_idle,
        "idle_thresholds": idle_thresholds,
    }


def _markdown_report(gate: Mapping[str, Any]) -> str:
    status = "PASS" if gate["technical_smoke_passed"] else "FAIL"
    lines = [
        "# Bench-CoE Query Router smoke report",
        "",
        f"Technical smoke: **{status}**. Full-run gate: **{'GO' if gate['ready_for_full_run'] else 'STOP'}**. "
        f"Artifact-maintenance gate (GPUs 0-3 only): "
        f"**{'GO' if gate['ready_for_artifact_maintenance'] else 'STOP'}**.",
        "",
        "## Router results (diagnostic smoke only)",
        "",
        "| Router | Test rows | Routed accuracy | Average cost | Rank Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in gate["router_metrics"].items():
        lines.append(
            f"| {name} | {metrics['num_samples']} | {metrics['routed_accuracy']:.6f} | "
            f"{metrics['average_cost']:.9g} | {metrics['rank_score']:.6f} |"
        )
    lines.extend(["", "These numbers use one seed, one epoch, and 64 test rows; they are not final model comparisons.", ""])
    lines.extend(["## Feature consistency", ""])
    for backbone, values in gate["feature_consistency"].items():
        cross = values["single_vs_two_gpu"]
        repeat = values["repeat_8"]
        lines.append(
            f"- `{backbone}`: repeat bitwise={repeat['bitwise_equal']}; single/two-GPU "
            f"bitwise={cross['bitwise_equal']}, max_abs={cross['max_absolute_difference']:.8g}, "
            f"min_cosine={cross['minimum_cosine_similarity']:.9f}, "
            f"max_relative_l2={cross['maximum_relative_l2']:.9f}."
        )
    lines.extend(["", "## 64-sample performance preflight", ""])
    for backbone, values in gate["performance_preflight"].items():
        lines.append(
            f"- `{backbone}`: 2-GPU wall {values['wall_seconds_excluding_model_load']:.3f}s, "
            f"{values['aggregate_samples_per_second']:.2f} samples/s, peak "
            f"{values['peak_cuda_memory_bytes'] / 2**30:.2f} GiB/GPU; estimated 8-GPU full extraction "
            f"{values['estimated_eight_gpu_seconds_at_assumed_85pct_scaling'] / 60:.2f} min "
            "(85% scaling assumption, model-load and multi-image policy excluded)."
        )
    lines.extend(["", "## Checks", ""])
    for check in gate["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- [{marker}] `{check['name']}` ({check['scope']}, {check['severity']})")
        if not check["passed"] and check.get("required_action"):
            lines.append(f"  Required: {check['required_action']}")
    lines.extend(["", "## Full-run blockers", ""])
    for blocker in gate["full_run_blockers"]:
        lines.append(f"- `{blocker['name']}`: {blocker['required_action']}")
    lines.extend(
        [
            "",
            "`RUN_FULL` and `RUN_API_REGEN` remained `0`. No expert inference, answer regeneration, remote model download, or API request was run.",
            "",
        ]
    )
    return "\n".join(lines)


def build_quality_gate(
    dataset_root: str | Path = ".",
    *,
    contract_path: str | Path = "configs/bench_coe_query/experiment_contract.yaml",
    output_dir: str | Path = "outputs/bench_coe_query/smoke",
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    contract: ExperimentContract = load_contract(contract_path)
    data = load_vlrb_strict(root, contract=contract)
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    safety = {
        "RUN_FULL": os.environ.get("RUN_FULL", "0"),
        "RUN_API_REGEN": os.environ.get("RUN_API_REGEN", "0"),
    }
    _add_check(
        checks,
        "safety_environment_is_smoke_offline",
        safety == {"RUN_FULL": "0", "RUN_API_REGEN": "0"},
        evidence=safety,
        required_action="Set RUN_FULL=0 and RUN_API_REGEN=0 before regenerating the smoke gate.",
    )
    data_evidence = {
        "shape_y": list(data.y.shape),
        "shape_c": list(data.c.shape),
        "model_order": list(data.model_order),
        "split_counts": {name: len(rows) for name, rows in data.split_to_rows.items()},
        "split_hashes": {
            name: sha256_file(root / str(contract.values["data"]["splits"][name]["path"]))
            for name in ("train", "dev", "test")
        },
    }
    expected_hashes = {
        name: str(contract.values["data"]["splits"][name]["sha256"])
        for name in ("train", "dev", "test")
    }
    _add_check(
        checks,
        "strict_data_registry_y_c_and_split_hashes",
        data_evidence["split_hashes"] == expected_hashes
        and data.y.shape == data.c.shape == (30540, 17)
        and data.model_order == contract.model_order,
        evidence=data_evidence,
    )

    train_rows = data.split_to_rows["train"]
    accuracy = compute_train_expert_accuracy(data.y, train_rows, data.model_order)
    labels = build_benchcoe_hard_labels(data.y, train_rows, accuracy, data.model_order)
    mutated = np.array(data.y, copy=True)
    non_train = np.asarray(data.split_to_rows["dev"] + data.split_to_rows["test"], dtype=np.int64)
    mutated[non_train] = 1 - mutated[non_train]
    mutated_accuracy = compute_train_expert_accuracy(mutated, train_rows, data.model_order)
    mutated_labels = build_benchcoe_hard_labels(mutated, train_rows, mutated_accuracy, data.model_order)
    label_meta = _json(root / "outputs/bench_coe_query/labels/label_manifest.meta.json")
    label_manifest_path = root / "outputs/bench_coe_query/labels/label_manifest.jsonl"
    stored_accuracy = np.asarray(label_meta.get("train_expert_accuracy", ()), dtype=np.float64)
    label_invariance = bool(
        np.array_equal(accuracy, mutated_accuracy)
        and np.array_equal(labels.labels, mutated_labels.labels)
        and label_meta.get("statistics_source") == "train_only"
        and label_meta.get("ignore_index") == -100
        and label_meta.get("model_order") == list(data.model_order)
        and stored_accuracy.shape == accuracy.shape
        and np.array_equal(stored_accuracy, accuracy)
        and label_meta.get("manifest_sha256") == sha256_file(label_manifest_path)
    )
    _add_check(
        checks,
        "labels_are_train_only_and_ignore_all_wrong",
        label_invariance,
        evidence={
            "dev_test_y_fully_inverted_for_probe": True,
            "accuracy_unchanged": bool(np.array_equal(accuracy, mutated_accuracy)),
            "train_labels_unchanged": bool(np.array_equal(labels.labels, mutated_labels.labels)),
            "train_all_wrong": int((labels.labels == -100).sum()),
            "label_manifest_sha256": label_meta.get("manifest_sha256"),
        },
    )
    all_rows = np.arange(len(data.samples), dtype=np.int64)
    training_positions = {
        split: np.asarray(data.split_to_rows[split], dtype=np.int64)
        for split in ("train", "dev", "test")
    }
    training_labels = build_head_training_labels(data, all_rows, training_positions)
    test_targets_absent = bool(np.all(training_labels[training_positions["test"]] == -100))
    _add_check(
        checks,
        "test_y_absent_from_head_training_targets",
        test_targets_absent,
        evidence={
            "test_rows": len(training_positions["test"]),
            "test_targets_materialized_for_training": int(
                np.count_nonzero(training_labels[training_positions["test"]] != -100)
            ),
            "training_label_splits": ["train", "dev"],
        },
    )

    parity = _json(root / "outputs/bench_coe_query/baseline_parity/PARITY_GATE.json")
    _add_check(checks, "baseline_parity", parity.get("passed") is True, evidence={"path": "outputs/bench_coe_query/baseline_parity/PARITY_GATE.json"})

    source_audit = _source_safety_audit(root)
    _add_check(
        checks,
        "no_network_api_expert_regeneration_or_resplitting_codepath",
        source_audit["passed"],
        evidence=source_audit,
    )
    try:
        target_free_evidence = target_free_loader_audit(contract)
        producer_source_checks = {
            "runtime_predict_does_not_import_strict_target_loader": "load_vlrb_strict"
            not in (root / "routers/bench_coe_query/runtime_predict.py").read_text(
                encoding="utf-8"
            ),
            "bert_training_does_not_write_or_evaluate_predictions": all(
                token
                not in (root / "routers/bench_coe_query/bert_router.py").read_text(
                    encoding="utf-8"
                )
                for token in ("write_logits_predictions", "evaluate_prediction_file")
            ),
            "head_training_does_not_write_or_evaluate_predictions": all(
                token
                not in (root / "routers/bench_coe_query/head_trainer.py").read_text(
                    encoding="utf-8"
                )
                for token in ("write_logits_predictions", "evaluate_prediction_file")
            ),
        }
        target_free_evidence["producer_source_checks"] = producer_source_checks
        target_free_evidence["passed"] = target_free_evidence.get("passed") is True and all(
            producer_source_checks.values()
        )
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        target_free_evidence = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _add_check(
        checks,
        "prediction_producer_has_enforced_target_free_process_boundary",
        target_free_evidence.get("passed") is True,
        severity="critical",
        scope="full_run",
        evidence=target_free_evidence,
        required_action=(
            "Restore the metadata-only prediction schema/loader, Y/C access guard, "
            "and separate freeze/evaluation entry points."
        ),
    )

    try:
        provenance_remediation = verify_remediation_report(
            contract, PROVENANCE_REMEDIATION_REPORT
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        provenance_remediation = {
            "passed": False,
            "path": PROVENANCE_REMEDIATION_REPORT.as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    _add_check(
        checks,
        "legacy_amp_ledger_and_complete_backbone_state_attestation",
        provenance_remediation.get("passed") is True,
        severity="critical",
        scope="full_run",
        evidence=provenance_remediation,
        required_action=(
            "Regenerate the hash-bound AF-004/AF-006 remediation after an "
            "independent 16-process full-state forward probe."
        ),
    )

    try:
        input_content_verification = verify_pinned_input_content(
            contract,
            model_names=("bert", "tinyllava", "qwen3vl"),
            include_visual_assets=True,
        )
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        input_content_verification = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _add_check(
        checks,
        "all_model_tokenizer_and_visual_input_bytes_content_addressed",
        input_content_verification.get("passed") is True,
        severity="critical",
        scope="full_run",
        evidence=input_content_verification,
        required_action=(
            "Rebuild and explicitly approve the input-content manifest, then restore every "
            "model/tokenizer/TSV byte to its pinned digest."
        ),
    )

    full_feature_cache_integrity: dict[str, Any] = {}
    full_pointer_paths = {
        backbone: root
        / str(contract.values["features"]["root"])
        / backbone
        / "LATEST_FULL.json"
        for backbone in ("tinyllava", "qwen3vl")
    }
    if any(path.is_file() for path in full_pointer_paths.values()):
        eligible_rows, _exclusions = partition_feature_rows(
            data,
            range(len(data.samples)),
            contract=contract,
            require_release_counts=True,
        )
        for backbone, pointer_path in full_pointer_paths.items():
            try:
                if not pointer_path.is_file():
                    raise FeatureCacheError(f"full feature pointer is missing: {pointer_path}")
                pointer = _json(pointer_path)
                features, records, _manifest, integrity = load_full_feature_cache(
                    pointer["cache_root"],
                    dataset_root=root,
                    expected_contract_sha256=contract.sha256,
                    expected_input_content=feature_input_content_fields(contract, backbone),
                )
                observed_rows = tuple(int(record["row_index"]) for record in records)
                aligned = observed_rows == eligible_rows and all(
                    data.samples[row].sample_id == str(record["sample_id"])
                    for row, record in zip(observed_rows, records)
                )
                integrity["eligible_rows_and_registry_aligned"] = aligned
                integrity["passed"] = integrity.get("passed") is True and aligned
                del features
            except (FeatureCacheError, OSError, ValueError, KeyError, TypeError) as exc:
                integrity = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            full_feature_cache_integrity[backbone] = integrity
            _add_check(
                checks,
                f"{backbone}_full_feature_cache_external_trust_chain",
                integrity.get("passed") is True,
                severity="critical",
                scope="full_run",
                evidence=integrity,
                required_action="Repair or regenerate the hash-bound full feature cache before reuse.",
            )

    tiny = {
        "single": root / "outputs/bench_coe_query/smoke/tinyllava/v2-single32",
        "multi": root / "outputs/bench_coe_query/smoke/tinyllava/v2-multi32",
        "repeat": root / "outputs/bench_coe_query/smoke/tinyllava/v2-repeat8",
        "all": root / "outputs/bench_coe_query/smoke/tinyllava/v2-features-all",
        "perf": root / "outputs/bench_coe_query/smoke/tinyllava/v2-perf64",
        "head": root / "outputs/bench_coe_query/smoke/tinyllava/head/smoke-final/seed-0",
    }
    qwen = {
        "single": root / "outputs/bench_coe_query/smoke/qwen3vl/v2-single32",
        "multi": root / "outputs/bench_coe_query/smoke/qwen3vl/v2-multi32",
        "repeat": root / "outputs/bench_coe_query/smoke/qwen3vl/v2-repeat8",
        "all": root / "outputs/bench_coe_query/smoke/qwen3vl/v2-features-all",
        "perf": root / "outputs/bench_coe_query/smoke/qwen3vl/v2-perf64",
        "head": root / "outputs/bench_coe_query/smoke/qwen3vl/head/smoke-final/seed-0",
    }
    layouts = {"tinyllava": tiny, "qwen3vl": qwen}
    feature_consistency: dict[str, Any] = {}
    feature_cache_checks: dict[str, Any] = {}
    performance: dict[str, Any] = {}
    head_checks: dict[str, Any] = {}
    for backbone, layout in layouts.items():
        cross = _feature_comparison(layout["single"], layout["multi"])
        repeat = _repeat_comparison(layout["single"], layout["repeat"])
        feature_consistency[backbone] = {"single_vs_two_gpu": cross, "repeat_8": repeat}
        _add_check(checks, f"{backbone}_single_two_gpu_feature_consistency", cross["passed"], evidence=cross)
        _add_check(checks, f"{backbone}_repeat_extraction_is_bitwise_stable", repeat["passed"], evidence=repeat)
        resource = _resource_evidence(layout["perf"])
        _add_check(checks, f"{backbone}_backbone_frozen_and_extraction_optimizer_empty", resource["passed"], evidence=resource)
        cache = _feature_cache_evidence(
            root,
            layout["all"],
            expected_rows=160,
            expected_contract_sha256=contract.sha256,
            expected_split_counts={"train": 64, "dev": 32, "test": 64},
        )
        feature_cache_checks[backbone] = cache
        _add_check(checks, f"{backbone}_signed_feature_cache_exact_smoke_coverage", cache["passed"], evidence=cache)
        performance[backbone] = _performance_evidence(
            layout["perf"], full_rows=30540, feature_dim=int(cache["shape"][1])
        )
        head = _head_evidence(
            layout["head"],
            expected_signature=str(cache["signature"]),
            expected_contract_sha256=contract.sha256,
        )
        head_checks[backbone] = head
        _add_check(checks, f"{backbone}_ln_linear_head_training_invariants", head["passed"], evidence=head)

    bert_path = root / "outputs/bench_coe_query/smoke/bert/smoke-main/seed-0"
    bert = _bert_evidence(bert_path, expected_contract_sha256=contract.sha256)
    _add_check(checks, "bert_full_parameter_training_invariants", bert["passed"], evidence=bert)
    bert_runtime_warnings = bert["resource"].get("runtime_warnings")
    _add_check(
        checks,
        "bert_runtime_nondeterminism_warnings_are_disclosed",
        isinstance(bert_runtime_warnings, list),
        severity="warning",
        evidence={"runtime_warnings": bert_runtime_warnings},
        required_action="Capture and disclose runtime determinism warnings in resource_usage.json.",
    )

    expected_test_ids = _expected_smoke_ids(root)
    router_paths = {
        "Query-BERT": bert_path,
        "Query-TinyLLaVA": tiny["head"],
        "Query-Qwen3VL": qwen["head"],
    }
    router_metrics: dict[str, Any] = {}
    prediction_evidence: dict[str, Any] = {}
    evaluator_dir = output / "independent_evaluator"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    for router, path in router_paths.items():
        prediction_path = path / "predictions_test.jsonl"
        records, before_hash = read_frozen_predictions(prediction_path)
        validate_predictions(records, data, expected_split="test", expected_sample_ids=expected_test_ids)
        logits_ok = all(
            len(record.get("logits", [])) == len(data.model_order)
            and np.isfinite(np.asarray(record["logits"], dtype=np.float64)).all()
            for record in records
        )
        metrics = evaluate_prediction_file(
            prediction_path,
            root,
            expected_fallback_policy="none",
            contract_path=contract.path,
            output_path=evaluator_dir / f"{router.lower().replace('-', '_')}.json",
            expected_split="test",
            expected_sample_ids=expected_test_ids,
            expected_seed=0,
            expected_checkpoint_path=path / "checkpoints/best/checkpoint.pt",
            expected_prediction_sha256=before_hash,
        )
        unchanged = metrics.get("prediction_sha256") == before_hash and metrics.get("prediction_unchanged") is True
        prediction_evidence[router] = {
            "count": len(records),
            "exact_id_set": {record["sample_id"] for record in records} == set(expected_test_ids),
            "logits_shape_and_finite": bool(logits_ok),
            "prediction_unchanged_by_evaluator": unchanged,
            "seed_checkpoint_and_prior_digest_verified": bool(
                metrics.get("prediction_provenance", {}).get("seed_verified")
                and metrics.get("prediction_provenance", {}).get("checkpoint_bytes_verified")
                and metrics.get("prediction_provenance", {}).get("prediction_sha256_verified")
            ),
            "sha256": before_hash,
        }
        _add_check(
            checks,
            f"{router.lower().replace('-', '_')}_prediction_and_independent_evaluation",
            len(records) == 64
            and logits_ok
            and unchanged
            and prediction_evidence[router]["exact_id_set"]
            and prediction_evidence[router]["seed_checkpoint_and_prior_digest_verified"],
            evidence=prediction_evidence[router],
        )
        router_metrics[router] = {
            key: metrics[key]
            for key in ("num_samples", "routed_accuracy", "average_cost", "rank_score", "hard_label_accuracy_non_all_wrong")
        }

    declared_review_topics = tuple(str(item) for item in contract.values.get("requires_human_approval", ()))
    approval_error = None
    try:
        approval, approval_sha256 = load_full_run_approval(contract, required=False)
    except ContractError as exc:
        approval, approval_sha256 = None, None
        approval_error = str(exc)
    decisions = approval.get("decisions", {}) if approval else {}
    reviewed_artifacts = approval.get("reviewed_artifacts", {}) if approval else {}
    expected_reviewed_artifacts = {
        "tinyllava_pooling_decision",
        "qwen3vl_pooling_decision",
        "model_adapter_audit",
    }
    reviewed_hashes_match = bool(approval) and set(reviewed_artifacts) == expected_reviewed_artifacts
    reviewed_evidence: dict[str, Any] = {}
    for name, item in reviewed_artifacts.items():
        path = root / str(item.get("path", ""))
        observed = sha256_file(path) if path.is_file() else None
        expected = item.get("sha256")
        matched = observed == expected
        reviewed_hashes_match = reviewed_hashes_match and matched
        reviewed_evidence[name] = {
            "path": str(item.get("path", "")),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matched": matched,
        }
    approval_checks = {
        "source_provenance_approved": decisions.get("source_provenance")
        == "approve_hash_pinned_candidate_as_adapted",
        "tinyllava_pooling_approved": decisions.get("tinyllava_pooling")
        == "approve_hash_pinned_adapted_contract"
        and reviewed_hashes_match,
        "qwen3vl_pooling_approved": decisions.get("qwen3vl_pooling")
        == "approve_hash_pinned_adapted_contract"
        and reviewed_hashes_match,
        "multi_image_policy_approved": decisions.get("multi_image_policy") == FULL_MULTI_IMAGE_POLICY
        and decisions.get("multi_image_fallback") == FULL_MULTI_IMAGE_FALLBACK,
        "full_run_authorized_after_review": decisions.get("full_run_authorized") is True,
    }
    for name, approved in approval_checks.items():
        approval_evidence = {
            "approval_path": str(FULL_RUN_APPROVAL_RELATIVE_PATH),
            "approval_sha256": approval_sha256,
            "approval_error": approval_error,
            "contract_declared_review_topics": list(declared_review_topics),
        }
        if name == "multi_image_policy_approved":
            approval_evidence["full_multi_image_policy"] = decisions.get("multi_image_policy")
            approval_evidence["full_multi_image_fallback"] = decisions.get("multi_image_fallback")
        if name in {"tinyllava_pooling_approved", "qwen3vl_pooling_approved"}:
            approval_evidence["reviewed_artifacts"] = reviewed_evidence
        _add_check(
            checks,
            name,
            approved,
            severity="critical",
            scope="full_run",
            evidence=approval_evidence,
            required_action=(
                "Create or correct the contract-hash-bound FULL_RUN_APPROVAL.json audit artifact."
            ),
        )

    disk = shutil.disk_usage(root)
    free_percent = 100.0 * disk.free / disk.total
    required_percent = float(contract.values["safety"]["minimum_full_run_free_disk_percent"])
    _add_check(
        checks,
        "full_run_free_disk",
        free_percent >= required_percent,
        severity="critical",
        scope="full_run",
        evidence={"free_bytes": disk.free, "total_bytes": disk.total, "free_percent": free_percent, "required_percent": required_percent},
        required_action="Free disk space until at least 20% is available, then rerun preflight and the quality gate.",
    )
    gpu = _gpu_inventory()
    _add_check(
        checks,
        "eight_a6000_inventory",
        gpu["passed"],
        severity="critical",
        scope="full_run",
        evidence=gpu,
        required_action="Run the gate where all eight A6000 GPUs are visible and idle.",
    )

    technical_smoke_passed = all(
        check["passed"] for check in checks if check["scope"] == "technical_smoke" and check["severity"] == "critical"
    )
    all_critical_passed = all(check["passed"] for check in checks if check["severity"] == "critical")
    non_inventory_critical_passed = all(
        check["passed"]
        for check in checks
        if check["severity"] == "critical" and check["name"] != "eight_a6000_inventory"
    )
    pooling_approved = approval_checks["tinyllava_pooling_approved"] and approval_checks["qwen3vl_pooling_approved"]
    ready = bool(all_critical_passed and pooling_approved and parity.get("passed") is True)
    ready_for_artifact_maintenance = bool(
        non_inventory_critical_passed
        and pooling_approved
        and parity.get("passed") is True
        and gpu.get("artifact_maintenance_first_four_idle") is True
    )
    blockers = [
        {
            "name": check["name"],
            "severity": check["severity"],
            "evidence": check["evidence"],
            "affected_stage": "Prompts 11-13 full execution",
            "required_action": check.get("required_action", "Resolve the failed check and rerun the quality gate."),
        }
        for check in checks
        if check["scope"] == "full_run" and not check["passed"]
    ]
    frozen_runtime_inputs = {
        "outputs/bench_coe_query/labels/label_manifest.jsonl": sha256_file(
            root / "outputs/bench_coe_query/labels/label_manifest.jsonl"
        ),
        "outputs/bench_coe_query/labels/label_manifest.meta.json": sha256_file(
            root / "outputs/bench_coe_query/labels/label_manifest.meta.json"
        ),
        "outputs/bench_coe_query/baseline_parity/PARITY_GATE.json": sha256_file(
            root / "outputs/bench_coe_query/baseline_parity/PARITY_GATE.json"
        ),
        INPUT_CONTENT_MANIFEST_RELATIVE_PATH.as_posix(): sha256_file(
            root / INPUT_CONTENT_MANIFEST_RELATIVE_PATH
        ),
    }
    for pointer_path in full_pointer_paths.values():
        if pointer_path.is_file():
            relative = pointer_path.relative_to(root).as_posix()
            frozen_runtime_inputs[relative] = sha256_file(pointer_path)
    remediation_path = root / PROVENANCE_REMEDIATION_REPORT
    if remediation_path.is_file():
        frozen_runtime_inputs[
            PROVENANCE_REMEDIATION_REPORT.as_posix()
        ] = sha256_file(remediation_path)

    gate = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract.sha256,
        "mode": "smoke",
        "limits": {"train": 64, "dev": 32, "test": 64, "epochs": 1},
        "checks": checks,
        "technical_smoke_passed": technical_smoke_passed,
        "all_critical_checks_passed": all_critical_passed,
        "pooling_approved": pooling_approved,
        "baseline_parity_passed": parity.get("passed") is True,
        "feature_consistency": feature_consistency,
        "feature_caches": feature_cache_checks,
        "full_feature_cache_integrity": full_feature_cache_integrity,
        "head_training": head_checks,
        "bert_training": bert,
        "prediction_evidence": prediction_evidence,
        "router_metrics": router_metrics,
        "performance_preflight": performance,
        "source_safety_audit": source_audit,
        "target_free_prediction_input": target_free_evidence,
        "provenance_remediation": provenance_remediation,
        "input_content_verification": input_content_verification,
        "frozen_runtime_inputs": frozen_runtime_inputs,
        "full_run_approval": {
            "path": str(FULL_RUN_APPROVAL_RELATIVE_PATH),
            "sha256": approval_sha256,
            "valid": approval is not None and approval_error is None and reviewed_hashes_match,
            "error": approval_error,
            "reviewed_artifacts": reviewed_evidence,
        },
        "full_run_blockers": blockers,
        "artifact_maintenance_scope": {
            "purpose": "regenerate predictions, target-free receipts, metrics, and aggregate reports from frozen checkpoints and feature caches",
            "allowed_cli_commands": ["freeze-run", "aggregate"],
            "required_environment": {"BENCHCOE_ARTIFACT_MAINTENANCE": "1"},
            "physical_gpu_indices": [0, 1, 2, 3],
            "training_allowed": False,
            "feature_extraction_allowed": False,
            "full_run_gate_unchanged": True,
        },
        "ready_for_artifact_maintenance": ready_for_artifact_maintenance,
        "ready_for_full_run": ready,
    }
    (output / "QUALITY_GATE.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SMOKE_REPORT.md").write_text(_markdown_report(gate), encoding="utf-8")
    return gate


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--output-dir", default="outputs/bench_coe_query/smoke")
    args = parser.parse_args()
    gate = build_quality_gate(args.dataset_dir, contract_path=args.config, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "technical_smoke_passed": gate["technical_smoke_passed"],
                "ready_for_artifact_maintenance": gate[
                    "ready_for_artifact_maintenance"
                ],
                "ready_for_full_run": gate["ready_for_full_run"],
                "full_run_blockers": [item["name"] for item in gate["full_run_blockers"]],
                "quality_gate": str(Path(args.output_dir) / "QUALITY_GATE.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if gate["technical_smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
