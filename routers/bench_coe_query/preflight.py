"""Read-only preflight checks for the frozen Bench-CoE query experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import effective_model_path, load_contract, load_full_run_approval, sha256_file
from .data import benchmark_manifest_sha256
from .errors import ContractError, FeatureCacheError
from .feature_cache import load_full_feature_cache
from .input_content import feature_input_content_fields, verify_pinned_input_content


def _check(name: str, passed: bool, *, severity: str = "critical", **evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def run_preflight(config_path: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    contract = load_contract(config_path)
    checks: list[dict[str, Any]] = []

    api_regen = os.environ.get("RUN_API_REGEN", str(contract.values["safety"]["RUN_API_REGEN"]))
    run_full = os.environ.get("RUN_FULL", str(contract.values["safety"]["RUN_FULL"]))
    checks.append(_check("RUN_API_REGEN_is_zero", api_regen == "0", observed=api_regen))
    checks.append(_check("RUN_FULL_value_valid", run_full in {"0", "1"}, observed=run_full))

    try:
        _approval, approval_sha256 = load_full_run_approval(contract, required=run_full == "1")
        approval_error = None
    except ContractError as exc:
        approval_sha256 = None
        approval_error = str(exc)
    checks.append(
        _check(
            "full_run_approval",
            approval_error is None and (run_full != "1" or approval_sha256 is not None),
            severity="critical" if run_full == "1" else "warning",
            sha256=approval_sha256,
            error=approval_error,
        )
    )

    if run_full == "1":
        try:
            input_content = verify_pinned_input_content(contract)
            input_content_error = None
        except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
            input_content = None
            input_content_error = f"{type(exc).__name__}: {exc}"
        checks.append(
            _check(
                "model_tokenizer_and_visual_input_content",
                input_content is not None and input_content.get("passed") is True,
                severity="critical",
                verification=input_content,
                error=input_content_error,
            )
        )

    if run_full == "1":
        for backbone in ("tinyllava", "qwen3vl"):
            pointer_path = (
                contract.dataset_root
                / str(contract.values["features"]["root"])
                / backbone
                / "LATEST_FULL.json"
            )
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                features, records, _manifest, integrity = load_full_feature_cache(
                    pointer["cache_root"],
                    dataset_root=contract.dataset_root,
                    expected_contract_sha256=contract.sha256,
                    expected_input_content=feature_input_content_fields(contract, backbone),
                )
                evidence = {
                    "rows": len(records),
                    "pointer_sha256": integrity["pointer_sha256"],
                    "manifest_file_sha256": integrity["manifest_file_sha256"],
                    "shards": len(integrity["shards"]),
                }
                passed = integrity.get("passed") is True
                del features
            except (OSError, ValueError, KeyError, TypeError, ContractError, FeatureCacheError) as exc:
                passed = False
                evidence = {"error": f"{type(exc).__name__}: {exc}"}
            checks.append(
                _check(
                    f"full_feature_cache_{backbone}",
                    passed,
                    severity="critical",
                    **evidence,
                )
            )

    gate_path = contract.dataset_root / str(contract.values["safety"]["output_root"]) / "smoke/QUALITY_GATE.json"
    gate = None
    if gate_path.is_file():
        try:
            with gate_path.open("r", encoding="utf-8") as handle:
                gate = json.load(handle)
        except (OSError, json.JSONDecodeError):
            gate = None
    gate_passed = bool(
        gate
        and gate.get("contract_sha256") == contract.sha256
        and gate.get("ready_for_full_run") is True
        and gate.get("full_run_approval", {}).get("sha256") == approval_sha256
    )
    checks.append(
        _check(
            "hash_bound_quality_gate",
            gate_passed if run_full == "1" else gate is not None,
            severity="critical" if run_full == "1" else "warning",
            path=str(gate_path),
            ready_for_full_run=gate.get("ready_for_full_run") if gate else None,
            contract_sha256=gate.get("contract_sha256") if gate else None,
        )
    )

    for split_name, entry in contract.values["data"]["splits"].items():
        path = contract.resolve(str(entry["path"]))
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            _check(
                f"split_{split_name}",
                observed == str(entry["sha256"]),
                path=str(path),
                expected_sha256=str(entry["sha256"]),
                observed_sha256=observed,
            )
        )

    for group_name in ("matrices", "registry"):
        for artifact_name, entry in contract.values["data"][group_name].items():
            path = contract.resolve(str(entry["path"]))
            observed = sha256_file(path) if path.is_file() else None
            checks.append(
                _check(
                    f"{group_name}_{artifact_name}",
                    observed == str(entry["sha256"]),
                    path=str(path),
                    expected_sha256=str(entry["sha256"]),
                    observed_sha256=observed,
                )
            )

    benchmarks = contract.resolve(str(contract.values["data"]["benchmarks"]["path"]))
    benchmark_files = list(benchmarks.glob("*/*_samples.jsonl")) if benchmarks.is_dir() else []
    benchmark_manifest, _, benchmark_bytes = (
        benchmark_manifest_sha256(contract.dataset_root, benchmarks)
        if benchmarks.is_dir()
        else (None, 0, 0)
    )
    expected_benchmark_manifest = str(contract.values["data"]["benchmarks"]["manifest_sha256"])
    checks.append(
        _check(
            "benchmarks_manifest",
            len(benchmark_files) == int(contract.values["data"]["benchmarks"]["files"])
            and benchmark_manifest == expected_benchmark_manifest,
            path=str(benchmarks),
            observed_files=len(benchmark_files),
            total_bytes=benchmark_bytes,
            expected_manifest_sha256=expected_benchmark_manifest,
            observed_manifest_sha256=benchmark_manifest,
        )
    )

    for model_name in ("bert", "tinyllava", "qwen3vl"):
        model = contract.values["models"][model_name]
        configured_path = Path(str(model["path"]))
        path = effective_model_path(contract, model_name)
        config_file = path / "config.json"
        observed = sha256_file(config_file) if config_file.is_file() else None
        readable = path.is_dir() and os.access(path, os.R_OK | os.X_OK)
        checks.append(
            _check(
                f"model_{model_name}_config",
                readable and observed == str(model["config_sha256"]),
                path=str(configured_path),
                resolved_path=str(path),
                configured_path_is_symlink=configured_path.is_symlink(),
                readable=readable,
                expected_sha256=str(model["config_sha256"]),
                observed_sha256=observed,
            )
        )

    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_lines = (
            [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]
            if gpu_query.returncode == 0
            else []
        )
    except (OSError, subprocess.SubprocessError):
        gpu_lines = []
    target = str(contract.values["scheduling"]["target_gpu_model"])
    expected_gpus = int(contract.values["scheduling"]["available_target_gpus"])
    parsed_gpu = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            parsed_gpu.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }
            )
    target_gpus = [item for item in parsed_gpu if target in item["name"]]
    target_count = len(target_gpus)
    idle = all(item["memory_used_mib"] <= 1024 and item["utilization_percent"] <= 10 for item in target_gpus)
    gpu_passed = (
        target_count >= expected_gpus and idle
        if run_full == "1"
        else target_count >= 1
    )
    checks.append(
        _check(
            "gpu_visibility",
            gpu_passed,
            severity="critical" if run_full == "1" else "warning",
            expected_full_count=expected_gpus,
            observed_count=len(parsed_gpu),
            observed_target_count=target_count,
            idle_for_full_run=idle,
            gpus=parsed_gpu,
            smoke_allows_fewer=True,
        )
    )

    usage = shutil.disk_usage(contract.dataset_root)
    free_percent = 100.0 * usage.free / usage.total
    required_free = float(contract.values["safety"]["minimum_full_run_free_disk_percent"])
    disk_passed = free_percent >= required_free
    checks.append(
        _check(
            "full_run_free_disk",
            disk_passed,
            severity="critical" if run_full == "1" else "warning",
            total_bytes=usage.total,
            free_bytes=usage.free,
            free_percent=free_percent,
            required_percent=required_free,
            blocks_current_mode=run_full == "1" and not disk_passed,
        )
    )

    failed_critical = [item["name"] for item in checks if not item["passed"] and item["severity"] == "critical"]
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full" if run_full == "1" else "smoke",
        "contract_sha256": contract.sha256,
        "passed": not failed_critical,
        "full_run_ready": not failed_critical
        and disk_passed
        and target_count >= expected_gpus
        and idle
        and gate_passed
        and run_full == "1",
        "failed_critical": failed_critical,
        "checks": checks,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--output", default="outputs/bench_coe_query/audit/PREFLIGHT.json")
    args = parser.parse_args()
    result = run_preflight(args.config, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
