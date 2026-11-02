"""Rebind archived runs to independently produced target-free predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import ExperimentContract, load_contract, sha256_file
from .target_free import (
    deny_target_matrix_access,
    load_and_validate_target_free_receipt,
    load_target_free_vlrb,
    prediction_receipt_path,
)


ROUTERS = ("bert", "tinyllava", "qwen3vl")
SPLITS = ("train", "dev", "test")
CORE_METRICS = (
    "routed_accuracy",
    "average_cost",
    "rank_score",
    "hard_label_accuracy_non_all_wrong",
)
DEFAULT_BEFORE = Path("outputs/bench_coe_query/audit/AF010_BEFORE.json")
DEFAULT_REPORT = Path("outputs/bench_coe_query/audit/AF010_TARGET_FREE_REFREEZE.json")
TRACE_ROOT = Path("outputs/bench_coe_query/audit/af010_strace")
LOG_ROOT = Path("outputs/bench_coe_query/audit/af010_process_logs")


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _run_dir(contract: ExperimentContract, router: str, run_id: str, seed: int) -> Path:
    return (
        contract.dataset_root
        / "outputs/bench_coe_query/runs"
        / router
        / "vlrb_adapted_v1"
        / run_id
        / f"seed-{seed}"
    )


def _relative(contract: ExperimentContract, path: Path) -> str:
    return path.resolve().relative_to(contract.dataset_root).as_posix()


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _snapshot_entry(contract: ExperimentContract, router: str, seed: int, run_id: str) -> dict[str, Any]:
    run = _run_dir(contract, router, run_id, seed)
    predictions = {}
    for split in SPLITS:
        prediction = run / f"predictions_{split}.jsonl"
        metrics_path = run / f"metrics_{split}.json"
        metrics = _load_json(metrics_path)
        predictions[split] = {
            "prediction_path": _relative(contract, prediction),
            "prediction_sha256": sha256_file(prediction),
            "prediction_records": _count_lines(prediction),
            "metrics_path": _relative(contract, metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "core_metrics": {name: metrics[name] for name in CORE_METRICS},
        }
    return {"router": router, "seed": seed, "run": _relative(contract, run), "splits": predictions}


def snapshot(contract: ExperimentContract, run_id: str, output: Path) -> dict[str, Any]:
    runs = [
        _snapshot_entry(contract, router, seed, run_id)
        for router in ROUTERS
        for seed in contract.seeds
    ]
    payload = {
        "schema_version": 1,
        "kind": "af010_pre_refreeze_snapshot",
        "contract_sha256": contract.sha256,
        "run_id": run_id,
        "runs": runs,
        "prediction_files": sum(len(item["splits"]) for item in runs),
        "prediction_records": sum(
            split["prediction_records"]
            for item in runs
            for split in item["splits"].values()
        ),
        "source_sha256": {
            name: sha256_file(contract.dataset_root / name)
            for name in (
                "routers/bench_coe_query/target_free.py",
                "routers/bench_coe_query/runtime_predict.py",
                "routers/bench_coe_query/evaluator.py",
                "routers/bench_coe_query/target_free_remediation.py",
                "scripts/bench_coe_query/refreeze_target_free.sh",
            )
        },
    }
    _atomic_json(output, payload)
    return payload


def _sanitize_nonfinite(value: Any) -> tuple[Any, int]:
    if isinstance(value, float) and not math.isfinite(value):
        return None, 1
    if isinstance(value, list):
        result = []
        count = 0
        for item in value:
            clean, changed = _sanitize_nonfinite(item)
            result.append(clean)
            count += changed
        return result, count
    if isinstance(value, dict):
        result = {}
        count = 0
        for key, item in value.items():
            clean, changed = _sanitize_nonfinite(item)
            result[key] = clean
            count += changed
        return result, count
    return value, 0


def _sanitize_history(path: Path) -> int:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    cleaned = []
    replacements = 0
    for record in records:
        value, count = _sanitize_nonfinite(record)
        cleaned.append(value)
        replacements += count
    payload = "".join(
        json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in cleaned
    )
    path.write_text(payload, encoding="utf-8")
    return replacements


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    return True


def _target_trace_records(trace: Path, contract: ExperimentContract) -> list[str]:
    targets = [str(path) for path in (
        contract.resolve(contract.values["data"]["matrices"]["y"]["path"]),
        contract.resolve(contract.values["data"]["matrices"]["c"]["path"]),
    )]
    return [
        line
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(target in line for target in targets)
    ]


def finalize(
    contract: ExperimentContract,
    run_id: str,
    before_path: Path,
    output: Path,
) -> dict[str, Any]:
    before = _load_json(before_path)
    if before.get("contract_sha256") != contract.sha256 or before.get("run_id") != run_id:
        raise RuntimeError("AF-010 before snapshot does not match this run")
    before_runs = {
        (item["router"], int(item["seed"])): item for item in before["runs"]
    }
    with deny_target_matrix_access(contract) as loader_guard:
        target_free_data = load_target_free_vlrb(contract.dataset_root, contract=contract)
    if loader_guard.get("passed") is not True:
        raise RuntimeError("target-free loader guard failed during remediation finalization")

    run_evidence = []
    producer_pids: set[int] = set()
    total_history_replacements = 0
    for router in ROUTERS:
        for seed in contract.seeds:
            run = _run_dir(contract, router, run_id, seed)
            checkpoint = run / "checkpoints/best/checkpoint.pt"
            trace = contract.dataset_root / TRACE_ROOT / f"{router}-seed-{seed}.targets.log"
            producer_log = contract.dataset_root / LOG_ROOT / f"{router}-seed-{seed}.producer.log"
            if not trace.is_file():
                raise RuntimeError(f"filtered Y/C open trace is missing: {trace}")
            target_trace_records = _target_trace_records(trace, contract)
            if target_trace_records:
                raise RuntimeError(f"prediction producer opened Y/C: {trace}")
            if not producer_log.is_file():
                raise RuntimeError(f"producer process log is missing: {producer_log}")
            splits = {}
            summary_for_resource = {}
            for split in SPLITS:
                prediction = run / f"predictions_{split}.jsonl"
                receipt_path = prediction_receipt_path(prediction)
                receipt = load_and_validate_target_free_receipt(
                    receipt_path,
                    contract=contract,
                    data=target_free_data,
                    prediction_path=prediction,
                    checkpoint_path=checkpoint,
                    expected_backbone=router,
                    expected_split=split,
                    expected_seed=seed,
                )
                producer_pids.add(int(receipt["producer_pid"]))
                metrics_path = run / f"metrics_{split}.json"
                metrics = _load_json(metrics_path)
                provenance = metrics.get("prediction_provenance", {})
                if (
                    provenance.get("target_free_producer_receipt_verified") is not True
                    or provenance.get("receipt_validated_before_evaluator_loaded_y_c") is not True
                    or provenance.get("target_free_producer_receipt_file_sha256")
                    != receipt["receipt_file_sha256"]
                ):
                    raise RuntimeError(f"metrics lack target-free receipt binding: {metrics_path}")
                if metrics_path.stat().st_mtime_ns < receipt_path.stat().st_mtime_ns:
                    raise RuntimeError("evaluator output predates the frozen producer receipt")
                prior = before_runs[(router, seed)]["splits"][split]
                current_sha = sha256_file(prediction)
                core_equal = all(
                    metrics[name] == prior["core_metrics"][name] for name in CORE_METRICS
                )
                if not core_equal or not _all_numbers_finite(metrics):
                    raise RuntimeError(f"AF-010 refreeze changed core metrics or emitted nonfinite JSON: {run}/{split}")
                splits[split] = {
                    "prediction_path": _relative(contract, prediction),
                    "prediction_sha256_before": prior["prediction_sha256"],
                    "prediction_sha256_after": current_sha,
                    "prediction_bytes_identical": current_sha == prior["prediction_sha256"],
                    "prediction_records": _count_lines(prediction),
                    "receipt_path": _relative(contract, receipt_path),
                    "receipt_file_sha256": receipt["receipt_file_sha256"],
                    "metrics_path": _relative(contract, metrics_path),
                    "metrics_sha256_before": prior["metrics_sha256"],
                    "metrics_sha256_after": sha256_file(metrics_path),
                    "core_metrics_identical": core_equal,
                    "metrics_strict_finite_json": True,
                    "producer_finished_before_evaluator_output": True,
                }
                summary_for_resource[split] = {
                    "path": str(prediction),
                    "sha256": current_sha,
                    "count": int(metrics["num_samples"]),
                    "routed_accuracy": metrics["routed_accuracy"],
                    "structured_fallback_rows": int(
                        provenance.get("expected_fallback_rows", 0)
                    ),
                    "fallback_policy_verified": provenance.get(
                        "fallback_policy_verified"
                    )
                    is True,
                    "target_free_receipt": _relative(contract, receipt_path),
                    "target_free_receipt_file_sha256": receipt[
                        "receipt_file_sha256"
                    ],
                }

            resource_path = run / "resource_usage.json"
            resource = _load_json(resource_path)
            resource["prediction_summaries"] = summary_for_resource
            resource["target_free_prediction_process"] = {
                "status": "verified",
                "producer_process_loaded_y_or_c": False,
                "producer_process_evaluated_predictions": False,
                "separate_evaluator_process_started_after_producer_exit": True,
                "target_free_input_manifest_sha256": target_free_data.input_manifest_sha256,
                "filtered_os_open_trace": _relative(contract, trace),
                "filtered_os_open_trace_bytes": trace.stat().st_size,
                "target_matrix_open_records": 0,
                "filtered_os_open_trace_sha256": sha256_file(trace),
                "producer_log": _relative(contract, producer_log),
                "producer_log_sha256": sha256_file(producer_log),
                "prediction_receipts": {
                    split: {
                        "path": values["receipt_path"],
                        "sha256": values["receipt_file_sha256"],
                    }
                    for split, values in splits.items()
                },
            }
            _atomic_json(resource_path, resource)

            manifest_path = run / "manifest.json"
            manifest = _load_json(manifest_path)
            manifest["target_free_prediction_process"] = {
                "status": "post_training_target_free_refreeze_verified",
                "training_process_did_not_generate_current_predictions": True,
                "producer_process_loaded_y_or_c": False,
                "separate_evaluator_process": True,
                "target_free_input_manifest_sha256": target_free_data.input_manifest_sha256,
                "receipts": {
                    split: {
                        "path": values["receipt_path"],
                        "sha256": values["receipt_file_sha256"],
                    }
                    for split, values in splits.items()
                },
            }
            _atomic_json(manifest_path, manifest)

            replacements = 0
            if router == "bert":
                replacements = _sanitize_history(run / "training_history.jsonl")
                total_history_replacements += replacements
            run_evidence.append(
                {
                    "router": router,
                    "seed": seed,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "trace_path": _relative(contract, trace),
                    "trace_bytes": trace.stat().st_size,
                    "trace_target_open_records": 0,
                    "history_nonfinite_values_replaced_with_null": replacements,
                    "splits": splits,
                }
            )

    prediction_files = sum(len(item["splits"]) for item in run_evidence)
    prediction_records = sum(
        value["prediction_records"]
        for item in run_evidence
        for value in item["splits"].values()
    )
    unchanged = sum(
        value["prediction_bytes_identical"]
        for item in run_evidence
        for value in item["splits"].values()
    )
    payload = {
        "schema_version": 1,
        "kind": "af010_target_free_prediction_refreeze",
        "contract_sha256": contract.sha256,
        "run_id": run_id,
        "numerical_runtime": {
            "python_executable": os.sys.executable,
            "numpy_version": np.__version__,
            "same_runtime_required_for_metrics_and_aggregation": True,
        },
        "process_protocol": {
            "producer": "one target-free freeze-run subprocess per router/seed",
            "producer_schema_has_y_or_c": False,
            "in_process_python_open_guard": True,
            "os_open_trace_filter": [
                str(path) for path in (
                    contract.resolve(contract.values["data"]["matrices"]["y"]["path"]),
                    contract.resolve(contract.values["data"]["matrices"]["c"]["path"]),
                )
            ],
            "producer_processes_exit_before_evaluators_start": True,
            "evaluator_processes_load_y_c_only_after_prediction_freeze": True,
        },
        "target_free_input_manifest_sha256": target_free_data.input_manifest_sha256,
        "producer_processes": len(producer_pids),
        "expected_producer_processes": 15,
        "prediction_files": prediction_files,
        "prediction_records": prediction_records,
        "prediction_files_byte_identical_to_pre_refreeze": unchanged,
        "core_metric_documents_unchanged": prediction_files,
        "filtered_trace_files": len(run_evidence),
        "target_matrix_open_records": 0,
        "bert_history_nonfinite_values_replaced_with_null": total_history_replacements,
        "runs": run_evidence,
        "passed": bool(
            len(run_evidence) == 15
            and len(producer_pids) == 15
            and prediction_files == 45
            and prediction_records == 458100
            and all(
                value["prediction_bytes_identical"]
                for item in run_evidence
                if item["router"] != "bert"
                for value in item["splits"].values()
            )
            and all(
                value["core_metrics_identical"]
                for item in run_evidence
                for value in item["splits"].values()
            )
        ),
    }
    if payload["passed"] is not True:
        raise RuntimeError("AF-010 target-free refreeze did not pass")
    _atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("snapshot", "finalize"))
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--run-id", default="full-vlrb-adapted-v1")
    parser.add_argument("--before", default=str(DEFAULT_BEFORE))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    contract = load_contract(args.config)
    if args.phase == "snapshot":
        result = snapshot(contract, args.run_id, contract.dataset_root / args.before)
    else:
        result = finalize(
            contract,
            args.run_id,
            contract.dataset_root / args.before,
            contract.dataset_root / args.output,
        )
    print(json.dumps({key: result[key] for key in result if key != "runs"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
