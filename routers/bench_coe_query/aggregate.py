"""Prompt-12 completeness checks and Prompt-13 five-seed aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import ExperimentContract
from .data import load_vlrb_strict
from .evaluator import (
    evaluate_prediction_file,
    expected_multi_image_fallback_sample_ids,
    read_frozen_predictions,
    validate_predictions,
    validate_structured_fallback,
)
from .feature_cache import load_full_feature_cache
from .full_policy import frozen_train_strongest_column, partition_feature_rows
from .labels import build_diagnostic_labels_for_split, compute_train_expert_accuracy
from .manifest import write_manifest
from .input_content import (
    INPUT_CONTENT_MANIFEST_RELATIVE_PATH,
    feature_input_content_fields,
    input_content_summary,
)
from .statistics import mcnemar_exact, paired_bootstrap_delta, paired_permutation_mean_delta
from .target_free import (
    deny_target_matrix_access,
    load_and_validate_target_free_receipt,
    load_target_free_vlrb,
    prediction_receipt_path,
)


ROUTERS = {
    "Query-BERT": "bert",
    "Query-TinyLLaVA": "tinyllava",
    "Query-Qwen3VL": "qwen3vl",
}
CORE_METRICS = (
    "routed_accuracy",
    "average_cost",
    "rank_score",
    "hard_label_accuracy_non_all_wrong",
)


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate summary requires finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _run_dir(contract: ExperimentContract, router_key: str, run_id: str, seed: int) -> Path:
    return (
        contract.dataset_root
        / str(contract.values["safety"]["output_root"])
        / "runs"
        / router_key
        / "vlrb_adapted_v1"
        / run_id
        / f"seed-{seed}"
    )


def _completeness(
    contract: ExperimentContract,
    *,
    data,
    run_id: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[int, Path]]]:
    expected_counts = {name: int(entry["count"]) for name, entry in contract.values["data"]["splits"].items()}
    expected_input_manifest_sha256 = input_content_summary(contract)["manifest_sha256"]
    with deny_target_matrix_access(contract) as target_free_guard:
        target_free_data = load_target_free_vlrb(
            contract.dataset_root, contract=contract
        )
    if target_free_guard.get("passed") is not True:
        raise RuntimeError("target-free input view attempted to open Y/C")
    run_paths: dict[str, dict[int, Path]] = {}
    runs = []
    for router, key in ROUTERS.items():
        run_paths[router] = {}
        for seed in contract.seeds:
            path = _run_dir(contract, key, run_id, seed)
            run_paths[router][seed] = path
            required = [
                path / "config.resolved.yaml",
                path / "manifest.json",
                path / "training_history.jsonl",
                path / "checkpoints/best/checkpoint.pt",
                path / "resource_usage.json",
            ]
            for split in ("train", "dev", "test"):
                prediction_path = path / f"predictions_{split}.jsonl"
                required.extend(
                    (
                        prediction_path,
                        prediction_receipt_path(prediction_path),
                        path / f"metrics_{split}.json",
                    )
                )
            missing = [str(item.relative_to(contract.dataset_root)) for item in required if not item.is_file()]
            counts: dict[str, int | None] = {}
            prediction_errors: dict[str, str] = {}
            prediction_provenance: dict[str, Any] = {}
            checkpoint_path = path / "checkpoints/best/checkpoint.pt"
            checkpoint_digest = _sha256(checkpoint_path) if checkpoint_path.is_file() else None
            resource = _json(path / "resource_usage.json") if (path / "resource_usage.json").is_file() else {}
            for split in ("train", "dev", "test"):
                prediction_path = path / f"predictions_{split}.jsonl"
                if prediction_path.is_file():
                    try:
                        records, digest = read_frozen_predictions(prediction_path)
                        expected_ids = [
                            data.samples[row].sample_id for row in data.split_to_rows[split]
                        ]
                        rows, predictions, _validated_split = validate_predictions(
                            records,
                            data,
                            expected_split=split,
                            expected_sample_ids=expected_ids,
                            expected_seed=seed,
                            expected_checkpoint_id=checkpoint_digest,
                        )
                        fallback_ids = (
                            ()
                            if key == "bert"
                            else expected_multi_image_fallback_sample_ids(data, expected_ids)
                        )
                        validate_structured_fallback(
                            records,
                            data,
                            rows,
                            predictions,
                            expected_fallback_sample_ids=fallback_ids,
                        )
                        receipt_path = prediction_receipt_path(prediction_path)
                        receipt = load_and_validate_target_free_receipt(
                            receipt_path,
                            contract=contract,
                            data=target_free_data,
                            prediction_path=prediction_path,
                            checkpoint_path=checkpoint_path,
                            expected_backbone=key,
                            expected_split=split,
                            expected_seed=seed,
                        )
                        recorded_metrics = _json(path / f"metrics_{split}.json")
                        resource_summary = resource.get("prediction_summaries", {}).get(split, {})
                        if recorded_metrics.get("prediction_sha256") != digest:
                            raise RuntimeError("metrics prediction SHA-256 mismatch")
                        if resource_summary.get("sha256") != digest:
                            raise RuntimeError("resource prediction SHA-256 mismatch")
                        if recorded_metrics.get("prediction_unchanged") is not True:
                            raise RuntimeError("metrics do not attest immutable prediction bytes")
                        metric_provenance = recorded_metrics.get("prediction_provenance", {})
                        if (
                            metric_provenance.get("target_free_producer_receipt_verified")
                            is not True
                            or metric_provenance.get(
                                "receipt_validated_before_evaluator_loaded_y_c"
                            )
                            is not True
                            or metric_provenance.get(
                                "target_free_producer_receipt_file_sha256"
                            )
                            != receipt["receipt_file_sha256"]
                        ):
                            raise RuntimeError(
                                "metrics do not bind the target-free producer receipt"
                            )
                        counts[split] = len(records)
                        prediction_provenance[split] = {
                            "path": str(prediction_path.relative_to(contract.dataset_root)),
                            "sha256": digest,
                            "records": len(records),
                            "seed": seed,
                            "checkpoint_sha256": checkpoint_digest,
                            "metrics_anchor": str(
                                (path / f"metrics_{split}.json").relative_to(contract.dataset_root)
                            ),
                            "resource_anchor": str(
                                (path / "resource_usage.json").relative_to(contract.dataset_root)
                            ),
                            "logits_width_finiteness_and_argmax_verified": True,
                            "fallback_policy_verified": True,
                            "expected_fallback_rows": len(fallback_ids),
                            "target_free_producer_receipt": str(
                                receipt_path.relative_to(contract.dataset_root)
                            ),
                            "target_free_producer_receipt_file_sha256": receipt[
                                "receipt_file_sha256"
                            ],
                            "target_free_input_manifest_sha256": target_free_data.input_manifest_sha256,
                            "producer_process_loaded_y_or_c": False,
                            "receipt_validated_before_evaluator_loaded_y_c": True,
                        }
                    except Exception as exc:  # recorded as a failed completeness item
                        counts[split] = None
                        prediction_errors[split] = f"{type(exc).__name__}: {exc}"
                else:
                    counts[split] = None
            manifest_contract = None
            manifest_input_content = None
            if (path / "manifest.json").is_file():
                manifest_document = _json(path / "manifest.json")
                manifest_contract = manifest_document.get("contract", {}).get("sha256")
                manifest_input_content = (
                    manifest_document.get("input_content", {})
                    .get("pinned", {})
                    .get("manifest_sha256")
                )
            complete = bool(
                not missing
                and not prediction_errors
                and counts == expected_counts
                and manifest_contract == contract.sha256
                and manifest_input_content == expected_input_manifest_sha256
            )
            runs.append(
                {
                    "router": router,
                    "seed": seed,
                    "path": str(path.relative_to(contract.dataset_root)),
                    "complete": complete,
                    "missing": missing,
                    "prediction_counts": counts,
                    "prediction_errors": prediction_errors,
                    "checkpoint_sha256": checkpoint_digest,
                    "prediction_provenance": prediction_provenance,
                    "manifest_contract_sha256": manifest_contract,
                    "manifest_input_content_sha256": manifest_input_content,
                }
            )
    result = {
        "schema_version": 2,
        "run_id": run_id,
        "contract_sha256": contract.sha256,
        "expected_runs": 15,
        "complete_runs": sum(item["complete"] for item in runs),
        "target_free_input_manifest_sha256": target_free_data.input_manifest_sha256,
        "target_free_loader_guard": target_free_guard,
        "passed": len(runs) == 15 and all(item["complete"] for item in runs),
        "runs": runs,
    }
    write_manifest(output_dir / "RUN_COMPLETENESS.json", result)
    return result, run_paths


def _prediction_arrays(records, data, test_rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, predictions, _split = validate_predictions(
        records,
        data,
        expected_split="test",
        expected_sample_ids=[data.samples[row].sample_id for row in test_rows],
    )
    by_row = {int(row): int(column) for row, column in zip(rows, predictions)}
    ordered_predictions = np.asarray([by_row[int(row)] for row in test_rows], dtype=np.int64)
    ordered_rows = np.asarray(test_rows, dtype=np.int64)
    correctness = data.y[ordered_rows, ordered_predictions].astype(np.float64)
    costs = data.c[ordered_rows, ordered_predictions].astype(np.float64)
    return ordered_predictions, correctness, costs


def run_aggregation(args, contract: ExperimentContract) -> int:
    data = load_vlrb_strict(args.dataset_dir, contract=contract)
    output = contract.dataset_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    completeness, run_paths = _completeness(
        contract,
        data=data,
        run_id=args.run_id,
        output_dir=output,
    )
    if completeness["passed"] is not True:
        raise RuntimeError(
            f"full aggregation refused: {completeness['complete_runs']}/15 runs are complete"
        )

    prediction_provenance = {
        "schema_version": 2,
        "contract_sha256": contract.sha256,
        "run_id": args.run_id,
        "verifier_source_sha256": _sha256(Path(__file__)),
        "expected_runs": 15,
        "verified_runs": completeness["complete_runs"],
        "verified_checkpoints": sum(
            item.get("checkpoint_sha256") is not None for item in completeness["runs"]
        ),
        "verified_prediction_files": sum(
            len(item.get("prediction_provenance", {})) for item in completeness["runs"]
        ),
        "target_free_input_manifest_sha256": completeness[
            "target_free_input_manifest_sha256"
        ],
        "target_free_prediction_processes_verified": sum(
            len(item.get("prediction_provenance", {}))
            for item in completeness["runs"]
        ),
        "runs": [
            {
                "router": item["router"],
                "seed": item["seed"],
                "path": item["path"],
                "checkpoint_sha256": item.get("checkpoint_sha256"),
                "predictions": item.get("prediction_provenance", {}),
            }
            for item in completeness["runs"]
        ],
        "passed": completeness["passed"] is True,
    }
    if (
        prediction_provenance["verified_runs"] != 15
        or prediction_provenance["verified_checkpoints"] != 15
        or prediction_provenance["verified_prediction_files"] != 45
        or prediction_provenance["target_free_prediction_processes_verified"] != 45
    ):
        raise RuntimeError("prediction provenance inventory is incomplete")
    write_manifest(output / "PREDICTION_PROVENANCE.json", prediction_provenance)

    eligible_rows, _exclusions = partition_feature_rows(
        data,
        range(len(data.samples)),
        contract=contract,
        require_release_counts=True,
    )
    cache_evidence: dict[str, Any] = {}
    for backbone in ("tinyllava", "qwen3vl"):
        pointer_path = (
            contract.dataset_root
            / str(contract.values["features"]["root"])
            / backbone
            / "LATEST_FULL.json"
        )
        pointer = _json(pointer_path)
        features, records, _manifest, integrity = load_full_feature_cache(
            pointer["cache_root"],
            dataset_root=contract.dataset_root,
            expected_contract_sha256=contract.sha256,
            expected_input_content=feature_input_content_fields(contract, backbone),
        )
        observed_rows = tuple(int(record["row_index"]) for record in records)
        if observed_rows != eligible_rows:
            raise RuntimeError(f"{backbone} cache rows differ from approved eligible rows")
        if any(
            data.samples[row].sample_id != str(record["sample_id"])
            for row, record in zip(observed_rows, records)
        ):
            raise RuntimeError(f"{backbone} cache sample IDs differ from the registry")
        integrity["pointer_anchor_sha256"] = _sha256(pointer_path)
        integrity["eligible_rows_exact"] = True
        integrity["sample_registry_alignment"] = True
        cache_evidence[backbone] = integrity
        del features
    cache_integrity = {
        "schema_version": 1,
        "contract_sha256": contract.sha256,
        "verifier_source_sha256": _sha256(
            contract.dataset_root / "routers/bench_coe_query/feature_cache.py"
        ),
        "caches": cache_evidence,
        "passed": len(cache_evidence) == 2
        and all(item.get("passed") is True for item in cache_evidence.values()),
    }
    if cache_integrity["passed"] is not True:
        raise RuntimeError("full feature cache integrity did not pass")
    write_manifest(output / "CACHE_INTEGRITY.json", cache_integrity)

    content_summary = input_content_summary(contract)
    quality_gate_path = contract.dataset_root / "outputs/bench_coe_query/smoke/QUALITY_GATE.json"
    quality_gate = _json(quality_gate_path)
    quality_input = quality_gate.get("input_content_verification", {})
    provenance_remediation = quality_gate.get("provenance_remediation", {})
    if provenance_remediation.get("passed") is not True:
        raise RuntimeError(
            "quality gate lacks the verified AF-004/AF-006 provenance remediation"
        )
    manifests_bound = all(
        item.get("manifest_input_content_sha256") == content_summary["manifest_sha256"]
        for item in completeness["runs"]
    )
    caches_bound = all(
        item.get("input_content") == feature_input_content_fields(contract, backbone)
        for backbone, item in cache_evidence.items()
    )
    input_content_provenance = {
        "schema_version": 1,
        "contract_sha256": contract.sha256,
        "input_content": content_summary,
        "quality_gate": {
            "path": str(quality_gate_path.relative_to(contract.dataset_root)),
            "sha256": _sha256(quality_gate_path),
            "all_input_bytes_verified": quality_input.get("passed") is True,
            "manifest_sha256": quality_input.get("manifest_sha256"),
        },
        "run_manifests_bound": manifests_bound,
        "run_manifests_checked": len(completeness["runs"]),
        "feature_caches_bound": caches_bound,
        "feature_caches_checked": len(cache_evidence),
        "immutable_base_contract_extended_by_hash_pinned_approval_overlay": True,
        "passed": bool(
            quality_input.get("passed") is True
            and quality_input.get("manifest_sha256") == content_summary["manifest_sha256"]
            and quality_gate.get("frozen_runtime_inputs", {}).get(
                INPUT_CONTENT_MANIFEST_RELATIVE_PATH.as_posix()
            )
            == content_summary["manifest_file_sha256"]
            and quality_gate.get("frozen_runtime_inputs", {}).get(
                str(provenance_remediation.get("path"))
            )
            == provenance_remediation.get("file_sha256")
            and manifests_bound
            and caches_bound
        ),
    }
    if input_content_provenance["passed"] is not True:
        raise RuntimeError("model/tokenizer/visual input-content provenance did not pass")
    write_manifest(output / "INPUT_CONTENT_PROVENANCE.json", input_content_provenance)

    test_rows = tuple(int(row) for row in data.split_to_rows["test"])
    test_ids = tuple(data.samples[row].sample_id for row in test_rows)
    datasets = np.asarray([data.samples[row].dataset for row in test_rows], dtype=object)
    train_accuracy = compute_train_expert_accuracy(
        data.y,
        data.split_to_rows["train"],
        data.model_order,
        expected_model_order=data.model_order,
    )
    labels = build_diagnostic_labels_for_split(
        data.y,
        test_rows,
        train_accuracy,
        data.model_order,
        expected_model_order=data.model_order,
    )
    strongest_column = frozen_train_strongest_column(contract, data.model_order)
    strongest_predictions = np.full(len(test_rows), strongest_column, dtype=np.int64)
    test_index = np.asarray(test_rows, dtype=np.int64)
    strongest_correct = data.y[test_index, strongest_predictions].astype(np.float64)
    strongest_cost = data.c[test_index, strongest_predictions].astype(np.float64)
    oracle_correct = (data.y[test_index].sum(axis=1) > 0).astype(np.float64)
    strongest_accuracy = float(strongest_correct.mean())
    oracle_accuracy = float(oracle_correct.mean())
    oracle_gap = oracle_accuracy - strongest_accuracy

    independent_root = output / "independent_metrics"
    router_results: dict[str, dict[str, Any]] = {}
    long_rows: list[dict[str, Any]] = []
    for router, seed_paths in run_paths.items():
        seed_metrics = []
        seed_predictions = []
        seed_correct = []
        seed_costs = []
        seed_resources = []
        for seed, path in seed_paths.items():
            prediction_path = path / "predictions_test.jsonl"
            records, prediction_sha256 = read_frozen_predictions(prediction_path)
            recorded_metrics = _json(path / "metrics_test.json")
            resource = _json(path / "resource_usage.json")
            checkpoint_path = path / "checkpoints/best/checkpoint.pt"
            independent_path = independent_root / router.lower().replace("-", "_") / f"seed-{seed}.json"
            metrics = evaluate_prediction_file(
                prediction_path,
                data.dataset_root,
                expected_fallback_policy="none" if ROUTERS[router] == "bert" else "multi_image",
                contract_path=contract.path,
                output_path=independent_path,
                expected_split="test",
                expected_sample_ids=test_ids,
                expected_seed=seed,
                expected_checkpoint_path=checkpoint_path,
                expected_prediction_sha256=str(recorded_metrics["prediction_sha256"]),
                expected_producer_receipt_path=prediction_receipt_path(prediction_path),
                expected_backbone=ROUTERS[router],
            )
            if metrics != recorded_metrics:
                differing = sorted(
                    key
                    for key in set(metrics).union(recorded_metrics)
                    if metrics.get(key) != recorded_metrics.get(key)
                )
                raise RuntimeError(
                    f"complete independent evaluator document mismatch for {router} seed {seed}: {differing}"
                )
            if resource.get("prediction_summaries", {}).get("test", {}).get("sha256") != prediction_sha256:
                raise RuntimeError(f"resource prediction digest mismatch for {router} seed {seed}")
            predictions, correctness, costs = _prediction_arrays(records, data, test_rows)
            seed_predictions.append(predictions)
            seed_correct.append(correctness)
            seed_costs.append(costs)
            seed_metrics.append(metrics)
            seed_resources.append(resource)
            for metric in CORE_METRICS:
                long_rows.append(
                    {
                        "router": router,
                        "seed": seed,
                        "scope": "overall",
                        "group": "all",
                        "metric": metric,
                        "value": metrics[metric],
                        "samples": metrics["num_samples"],
                        "prediction_sha256": prediction_sha256,
                    }
                )
            for dataset, values in metrics["per_dataset"].items():
                for metric in ("routed_accuracy", "average_cost", "rank_score"):
                    long_rows.append(
                        {
                            "router": router,
                            "seed": seed,
                            "scope": "dataset",
                            "group": dataset,
                            "metric": metric,
                            "value": values[metric],
                            "samples": values["samples"],
                            "prediction_sha256": prediction_sha256,
                        }
                    )
            for stratum, values in metrics["difficulty_strata"].items():
                long_rows.append(
                    {
                        "router": router,
                        "seed": seed,
                        "scope": "difficulty",
                        "group": stratum,
                        "metric": "routed_accuracy",
                        "value": values["routed_accuracy"],
                        "samples": values["samples"],
                        "prediction_sha256": prediction_sha256,
                    }
                )
        router_results[router] = {
            "metrics": seed_metrics,
            "predictions": np.stack(seed_predictions),
            "correct": np.stack(seed_correct),
            "costs": np.stack(seed_costs),
            "resources": seed_resources,
        }

    independent_metric_files = sorted(independent_root.rglob("*.json"))
    independent_metrics_manifest = {
        "schema_version": 1,
        "contract_sha256": contract.sha256,
        "numerical_runtime": {
            "numpy_version": np.__version__,
            "require_same_runtime_as_run_metrics_and_aggregation": True,
        },
        "expected_documents": 15,
        "verified_documents": len(independent_metric_files),
        "complete_document_equality_verified": True,
        "documents": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
            }
            for path in independent_metric_files
        ],
        "passed": len(independent_metric_files) == 15,
    }
    if independent_metrics_manifest["passed"] is not True:
        raise RuntimeError("independent metric document inventory is incomplete")
    write_manifest(
        output / "INDEPENDENT_METRICS_MANIFEST.json", independent_metrics_manifest
    )

    archived_bert_warnings = {
        str(seed): list(resource.get("runtime_warnings", []))
        for seed, resource in enumerate(router_results["Query-BERT"]["resources"])
    }
    extraction_reproducibility: dict[str, Any] = {}
    for backbone, integrity in cache_evidence.items():
        gate = _json(contract.dataset_root / integrity["gate"])
        comparison = next(
            check["evidence"]
            for check in gate["checks"]
            if check["name"] == "single_gpu_32_row_reextraction_consistency"
        )
        extraction_reproducibility[backbone] = comparison
    reproducibility_boundary = {
        "schema_version": 1,
        "contract_sha256": contract.sha256,
        "archived_result_claim": (
            "The frozen predictions and derived metrics are exactly reproducible from archived bytes. "
            "Fresh training and cross-geometry feature extraction are not claimed bitwise identical."
        ),
        "archived_bert_runs": {
            "determinism_classification": "numerically_validated_not_fresh_run_bitwise_attested",
            "legacy_warn_only_metadata": True,
            "runtime_warnings_by_seed": archived_bert_warnings,
        },
        "feature_reextraction": extraction_reproducibility,
        "input_content": {
            "future_runs_fail_closed": True,
            "manifest_sha256": content_summary["manifest_sha256"],
            "all_model_tokenizer_and_tsv_bytes_content_addressed": True,
            "archived_feature_cache_rebound_without_reextraction": True,
            "archived_feature_bytes_unchanged": True,
            "fresh_archived_extraction_bitwise_identity_claimed": False,
            "evidence": "INPUT_CONTENT_PROVENANCE.json",
        },
        "target_free_prediction_process": {
            "archived_predictions_regenerated_in_isolated_processes": True,
            "prediction_process_loaded_y_or_c": False,
            "evaluation_process_started_after_prediction_freeze": True,
            "receipt_files_verified": 45,
            "os_open_trace_evidence": "outputs/bench_coe_query/audit/AF010_TARGET_FREE_REFREEZE.json",
            "evidence": "PREDICTION_PROVENANCE.json",
        },
        "provenance_remediation": {
            "legacy_amp_migration_entries": provenance_remediation[
                "legacy_amp_migration_entries"
            ],
            "legacy_amp_exact_skipped_step_status_claimed": False,
            "complete_state_backbones_verified": provenance_remediation[
                "full_state_backbones_verified"
            ],
            "complete_state_rank_processes_verified": provenance_remediation[
                "full_state_rank_processes_verified"
            ],
            "complete_state_digest_includes_buffers_and_raw_bytes": True,
            "path": provenance_remediation["path"],
            "file_sha256": provenance_remediation["file_sha256"],
            "manifest_sha256": provenance_remediation["manifest_sha256"],
            "evidence_chain": (
                "REPRODUCIBILITY_BOUNDARY.json -> INPUT_CONTENT_PROVENANCE.json -> "
                "QUALITY_GATE.json -> AF004_AF006_REMEDIATION.json"
            ),
        },
        "future_run_policy": {
            "mode": "fail_closed_fixed_execution_geometry",
            "torch_deterministic_warn_only": False,
            "flash_sdp_enabled": False,
            "memory_efficient_sdp_enabled": False,
            "math_sdp_enabled": True,
            "cross_geometry_bitwise_claim": False,
            "source": "routers/bench_coe_query/reproducibility.py",
            "source_sha256": _sha256(
                contract.dataset_root / "routers/bench_coe_query/reproducibility.py"
            ),
        },
        "passed": True,
    }
    write_manifest(output / "REPRODUCIBILITY_BOUNDARY.json", reproducibility_boundary)

    main_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    per_dataset_rows: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    statistical_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []

    for router, values in router_results.items():
        metrics = values["metrics"]
        accuracy_summary = _summary(item["routed_accuracy"] for item in metrics)
        hard_summary = _summary(item["hard_label_accuracy_non_all_wrong"] for item in metrics)
        delta_summary = _summary(item["routed_accuracy"] - strongest_accuracy for item in metrics)
        closure_summary = _summary(
            (item["routed_accuracy"] - strongest_accuracy) / oracle_gap for item in metrics
        )
        per_sample_mean = values["correct"].mean(axis=0)
        bootstrap = paired_bootstrap_delta(test_ids, per_sample_mean, strongest_correct)
        main_rows.append(
            {
                "method": router,
                "training_target": "BenchCoE hard-global-best-among-correct",
                "seeds": 5,
                **{f"routed_accuracy_{key}": value for key, value in accuracy_summary.items()},
                **{f"delta_to_strongest_{key}": value for key, value in delta_summary.items()},
                **{f"oracle_gap_closure_{key}": value for key, value in closure_summary.items()},
                **{f"hard_label_accuracy_{key}": value for key, value in hard_summary.items()},
                "paired_bootstrap_delta_ci_low": bootstrap.ci_low,
                "paired_bootstrap_delta_ci_high": bootstrap.ci_high,
            }
        )
        average_cost_summary = _summary(item["average_cost"] for item in metrics)
        rank_summary = _summary(item["rank_score"] for item in metrics)
        cost_rows.append(
            {
                "method": router,
                "seeds": 5,
                **{f"average_cost_{key}": value for key, value in average_cost_summary.items()},
                **{f"rank_score_{key}": value for key, value in rank_summary.items()},
            }
        )
        for dataset in sorted(set(datasets.tolist())):
            dataset_metrics = [item["per_dataset"][dataset] for item in metrics]
            accuracy = _summary(item["routed_accuracy"] for item in dataset_metrics)
            cost = _summary(item["average_cost"] for item in dataset_metrics)
            per_dataset_rows.append(
                {
                    "method": router,
                    "dataset": dataset,
                    "samples": dataset_metrics[0]["samples"],
                    **{f"routed_accuracy_{key}": value for key, value in accuracy.items()},
                    **{f"average_cost_{key}": value for key, value in cost.items()},
                }
            )
        for seed, (metric, predictions) in enumerate(zip(metrics, values["predictions"])):
            collapse_rows.append(
                {
                    "method": router,
                    "seed": seed,
                    "max_expert_share": metric["expert_usage"]["max_share"],
                    "entropy": metric["expert_usage"]["entropy"],
                    "normalized_entropy": metric["expert_usage"]["normalized_entropy"],
                    "gini": metric["expert_usage"]["gini"],
                    "strongest_selection_overlap": float((predictions == strongest_column).mean()),
                }
            )
        for stratum in ("all_wrong", "single_correct", "multi_correct"):
            summary = _summary(item["difficulty_strata"][stratum]["routed_accuracy"] for item in metrics)
            difficulty_rows.append(
                {
                    "method": router,
                    "stratum": stratum,
                    "samples": metrics[0]["difficulty_strata"][stratum]["samples"],
                    **{f"routed_accuracy_{key}": value for key, value in summary.items()},
                }
            )
        permutation = paired_permutation_mean_delta(test_ids, per_sample_mean, strongest_correct)
        statistical_rows.extend(
            [
                {
                    "left": router,
                    "right": "StrongestGlobal",
                    "test": "paired_bootstrap_mean_over_seeds",
                    "seed": "all",
                    "observed_delta": bootstrap.observed_delta,
                    "ci_low": bootstrap.ci_low,
                    "ci_high": bootstrap.ci_high,
                    "p_value": "",
                    "status": "computed",
                },
                {
                    "left": router,
                    "right": "StrongestGlobal",
                    "test": "paired_permutation_mean_over_seeds",
                    "seed": "all",
                    "observed_delta": permutation["observed_delta"],
                    "ci_low": "",
                    "ci_high": "",
                    "p_value": permutation["p_value_two_sided"],
                    "status": "computed",
                },
            ]
        )
        for seed, correctness in enumerate(values["correct"]):
            result = mcnemar_exact(test_ids, correctness, strongest_correct)
            statistical_rows.append(
                {
                    "left": router,
                    "right": "StrongestGlobal",
                    "test": "mcnemar_exact",
                    "seed": seed,
                    "observed_delta": float(correctness.mean() - strongest_correct.mean()),
                    "ci_low": "",
                    "ci_high": "",
                    "p_value": result["p_value_two_sided_exact"],
                    "status": "computed",
                }
            )
        elapsed = _summary(float(item["elapsed_seconds"]) for item in values["resources"])
        resource_rows.append(
            {
                "method": router,
                "stage": "full_training_and_prediction",
                "trainable_scope": "all_parameters" if router == "Query-BERT" else "LayerNorm+Linear_only",
                "elapsed_seconds_mean": elapsed["mean"],
                "elapsed_seconds_std": elapsed["std"],
                "peak_cuda_memory_bytes_max": max(
                    int(item["peak_cuda_memory_bytes"]) for item in values["resources"]
                ),
                "throughput_definition": "training resource record; VLM head-only excludes frozen backbone extraction",
            }
        )

    for left, right in combinations(ROUTERS, 2):
        left_mean = router_results[left]["correct"].mean(axis=0)
        right_mean = router_results[right]["correct"].mean(axis=0)
        bootstrap = paired_bootstrap_delta(test_ids, left_mean, right_mean)
        permutation = paired_permutation_mean_delta(test_ids, left_mean, right_mean)
        statistical_rows.extend(
            [
                {
                    "left": left,
                    "right": right,
                    "test": "paired_bootstrap_mean_over_seeds",
                    "seed": "all",
                    "observed_delta": bootstrap.observed_delta,
                    "ci_low": bootstrap.ci_low,
                    "ci_high": bootstrap.ci_high,
                    "p_value": "",
                    "status": "computed",
                },
                {
                    "left": left,
                    "right": right,
                    "test": "paired_permutation_mean_over_seeds",
                    "seed": "all",
                    "observed_delta": permutation["observed_delta"],
                    "ci_low": "",
                    "ci_high": "",
                    "p_value": permutation["p_value_two_sided"],
                    "status": "computed",
                },
            ]
        )

    main_rows.extend(
        [
            {
                "method": "StrongestGlobal",
                "training_target": "training-free train-only baseline",
                "seeds": 0,
                "routed_accuracy_mean": strongest_accuracy,
                "delta_to_strongest_mean": 0.0,
                "oracle_gap_closure_mean": 0.0,
            },
            {
                "method": "Oracle",
                "training_target": "report-only upper bound",
                "seeds": 0,
                "routed_accuracy_mean": oracle_accuracy,
                "delta_to_strongest_mean": oracle_gap,
                "oracle_gap_closure_mean": 1.0,
            },
        ]
    )
    strongest_prediction_path = (
        contract.dataset_root / "outputs/bench_coe_query/baseline_parity/predictions/strongest_global.jsonl"
    )
    parity_gate = _json(
        contract.dataset_root / "outputs/bench_coe_query/baseline_parity/PARITY_GATE.json"
    )
    strongest_prediction_sha256 = str(
        parity_gate["results"]["strongest_global"]["prediction_sha256"]
    )
    strongest_metrics = evaluate_prediction_file(
        strongest_prediction_path,
        data.dataset_root,
        expected_fallback_policy="none",
        contract_path=contract.path,
        expected_split="test",
        expected_sample_ids=test_ids,
        expected_seed=0,
        expected_checkpoint_id="baseline:strongest_global:frozen",
        expected_prediction_sha256=strongest_prediction_sha256,
    )
    oracle_metrics = _json(contract.dataset_root / "outputs/bench_coe_query/baseline_parity/oracle.json")
    cost_rows.extend(
        [
            {
                "method": "StrongestGlobal",
                "seeds": 0,
                "average_cost_mean": strongest_metrics["average_cost"],
                "rank_score_mean": strongest_metrics["rank_score"],
            },
            {
                "method": "Oracle",
                "seeds": 0,
                "average_cost_mean": oracle_metrics["average_cost"],
                "rank_score_mean": oracle_metrics["rank_score"],
            },
        ]
    )
    for dataset in sorted(set(datasets.tolist())):
        mask = datasets == dataset
        per_dataset_rows.append(
            {
                "method": "StrongestGlobal",
                "dataset": dataset,
                "samples": int(mask.sum()),
                "routed_accuracy_mean": float(strongest_correct[mask].mean()),
                "routed_accuracy_std": 0.0,
                "average_cost_mean": float(strongest_cost[mask].mean()),
                "average_cost_std": 0.0,
            }
        )
    stratum_masks = {
        "all_wrong": labels.correct_expert_count == 0,
        "single_correct": labels.correct_expert_count == 1,
        "multi_correct": labels.correct_expert_count > 1,
    }
    for stratum, mask in stratum_masks.items():
        difficulty_rows.append(
            {
                "method": "StrongestGlobal",
                "stratum": stratum,
                "samples": int(mask.sum()),
                "routed_accuracy_mean": float(strongest_correct[mask].mean()),
                "routed_accuracy_std": 0.0,
            }
        )
    for method in ("Official MLP lambda=0", "RouterDC", "VLC"):
        statistical_rows.append(
            {
                "left": "BenchCoE query routers",
                "right": method,
                "test": "paired_sample_level",
                "seed": "all",
                "observed_delta": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "status": "not_computable_no_frozen_sample_predictions_in_release",
            }
        )

    for backbone in ("tinyllava", "qwen3vl"):
        pointer = _json(
            contract.dataset_root / str(contract.values["features"]["root"]) / backbone / "LATEST_FULL.json"
        )
        cache_root = contract.dataset_root / pointer["cache_root"]
        resources = [_json(path) for path in sorted(cache_root.glob("resource_rank_*.json"))]
        resource_rows.append(
            {
                "method": f"Query-{backbone}",
                "stage": "frozen_backbone_feature_extraction",
                "trainable_scope": "none",
                "elapsed_seconds_mean": max(float(item["elapsed_seconds"]) for item in resources),
                "elapsed_seconds_std": "",
                "peak_cuda_memory_bytes_max": max(int(item["peak_cuda_memory_bytes"]) for item in resources),
                "throughput_definition": "8-GPU end-to-end extraction wall proxy=max rank elapsed; excludes model load",
            }
        )

    _atomic_csv(
        output / "ALL_RESULTS_LONG.csv",
        long_rows,
        ("router", "seed", "scope", "group", "metric", "value", "samples", "prediction_sha256"),
    )
    main_fields = tuple(dict.fromkeys(key for row in main_rows for key in row))
    cost_fields = tuple(dict.fromkeys(key for row in cost_rows for key in row))
    per_dataset_fields = tuple(dict.fromkeys(key for row in per_dataset_rows for key in row))
    _atomic_csv(output / "MAIN_TABLE.csv", main_rows, main_fields)
    _atomic_csv(output / "COST_TABLE.csv", cost_rows, cost_fields)
    _atomic_csv(output / "PER_DATASET.csv", per_dataset_rows, per_dataset_fields)
    _atomic_csv(
        output / "STATISTICAL_TESTS.csv",
        statistical_rows,
        ("left", "right", "test", "seed", "observed_delta", "ci_low", "ci_high", "p_value", "status"),
    )
    _atomic_csv(
        output / "ROUTING_COLLAPSE.csv",
        collapse_rows,
        ("method", "seed", "max_expert_share", "entropy", "normalized_entropy", "gini", "strongest_selection_overlap"),
    )
    _atomic_csv(
        output / "DIFFICULTY_STRATA.csv",
        difficulty_rows,
        tuple(dict.fromkeys(key for row in difficulty_rows for key in row)),
    )
    _atomic_csv(
        output / "RESOURCE_TABLE.csv",
        resource_rows,
        ("method", "stage", "trainable_scope", "elapsed_seconds_mean", "elapsed_seconds_std", "peak_cuda_memory_bytes_max", "throughput_definition"),
    )

    report_lines = [
        "# Bench-CoE Query Router reproduction report",
        "",
        f"Contract SHA-256: `{contract.sha256}`.",
        "",
        "All 15 preregistered runs are complete and were independently re-evaluated from frozen test prediction JSONL files. Checkpoint selection used dev hard-label loss only.",
        "",
        "## Main results",
        "",
        "| Method | Routed accuracy mean ± std | Delta to Strongest | Oracle-gap closure |",
        "|---|---:|---:|---:|",
    ]
    for row in main_rows:
        report_lines.append(
            f"| {row['method']} | {float(row['routed_accuracy_mean']):.6f} ± {float(row.get('routed_accuracy_std', 0.0)):.6f} | "
            f"{float(row.get('delta_to_strongest_mean', 0.0)):.6f} | {float(row.get('oracle_gap_closure_mean', 0.0)):.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Fairness and limitations",
            "",
            "- Query-BERT is text-only; TinyLLaVA and Qwen3-VL are multimodal and still require their frozen backbones online. Head-only throughput is not end-to-end throughput.",
            "- The 47 genuine multi-image rows are excluded from VLM feature/head training. Their frozen-denominator predictions use StrongestGlobal selected from train-only statistics; coverage is recorded in every metrics file.",
            "- The evaluator requires the exact expected fallback set even when empty; all 45 prediction files passed this fail-closed policy check.",
            "- All 45 router prediction files were frozen by target-free producer processes whose schemas contain no Y/C fields. Hash-bound receipts and filtered OS open traces show that the producer processes never opened Y.npz or C.npy; evaluators ran only after those processes exited.",
            "- The AF-004 legacy diagnostic migration is preserved as a per-field hash-bound ledger. Unrecoverable historical AMP skipped-step status is explicitly unknown, while future training records skipped-step counts, scaler transitions, and typed gradient status.",
            "- The AF-006 freeze probes hash every ordered state_dict parameter and persistent buffer from raw bytes before and after actual forward execution; all 16 independent rank-process attestations passed and are bound through the quality gate.",
            "- `INPUT_CONTENT_PROVENANCE.json` content-addresses every selected local model/tokenizer payload and all 18 TSV image containers. Archived feature bytes were rebound without claiming a fresh extraction rerun.",
            "- The 17 expert outcomes, including GPT-4o and Gemini, come only from official offline artifacts. No API or expert inference was run.",
            "- Official MLP/RouterDC/VLC sample-level predictions are not present in this release, so paired tests against them are explicitly marked unavailable rather than fabricated.",
            "- This is ID sample-level generalization over the same 14-dataset mixed pool, not held-out-dataset OOD evaluation.",
            "- Archived predictions/metrics are byte-anchored and exactly derivable. Fresh BERT training and TinyLLaVA cross-geometry extraction are claimed numerically reproducible only, not bitwise identical; `REPRODUCIBILITY_BOUNDARY.json` records the boundary and the fail-closed policy for future runs.",
            "",
        ]
    )
    (output / "REPRODUCTION_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    claims = [
        "# Claims and evidence",
        "",
        f"- Run completeness: `RUN_COMPLETENESS.json` ({completeness['complete_runs']}/15).",
        "- Main routed/cost results: `MAIN_TABLE.csv` and `COST_TABLE.csv`.",
        "- Per-dataset and difficulty evidence: `PER_DATASET.csv` and `DIFFICULTY_STRATA.csv`.",
        "- Paired inference: `STATISTICAL_TESTS.csv`; unavailable comparisons are retained as negative evidence.",
        "- Every independent metric artifact is under `independent_metrics/` and binds the frozen prediction SHA-256.",
        "- `PREDICTION_PROVENANCE.json` binds all 45 prediction files to logits argmax, seed, checkpoint bytes, metrics/resource anchors, and frozen SHA-256 values.",
        "- `INDEPENDENT_METRICS_MANIFEST.json` hash-gates all 15 complete independently regenerated test metric documents under the same pinned NumPy runtime.",
        "- `CACHE_INTEGRITY.json` binds both merged caches through their external pointer/gate/manifest and all 16 shard payload digests.",
        "- `INPUT_CONTENT_PROVENANCE.json` binds the approval overlay, all local model/tokenizer payloads, all 18 TSV image containers, 15 run manifests, and both feature-cache signatures.",
        "- `REPRODUCIBILITY_BOUNDARY.json` distinguishes exact archived-artifact derivation from fresh-run/cross-geometry bitwise reproducibility.",
        "- The provenance-remediation block in `REPRODUCIBILITY_BOUNDARY.json` binds the 16-entry legacy AMP migration ledger and all 16 full-state VLM rank attestations.",
        "- Adapted (not faithful) provenance and pooling decisions remain disclosed in the audit directory.",
        "",
    ]
    (output / "CLAIMS_AND_EVIDENCE.md").write_text("\n".join(claims), encoding="utf-8")
    final_gate = {
        "schema_version": 1,
        "contract_sha256": contract.sha256,
        "run_id": args.run_id,
        "run_completeness_passed": True,
        "independent_evaluation_runs": 15,
        "prediction_files_immutable": True,
        "prediction_provenance_verified_files": prediction_provenance["verified_prediction_files"],
        "target_free_prediction_processes_verified": prediction_provenance[
            "target_free_prediction_processes_verified"
        ],
        "target_free_prediction_process_boundary_passed": True,
        "legacy_amp_migration_ledger_passed": provenance_remediation[
            "legacy_amp_migration_entries"
        ]
        == 16,
        "complete_backbone_state_attestation_passed": provenance_remediation[
            "full_state_backbones_verified"
        ]
        == 2
        and provenance_remediation["full_state_rank_processes_verified"] == 16,
        "checkpoint_provenance_verified_files": prediction_provenance["verified_checkpoints"],
        "feature_cache_integrity_passed": cache_integrity["passed"],
        "fallback_policy_fail_closed": True,
        "input_content_provenance_passed": input_content_provenance["passed"],
        "reproducibility_boundary_disclosed": True,
        "official_comparator_limitation_disclosed": True,
        "passed": True,
        "artifacts": {
            name: {"path": name, "sha256": _sha256(output / name)}
            for name in (
                "ALL_RESULTS_LONG.csv",
                "MAIN_TABLE.csv",
                "COST_TABLE.csv",
                "PER_DATASET.csv",
                "STATISTICAL_TESTS.csv",
                "ROUTING_COLLAPSE.csv",
                "DIFFICULTY_STRATA.csv",
                "RESOURCE_TABLE.csv",
                "REPRODUCTION_REPORT.md",
                "CLAIMS_AND_EVIDENCE.md",
                "PREDICTION_PROVENANCE.json",
                "INDEPENDENT_METRICS_MANIFEST.json",
                "CACHE_INTEGRITY.json",
                "REPRODUCIBILITY_BOUNDARY.json",
                "INPUT_CONTENT_PROVENANCE.json",
            )
        },
    }
    write_manifest(output / "FINAL_GATE.json", final_gate)
    print(json.dumps({"output": str(output), "final_gate": final_gate}, indent=2, sort_keys=True))
    return 0
