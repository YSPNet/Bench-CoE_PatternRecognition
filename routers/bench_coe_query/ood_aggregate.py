"""Recompute and aggregate all 3 x 5 Prompt-15 ID/OOD results."""

from __future__ import annotations

import argparse
import csv
import io
import json
from typing import Any, Iterable, Mapping

import numpy as np

from .config import ExperimentContract, load_contract, sha256_file
from .errors import DataIntegrityError
from .input_content import canonical_sha256
from .ood_common import (
    OOD_BACKBONES,
    OOD_CONFIG_PATH,
    OOD_EVAL_ROLES,
    OOD_GATE_PATH,
    OOD_OUTPUT_ROOT,
    OOD_PROTOCOL_PATH,
    OOD_SEEDS,
    atomic_json,
    atomic_text,
    load_strict_json,
    ood_run_dir,
)
from .ood_evaluator import BARRIER_PATH, evaluate_ood_prediction_file


ROUTER_LABELS = {
    "bert": "Query-BERT",
    "tinyllava": "Query-TinyLLaVA",
    "qwen3vl": "Query-Qwen3-VL",
}


def _csv_text(fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buffer.getvalue()


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(array.mean())


def _std(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _require_base_evidence(contract: ExperimentContract) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = load_strict_json(contract.dataset_root / "outputs/bench_coe_query/audit_final/VERDICT.json")
    main_gate = load_strict_json(contract.dataset_root / "outputs/bench_coe_query/final/FINAL_GATE.json")
    ood_gate = load_strict_json(contract.dataset_root / OOD_GATE_PATH)
    if audit.get("verdict") != "pass" or int(audit.get("issue_count", -1)) != 0:
        raise DataIntegrityError("Prompt 14 pass is no longer present")
    if main_gate.get("passed") is not True or ood_gate.get("passed") is not True:
        raise DataIntegrityError("main or OOD execution gate is not passing")
    return audit, main_gate, ood_gate


def aggregate(contract: ExperimentContract) -> dict[str, Any]:
    audit, main_gate, ood_gate = _require_base_evidence(contract)
    output = contract.dataset_root / OOD_OUTPUT_ROOT
    metrics: dict[tuple[str, int, str], dict[str, Any]] = {}
    recompute_records = []
    run_records = []
    for backbone in OOD_BACKBONES:
        for seed in OOD_SEEDS:
            run_dir = ood_run_dir(contract, backbone, seed)
            completion = load_strict_json(run_dir / "OOD_TRAINING_COMPLETE.json")
            freeze = load_strict_json(run_dir / "PREDICTION_FREEZE.json")
            if completion.get("passed") is not True or freeze.get("passed") is not True:
                raise DataIntegrityError(f"incomplete OOD run: {backbone}/seed-{seed}")
            role_hashes = {}
            for role in OOD_EVAL_ROLES:
                path = run_dir / f"metrics_{role}.json"
                stored = load_strict_json(path)
                recomputed = evaluate_ood_prediction_file(
                    contract,
                    backbone=backbone,
                    seed=seed,
                    role=role,
                    write_outputs=False,
                )
                if stored != recomputed:
                    raise DataIntegrityError(
                        f"stored OOD metrics differ from independent recomputation: {backbone}/{seed}/{role}"
                    )
                metrics[(backbone, seed, role)] = stored
                role_hashes[role] = sha256_file(path)
                recompute_records.append(
                    {
                        "backbone": backbone,
                        "seed": seed,
                        "role": role,
                        "metrics": path.relative_to(contract.dataset_root).as_posix(),
                        "metrics_sha256": role_hashes[role],
                        "strict_equal_to_recomputation": True,
                    }
                )
            run_records.append(
                {
                    "backbone": backbone,
                    "router": ROUTER_LABELS[backbone],
                    "seed": seed,
                    "training_completion_sha256": sha256_file(
                        run_dir / "OOD_TRAINING_COMPLETE.json"
                    ),
                    "prediction_freeze_sha256": sha256_file(
                        run_dir / "PREDICTION_FREEZE.json"
                    ),
                    "metrics_sha256": role_hashes,
                    "passed": True,
                }
            )
    if len(run_records) != 15 or len(recompute_records) != 30:
        raise DataIntegrityError("OOD aggregation does not contain exactly 15 runs/30 role metrics")

    seed_rows = []
    collapse_rows = []
    per_dataset_rows = []
    for backbone in OOD_BACKBONES:
        for seed in OOD_SEEDS:
            id_metrics = metrics[(backbone, seed, "id_test")]
            ood_metrics = metrics[(backbone, seed, "ood_test")]
            seed_rows.append(
                {
                    "router": ROUTER_LABELS[backbone],
                    "backbone": backbone,
                    "seed": seed,
                    "id_samples": id_metrics["num_samples"],
                    "ood_samples": ood_metrics["num_samples"],
                    "id_accuracy": id_metrics["routed_accuracy"],
                    "ood_accuracy": ood_metrics["routed_accuracy"],
                    "id_to_ood_drop": id_metrics["routed_accuracy"]
                    - ood_metrics["routed_accuracy"],
                    "id_average_cost": id_metrics["average_cost"],
                    "ood_average_cost": ood_metrics["average_cost"],
                    "id_rank_score": id_metrics["rank_score"],
                    "ood_rank_score": ood_metrics["rank_score"],
                    "worst_ood_dataset": ood_metrics["worst_dataset"]["dataset"],
                    "worst_ood_domain": ood_metrics["worst_dataset"]["domain"],
                    "worst_ood_accuracy": ood_metrics["worst_dataset"]["routed_accuracy"],
                }
            )
            for role in OOD_EVAL_ROLES:
                value = metrics[(backbone, seed, role)]
                collapse = value["expert_collapse"]
                collapse_rows.append(
                    {
                        "router": ROUTER_LABELS[backbone],
                        "backbone": backbone,
                        "seed": seed,
                        "protocol_role": role,
                        "max_share": collapse["max_share"],
                        "normalized_entropy": collapse["normalized_entropy"],
                        "gini": collapse["gini"],
                        "dominant_expert_id": collapse["dominant_expert_id"],
                    }
                )
                for dataset, dataset_value in value["per_dataset"].items():
                    per_dataset_rows.append(
                        {
                            "router": ROUTER_LABELS[backbone],
                            "backbone": backbone,
                            "seed": seed,
                            "protocol_role": role,
                            "dataset": dataset,
                            "domain": dataset_value["domain"],
                            "samples": dataset_value["samples"],
                            "routed_accuracy": dataset_value["routed_accuracy"],
                            "average_cost": dataset_value["average_cost"],
                            "rank_score": dataset_value["rank_score"],
                        }
                    )

    summary_rows = []
    for backbone in OOD_BACKBONES:
        subset = [row for row in seed_rows if row["backbone"] == backbone]
        dataset_means = {}
        for dataset in ("MMMU_DEV_VAL", "MathVision_MINI", "ChartQA_TEST"):
            values = [
                row["routed_accuracy"]
                for row in per_dataset_rows
                if row["backbone"] == backbone and row["dataset"] == dataset
            ]
            dataset_means[dataset] = _mean(values)
        worst_dataset = min(dataset_means, key=lambda name: (dataset_means[name], name))
        worst_domain = next(
            row["domain"]
            for row in per_dataset_rows
            if row["backbone"] == backbone and row["dataset"] == worst_dataset
        )
        id_collapse = [
            row for row in collapse_rows if row["backbone"] == backbone and row["protocol_role"] == "id_test"
        ]
        ood_collapse = [
            row for row in collapse_rows if row["backbone"] == backbone and row["protocol_role"] == "ood_test"
        ]
        summary_rows.append(
            {
                "router": ROUTER_LABELS[backbone],
                "backbone": backbone,
                "seeds": 5,
                "id_accuracy_mean": _mean(row["id_accuracy"] for row in subset),
                "id_accuracy_std": _std(row["id_accuracy"] for row in subset),
                "ood_accuracy_mean": _mean(row["ood_accuracy"] for row in subset),
                "ood_accuracy_std": _std(row["ood_accuracy"] for row in subset),
                "id_to_ood_drop_mean": _mean(row["id_to_ood_drop"] for row in subset),
                "id_to_ood_drop_std": _std(row["id_to_ood_drop"] for row in subset),
                "worst_ood_dataset": worst_dataset,
                "worst_ood_domain": worst_domain,
                "worst_ood_accuracy_mean": dataset_means[worst_dataset],
                "id_max_share_mean": _mean(row["max_share"] for row in id_collapse),
                "ood_max_share_mean": _mean(row["max_share"] for row in ood_collapse),
                "id_normalized_entropy_mean": _mean(
                    row["normalized_entropy"] for row in id_collapse
                ),
                "ood_normalized_entropy_mean": _mean(
                    row["normalized_entropy"] for row in ood_collapse
                ),
                "id_gini_mean": _mean(row["gini"] for row in id_collapse),
                "ood_gini_mean": _mean(row["gini"] for row in ood_collapse),
            }
        )

    aggregate_dataset_rows = []
    keys = sorted(
        {
            (row["backbone"], row["protocol_role"], row["dataset"], row["domain"])
            for row in per_dataset_rows
        }
    )
    for backbone, role, dataset, domain in keys:
        values = [
            row
            for row in per_dataset_rows
            if row["backbone"] == backbone
            and row["protocol_role"] == role
            and row["dataset"] == dataset
        ]
        aggregate_dataset_rows.append(
            {
                "router": ROUTER_LABELS[backbone],
                "backbone": backbone,
                "protocol_role": role,
                "dataset": dataset,
                "domain": domain,
                "samples_per_seed": values[0]["samples"],
                "seeds": len(values),
                "routed_accuracy_mean": _mean(row["routed_accuracy"] for row in values),
                "routed_accuracy_std": _std(row["routed_accuracy"] for row in values),
                "average_cost_mean": _mean(row["average_cost"] for row in values),
                "average_cost_std": _std(row["average_cost"] for row in values),
                "rank_score_mean": _mean(row["rank_score"] for row in values),
                "rank_score_std": _std(row["rank_score"] for row in values),
            }
        )

    completeness = {
        "schema_version": 1,
        "kind": "prompt15_ood_run_completeness",
        "passed": True,
        "expected_runs": 15,
        "observed_runs": len(run_records),
        "expected_role_metrics": 30,
        "observed_role_metrics": len(recompute_records),
        "backbones": list(OOD_BACKBONES),
        "seeds": list(OOD_SEEDS),
        "roles": list(OOD_EVAL_ROLES),
        "runs": run_records,
    }
    independent = {
        "schema_version": 1,
        "kind": "prompt15_independent_metric_recomputation",
        "passed": True,
        "strict_equal_files": len(recompute_records),
        "records": recompute_records,
    }
    provenance = {
        "schema_version": 1,
        "kind": "prompt15_ood_provenance",
        "claim": "reconstructed held-out protocol; not exact Table A2 membership",
        "author_original_ood_manifest_found": False,
        "protocol": {
            "path": str(OOD_PROTOCOL_PATH),
            "sha256": sha256_file(contract.dataset_root / OOD_PROTOCOL_PATH),
        },
        "config": {
            "path": str(OOD_CONFIG_PATH),
            "sha256": sha256_file(contract.dataset_root / OOD_CONFIG_PATH),
        },
        "prediction_barrier": {
            "path": str(BARRIER_PATH),
            "sha256": sha256_file(contract.dataset_root / BARRIER_PATH),
        },
        "execution_gate": {
            "path": str(OOD_GATE_PATH),
            "sha256": sha256_file(contract.dataset_root / OOD_GATE_PATH),
        },
        "prompt14_verdict_sha256": sha256_file(
            contract.dataset_root / "outputs/bench_coe_query/audit_final/VERDICT.json"
        ),
        "main_final_gate_sha256": sha256_file(
            contract.dataset_root / "outputs/bench_coe_query/final/FINAL_GATE.json"
        ),
        "feature_reuse": {
            "existing_unlabeled_query_features_reused": True,
            "new_extraction_performed": False,
            "caveat": "The pretrained backbones may have seen these benchmark datasets.",
        },
        "held_out_targets": {
            "used_for_labels_training_or_tuning": False,
            "first_allowed_use": "post-freeze independent evaluation",
        },
        "base_evidence": {
            "prompt14_verdict": audit["verdict"],
            "main_final_passed": main_gate["passed"],
            "ood_execution_gate_passed": ood_gate["passed"],
        },
    }

    atomic_text(
        output / "SEED_RESULTS.csv",
        _csv_text(seed_rows[0].keys(), seed_rows),
    )
    atomic_text(
        output / "ID_OOD_SUMMARY.csv",
        _csv_text(summary_rows[0].keys(), summary_rows),
    )
    atomic_text(
        output / "PER_DATASET.csv",
        _csv_text(aggregate_dataset_rows[0].keys(), aggregate_dataset_rows),
    )
    atomic_text(
        output / "EXPERT_COLLAPSE.csv",
        _csv_text(collapse_rows[0].keys(), collapse_rows),
    )
    atomic_json(output / "RUN_COMPLETENESS.json", completeness)
    atomic_json(output / "INDEPENDENT_RECOMPUTE.json", independent)
    atomic_json(output / "PROVENANCE.json", provenance)

    report_lines = [
        "# Prompt 15 held-out-dataset OOD report",
        "",
        "## Protocol status",
        "",
        "This is a **reconstructed held-out protocol**, because no author OOD membership manifest was found in the pinned repository. It does not claim exact per-sample reproduction of Table A2. Official SPLITS files were not changed.",
        "",
        "The complete held-out datasets are MMMU (General), MathVision (STEM), and ChartQA (Charts/OCR). The other 11 datasets form train_seen/dev_seen/ID test. Labels and expert priority use train_seen only; held-out targets first become available after all predictions are frozen.",
        "",
        "## Five-seed ID/OOD results",
        "",
        "| Router | ID accuracy | OOD accuracy | ID→OOD drop | Worst OOD domain/dataset | Worst accuracy | OOD max-share |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            "| {router} | {id_accuracy_mean:.6f} ± {id_accuracy_std:.6f} | "
            "{ood_accuracy_mean:.6f} ± {ood_accuracy_std:.6f} | "
            "{id_to_ood_drop_mean:.6f} ± {id_to_ood_drop_std:.6f} | "
            "{worst_ood_domain}/{worst_ood_dataset} | {worst_ood_accuracy_mean:.6f} | "
            "{ood_max_share_mean:.6f} |".format(**row)
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "All positive, null, and negative ID→OOD changes are reported as measured; no seed or router was removed based on test performance. Architecture, training hyperparameters, and dev-loss-only checkpoint selection match the approved main experiment.",
            "",
            "The VLM query features are reused from the audited unlabeled feature caches. This does not use target labels, but the pretrained backbones may have encountered benchmark content during pretraining; the results therefore measure held-out-dataset supervised routing transfer, not guaranteed pretraining-level novelty.",
            "",
            "Detailed seed, per-dataset, expert-collapse, provenance, and independent-recomputation records are stored beside this report.",
            "",
        ]
    )
    atomic_text(output / "OOD_REPORT.md", "\n".join(report_lines))

    artifact_names = (
        "ID_OOD_SUMMARY.csv",
        "SEED_RESULTS.csv",
        "PER_DATASET.csv",
        "EXPERT_COLLAPSE.csv",
        "RUN_COMPLETENESS.json",
        "INDEPENDENT_RECOMPUTE.json",
        "PROVENANCE.json",
        "OOD_REPORT.md",
    )
    final_gate: dict[str, Any] = {
        "schema_version": 1,
        "kind": "prompt15_ood_final_gate",
        "passed": True,
        "contract_sha256": contract.sha256,
        "protocol_claim": "reconstructed_held_out_protocol",
        "exact_table_a2_reproduction_claimed": False,
        "author_original_ood_manifest_found": False,
        "prompt14_pass_required_and_verified": True,
        "main_final_gate_verified": True,
        "official_splits_overwritten": False,
        "held_out_y_used_for_training_or_tuning": False,
        "all_predictions_frozen_before_evaluation": True,
        "run_completeness_passed": True,
        "independent_metric_recomputation_passed": True,
        "expected_runs": 15,
        "observed_runs": 15,
        "physical_gpu_indices_used": [0, 1, 2, 3],
        "feature_reuse_pretraining_caveat_disclosed": True,
        "artifacts": {
            name: {
                "path": name,
                "sha256": sha256_file(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in artifact_names
        },
    }
    final_gate["gate_manifest_sha256"] = canonical_sha256(final_gate)
    atomic_json(output / "OOD_FINAL_GATE.json", final_gate)
    return final_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="configs/bench_coe_query/experiment_contract.yaml")
    args = parser.parse_args()
    result = aggregate(load_contract(args.contract))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
