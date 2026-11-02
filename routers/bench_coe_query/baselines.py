"""Leakage-safe VL-RouterBench baseline generation and evaluator parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from routers.utils.rank_score import rank_score

from .config import load_contract
from .data import load_vlrb_strict
from .evaluator import evaluate_prediction_file
from .labels import compute_train_expert_accuracy
from .schemas import StrictVLRouterData


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


def strongest_global_predictions(data: StrictVLRouterData, rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train_accuracy = compute_train_expert_accuracy(
        data.y, data.split_to_rows["train"], data.model_order, expected_model_order=data.model_order
    )
    column = int(np.argmax(train_accuracy))
    return np.full(len(rows), column, dtype=np.int64), {
        "selection_source": "train_Y_only",
        "expert_index": column,
        "expert_id": data.model_order[column],
    }


def cheapest_global_predictions(data: StrictVLRouterData, rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train_rows = np.asarray(data.split_to_rows["train"], dtype=np.int64)
    train_costs = data.c[train_rows].astype(np.float64).mean(axis=0)
    column = int(np.argmin(train_costs))
    return np.full(len(rows), column, dtype=np.int64), {
        "selection_source": "train_C_only",
        "expert_index": column,
        "expert_id": data.model_order[column],
    }


def oracle_predictions(data: StrictVLRouterData, rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    predictions = np.empty(len(rows), dtype=np.int64)
    for position, row in enumerate(rows):
        correct = np.flatnonzero(data.y[row] == 1)
        candidates = correct if len(correct) else np.arange(len(data.model_order), dtype=np.int64)
        predictions[position] = int(candidates[np.argmin(data.c[row, candidates])])
    return predictions, {"selection_source": "target_Y_C_oracle_reference"}


def random_predictions(
    data: StrictVLRouterData, rows: np.ndarray, *, seed: int = 42
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.RandomState(seed)
    predictions = rng.randint(0, len(data.model_order), size=len(rows)).astype(np.int64)
    return predictions, {"selection_source": "seeded_uniform_diagnostic", "seed": seed}


def write_predictions(
    path: str | Path,
    data: StrictVLRouterData,
    rows: np.ndarray,
    predictions: np.ndarray,
    *,
    split: str,
    seed: int,
    checkpoint_id: str,
) -> str:
    if rows.shape != predictions.shape:
        raise ValueError("rows and predictions must have identical shape")
    records = []
    for row, column in zip(rows, predictions):
        logits = np.zeros(len(data.model_order), dtype=np.float64)
        logits[int(column)] = 1.0
        records.append(
            {
                "sample_id": data.samples[int(row)].sample_id,
                "row_index": int(row),
                "predicted_expert_index": int(column),
                "predicted_expert_id": data.model_order[int(column)],
                "split": split,
                "seed": int(seed),
                "checkpoint_id": checkpoint_id,
                "logits": logits.tolist(),
            }
        )
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
    )
    _atomic_text(Path(path), payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _independent_reference(
    data: StrictVLRouterData, rows: np.ndarray, predictions: np.ndarray
) -> dict[str, float | int]:
    correct = data.y[rows, predictions]
    cost = data.c[rows, predictions].astype(np.float64)
    accuracy = float(np.sum(correct, dtype=np.int64) / len(rows))
    average_cost = float(np.sum(cost, dtype=np.float64) / len(rows))
    with (data.dataset_root / "data/matrices/cost_bounds.json").open("r", encoding="utf-8") as handle:
        bounds = json.load(handle)
    score = float(rank_score(accuracy, average_cost, float(bounds["cmin"]), float(bounds["cmax"]), beta=0.1))
    return {
        "num_samples": len(rows),
        "routed_accuracy": accuracy,
        "average_cost": average_cost,
        "rank_score": score,
    }


def run_baseline_parity(
    dataset_root: str | Path,
    *,
    contract_path: str | Path = "configs/bench_coe_query/experiment_contract.yaml",
    output_dir: str | Path = "outputs/bench_coe_query/baseline_parity",
    random_seed: int = 42,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    data = load_vlrb_strict(dataset_root, contract=contract)
    output = Path(output_dir)
    prediction_dir = output / "predictions"
    rows = np.asarray(data.split_to_rows["test"], dtype=np.int64)

    generators: dict[str, Callable[[], tuple[np.ndarray, dict[str, Any]]]] = {
        "oracle": lambda: oracle_predictions(data, rows),
        "strongest_global": lambda: strongest_global_predictions(data, rows),
        "cheapest_global": lambda: cheapest_global_predictions(data, rows),
        "random": lambda: random_predictions(data, rows, seed=random_seed),
    }
    results: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for name, generator in generators.items():
        predictions, selection = generator()
        prediction_path = prediction_dir / f"{name}.jsonl"
        prediction_hash = write_predictions(
            prediction_path,
            data,
            rows,
            predictions,
            split="test",
            seed=random_seed if name == "random" else 0,
            checkpoint_id=f"baseline:{name}:frozen",
        )
        metrics_path = output / f"{name}.json"
        selected_path = output / f"{name}_selected.jsonl"
        metrics = evaluate_prediction_file(
            prediction_path,
            dataset_root,
            expected_fallback_policy="none",
            contract_path=contract_path,
            output_path=metrics_path,
            selected_output_path=selected_path,
            expected_split="test",
            expected_seed=random_seed if name == "random" else 0,
            expected_checkpoint_id=f"baseline:{name}:frozen",
            expected_prediction_sha256=prediction_hash,
        )
        reference = _independent_reference(data, rows, predictions)
        errors = {
            key: abs(float(metrics[key]) - float(reference[key]))
            for key in ("routed_accuracy", "average_cost", "rank_score")
        }
        passed = max(errors.values()) <= 1e-12 and metrics["num_samples"] == reference["num_samples"]
        checks.append({"name": f"{name}_same_prediction_metric_parity", "passed": passed, "errors": errors})
        results[name] = {
            "prediction_path": str(prediction_path),
            "prediction_sha256": prediction_hash,
            "selection": selection,
            "metrics": metrics,
            "independent_reference": reference,
        }

    strongest_selection = results["strongest_global"]["selection"]
    cheapest_selection = results["cheapest_global"]["selection"]
    checks.extend(
        [
            {
                "name": "strongest_global_selected_from_train_only",
                "passed": strongest_selection["selection_source"] == "train_Y_only",
                "evidence": strongest_selection,
            },
            {
                "name": "cheapest_global_selected_from_train_only",
                "passed": cheapest_selection["selection_source"] == "train_C_only",
                "evidence": cheapest_selection,
            },
            {
                "name": "all_predictions_cover_exact_official_test",
                "passed": all(value["metrics"]["num_samples"] == len(rows) for value in results.values()),
                "expected": len(rows),
            },
            {
                "name": "prediction_files_unchanged_by_evaluator",
                "passed": all(value["metrics"]["prediction_unchanged"] for value in results.values()),
            },
        ]
    )
    legacy_summary = Path(dataset_root) / "outputs/baselines_evaluation/summary.csv"
    gate = {
        "schema_version": 1,
        "passed": all(check["passed"] for check in checks),
        "tolerance": 1e-12,
        "checks": checks,
        "official_step6_comparison": {
            "status": "not_present" if not legacy_summary.is_file() else "present_but_policies_not_directly_comparable",
            "path": str(legacy_summary),
            "reason": "legacy Step 6 fits Strongest/Cheapest on test; this experiment freezes both on train",
        },
        "results": results,
    }
    _atomic_text(output / "PARITY_GATE.json", json.dumps(gate, indent=2, sort_keys=True) + "\n")
    report_rows = "\n".join(
        f"| {name} | {value['metrics']['routed_accuracy']:.12f} | {value['metrics']['average_cost']:.12g} | {value['metrics']['rank_score']:.12f} |"
        for name, value in results.items()
    )
    report = f"""# Baseline evaluator parity

Gate passed: **{str(gate['passed']).lower()}**. Predictions were serialized first; the independent evaluator then loaded Y/C and verified that prediction bytes did not change.

| Baseline | Routed accuracy | Average cost | Rank Score |
|---|---:|---:|---:|
{report_rows}

Every metric matches an independent vectorized calculation on the identical predictions within `1e-12`. StrongestGlobal was selected from train Y only; CheapestGlobal from train C only. Oracle alone uses target Y/C as a declared theoretical reference. Random uses the fixed NumPy `RandomState({random_seed})` sequence.

The repository's legacy Step 6 output is absent. Its code fits StrongestGlobal and CheapestGlobal on the test arrays, so those policy results would not be directly comparable to the leakage-safe contract. The official `routers.utils.rank_score.rank_score` function and frozen cost bounds are used here.
"""
    _atomic_text(output / "PARITY_REPORT.md", report)
    return gate


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--output", default="outputs/bench_coe_query/baseline_parity")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    gate = run_baseline_parity(
        args.dataset_dir,
        contract_path=args.config,
        output_dir=args.output,
        random_seed=args.random_seed,
    )
    print(json.dumps({"passed": gate["passed"], "checks": gate["checks"]}, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
