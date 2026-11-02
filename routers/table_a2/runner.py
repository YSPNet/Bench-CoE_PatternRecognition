from __future__ import annotations

import gc
import importlib.metadata
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import create_router, fit_classifier, load_router, predictions
from .config import END_TO_END, FEATURE_METHODS, METHODS, STOCHASTIC, ExperimentError, canonical_hash, lambda_tag, resolve
from .data import Dataset, atomic_bytes, atomic_json, file_hash, load_features, read_json


def configure_runtime(config: dict) -> None:
    if config["runtime"]["offline"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [root / name / "router.py" for name in
              ("knn", "prknn", "ovr", "kmeans", "linear", "mlp", "cosinecls", "routerdc", "zooter", "vlc")]
    paths += list((root / "baselines").glob("*.py"))
    paths += list((root / "utils").glob("*.py"))
    paths += [root / "bench_coe_query/assets.py", root / "bench_coe_query/schemas.py"]
    paths += list((root / "features").glob("*.py"))
    paths += list((root.parent / "scripts/table_a2").glob("*.py"))
    return canonical_hash({str(p): file_hash(p) for p in sorted(paths)})


def freeze_experiment(data: Dataset, config: dict, output: Path) -> str:
    if output == data.root or any(output.is_relative_to(data.root / directory) for directory in
                                  ("SPLITS", "BENCHMARKS", "ORACLE", "data", "models", "vlm_router_data")):
        raise ExperimentError("output directory must be independent of benchmark inputs")
    path = output / "EXPERIMENT.json"
    recipe = {"schema_version": 1, "claim": "reconstructed held-out protocol; not exact Table A2 reproduction",
              "config": config, "input_hashes": data.hashes, "data_fingerprint": data.fingerprint,
              "code_sha256": code_hash(), "model_order": list(data.models),
              "view_counts": {role: len(rows) for role, rows in data.roles.items()},
              "selection_view": "dev_seen", "target_statistics_used_for_selection": False}
    recipe["python_version"] = sys.version.split()[0]
    pretrained_manifest = config["runtime"].get("pretrained_manifest")
    if pretrained_manifest:
        recipe["pretrained_models"] = read_json(resolve(data.root, pretrained_manifest))
        pretrained = recipe["pretrained_models"]
        for name, snapshot in pretrained["models"].items():
            reference = Path(pretrained["cache_dir"]) / ("models--" + name.replace("/", "--")) / "refs/main"
            if not reference.is_file() or reference.read_text().strip() != Path(snapshot["snapshot"]).name:
                raise ExperimentError(f"pretrained cache revision changed or missing: {name}")
            for filename, digest in snapshot["files"].items():
                if file_hash(Path(filename)) != digest:
                    raise ExperimentError(f"pretrained file changed: {filename}")
    recipe["dependency_versions"] = {}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch", "torchvision", "transformers", "sentence-transformers"):
        try:
            recipe["dependency_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            recipe["dependency_versions"][package] = None
    if path.exists():
        if read_json(path) != recipe:
            raise ExperimentError("frozen experiment inputs/config/code changed; select a new output directory")
    else:
        if output.exists() and any(p.name != ".run.lock" for p in output.iterdir()):
            raise ExperimentError("refusing to reuse a nonempty directory without a Table A2 experiment manifest")
        atomic_json(path, recipe)
    return file_hash(path)


def validate_choices(choice, count: int, num_models: int) -> np.ndarray:
    choice = np.asarray(choice)
    if choice.shape != (count,) or choice.dtype.kind not in ("i", "u"):
        raise ExperimentError("predictions must contain exactly one integer model column per sample")
    if np.any(choice < 0) or np.any(choice >= num_models):
        raise ExperimentError("predicted model column is outside the canonical model pool")
    return choice.astype(np.int64)


def oracle_choices(y, c) -> np.ndarray:
    """Total Oracle rule: prefer correctness, then lower cost, then column.

    This lexicographic definition covers rows on which every expert is wrong
    without a separate fallback branch.  On such rows, correctness ties and the
    cheapest expert is selected; accuracy necessarily remains zero.
    """
    y, c = np.asarray(y), np.asarray(c)
    if y.ndim != 2 or c.shape != y.shape or y.shape[1] == 0:
        raise ExperimentError("Oracle requires aligned nonempty Y/C matrices")
    columns = range(y.shape[1])
    return np.asarray([
        min(columns, key=lambda column: (int(y[row, column] <= 0),
                                         float(c[row, column]), column))
        for row in range(len(y))
    ], dtype=np.int64)


def observed_metrics(y, c, choice, bounds, beta) -> dict:
    choice = validate_choices(choice, len(y), y.shape[1])
    correct = y[np.arange(len(choice)), choice]
    costs = c[np.arange(len(choice)), choice]
    accuracy, avg_cost = float(correct.mean()), float(costs.mean())
    from routers.utils.rank_score import rank_score
    score = rank_score(accuracy, avg_cost, *bounds, beta=beta)
    return {"num_samples": len(choice), "num_correct": int(correct.sum()),
            "all_wrong_n": int((y.sum(axis=1) == 0).sum()), "accuracy": accuracy,
            "avg_cost": avg_cost, "rank_score": score,
            "accuracy_pct": accuracy * 100, "cost_usd_per_10k": avg_cost * 10000,
            "rank_score_pct": score * 100}


def save_array(path: Path, rows, ids, columns) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, rows=np.asarray(rows, dtype=np.int64),
                        sample_ids=np.asarray(ids, dtype=str), columns=np.asarray(columns, dtype=np.int64))
    atomic_bytes(path, buffer.getvalue())


def seed_list(method: str, seeds: list[int], config: dict) -> list[int]:
    return seeds if method in STOCHASTIC else config["seeds"][:1]


def verify_candidate(record: dict, directory: Path, signature: str) -> None:
    if record["signature"] != signature:
        raise ExperimentError(f"candidate inputs changed: {directory}")
    for name, digest in record["artifacts"].items():
        if not (directory / name).is_file() or file_hash(directory / name) != digest:
            raise ExperimentError(f"candidate artifact missing or modified: {directory / name}")


def verify_model_artifacts(selected: dict) -> None:
    model_path = Path(selected["model_path"])
    artifacts = selected.get("model_artifacts", {model_path.name: selected["model_sha256"]})
    for name, digest in artifacts.items():
        path = model_path.parent / name
        if not path.is_file() or file_hash(path) != digest:
            raise ExperimentError(f"selected checkpoint missing or changed: {path}")


def train_candidate(data, config, output, method, seed, lam, recipe_hash, inputs_hash,
                    source, feature_arrays, multimodal_meta, progress):
    directory = output / "runs" / method / f"seed_{seed}" / f"lambda_{lam}"
    signature = canonical_hash({"recipe": recipe_hash, "inputs": inputs_hash,
                                "method": method, "seed": seed, "lambda": lam})
    result_path = directory / "candidate.json"
    if result_path.exists():
        record = read_json(result_path)
        verify_candidate(record, directory, signature)
        progress(f"resume: {method} seed={seed} lambda={lam}")
        return record
    directory.mkdir(parents=True, exist_ok=True)
    tr, dv = data.roles["train_seen"], data.roles["dev_seen"]
    y, c, yd, cd = source
    model_files = []
    if method == "oracle":
        choice = oracle_choices(yd, cd)
        model_file = "policy.json"
        atomic_json(directory / model_file, {"method": "oracle", "theoretical_upper_bound": True})
    else:
        router = create_router(method, config, lam if lam != "na" else "0", seed)
        if method in ("linear", "mlp"):
            text, vision = feature_arrays
            fit_classifier(router, method, config, y, c, text[tr], vision[tr], yd, cd,
                           text[dv], vision[dv], data.models, data.cost_bounds, directory)
        else:
            kwargs = {}
            meta = data.meta if multimodal_meta is None else multimodal_meta
            if method in FEATURE_METHODS:
                text, vision = feature_arrays
                kwargs.update(X_text=text[tr], X_vision=vision[tr])
            if method in FEATURE_METHODS or method in END_TO_END:
                kwargs.update(model_mapping=dict(enumerate(data.models)), costs=c.mean(axis=0),
                              cmin=data.cost_bounds[0], cmax=data.cost_bounds[1],
                              rank_score_beta=config["rank_score_beta"])
            if method in END_TO_END:
                kwargs.update(Y_dev=yd, C_dev=cd, meta_dev=meta.iloc[dv].copy(),
                              monitor_output_dir=directory / "training_monitor")
            router.fit(y, c, meta.iloc[tr].copy(), **kwargs)
        meta = data.meta if multimodal_meta is None else multimodal_meta
        text, vision = feature_arrays if feature_arrays is not None else (None, None)
        choice = predictions(router, method, meta.iloc[dv].copy(),
                             None if text is None else text[dv], None if vision is None else vision[dv],
                             config["training"]["batch_size"])
        if method in ("strongest", "cheapest"):
            model_file = "policy.json"
            column = router.best_j if method == "strongest" else router.cheapest_j
            atomic_json(directory / model_file, {"method": method, "column": int(column),
                        "expert_id": data.models[column], "statistics_view": "train_seen",
                        "train_expert_accuracy": y.mean(axis=0).tolist(), "train_mean_cost": c.mean(axis=0).tolist()})
        else:
            model_file = "model.pkl"
            separate_weights = method == "vlc" and getattr(router, "model", None) is not None
            router.save(str(directory / model_file))
            if separate_weights:
                weights = directory / "model_model.pt"
                if not weights.is_file():
                    raise ExperimentError("VLC did not save its separate model weights")
                model_files.append(weights.name)
        del router
    choice = validate_choices(choice, len(dv), len(data.models))
    save_array(directory / "dev_predictions.npz", dv, data.meta.iloc[dv].sample_id, choice)
    metrics = observed_metrics(yd, cd, choice, data.cost_bounds, config["rank_score_beta"])
    record = {"signature": signature, "method": method, "seed": seed, "lambda": lam,
              "model_file": model_file, "dev_metrics": metrics, "selection_view": "dev_seen",
              "artifacts": {name: file_hash(directory / name) for name in [model_file, *model_files, "dev_predictions.npz"]}}
    atomic_json(result_path, record)
    progress(f"trained: {method} seed={seed} lambda={lam} dev_RS={metrics['rank_score_pct']:.3f}")
    gc.collect()
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        sys.modules["torch"].cuda.empty_cache()
    return record


def selection(data, config, output, method, seed, recipe_hash, inputs_hash,
              source, features, multimodal_meta, progress):
    fixed = config.get("runtime", {}).get("fixed_lambda_by_method_seed", {})
    fixed_lambda = fixed.get(f"{method}/seed_{seed}")
    if fixed_lambda is not None and method not in STOCHASTIC:
        raise ExperimentError("fixed lambda is valid only for stochastic methods")
    candidates = ([lambda_tag(fixed_lambda)] if fixed_lambda is not None
                  else [lambda_tag(lam) for lam in config["lambdas"]] if method in STOCHASTIC
                  else ["na"])
    records = [train_candidate(data, config, output, method, seed, lam, recipe_hash, inputs_hash,
                               source, features, multimodal_meta, progress) for lam in candidates]
    # Stable ties retain the first configured lambda, never target performance.
    best = max(records, key=lambda record: record["dev_metrics"]["rank_score"])
    directory = output / "runs" / method / f"seed_{seed}"
    result = {"method": method, "seed": seed, "selected_lambda": best["lambda"],
              "selection_view": "dev_seen", "selection_metric": "rank_score", "tie_break": "configured_lambda_order",
              "candidate_signatures": [record["signature"] for record in records],
              "model_path": str(directory / f"lambda_{best['lambda']}" / best["model_file"]),
              "model_sha256": best["artifacts"][best["model_file"]], "dev_metrics": best["dev_metrics"],
              "model_artifacts": {name: digest for name, digest in best["artifacts"].items() if name != "dev_predictions.npz"},
              "recipe_sha256": recipe_hash, "inputs_sha256": inputs_hash}
    path = directory / "selection.json"
    if path.exists():
        if read_json(path) != result:
            raise ExperimentError(f"frozen dev selection changed: {path}")
    else:
        atomic_json(path, result)
    return path


def freeze_predictions(data, config, output, selection_path, feature_arrays, multimodal_meta):
    selected = read_json(selection_path)
    model_path = Path(selected["model_path"])
    verify_model_artifacts(selected)
    method, seed = selected["method"], selected["seed"]
    if method == "oracle":
        return  # Oracle is explicitly generated by the outcome-aware evaluator.
    policy = read_json(model_path) if method in ("strongest", "cheapest") else None
    router = None
    meta = data.meta if multimodal_meta is None else multimodal_meta
    for role in ("id_test", "ood_test"):
        rows = data.roles[role]
        path = output / "predictions" / method / f"seed_{seed}" / f"{role}.npz"
        sidecar = path.with_suffix(".json")
        signature = {"method": method, "seed": seed, "role": role, "selected_lambda": selected["selected_lambda"],
                     "selection_path": str(selection_path), "selection_sha256": file_hash(selection_path),
                     "model_sha256": selected["model_sha256"], "data_fingerprint": data.fingerprint,
                     "recipe_sha256": selected["recipe_sha256"], "theoretical_upper_bound": False}
        if sidecar.exists():
            saved = read_json(sidecar)
            if {key: saved[key] for key in signature} != signature:
                raise ExperimentError("frozen prediction inputs changed")
            read_predictions(data, path)
            continue
        if policy is not None:
            choice = np.full(len(rows), policy["column"], dtype=np.int64)
        else:
            if router is None:
                router = load_router(method, model_path, config)
            text, vision = feature_arrays if feature_arrays is not None else (None, None)
            choice = predictions(router, method, meta.iloc[rows].copy(),
                                 None if text is None else text[rows], None if vision is None else vision[rows],
                                 config["training"]["batch_size"])
        choice = validate_choices(choice, len(rows), len(data.models))
        if path.exists():
            # A crash between the array and sidecar commits can be recovered
            # only by reproducing the predictions from the selected checkpoint.
            with np.load(path, allow_pickle=False) as archive:
                if (not np.array_equal(archive["rows"], rows)
                        or not np.array_equal(archive["columns"], choice)
                        or archive["sample_ids"].tolist() != data.meta.iloc[rows].sample_id.tolist()):
                    raise ExperimentError(f"uncommitted prediction disagrees with checkpoint: {path}")
        else:
            save_array(path, rows, data.meta.iloc[rows].sample_id, choice)
        atomic_json(sidecar, {**signature, "prediction_sha256": file_hash(path)})


def read_predictions(data: Dataset, path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    saved = read_json(path.with_suffix(".json"))
    if saved["data_fingerprint"] != data.fingerprint or file_hash(path) != saved["prediction_sha256"]:
        raise ExperimentError("frozen predictions or benchmark inputs changed")
    selection_path = Path(saved["selection_path"])
    if file_hash(selection_path) != saved["selection_sha256"]:
        raise ExperimentError("prediction dev selection changed")
    selected = read_json(selection_path)
    verify_model_artifacts(selected)
    if (selected["method"] != saved["method"] or selected["seed"] != saved["seed"]
            or selected["selected_lambda"] != saved["selected_lambda"]
            or selected["recipe_sha256"] != saved["recipe_sha256"]
            or selected["model_sha256"] != saved["model_sha256"]
            or file_hash(Path(selected["model_path"])) != saved["model_sha256"]):
        raise ExperimentError("prediction method, seed or checkpoint binding is invalid")
    with np.load(path, allow_pickle=False) as archive:
        rows, choice, ids = archive["rows"], archive["columns"], archive["sample_ids"]
    if (not np.array_equal(rows, data.roles[saved["role"]])
            or ids.tolist() != data.meta.iloc[rows].sample_id.tolist()):
        raise ExperimentError("prediction rows do not exactly cover their protocol view")
    choice = validate_choices(choice, len(rows), len(data.models))
    return saved, rows, choice


def evaluate(data: Dataset, config: dict, output: Path, methods=None, seeds=None) -> list[dict]:
    """Validate frozen predictions before materializing any target Y/C rows."""
    methods, seeds = methods or config["methods"], seeds or config["seeds"]
    results = []
    for method in methods:
        for seed in seed_list(method, seeds, config):
            selection_path = output / "runs" / method / f"seed_{seed}" / "selection.json"
            selected = read_json(selection_path)
            for role in ("id_test", "ood_test"):
                path = output / "predictions" / method / f"seed_{seed}" / f"{role}.npz"
                if method == "oracle" and not path.with_suffix(".json").exists():
                    rows = data.roles[role]
                    y, c = data.matrices(rows)
                    choice = oracle_choices(y, c)
                    save_array(path, rows, data.meta.iloc[rows].sample_id, choice)
                    atomic_json(path.with_suffix(".json"), {
                        "method": method, "seed": seed, "role": role, "selected_lambda": selected["selected_lambda"],
                        "selection_path": str(selection_path), "selection_sha256": file_hash(selection_path),
                        "model_sha256": selected["model_sha256"], "recipe_sha256": selected["recipe_sha256"],
                        "data_fingerprint": data.fingerprint, "prediction_sha256": file_hash(path),
                        "theoretical_upper_bound": True})
                saved, rows, choice = read_predictions(data, path)
                y, c = data.matrices(rows)
                metrics = observed_metrics(y, c, choice, data.cost_bounds, config["rank_score_beta"])
                per_dataset = []
                datasets = data.meta.iloc[rows].dataset.to_numpy()
                for dataset in sorted(set(datasets)):
                    mask = datasets == dataset
                    per_dataset.append({"dataset": dataset, **observed_metrics(y[mask], c[mask], choice[mask],
                                                                               data.cost_bounds, config["rank_score_beta"])})
                result = {**saved, **metrics, "per_dataset": per_dataset,
                          "cost_unit": "USD per sample; cost_usd_per_10k is USD/10000 samples",
                          "rank_score_beta": config["rank_score_beta"], "cost_bounds": list(data.cost_bounds)}
                metric_path = output / "metrics" / method / f"seed_{seed}" / f"{role}.json"
                atomic_json(metric_path, result)
                atomic_json(metric_path.with_suffix(".digest.json"), {"sha256": file_hash(metric_path)})
                results.append(result)
    return results


def aggregate(data: Dataset, config: dict, output: Path, methods=None, seeds=None,
              allow_partial=False) -> dict:
    methods, seeds = methods or config["methods"], seeds or config["seeds"]
    raw, per_dataset, missing = [], [], []
    for method in methods:
        for seed in seed_list(method, seeds, config):
            for role in ("id_test", "ood_test"):
                path = output / "metrics" / method / f"seed_{seed}" / f"{role}.json"
                digest_path = path.with_suffix(".digest.json")
                if not path.exists() or (allow_partial and not digest_path.is_file()):
                    missing.append(f"{method}/seed_{seed}/{role}")
                    continue
                result = read_json(path)
                if file_hash(path) != read_json(digest_path)["sha256"]:
                    raise ExperimentError("evaluation metrics were modified")
                prediction = output / "predictions" / method / f"seed_{seed}" / f"{role}.npz"
                saved, _, _ = read_predictions(data, prediction)
                if any(result[key] != saved[key] for key in saved):
                    raise ExperimentError("metrics do not match their frozen predictions")
                raw.append({key: value for key, value in result.items()
                            if key not in ("per_dataset", "cost_bounds")})
                per_dataset.extend({"method": method, "seed": seed, "role": role,
                                    "selected_lambda": result["selected_lambda"], **row} for row in result["per_dataset"])
    if missing and not allow_partial:
        raise ExperimentError(f"incomplete experiment: {len(missing)} missing view results; use --allow-partial to inspect")
    table = []
    for method in methods:
        entries = [row for row in raw if row["method"] == method]
        expected = seed_list(method, seeds, config)
        complete_seeds = [seed for seed in expected
                          if all(any(row["seed"] == seed and row["role"] == role for row in entries)
                                 for role in ("id_test", "ood_test"))]
        row = {"method": method, "num_seeds": len(complete_seeds), "expected_seeds": len(expected),
               "status": "complete" if len(complete_seeds) == len(expected) else "incomplete"}
        for role, prefix in (("id_test", "id"), ("ood_test", "ood")):
            values = [entry for entry in entries if entry["role"] == role and entry["seed"] in complete_seeds]
            for metric in ("accuracy_pct", "cost_usd_per_10k", "rank_score_pct"):
                numbers = [entry[metric] for entry in values]
                row[f"{prefix}_{metric}_mean"] = float(np.mean(numbers)) if numbers else None
                row[f"{prefix}_{metric}_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else (0.0 if numbers else None)
        row["selected_lambdas"] = ",".join(str(entry["selected_lambda"]) for entry in entries
                                          if entry["role"] == "id_test" and entry["seed"] in complete_seeds)
        if complete_seeds:
            row["accuracy_drop_pp_mean"] = row["id_accuracy_pct_mean"] - row["ood_accuracy_pct_mean"]
        table.append(row)
    summary = output / "summary"
    for name, rows in (("per_seed.csv", raw), ("per_dataset.csv", per_dataset), ("table_a2.csv", table)):
        atomic_bytes(summary / name, pd.DataFrame(rows).to_csv(index=False).encode())
    lines = ["# Table A2：重建协议实验", "", "成本单位：USD/万样本；Accuracy 与 Rank Score 为百分制。",
             "每个 seed 先在完整视图上计算 Rank Score，再汇总 seed 均值；标准差使用 ddof=1。", "",
             "| 方法 | seed | ID Acc | ID Cost | ID RS | OOD Acc | OOD Cost | OOD RS | 状态 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    def formatted(row, prefix, metric):
        mean, std = row[f"{prefix}_{metric}_mean"], row[f"{prefix}_{metric}_std"]
        return "—" if mean is None else f"{mean:.2f} ± {std:.2f}"
    for row in table:
        cells = [formatted(row, prefix, metric) for prefix in ("id", "ood")
                 for metric in ("accuracy_pct", "cost_usd_per_10k", "rank_score_pct")]
        lines.append(f"| {row['method']} | {row['num_seeds']}/{row['expected_seeds']} | " + " | ".join(cells) + f" | {row['status']} |")
    atomic_bytes(summary / "table_a2.md", ("\n".join(lines) + "\n").encode())
    coverage = {"claim": "reconstructed held-out protocol; not exact Table A2 reproduction",
                "methods": methods, "seeds": seeds, "complete": not missing, "missing": missing,
                "data_fingerprint": data.fingerprint}
    atomic_json(summary / "COVERAGE.json", coverage)
    return coverage


def run(data: Dataset, config: dict, output: Path, methods=None, seeds=None, progress=print,
        report=True, prepared_images_only=False) -> dict:
    methods, seeds = methods or config["methods"], seeds or config["seeds"]
    if not set(methods).issubset(config["methods"]) or not set(seeds).issubset(config["seeds"]):
        raise ExperimentError("requested methods/seeds are outside the configured experiment")
    configure_runtime(config)
    recipe_hash = freeze_experiment(data, config, output)
    features, feature_hash, multimodal, image_hash = None, "unused", None, "unused"
    if set(methods) & FEATURE_METHODS:
        text, vision, feature_hash = load_features(data, config)
        features = (text, vision)
    if set(methods) & END_TO_END:
        from .inputs import prepare_images
        image_output = resolve(data.root, config["runtime"]["image_cache_dir"]) if config["runtime"].get("image_cache_dir") else output
        multimodal, image_hash = prepare_images(data, image_output, progress, read_only=prepared_images_only)
    y, c = data.matrices(data.roles["train_seen"])
    yd, cd = data.matrices(data.roles["dev_seen"])
    source = (y, c, yd, cd)
    from contextlib import nullcontext
    strict_images = nullcontext()
    if set(methods) & END_TO_END:
        from .inputs import forbid_synthetic_fallback_images
        strict_images = forbid_synthetic_fallback_images()
    with strict_images:
        for method in methods:
            inputs_hash = feature_hash if method in FEATURE_METHODS else image_hash if method in END_TO_END else "unused"
            for seed in seed_list(method, seeds, config):
                path = selection(data, config, output, method, seed, recipe_hash, inputs_hash,
                                 source, features, multimodal, progress)
                freeze_predictions(data, config, output, path, features if method in FEATURE_METHODS else None,
                                   multimodal if method in END_TO_END else None)
                progress(f"predictions frozen: {method} seed={seed}")
    # All learned predictions in the requested scope exist before target scoring.
    results = evaluate(data, config, output, methods, seeds)
    if not report:
        return {"methods": methods, "seeds": seeds, "evaluated_views": len(results)}
    return aggregate(data, config, output, methods, seeds)
