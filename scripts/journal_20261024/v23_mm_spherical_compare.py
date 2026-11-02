#!/usr/bin/env python3
"""Add a spherical Qwen3-VL State-Bench-CoE-MM variant for v23 comparison.

The existing Euclidean mini-batch method is retained as a frozen reference.
All new centroids and cost maps use the 11-source fit groups only. Test labels
are read only after the five sets of predictions have been written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

import e12_matched as matched
from common import (OUTPUT, RESULT_COLUMNS, ROOT, SUMMARY_COLUMNS, eval_matrices,
                    make_audit, read_json, scores, summarize, utc_now, view_uids,
                    write_csv, write_json)
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source
from src.next_round_x.models import normalized
from src.subject_routing.clustering.spherical_kmeans import (
    SphericalKMeans, SphericalKMeansArtifact, assign_frozen)
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit

OUT = OUTPUT / "V23_MM_SPHERICAL_COMPARE"
METHOD = "StateRoute-MM"
SEEDS = tuple(range(5))
LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
KAPPA = 64.0
K = 2
VARIANTS = ("euclidean_original", "cosine_original_centroids",
            "spherical_fixed_lambda", "spherical_source_cv")


def log(**values) -> None:
    print(json.dumps(values, sort_keys=True), flush=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def integer_states(artifact: SphericalKMeansArtifact, states) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(artifact.cluster_ids)}
    return np.fromiter((lookup[value] for value in states), dtype=np.int8,
                       count=len(states))


def fit_spherical(x: np.ndarray, uids: list[str], seed: int, tag: str,
                  feature_sha: str) -> SphericalKMeansArtifact:
    path = OUT / "models" / f"seed-{seed}" / f"{tag}.json"
    if path.is_file():
        artifact = SphericalKMeansArtifact.load(path)
        if (artifact.fit_uids != tuple(uids) or
                artifact.feature_artifact_sha256 != feature_sha or
                artifact.seed != seed or artifact.requested_k != K or
                artifact.n_init != 10 or artifact.max_iter != 100):
            raise RuntimeError(f"frozen spherical artifact identity changed: {path}")
        log(stage="spherical_artifact_reused", seed=seed, tag=tag)
        return artifact
    start = time.perf_counter()
    artifact = SphericalKMeans(K, seed=seed, n_init=10, max_iter=100,
                               tolerance=1e-6).fit(
        x, uids, environment_id="ood11_3", fit_role="train_seen",
        feature_artifact_sha256=feature_sha,
        audit=UIDAccessAudit.create(Operation.FIT, uids))
    artifact.save(path)
    log(stage="spherical_fit", seed=seed, tag=tag,
        seconds=round(time.perf_counter() - start, 2),
        support=artifact.support, converged=artifact.converged,
        iterations=artifact.iterations, objective=artifact.objective_cosine_distance_sum)
    return artifact


def profile(y: np.ndarray, c: np.ndarray, states: np.ndarray):
    if len(y) != len(states) or len(c) != len(states):
        raise RuntimeError("source profile row alignment changed")
    support = np.bincount(states, minlength=K)
    if np.any(support == 0):
        raise RuntimeError("empty source state")
    global_q = y.mean(axis=0, dtype=np.float64)
    successes = np.zeros((K, y.shape[1]), dtype=np.float64)
    np.add.at(successes, states, y)
    quality = (successes + KAPPA * global_q) / (support[:, None] + KAPPA)
    cost = c.mean(axis=0, dtype=np.float64)
    scaled_cost = cost / float(np.median(c[c > 0]))
    return quality, scaled_cost, support


def decision_table(quality: np.ndarray, scaled_cost: np.ndarray,
                   lam: float) -> np.ndarray:
    return (quality - lam * scaled_cost[None, :]).argmax(axis=1).astype(np.int8)


def source_cv(seed: int, data: dict, fit: np.ndarray, feature_sha: str,
              bounds: dict) -> tuple[float, list[dict]]:
    fit_position = {int(row): index for index, row in enumerate(fit)}
    oof = {lam: np.full(len(fit), -1, dtype=np.int8) for lam in LAMBDAS}
    for fold in range(3):
        train = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][train]) & set(data["groups"][valid]):
            raise RuntimeError(f"source group overlap in fold {fold}")
        uids = data["uids"][train].tolist()
        artifact = fit_spherical(data["x"][train], uids, seed,
                                 f"fold-{fold}", feature_sha)
        source_states = integer_states(artifact, artifact.assignments)
        valid_states = integer_states(artifact, assign_frozen(
            artifact, data["x"][valid],
            feature_artifact_sha256=feature_sha).states)
        quality, scaled_cost, _ = profile(data["y"][train], data["c"][train],
                                          source_states)
        positions = np.asarray([fit_position[int(row)] for row in valid])
        for lam in LAMBDAS:
            oof[lam][positions] = decision_table(quality, scaled_cost, lam)[valid_states]
        log(stage="spherical_oof_frozen", seed=seed, fold=fold,
            train=len(train), valid=len(valid))
    rows = []
    for lam in LAMBDAS:
        if np.any(oof[lam] < 0):
            raise RuntimeError(f"incomplete spherical OOF actions: lambda={lam}")
        A, C, R = scores(data["y"][fit], data["c"][fit], oof[lam], bounds)
        rows.append(dict(lambda_cost=lam, A=A, C=C, R=R))
    selected = max(rows, key=lambda row: (row["R"], row["A"], -row["C"],
                                          -LAMBDAS.index(row["lambda_cost"])))
    return selected["lambda_cost"], rows


def inputs() -> tuple[dict, np.ndarray, np.ndarray, list[str], dict, dict, str]:
    views = view_uids()
    data = load_source("heldout11")
    fit = np.flatnonzero(data["role"] == "fit")
    if len(fit) != 16269 or not data["available"][fit].all():
        raise RuntimeError("frozen heldout11 source-fit feature scope changed")
    source_manifest = read_json(ROOT / "outputs/next_round_x/source/heldout11/MANIFEST.json")
    feature_sha = source_manifest["ancestors"][0]["feature_sha256"]
    x14, overlay, cache = subject.qwen_features(views["14ID"], allow_multi=True)
    if x14.shape != (6108, 2048) or len(overlay) != 10:
        raise RuntimeError("frozen Qwen3-VL test feature/overlay scope changed")
    held_uids = views["11ID"] + views["3OOD"]
    position = {uid: index for index, uid in enumerate(views["14ID"])}
    test = x14[np.asarray([position[uid] for uid in held_uids])]
    if len(test) != 6108 or not np.isfinite(test).all():
        raise RuntimeError("held-out Qwen3-VL test feature alignment changed")
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    provenance = dict(source_manifest_sha256=sha(ROOT /
        "outputs/next_round_x/source/heldout11/MANIFEST.json"),
        source_feature_sha256=source_manifest["files"][1]["sha256"],
        qwen_cache_feature_sha256=cache.manifest["features_sha256"],
        qwen_overlay_manifest_sha256=sha(subject.OVERLAY_MANIFEST),
        source_fit_n=len(fit), test_n=len(test), test_multi_image_overlay_n=len(overlay),
        raw_source_feature_norm_quantiles=np.quantile(
            np.linalg.norm(np.asarray(data["x"][fit]), axis=1),
            [0, .1, .5, .9, 1]).tolist())
    return data, fit, test, held_uids, views, bounds, feature_sha, provenance


def fit_seed(seed: int, data: dict, fit: np.ndarray, test: np.ndarray,
             held_uids: list[str], feature_sha: str, bounds: dict,
             baseline_lam: float) -> None:
    root = OUT / "models" / f"seed-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    prediction = root / "PREDICTIONS.npz"
    metadata = root / "COMPLETE.json"
    if prediction.is_file() and metadata.is_file():
        with np.load(prediction, allow_pickle=False) as frozen:
            if frozen["uids"].tolist() != held_uids or set(VARIANTS) - set(frozen.files):
                raise RuntimeError(f"seed-{seed} prediction checkpoint changed")
        log(stage="seed_reused", seed=seed)
        return
    original = matched.model_path("heldout11", "KMeans2", seed)
    with np.load(original, allow_pickle=False) as model:
        centers = np.asarray(model["centers"], dtype=np.float64)
        original_table = np.asarray(model["table"], dtype=np.int8)
        original_support = np.asarray(model["support"], dtype=np.int64)
        original_rates = np.asarray(model["rates"], dtype=np.float64)
    if centers.shape != (2, 2048) or int(original_support.sum()) != len(fit):
        raise RuntimeError(f"original seed-{seed} model shape/support changed")
    source_original = matched.assign(data["x"][fit], centers, "KMeans2")
    test_original = matched.assign(test, centers, "KMeans2")
    q_original, c_original, support_original = profile(
        data["y"][fit], data["c"][fit], source_original)
    if (not np.array_equal(support_original, original_support) or
            not np.allclose(q_original, original_rates, atol=2e-5) or
            not np.array_equal(decision_table(q_original, c_original, baseline_lam),
                               original_table)):
        raise RuntimeError(f"original seed-{seed} source profile did not reproduce")
    actions = {"euclidean_original": original_table[test_original]}

    # Diagnostic: retain the original minibatch centroids, change only the
    # assignment geometry, then reprofile on source data under that geometry.
    unit_centers = normalized(centers)
    cosine_source = (normalized(data["x"][fit]) @ unit_centers.T).argmax(axis=1)
    cosine_test = (normalized(test) @ unit_centers.T).argmax(axis=1)
    q_cos, c_cos, support_cos = profile(data["y"][fit], data["c"][fit], cosine_source)
    cosine_table = decision_table(q_cos, c_cos, baseline_lam)
    actions["cosine_original_centroids"] = cosine_table[cosine_test]

    selected_lam, cv_rows = source_cv(seed, data, fit, feature_sha, bounds)
    artifact = fit_spherical(data["x"][fit], data["uids"][fit].tolist(), seed,
                             "full", feature_sha)
    spherical_source = integer_states(artifact, artifact.assignments)
    spherical_test = integer_states(artifact, assign_frozen(
        artifact, test, feature_artifact_sha256=feature_sha).states)
    q_sphere, c_sphere, support_sphere = profile(
        data["y"][fit], data["c"][fit], spherical_source)
    fixed_table = decision_table(q_sphere, c_sphere, baseline_lam)
    cv_table = decision_table(q_sphere, c_sphere, selected_lam)
    actions["spherical_fixed_lambda"] = fixed_table[spherical_test]
    actions["spherical_source_cv"] = cv_table[spherical_test]
    np.savez_compressed(prediction, uids=np.asarray(held_uids),
                        **{name: value.astype(np.int8) for name, value in actions.items()})
    write_json(metadata, dict(seed=seed, status="complete", protocol="heldout11",
        source_fit_n=len(fit), test_n=len(test), k=K, kappa=KAPPA,
        original_lambda=baseline_lam, selected_spherical_lambda=selected_lam,
        spherical_source_cv=cv_rows, source_only_selection=True,
        original_model_sha256=sha(original), original_center_norms=np.linalg.norm(
            centers, axis=1).tolist(), original_source_support=support_original.tolist(),
        cosine_original_source_support=support_cos.tolist(),
        spherical_source_support=support_sphere.tolist(),
        original_expert_table=original_table.tolist(), cosine_original_expert_table=cosine_table.tolist(),
        spherical_fixed_expert_table=fixed_table.tolist(),
        spherical_cv_expert_table=cv_table.tolist(),
        source_ari_original_vs_spherical=float(adjusted_rand_score(
            source_original, spherical_source)),
        test_ari_original_vs_spherical=float(adjusted_rand_score(
            test_original, spherical_test)),
        source_agreement_original_vs_spherical=float(max(
            np.mean(source_original == spherical_source),
            np.mean(source_original != spherical_source))),
        test_agreement_original_vs_spherical=float(max(
            np.mean(test_original == spherical_test),
            np.mean(test_original != spherical_test))),
        spherical_center_norms=np.linalg.norm(artifact.centers, axis=1).tolist(),
        spherical_objective=artifact.objective_cosine_distance_sum,
        spherical_iterations=artifact.iterations,
        spherical_converged=artifact.converged,
        predictions_sha256=sha(prediction)))
    log(stage="seed_complete", seed=seed, selected_lambda=selected_lam,
        original_support=support_original.tolist(),
        spherical_support=support_sphere.tolist(),
        original_table=original_table.tolist(), spherical_table=cv_table.tolist())


def evaluate(views: dict, bounds: dict, baseline_lam: float) -> None:
    # Every seed is frozen before this function opens test Y/C.
    held_uids = views["11ID"] + views["3OOD"]
    predictions = {}
    for seed in SEEDS:
        root = OUT / "models" / f"seed-{seed}"
        metadata = read_json(root / "COMPLETE.json")
        path = root / "PREDICTIONS.npz"
        if metadata.get("status") != "complete" or metadata["predictions_sha256"] != sha(path):
            raise RuntimeError(f"seed-{seed} is not frozen")
        with np.load(path, allow_pickle=False) as archive:
            if archive["uids"].tolist() != held_uids:
                raise RuntimeError(f"seed-{seed} test UID order changed")
            predictions[seed] = {name: archive[name].astype(np.int8) for name in VARIANTS}
    write_json(OUT / "ACTION_LOCK.json", dict(status="predictions_frozen_before_test_scoring",
        protocol="heldout11", seeds=list(SEEDS), variants=list(VARIANTS),
        prediction_files={str((OUT / "models" / f"seed-{seed}" / "PREDICTIONS.npz").relative_to(ROOT)):
                          sha(OUT / "models" / f"seed-{seed}" / "PREDICTIONS.npz")
                          for seed in SEEDS}))
    matrices, _, _ = eval_matrices(views)
    rows = []
    for seed in SEEDS:
        for variant, action in predictions[seed].items():
            for view, section in (("11ID", slice(0, 5341)),
                                  ("3OOD", slice(5341, 6108))):
                A, C, R = scores(*matrices[view], action[section], bounds)
                rows.append(dict(experiment="V23_MM_SPHERICAL_COMPARE", method=METHOD,
                    variant=variant, view=view, seed=seed, n_test=len(matrices[view][0]),
                    A=A, C=C, R=R,
                    notes=("K=2; kappa=64; lambda=0.03" if variant != "spherical_source_cv"
                           else "K=2; kappa=64; lambda selected by three source group folds")))
    # Exact seed-by-seed reproduction of the manuscript's frozen Euclidean MM.
    with (OUTPUT / "V7_MAIN_FIVE_SEED/RESULTS.csv").open(newline="") as handle:
        reference = {(int(row["seed"]), row["view"]): row for row in csv.DictReader(handle)
                     if row["method"] == METHOD and row["variant"] == "source_cv"
                     and row["view"] in ("11ID", "3OOD")}
    if len(reference) != 10:
        raise RuntimeError("frozen Euclidean reference coverage changed")
    checks = []
    for row in rows:
        if row["variant"] != "euclidean_original":
            continue
        old = reference[row["seed"], row["view"]]
        differences = {key: abs(row[key] - float(old[key])) for key in ("A", "C", "R")}
        checks.append(dict(seed=row["seed"], view=row["view"], differences=differences,
                           passed=max(differences.values()) < 1e-8))
    if not all(item["passed"] for item in checks):
        raise RuntimeError("original Euclidean method did not reproduce all ten cells")
    write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, rows)
    summary = summarize(rows)
    write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
    metadata = [read_json(OUT / "models" / f"seed-{seed}" / "COMPLETE.json")
                for seed in SEEDS]
    write_json(OUT / "COMPARISON.json", dict(status="complete", seed0_reference_checks=checks,
        variants=list(VARIANTS), original_lambda=baseline_lam,
        spherical_selected_lambda_by_seed={str(x["seed"]): x["selected_spherical_lambda"]
                                           for x in metadata},
        source_ari_mean=float(np.mean([x["source_ari_original_vs_spherical"] for x in metadata])),
        test_ari_mean=float(np.mean([x["test_ari_original_vs_spherical"] for x in metadata])),
        source_support_by_seed={str(x["seed"]): x["spherical_source_support"] for x in metadata},
        table_by_seed={str(x["seed"]): x["spherical_cv_expert_table"] for x in metadata}))
    log(stage="evaluation_complete", rows=len(rows), summaries=len(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    seeds = tuple(sorted(set(args.seeds)))
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise RuntimeError("seeds must be a nonempty subset of 0–4")
    OUT.mkdir(parents=True, exist_ok=True)
    data, fit, test, held_uids, views, bounds, feature_sha, provenance = inputs()
    baseline_lam = matched.selected_lambdas()["heldout11", "KMeans2"]
    if baseline_lam != .03:
        raise RuntimeError("v23 heldout State-MM lambda changed")
    audit_path = OUT / "AUDIT.json"
    audit = make_audit("V23_MM_SPHERICAL_COMPARE",
        "python -B scripts/journal_20261024/v23_mm_spherical_compare.py",
        "K=2 Qwen3-VL 2048-d heldout11 source-fit. Compare frozen Euclidean minibatch "
        "centroids, cosine assignment to original centroids with source reprofiling, "
        "and exact full-batch spherical k-means with unit centers. Kappa=64. "
        "Report both lambda=0.03 fixed from v23 and lambda independently selected "
        "by three source-only group folds. No test outcomes enter fitting or selection.",
        {view: len(views[view]) for view in ("11ID", "3OOD")})
    audit.update(status="fitting", seeds_requested=list(seeds), k=K, kappa=KAPPA,
        lambda_grid=list(LAMBDAS), original_lambda=baseline_lam,
        gpu="none; frozen Qwen3-VL features, four CPU BLAS threads",
        representation_provenance=provenance, feature_artifact_sha256=feature_sha,
        original_fitter="MiniBatchKMeans(n_init=3,batch_size=1024,max_iter=100) on L2-normalized inputs",
        spherical_fitter="SphericalKMeans(n_init=10,max_iter=100,tolerance=1e-6) with unit centers")
    write_json(audit_path, audit)
    try:
        for seed in seeds:
            fit_seed(seed, data, fit, test, held_uids, feature_sha, bounds, baseline_lam)
        if args.evaluate:
            if seeds != SEEDS:
                raise RuntimeError("test scoring requires all five frozen seeds")
            evaluate(views, bounds, baseline_lam)
            audit["status"] = "complete"
        else:
            audit["status"] = "predictions_partial" if seeds != SEEDS else "predictions_frozen"
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = str(exc)
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(audit_path, audit)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
