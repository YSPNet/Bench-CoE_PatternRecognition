#!/usr/bin/env python3
"""Complete the spherical State-Bench-CoE measurements requested in v25.

Every map and cost knob is fitted on the 11-source fit groups. Test outcomes
are opened only after all per-query actions have been frozen and hashed.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from threadpoolctl import threadpool_limits

import v23_mm_spherical_compare as previous
from common import (OUTPUT, RESULT_COLUMNS, ROOT, SUMMARY_COLUMNS, eval_matrices,
                    make_audit, read_json, scores, summarize, utc_now, write_csv,
                    write_json)
from src.subject_routing.clustering.spherical_kmeans import (
    SphericalKMeans, SphericalKMeansArtifact, assign_frozen)
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit

OUT = OUTPUT / "V25_SPHERICAL_COMPLETION"
KS = (1, 2, 4, 6, 8, 10, 12)
KAPPAS = (0., 5., 20., 64., 128., 512.)
LAMBDAS = previous.LAMBDAS
SEEDS = tuple(range(5))
N_ID = 5341
VARIANTS = ("main", "no_cost", "state_conditional_cost", "eligibility_filter")


def log(**fields):
    print(json.dumps(fields, sort_keys=True), flush=True)


def profile(y, c, states, k, kappa=64.):
    support = np.bincount(states, minlength=k).astype(np.float64)
    if np.any(support <= 0):
        raise RuntimeError("empty state in source profile")
    successes = np.zeros((k, y.shape[1]), dtype=np.float64)
    cost_sums = np.zeros_like(successes)
    np.add.at(successes, states, y)
    np.add.at(cost_sums, states, c)
    global_q = y.mean(axis=0, dtype=np.float64)
    global_c = c.mean(axis=0, dtype=np.float64)
    quality = (successes + kappa * global_q) / (support[:, None] + kappa)
    conditional_c = cost_sums / support[:, None]
    scale = float(np.median(c[c > 0]))
    return quality, conditional_c, global_c, scale, support, successes, global_q


def fit_states(k, seed, tag, train, train_uids, test, feature_sha):
    if k == 1:
        return (np.zeros(len(train), dtype=np.int16),
                np.zeros(len(test), dtype=np.int16), None)
    path = (OUTPUT / "V23_MM_SPHERICAL_COMPARE" / "models" / f"seed-{seed}" /
            f"{tag}.json") if k == 2 else OUT / "models" / f"K{k}" / f"seed-{seed}" / f"{tag}.json"
    if path.is_file():
        model = SphericalKMeansArtifact.load(path)
        if (model.requested_k != k or model.seed != seed or model.fit_uids != tuple(train_uids)
                or model.feature_artifact_sha256 != feature_sha or model.n_init != 10
                or model.max_iter != 100):
            raise RuntimeError(f"cluster artifact identity differs: {path}")
    else:
        if k == 2:
            raise RuntimeError("the previously frozen K=2 artifact is missing")
        model = SphericalKMeans(k, seed=seed, n_init=10, max_iter=100,
                                tolerance=1e-6).fit(
            train, train_uids, environment_id="ood11_3", fit_role="train_seen",
            feature_artifact_sha256=feature_sha,
            audit=UIDAccessAudit.create(Operation.FIT, train_uids))
        model.save(path)
        log(stage="cluster_fit", k=k, seed=seed, tag=tag,
            support=model.support, iterations=model.iterations)
    source = previous.integer_states(model, model.assignments).astype(np.int16)
    target = previous.integer_states(model, assign_frozen(
        model, test, feature_artifact_sha256=feature_sha).states).astype(np.int16)
    return source, target, model


def choose_lambda(k, data, fit, feature_sha, bounds):
    path = OUT / f"SOURCE_CV_K{k}.json"
    if path.is_file():
        return read_json(path)
    position = {int(index): order for order, index in enumerate(fit)}
    oof = {lam: np.full(len(fit), -1, dtype=np.int8) for lam in LAMBDAS}
    for fold in range(3):
        train = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][train]) & set(data["groups"][valid]):
            raise RuntimeError("source group overlap")
        source, target, _ = fit_states(k, 0, f"fold-{fold}", data["x"][train],
                                      data["uids"][train].tolist(), data["x"][valid],
                                      feature_sha)
        q, _, cost, scale, _, _, _ = profile(data["y"][train], data["c"][train],
                                             source, k)
        positions = np.asarray([position[int(index)] for index in valid])
        for lam in LAMBDAS:
            table = (q - lam * cost[None, :] / scale).argmax(axis=1)
            oof[lam][positions] = table[target]
    rows = []
    for lam in LAMBDAS:
        action = oof[lam]
        if np.any(action < 0):
            raise RuntimeError("incomplete source OOF actions")
        a, c, r = scores(data["y"][fit], data["c"][fit], action, bounds)
        rows.append(dict(lambda_cost=lam, A=a, C=c, R=r))
    winner = max(rows, key=lambda row: (row["R"], row["A"], -row["C"],
                                        -LAMBDAS.index(row["lambda_cost"])))
    result = dict(k=k, seed=0, folds=3, group_isolated=True,
                  source_fit_n=len(fit), selected_lambda=winner["lambda_cost"],
                  candidates=rows, test_outcomes_used=False)
    write_json(path, result)
    log(stage="source_cv", k=k, selected_lambda=winner["lambda_cost"])
    return result


def proposition_fraction(successes, support, global_q, kappa):
    if kappa == 0:
        return None
    q_full = successes / support[:, None]
    rhs = (2. / kappa + 1. / support[:, None]) * q_full * (1. - q_full)
    condition = (global_q[None, :] - q_full) ** 2 < rhs
    weights = np.broadcast_to(support[:, None], condition.shape)
    return float(weights[condition].sum() / weights.sum())


def freeze_actions(data, fit, test, uids, feature_sha, bounds):
    source_y, source_c = data["y"][fit], data["c"][fit]
    selections = {k: choose_lambda(k, data, fit, feature_sha, bounds) for k in KS}
    if selections[2]["selected_lambda"] != .03:
        raise RuntimeError("K=2 source-selected lambda differs from frozen v23")
    details = []
    for k in KS:
        for seed in SEEDS:
            target = OUT / "actions" / f"K{k}_seed-{seed}.npz"
            detail = OUT / "actions" / f"K{k}_seed-{seed}.json"
            if target.is_file() and detail.is_file():
                saved = read_json(detail)
                if saved["action_sha256"] != previous.sha(target):
                    raise RuntimeError(f"frozen action hash differs: {target}")
                details.append(saved)
                log(stage="actions_reused", k=k, seed=seed)
                continue
            source, assigned, model = fit_states(k, seed, "full", data["x"][fit],
                data["uids"][fit].tolist(), test, feature_sha)
            q, conditional_c, global_c, scale, support, successes, global_q = profile(
                source_y, source_c, source, k)
            lam = selections[k]["selected_lambda"]
            table = (q - lam * global_c[None, :] / scale).argmax(axis=1).astype(np.int8)
            actions = {"uids": np.asarray(uids), "main": table[assigned].astype(np.int8)}
            extra = {}
            if k == 2:
                extra["no_cost"] = q.argmax(axis=1).astype(np.int8)
                extra["state_conditional_cost"] = (
                    q - lam * conditional_c / scale).argmax(axis=1).astype(np.int8)
                cap = float(np.quantile(global_c, .5))
                eligible = np.flatnonzero(global_c <= cap + 1e-15)
                extra["eligibility_filter"] = np.asarray([
                    min(eligible, key=lambda j: (-q[state, j], global_c[j], j))
                    for state in range(k)], dtype=np.int8)
                for name, values in extra.items():
                    actions[name] = values[assigned].astype(np.int8)
                for alternative in LAMBDAS:
                    curve_table = (q - alternative * global_c[None, :] / scale).argmax(axis=1)
                    actions[f"lambda_{alternative:g}"] = curve_table[assigned].astype(np.int8)
                fractions = {}
                kappa_tables = {}
                for kappa in KAPPAS:
                    q_alt = (successes + kappa * global_q) / (support[:, None] + kappa)
                    alt_table = (q_alt - lam * global_c[None, :] / scale).argmax(axis=1)
                    fractions[f"{kappa:g}"] = proposition_fraction(
                        successes, support, global_q, kappa)
                    kappa_tables[f"{kappa:g}"] = alt_table.tolist()
                extra = {name: values.tolist() for name, values in extra.items()}
            else:
                fractions, kappa_tables = {}, {}
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **actions)
            saved = dict(k=k, seed=seed, source_fit_n=len(fit),
                selected_lambda=lam, source_support=support.astype(int).tolist(),
                expert_table=table.tolist(), cost_variant_tables=extra,
                kappa_fractions=fractions, kappa_tables=kappa_tables,
                cluster_artifact_sha256=model.sha256 if model is not None else None,
                action_sha256=previous.sha(target))
            write_json(detail, saved)
            details.append(saved)
            if k == 2:
                old = OUTPUT / "V23_MM_SPHERICAL_COMPARE" / "models" / f"seed-{seed}" / "PREDICTIONS.npz"
                with np.load(old, allow_pickle=False) as frozen:
                    if frozen["uids"].tolist() != uids or not np.array_equal(
                            actions["main"], frozen["spherical_source_cv"]):
                        raise RuntimeError(f"K=2 main actions did not reproduce v23 seed {seed}")
            log(stage="actions_frozen", k=k, seed=seed, support=saved["source_support"],
                table=saved["expert_table"])
    sources = {str((OUT / "actions" / f"K{k}_seed-{seed}.npz").relative_to(ROOT)):
               previous.sha(OUT / "actions" / f"K{k}_seed-{seed}.npz")
               for k in KS for seed in SEEDS}
    write_json(OUT / "ACTION_LOCK.json", dict(
        status="predictions_frozen_before_test_scoring", protocol="heldout11",
        source_fit_n=len(fit), test_n=len(uids), seeds=list(SEEDS), K_grid=list(KS),
        source_lambda_by_k={str(k): selections[k]["selected_lambda"] for k in KS},
        files=sources))
    return details


def evaluate(details, views, bounds):
    lock = read_json(OUT / "ACTION_LOCK.json")
    if lock["status"] != "predictions_frozen_before_test_scoring":
        raise RuntimeError("test actions have not been locked")
    matrices, _, _ = eval_matrices(views)
    held_uids = views["11ID"] + views["3OOD"]
    rows, dataset_rows, curve_rows = [], [], []
    for detail in details:
        k, seed = detail["k"], detail["seed"]
        path = OUT / "actions" / f"K{k}_seed-{seed}.npz"
        if previous.sha(path) != detail["action_sha256"]:
            raise RuntimeError("action changed after freeze")
        with np.load(path, allow_pickle=False) as frozen:
            if frozen["uids"].tolist() != held_uids:
                raise RuntimeError("test UID order changed")
            variants = ("main",) if k != 2 else VARIANTS
            for variant in variants:
                action = frozen[variant]
                for view, section in (("11ID", slice(0, N_ID)),
                                      ("3OOD", slice(N_ID, None))):
                    a, c, r = scores(*matrices[view], action[section], bounds)
                    rows.append(dict(experiment="V25_SPHERICAL_COMPLETION",
                        method="StateRoute-MM", variant=f"K{k}/{variant}", view=view,
                        seed=seed, n_test=len(matrices[view][0]), A=a, C=c, R=r,
                        notes="spherical Qwen3-VL-2B; kappa=64; source-only lambda"))
            if k == 2:
                for lam in LAMBDAS:
                    action = frozen[f"lambda_{lam:g}"]
                    for view, section in (("11ID", slice(0, N_ID)),
                                          ("3OOD", slice(N_ID, None))):
                        a, c, r = scores(*matrices[view], action[section], bounds)
                        curve_rows.append(dict(seed=seed, view=view, lambda_cost=lam,
                                               A=a, C=c, R=r,
                                               source_selected=lam == .03))
                if seed == 0:
                    for dataset in ("MMMU", "MathVision", "ChartQA"):
                        index = np.flatnonzero(np.asarray([
                            uid.split("/", 1)[0].startswith(dataset)
                            for uid in held_uids[N_ID:]]))
                        if not len(index):
                            raise RuntimeError(f"missing OOD dataset: {dataset}")
                        action = frozen["main"][N_ID:][index]
                        a, c, r = scores(matrices["3OOD"][0][index],
                                         matrices["3OOD"][1][index], action, bounds)
                        unique, counts = np.unique(action, return_counts=True)
                        dataset_rows.append(dict(dataset=dataset, n_test=len(index),
                            A=a, C=c, R=r,
                            expert_counts={str(int(e)): int(n) for e, n in zip(unique, counts)}))
    write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, rows)
    write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summarize(rows))
    write_csv(OUT / "COST_CURVE.csv",
              ("seed", "view", "lambda_cost", "A", "C", "R", "source_selected"), curve_rows)
    write_json(OUT / "OOD_BY_DATASET.json", dataset_rows)
    kappas = []
    for kappa in KAPPAS[1:]:
        group = [item for item in details if item["k"] == 2]
        fractions = [item["kappa_fractions"][f"{kappa:g}"] for item in group]
        same = [item["kappa_tables"][f"{kappa:g}"] == item["expert_table"] for item in group]
        kappas.append(dict(kappa=kappa, fraction_mean=float(np.mean(fractions)),
            fraction_sd=float(np.std(fractions, ddof=1)),
            same_experts_all_seeds=all(same), same_experts_by_seed=same))
    write_json(OUT / "KAPPA.json", kappas)
    log(stage="evaluation_complete", rows=len(rows), datasets=dataset_rows,
        kappas=kappas)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, fit, test, uids, views, bounds, feature_sha, provenance = previous.inputs()
    audit = make_audit("V25_SPHERICAL_COMPLETION",
        "python -B scripts/journal_20261024/v25_spherical_experiments.py",
        "11-source fit groups only; K-specific lambda from group-isolated source OOF "
        "using seed 0, then five final fits. K=2 reproduces the existing frozen "
        "five-seed source-selected spherical actions. No test outcome influences a map.",
        {view: len(uids_) for view, uids_ in views.items()})
    audit.update(status="running", c_min=bounds["cmin"], c_max=bounds["cmax"],
                 k_grid=list(KS), kappa_grid=list(KAPPAS), lambda_grid=list(LAMBDAS),
                 source_provenance=provenance)
    write_json(OUT / "AUDIT.json", audit)
    try:
        details = freeze_actions(data, fit, test, uids, feature_sha, bounds)
        evaluate(details, views, bounds)
        audit["status"] = "complete"
        audit["action_lock_sha256"] = previous.sha(OUT / "ACTION_LOCK.json")
    except Exception as exc:
        audit["status"] = "failed"
        audit["failures"].append(dict(error=str(exc)))
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
