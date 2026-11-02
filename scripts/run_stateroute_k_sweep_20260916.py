#!/usr/bin/env python3
"""Extend the audited StateRoute configurations to K=6/8 using frozen caches.

Run each phase in a separate process: train-text, train-mm, freeze, predict,
evaluate. No encoder training, expert generation, or remote calls are needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RUN_API_REGEN", "0")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from src.next_round_x.data import FeatureCache, load_source, manifests, role_uids
from src.next_round_x.models import cluster_quality, normalized
from src.next_round_x.runtime import canonical, digest, read_json, record, write_csv, write_json
from src.subject_routing.clustering.spherical_kmeans import (
    SphericalKMeans, SphericalKMeansArtifact, assign_frozen,
)
from src.subject_routing.data_contracts import TrainingLabels
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit
from src.subject_routing.policy_tables import build_strict_policy_tables, derive_training_cost_budgets
from src.subject_routing.profiling import CapabilityProfileArtifact, profile_subject_states
from routers.utils.rank_score import rank_score

OUT = ROOT / "outputs/stateroute_k_sweep_20260916"
REPORT = ROOT / "reports/stateroute_k_sweep_20260916"
LEGACY = ROOT / "artifacts/subject_routing/ood11_3/full_v1"
KS = (2, 4, 6, 8)
TEXT_SEEDS = (1013, 1021, 1042, 1087, 3026)
MM_SEEDS = (0, 1, 2, 3, 4)
TEXT_LAMBDAS = (0., .01, .03, .1, .3, 1., 3., 10., 30., 100.)
MM_LAMBDAS = (0., .001, .003, .01, .03, .1, .3)


def say(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify(records):
    for item in records:
        if not item["exists"] or digest(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Frozen artifact changed: {item['path']}")


def plan():
    p = OUT / "EXPERIMENT_PLAN.json"
    if p.exists():
        value = read_json(p)
        verify(value["implementation"])
        return value
    value = dict(
        protocol="heldout11; same fitted router for 11-ID and 3-OOD",
        k_values=list(KS), text_seeds=list(TEXT_SEEDS), mm_seeds=list(MM_SEEDS),
        text=dict(kappa=20, n_init=10, max_iter=100, tolerance=1e-6,
                  budget_quantile=.5, policy_lambdas=list(TEXT_LAMBDAS),
                  source_selection="original dev accuracy, subject to mean cost cap; deterministic tie order"),
        mm=dict(kappa=64, n_init=3, batch_size=1024, max_iter=100,
                lambda_grid=list(MM_LAMBDAS), selection_seed=0,
                source_selection="original three source fit group folds; maximum pooled OOF Rank; first grid point on ties"),
        status="FIXED_GRID_BEFORE_NEW_TEST_EVALUATION",
        historical_exposure="K2/K4 and historical test results already exposed; exploratory extension",
        test_selection=False, official_splits_unchanged=True,
        cost_scope="cached expert USD proxy; excludes encoder/router overhead",
        implementation=[record(p) for p in (
            Path(__file__), ROOT / "src/next_round_x/models.py",
            ROOT / "src/subject_routing/clustering/spherical_kmeans.py",
            ROOT / "src/subject_routing/profiling.py",
            ROOT / "src/subject_routing/policy_tables.py",
            ROOT / "routers/utils/rank_score.py")],
        environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    write_json(p, value)
    return value


def scores(y, c, actions):
    rows = np.arange(len(actions))
    quality = np.asarray(y[rows, actions], dtype=np.float64)
    costs = np.asarray(c[rows, actions], dtype=np.float64)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    return dict(n=len(actions), accuracy_percent=float(quality.mean() * 100),
                cost_usd_per_10k=float(costs.mean() * 10000),
                rank_score=100 * rank_score(float(quality.mean()), float(costs.mean()), bounds["cmin"], bounds["cmax"]))


def label_args():
    from scripts.subject_routing_v2.run_b00_query_teacher import _common_label_arguments
    split, pool = manifests()
    return _common_label_arguments(
        experts=pool["canonical_order"], model_pool=pool, split_manifest=split,
        matrix_audit=read_json(ROOT / "artifacts/subject_routing/audits/P02_matrix_semantics.json"))


def train_text():
    plan()
    data = load_source("heldout11")
    split, pool = manifests()
    experts = pool["canonical_order"]
    uids = tuple(data["uids"])
    dev_uids = role_uids("heldout11", "dev")
    assert list(uids) == role_uids("heldout11", "train")
    metadata = read_json(LEGACY / "features/metadata.json")
    features = {}
    for role in ("train_seen", "dev_seen"):
        item = metadata["roles"][role]
        assert digest(ROOT / item["path"]) == item["file_sha256"]
        assert item["ordered_uid_sha256"] == split["environments"]["ood11_3"]["roles"][role]["ordered_uid_sha256"]
        features[role] = np.load(ROOT / item["path"], allow_pickle=False)
    labels = TrainingLabels.create(uids=uids, expert_ids=experts, y=data["y"], c=data["c"],
                                   observed_mask=None, complete_matrix=True)
    feature_sha = metadata["feature_artifact_sha256"]
    label_sha = digest(ROOT / "outputs/next_round_x/source/heldout11/labels.npz")
    candidates, fitted, comparisons = [], [], []
    for k in KS:
        for seed in TEXT_SEEDS:
            started = time.perf_counter()
            dest = OUT / "models/T" / f"k{k}" / f"seed-{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            artifact = SphericalKMeans(k, seed=seed, n_init=10, max_iter=100, tolerance=1e-6).fit(
                features["train_seen"], uids, environment_id="ood11_3", fit_role="train_seen",
                feature_artifact_sha256=feature_sha, audit=UIDAccessAudit.create(Operation.FIT, uids))
            artifact.save(dest / "assigner.json")
            profile = profile_subject_states(
                method_id=f"StateRoute-T-K{k}-seed{seed}", environment_id="ood11_3",
                subject_artifact_sha256=artifact.sha256, assigner_artifact_sha256=artifact.sha256,
                label_artifact_sha256=label_sha, profile_assignments=dict(zip(uids, artifact.assignments)),
                state_ids=(*artifact.cluster_ids, "__global__"),
                parent_by_state={**{s: "__global__" for s in artifact.cluster_ids}, "__global__": None},
                global_state_id="__global__", labels=labels, canonical_expert_order=experts,
                kappa=20., audit=UIDAccessAudit.create(Operation.TRAIN, uids))
            profile.save(dest / "profile.json")
            budget = derive_training_cost_budgets(profile, (.5,))
            policy = build_strict_policy_tables(profile, budget, lambdas=TEXT_LAMBDAS)
            policy.save(dest / "policy.json")
            dev_states = assign_frozen(artifact, features["dev_seen"], feature_artifact_sha256=feature_sha).states
            if k in (2, 4):
                old_name = f"matched_k2_p{seed}" if k == 2 else f"b02_k4_p{seed}"
                old_method = f"n1_matched_fixed_k2__p{seed}" if k == 2 else f"b02_kmeans_k4__p{seed}"
                old = SphericalKMeansArtifact.load(LEGACY / "assigners" / f"{old_name}.json")
                old_profile = CapabilityProfileArtifact.load(LEGACY / "profiles" / f"{old_method}.json")
                same = artifact.assignments == old.assignments
                center_diff = float(np.max(np.abs(artifact.centers - old.centers)))
                qdiff = float(np.max(np.abs(profile.shrunk_quality_mean - old_profile.shrunk_quality_mean)))
                cdiff = float(np.max(np.abs(profile.shrunk_cost_mean - old_profile.shrunk_cost_mean)))
                comparisons.append(dict(k=k, seed=seed, identical_train_assignments=same,
                                        max_center_difference=center_diff, max_quality_difference=qdiff,
                                        max_cost_difference=cdiff))
                assert same and center_diff < 1e-10 and qdiff < 1e-12 and cdiff < 1e-14
            for order, strategy in enumerate(policy.strategy_lambdas):
                table = [experts.index(policy.decisions[strategy]["q050"][s].expert_id) for s in artifact.cluster_ids]
                actions = np.array([experts.index(e) for e in policy.route(dev_states, budget_id="q050", strategy_id=strategy)])
                pred_path = dest / f"dev-{order}.npz"
                np.savez_compressed(pred_path, uids=np.array(dev_uids), actions=actions)
                candidates.append(dict(k=k, seed=seed, strategy=strategy, strategy_order=order,
                                       lambda_cost=policy.strategy_lambdas[strategy], table=table,
                                       cap=budget[0].cap, predictions=record(pred_path)))
            fitted.append(dict(k=k, seed=seed, path=str(dest.relative_to(ROOT)), support=list(artifact.support),
                               wall_seconds=time.perf_counter()-started))
            say(stage="fit-text", k=k, seed=seed, seconds=round(time.perf_counter()-started, 2))
    candidate_path = OUT / "T_DEV_CANDIDATES.json"
    write_json(candidate_path, dict(candidates=candidates, models=fitted, train_n=len(uids), dev_n=len(dev_uids),
                                    existing_K2_K4_reproduction=comparisons))
    receipt_path = OUT / "T_DEV_RECEIPT.json"
    write_json(receipt_path, dict(predictions_frozen=True, candidates=record(candidate_path),
                                 prediction_files=[c["predictions"] for c in candidates]))
    verify([c["predictions"] for c in candidates])
    from src.subject_routing.dev_evaluation import materialize_scoped_dev_labels
    dev_labels, provenance = materialize_scoped_dev_labels(
        dev_uids, allowed_dev_uids=dev_uids, **label_args(),
        verified_prediction_file_sha256=digest(candidate_path),
        verified_prediction_receipt_sha256=digest(receipt_path),
        audit=UIDAccessAudit.create(Operation.DEV_EVALUATE, dev_uids))
    metric_rows, selections = [], []
    for candidate in candidates:
        prediction = np.load(ROOT / candidate["predictions"]["path"], allow_pickle=False)
        assert prediction["uids"].tolist() == dev_uids
        metric = scores(dev_labels.matrix.y, dev_labels.matrix.c, prediction["actions"])
        metric_rows.append({**candidate, **metric,
                            "feasible": metric["cost_usd_per_10k"] / 10000 <= candidate["cap"] + 1e-15})
    for model in fitted:
        feasible = [r for r in metric_rows if r["k"] == model["k"] and r["seed"] == model["seed"] and r["feasible"]]
        if not feasible:
            raise RuntimeError(f"No feasible source-dev policy: {model}")
        best = min(feasible, key=lambda r: (-r["accuracy_percent"], r["cost_usd_per_10k"], r["strategy_order"]))
        dest = ROOT / model["path"]
        np.savez_compressed(dest / "route.npz", table=np.array(best["table"]))
        selected = {**model, **{key: best[key] for key in ("strategy", "lambda_cost", "table", "cap", "accuracy_percent", "cost_usd_per_10k", "rank_score")}}
        write_json(dest / "selection.json", selected)
        selections.append(selected)
        say(stage="select-text", k=model["k"], seed=model["seed"], strategy=best["strategy"])
    write_json(OUT / "T_SOURCE_SELECTION.json", dict(selections=selections, dev_provenance=provenance.to_dict(),
                                                    original_K2_K4_reproduction=comparisons))
    write_csv(OUT / "T_SOURCE_DEV_RESULTS.csv", [{k:v for k,v in r.items() if k not in ("predictions", "table")} for r in metric_rows])


def train_mm():
    plan()
    data = load_source("heldout11")
    fit = np.flatnonzero(data["role"] == "fit")
    available = fit[data["available"][fit]]
    assert len(fit) == 16269 and len(available) == 16269
    position = {int(idx): row for row, idx in enumerate(fit)}
    mean_c = data["c"][fit].mean(0)
    scale = float(np.median(data["c"][fit][data["c"][fit] > 0]))
    stats = dict(cost_mean=mean_c.tolist(), cost_scale=scale, fit_uids=data["uids"][fit].tolist(),
                 fallback=int(data["y"][fit].mean(0).argmax()), qcal_cert_excluded=True)
    write_json(OUT / "models/MM/stats.json", stats)
    selections, source_results, comparisons = [], [], []
    for k in KS:
        oof = np.zeros((len(fit), 17))
        cost = np.zeros_like(oof)
        fallbacks = np.zeros(len(fit), dtype=np.int64)
        for fold in range(3):
            train = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
            valid = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
            assert not set(data["groups"][train]) & set(data["groups"][valid])
            target = np.array([position[int(i)] for i in valid])
            usable = train[data["available"][train]]
            oof[target], _, _ = cluster_quality(data["x"][usable], data["y"][usable], data["x"][valid], k, 0)
            cost[target] = data["c"][train].mean(0) / float(np.median(data["c"][train][data["c"][train] > 0]))
            fallbacks[target] = int(data["y"][train].mean(0).argmax())
        rows = []
        for lam in MM_LAMBDAS:
            actions = (oof - lam * cost).argmax(1)
            missing = ~data["available"][fit]
            actions[missing] = fallbacks[missing]
            # Original source selection used float32 means for binary quality.
            metric = scores(data["y"][fit], data["c"][fit], actions)
            source_a = float(data["y"][fit][np.arange(len(fit)), actions].mean())
            bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
            selection_rank = rank_score(source_a, metric["cost_usd_per_10k"] / 10000, bounds["cmin"], bounds["cmax"]) * 100
            rows.append(dict(k=k, lambda_cost=lam, selection_rank=selection_rank, **metric))
        selected = max(rows, key=lambda r: r["selection_rank"])
        source_results.extend(rows)
        oof_path = OUT / "oof" / f"MM-k{k}.npz"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof_path, uids=data["uids"][fit], quality=oof, scaled_cost=cost)
        if k in (2, 4):
            old_lock = read_json(ROOT / "outputs/next_round_x/SELECTION_LOCK.json")
            assert selected["lambda_cost"] == old_lock["protocols"]["heldout11"]["methods"][f"k{k}"]["lambda_cost"]
            old_oof = np.load(ROOT / "outputs/next_round_x/baselines/OOF_TRACES/heldout11.npz", allow_pickle=False)
            assert np.array_equal(old_oof["uids"], data["uids"][fit])
            assert np.allclose(oof, old_oof[f"p_k{k}"], atol=1e-7, rtol=0)
        selections.append(selected)
        say(stage="select-mm", k=k, lambda_cost=selected["lambda_cost"], source_rank=selected["rank_score"])
        for seed in MM_SEEDS:
            started = time.perf_counter()
            _, km, rates = cluster_quality(data["x"][available], data["y"][available], data["x"][available[:1]], k, seed)
            table = (rates - selected["lambda_cost"] * mean_c / scale).argmax(1)
            support = np.bincount(km.predict(normalized(data["x"][available])), minlength=k)
            dest = OUT / "models/MM" / f"k{k}" / f"seed-{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / "model.npz", centers=km.cluster_centers_, rates=rates, table=table, support=support)
            write_json(dest / "selection.json", dict(k=k, seed=seed, lambda_cost=selected["lambda_cost"],
                       support=support.tolist(), table=table.tolist(), wall_seconds=time.perf_counter()-started))
            if k in (2, 4):
                old = np.load(ROOT / "outputs/next_round_x/final/heldout11" / f"seed-{seed}" / f"k{k}.npz", allow_pickle=False)
                center_diff = float(np.max(np.abs(km.cluster_centers_ - old["centers"])))
                rate_diff = float(np.max(np.abs(rates - old["rates"])))
                comparisons.append(dict(k=k, seed=seed, max_center_difference=center_diff, max_quality_difference=rate_diff))
                assert center_diff < 1e-6 and rate_diff < 1e-7
            say(stage="fit-mm", k=k, seed=seed, seconds=round(time.perf_counter()-started, 2))
    write_json(OUT / "MM_SOURCE_SELECTION.json", dict(selections=selections, original_K2_K4_reproduction=comparisons))
    write_csv(OUT / "MM_SOURCE_OOF_RESULTS.csv", source_results)


def freeze():
    value = plan()
    assert len(read_json(OUT / "T_SOURCE_SELECTION.json")["selections"]) == 20
    assert len(read_json(OUT / "MM_SOURCE_SELECTION.json")["selections"]) == 4
    files = [record(p) for p in sorted((OUT / "models").rglob("*")) if p.is_file()]
    files += [record(OUT / name) for name in ("EXPERIMENT_PLAN.json", "T_SOURCE_SELECTION.json", "MM_SOURCE_SELECTION.json")]
    files += [record(ROOT / p) for p in (
        "artifacts/subject_routing/manifests/splits.json", "artifacts/subject_routing/manifests/model_pool.json",
        "outputs/next_round_x/source/heldout11/MANIFEST.json", "data/matrices/cost_bounds.json")]
    write_json(OUT / "MODEL_LOCK.json", dict(files=files, implementation=value["implementation"],
               all_40_models_selected_before_test=True, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    say(stage="freeze", models=40, files=len(files))


def predict():
    from src.next_round_x.read_fence import install
    denied = install()
    lock = read_json(OUT / "MODEL_LOCK.json")
    verify(lock["files"] + lock["implementation"])
    split, pool = manifests()
    uids = role_uids("heldout11", "id") + role_uids("heldout11", "ood")
    assert len(uids) == 6108 and len(set(uids)) == 6108
    old_meta = read_json(ROOT / "artifacts/subject_routing/final/predictions/prediction_metadata.json")
    text_file = "artifacts/subject_routing/final/predictions/test_minilm.npy"
    assert digest(ROOT / text_file) == old_meta["outputs"][text_file]["sha256"]
    official = split["official_router_splits"]["test"]["ordered_sample_ids"]
    assert canonical(official) == old_meta["ordered_test_uid_sha256"]
    positions = {u:i for i,u in enumerate(official)}
    text_x = np.load(ROOT / text_file, allow_pickle=False)[[positions[u] for u in uids]]
    mm_x, available = FeatureCache().rows(uids)
    nx = normalized(mm_x)
    stats = read_json(OUT / "models/MM/stats.json")
    actions, state_rows, methods, seeds = [], [], [], []
    for family, seed_values in (("T", TEXT_SEEDS), ("MM", MM_SEEDS)):
        for k in KS:
            for seed in seed_values:
                dest = OUT / "models" / family / f"k{k}" / f"seed-{seed}"
                if family == "T":
                    model = SphericalKMeansArtifact.load(dest / "assigner.json")
                    states = assign_frozen(model, text_x, feature_artifact_sha256=model.feature_artifact_sha256).states
                    assigned = np.array([model.cluster_ids.index(s) for s in states])
                    table = np.load(dest / "route.npz", allow_pickle=False)["table"]
                    a = table[assigned]
                else:
                    model = np.load(dest / "model.npz", allow_pickle=False)
                    centers = model["centers"]
                    distance = np.square(nx).sum(1)[:,None] + np.square(centers).sum(1)[None,:] - 2 * nx @ centers.T
                    assigned = distance.argmin(1)
                    a = model["table"][assigned].copy()
                    a[~available] = stats["fallback"]
                    assigned[~available] = -1
                assert a.shape == (6108,) and (a >= 0).all() and (a < 17).all()
                methods.append(f"StateRoute-{family}-K{k}")
                seeds.append(seed)
                actions.append(a)
                state_rows.append(assigned)
    path = OUT / "predictions.npz"
    np.savez_compressed(path, uids=np.array(uids), methods=np.array(methods), seeds=np.array(seeds),
                        actions=np.stack(actions), states=np.stack(state_rows), mm_available=available,
                        views=np.array(["11-ID"] * 5341 + ["3-OOD"] * 767))
    receipt = dict(kind="subject_routing_final_evaluation_unlock", target_split="test", predictions_frozen=True,
                   method_selection_complete=True, approved=True, approval_source="user requested K=6/8 experiments",
                   prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
                   methods=methods, seeds=seeds, test_outcomes_opened=0, labels_or_actual_cost_used=False,
                   read_fence_denied_attempts=denied, model_lock=record(OUT / "MODEL_LOCK.json"),
                   test_fit_performed=False, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_json(OUT / "PREDICTION_RECEIPT.json", receipt)
    say(stage="predict", routes=len(actions), requests=len(uids), denied_attempts=denied)


def evaluate():
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import evaluate_final_unlock
    lock = read_json(OUT / "MODEL_LOCK.json")
    verify(lock["files"] + lock["implementation"])
    pred_path, receipt_path = OUT / "predictions.npz", OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    assert digest(pred_path) == receipt["prediction_sha256"]
    prediction = np.load(pred_path, allow_pickle=False)
    uids = prediction["uids"].tolist()
    assert uids == role_uids("heldout11", "id") + role_uids("heldout11", "ood")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(pred_path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **label_args())
    write_json(OUT / "OUTCOME_PROVENANCE.json", dict(provenance=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    per_seed, per_dataset, selected_q, selected_c = [], [], [], []
    for method, seed, a in zip(prediction["methods"], prediction["seeds"], prediction["actions"]):
        family = method.split("-")[1]
        k = int(method.rsplit("K",1)[1])
        q = y[np.arange(len(uids)), a]
        costs = c[np.arange(len(uids)), a]
        selected_q.append(q)
        selected_c.append(costs)
        for view in ("11-ID", "3-OOD"):
            mask = prediction["views"] == view
            per_seed.append(dict(method=str(method), family=family, k=k, seed=int(seed), view=view,
                **scores(y[mask], c[mask], a[mask]), distinct_experts=int(len(set(a[mask]))),
                all_wrong_rows_retained=int((y[mask].sum(1) == 0).sum())))
        datasets = np.array([u.split("/",1)[0] for u in uids])
        for dataset in sorted(set(datasets)):
            mask = datasets == dataset
            per_dataset.append(dict(method=str(method), family=family, k=k, seed=int(seed), dataset=str(dataset),
                                    **scores(y[mask], c[mask], a[mask])))
    summary = []
    for family in ("T", "MM"):
        for k in KS:
            for view in ("11-ID", "3-OOD"):
                rows = [r for r in per_seed if r["family"] == family and r["k"] == k and r["view"] == view]
                assert len(rows) == 5 and len({r["seed"] for r in rows}) == 5
                result = dict(method=rows[0]["method"], family=family, k=k, view=view, n=rows[0]["n"], seeds=5)
                for metric in ("accuracy_percent", "cost_usd_per_10k", "rank_score"):
                    values = np.array([r[metric] for r in rows])
                    result[f"{metric}_mean"] = float(values.mean())
                    result[f"{metric}_sd"] = float(values.std(ddof=1))
                summary.append(result)
    deltas = []
    for row in summary:
        if row["k"] == 2:
            continue
        base = next(r for r in summary if r["family"] == row["family"] and r["k"] == 2 and r["view"] == row["view"])
        deltas.append(dict(method=row["method"], view=row["view"], comparison="K minus K2",
            **{f"delta_{m}":row[f"{m}_mean"]-base[f"{m}_mean"] for m in ("accuracy_percent", "cost_usd_per_10k", "rank_score")}))
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "per_seed.csv", per_seed)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    write_csv(REPORT / "summary.csv", summary)
    write_csv(REPORT / "delta_vs_k2.csv", deltas)
    np.savez_compressed(OUT / "evaluation_traces.npz", uids=prediction["uids"], methods=prediction["methods"],
                        seeds=prediction["seeds"], views=prediction["views"], actions=prediction["actions"],
                        correct=np.stack(selected_q), cost=np.stack(selected_c))
    write_json(REPORT / "RESULTS.json", dict(summary=summary, deltas=deltas,
               prediction=record(pred_path), plan=record(OUT / "EXPERIMENT_PLAN.json"),
               all_wrong_retained=True, seed_sd_ddof=1, interpretation="exposed-test exploratory extension"))
    for row in summary:
        say(**row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train-text", "train-mm", "freeze", "predict", "evaluate"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"train-text": train_text, "train-mm": train_mm, "freeze": freeze, "predict": predict, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
