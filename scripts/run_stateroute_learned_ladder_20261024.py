#!/usr/bin/env python3
"""Five-seed learned-state refinement ladder on frozen CPU feature caches.

Run --stage source, predict, evaluate for each protocol, then --stage report.
Source selection uses only grouped source-fit out-of-fold outcomes. Test labels
are materialized only after all actions have been frozen and hash-bound.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from scripts import run_stateroute_tiny_knn_20260928 as tiny
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from scripts import run_stateroute_mmmu_subject_four_rules_20260928 as subject_rules
from src.next_round_x.data import manifests, role_uids
from src.next_round_x.models import normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json
from src.subject_routing.clustering.spherical_kmeans import (
    SphericalKMeans, SphericalKMeansArtifact, assign_frozen)
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit

OUT = ROOT / "outputs/journal_ladder_20261024"
REPORT = ROOT / "reports/journal_supplement_20261024/ladder"
ARCHIVE = Path("/home/sm5/ys/ICASSP/archive/non_paper_research_20261011")
T_SEEDS = (1013, 1021, 1042, 1087, 3026)
MM_SEEDS = (0, 1, 2, 3, 4)
ROWS = ("K1", "K2", "LocalAggressive", "SwitchCostCap", "Guard10", "Guard25")
CONFIDENCE_QUANTILES = (.2, .35, .5, .65, .8)
GAIN_MARGINS = (0., .01, .02, .03, .04, .05, .06, .08, .1, .12)
PRICE_WEIGHTS = (0., .01, .02, .04, .06)
CAP_QUANTILES = (.95, .975, .99, .995, 1.)
METRICS = ("accuracy_percent", "cost_usd_per_10k", "rank_score")
TEXT_METADATA = ROOT / "artifacts/subject_routing/ood11_3/full_v1/features/metadata.json"
TEXT_TRAIN = ROOT / "artifacts/subject_routing/ood11_3/full_v1/features/minilm_train_seen.npy"
TEXT_TEST = ROOT / "artifacts/subject_routing/final/predictions/test_minilm.npy"


def say(**kwargs):
    print(json.dumps(kwargs, sort_keys=True, ensure_ascii=False), flush=True)


def where(protocol):
    if protocol not in ("heldout11", "full14"):
        raise ValueError(protocol)
    return OUT / protocol


def families(protocol):
    return (("T", T_SEEDS), ("MM", MM_SEEDS)) if protocol == "heldout11" else (("MM", MM_SEEDS),)


def text_train(data):
    meta = read_json(TEXT_METADATA)
    path_record = meta["roles"]["train_seen"]
    if digest(TEXT_TRAIN) != path_record["file_sha256"] or data["uids"].tolist() != role_uids("heldout11", "train"):
        raise RuntimeError("MiniLM source alignment or hash changed")
    values = np.load(TEXT_TRAIN, mmap_mode="r", allow_pickle=False)
    if len(values) != len(data["uids"]):
        raise RuntimeError("MiniLM train row count changed")
    return values


def text_test(uids):
    meta = read_json(ROOT / "artifacts/subject_routing/final/predictions/prediction_metadata.json")
    rel = str(TEXT_TEST.relative_to(ROOT))
    if digest(TEXT_TEST) != meta["outputs"][rel]["sha256"]:
        raise RuntimeError("MiniLM test hash changed")
    official = manifests()[0]["official_router_splits"]["test"]["ordered_sample_ids"]
    positions = {uid: i for i, uid in enumerate(official)}
    values = np.load(TEXT_TEST, mmap_mode="r", allow_pickle=False)
    return np.asarray(values[[positions[uid] for uid in uids]], dtype=np.float32)


def text_model(k, seed):
    folder = "stateroute_k1_20260918" if k == 1 else "stateroute_k_sweep_20260916"
    return ROOT / "outputs" / folder / f"models/T/k{k}/seed-{seed}"


def mm_model(k, seed, protocol):
    if protocol == "full14":
        if k != 2:
            raise ValueError("full14 MM K1 has a source-selected singleton table")
        return ARCHIVE / f"outputs/stateroute_mmmu_full14_20260928/models/StateRoute-MM-K2/seed-{seed}.npz"
    folder = "stateroute_k1_20260918" if k == 1 else "stateroute_k_sweep_20260916"
    return ROOT / "outputs" / folder / f"models/MM/k{k}/seed-{seed}/model.npz"


def text_states(model, x):
    assigned = assign_frozen(model, x, feature_artifact_sha256=model.feature_artifact_sha256).states
    return np.array([model.cluster_ids.index(value) for value in assigned], dtype=np.int8)


def mm_states(centers, x):
    nx = normalized(x)
    d = (np.square(nx).sum(axis=1)[:, None] + np.square(centers).sum(axis=1)[None, :]
         - 2 * nx @ centers.T)
    return d.argmin(axis=1).astype(np.int8)


def fit_fold_states(family, seed, tr, va, data, xq, tx):
    if family == "T":
        metadata = read_json(TEXT_METADATA)
        model = SphericalKMeans(2, seed=seed, n_init=10, max_iter=100, tolerance=1e-6).fit(
            tx[tr], data["uids"][tr].tolist(), environment_id="ood11_3", fit_role="train_seen",
            feature_artifact_sha256=metadata["feature_artifact_sha256"],
            audit=UIDAccessAudit.create(Operation.FIT, data["uids"][tr].tolist()))
        source = np.array([model.cluster_ids.index(v) for v in model.assignments], dtype=np.int8)
        return source, text_states(model, tx[va])
    km = MiniBatchKMeans(n_clusters=2, random_state=seed, n_init=3, batch_size=1024, max_iter=100)
    source = km.fit_predict(normalized(xq[tr])).astype(np.int8)
    target = km.predict(normalized(xq[va])).astype(np.int8)
    return source, target


def profile(y, c, states):
    """Same kappa=64/128 profile and median-positive-cost scale as subject rules."""
    count = np.bincount(states, minlength=2)
    global_y = y.mean(axis=0, dtype=np.float64)
    global_c = c.mean(axis=0, dtype=np.float64)
    median = float(np.median(c[c > 0]))
    sum_y = np.zeros((2, 17), dtype=np.float64)
    sum_c = np.zeros((2, 17), dtype=np.float64)
    np.add.at(sum_y, states, y)
    np.add.at(sum_c, states, c)
    quality = ((sum_y + 64. * global_y) / (count[:, None] + 64.)).astype(np.float32)
    state_cost = ((sum_c + 128. * global_c) / (count[:, None] + 128.) / median).astype(np.float32)
    return quality, state_cost, (global_c / median).astype(np.float32), count


def candidates(base, all64, same32, state_cost, global_cost, states, available, full14):
    qr = .8 * base + .2 * all64
    if full14:
        qa = .6 * base + .4 * same32
        cost_a = state_cost[states]
    else:
        qa = .8 * base + .2 * same32
        cost_a = np.broadcast_to(global_cost, qa.shape)
    missing = ~available
    qr[missing] = base[missing]
    qa[missing] = base[missing]
    rows = np.arange(len(states))
    rank = (qr - .06 * state_cost[states]).argmax(axis=1).astype(np.int8)
    aggressive = (qa - .025 * cost_a).argmax(axis=1).astype(np.int8)
    gain = qa[rows, aggressive] - qa[rows, rank]
    delta = cost_a[rows, aggressive] - cost_a[rows, rank]
    return rank, aggressive, gain.astype(np.float32), delta.astype(np.float32)


def neighborhood_pass(query, bank, bank_y, bank_states, query_states, profiles, available, full14, sink):
    """One cosine matrix per batch is shared by all family/seed state maps."""
    by_state = {key: [np.flatnonzero(states == state) for state in (0, 1)]
                for key, states in bank_states.items()}
    for start in range(0, len(query), 128):
        stop = min(start + 128, len(query))
        sim = np.asarray(query[start:stop] @ bank.T, dtype=np.float32)
        all64 = subject_rules.weighted_neighbors(sim, bank_y, 64)
        for key, states_all in query_states.items():
            states = states_all[start:stop]
            same32 = np.empty_like(all64)
            for state in (0, 1):
                rows = np.flatnonzero(states == state)
                if not len(rows):
                    continue
                columns = by_state[key][state]
                if len(columns):
                    same32[rows] = subject_rules.weighted_neighbors(sim[rows][:, columns], bank_y[columns], 32)
                else:
                    same32[rows] = subject_rules.weighted_neighbors(sim[rows], bank_y, 32)
            quality, state_cost, global_cost, _ = profiles[key]
            base = quality[states]
            r, a, g, d = candidates(base, all64, same32, state_cost, global_cost,
                                     states, available[start:stop], full14)
            for name, values in (("rank", r), ("aggressive", a), ("gain", g), ("delta", d)):
                sink[key][name][start:stop] = values
        if stop == len(query) or stop % 2048 < 128:
            say(stage="neighbors", rows=stop, total=len(query), maps=len(bank_states))


def choose_rule(y, c, oof, maxscore, available, bounds):
    rank, aggressive = oof["rank"], oof["aggressive"]
    gain, delta = oof["gain"], oof["delta"]
    baseline = tiny.metrics(y, c, rank, bounds)
    grid = []
    for quantile in CONFIDENCE_QUANTILES:
        threshold = float(np.quantile(maxscore[available], quantile))
        for margin in GAIN_MARGINS:
            for weight in PRICE_WEIGHTS:
                switch = ((maxscore >= threshold) & (gain - weight * delta > margin)
                          & (aggressive != rank) & available)
                action = np.where(switch, aggressive, rank)
                grid.append(dict(confidence_quantile=quantile, confidence_threshold=threshold,
                                 gain_margin=margin, price_weight=weight,
                                 source_switch_n=int(switch.sum()),
                                 **tiny.metrics(y, c, action, bounds)))
    assert len(grid) == 250
    rank_safe = [r for r in grid if r["rank_score"] >= baseline["rank_score"] - .02
                 and r["cost_usd_per_10k"] <= .75]
    accuracy_safe = [r for r in grid if r["rank_score"] >= baseline["rank_score"] - .3
                     and r["cost_usd_per_10k"] <= .95]
    if not rank_safe or not accuracy_safe:
        raise RuntimeError(f"frozen 250-candidate feasibility failed: rank={len(rank_safe)}, accuracy={len(accuracy_safe)}")
    order = lambda r: (r["accuracy_percent"], r["rank_score"], -r["cost_usd_per_10k"])
    rank_rule = max(rank_safe, key=order)
    selected = max(accuracy_safe, key=order)
    switched = ((maxscore >= selected["confidence_threshold"])
                & (gain - selected["price_weight"] * delta > selected["gain_margin"])
                & (aggressive != rank) & available)
    uncapped = tiny.metrics(y, c, np.where(switched, aggressive, rank), bounds)
    cap_rows = []
    for quantile in CAP_QUANTILES:
        cap = float(np.quantile(delta[switched], quantile)) if switched.any() else float("inf")
        capped = switched & (delta <= cap)
        cap_rows.append(dict(cap_quantile=quantile, cost_delta_cap=cap,
                             source_switch_n=int(capped.sum()),
                             **tiny.metrics(y, c, np.where(capped, aggressive, rank), bounds)))
    eligible = [r for r in cap_rows if r["accuracy_percent"] >= uncapped["accuracy_percent"]
                and r["cost_usd_per_10k"] <= uncapped["cost_usd_per_10k"]]
    if not eligible:
        raise RuntimeError("frozen source-only cap feasibility failed")
    cap = max(eligible, key=lambda r: (r["rank_score"], r["accuracy_percent"], -r["cost_usd_per_10k"]))
    return dict(source_rank_baseline=baseline, rank_safe=rank_rule, accuracy=selected,
                cost_cap=cap, guard10=float(np.quantile(maxscore[available], .1)),
                guard25=float(np.quantile(maxscore[available], .25))), grid, cap_rows


def singleton_full14(data, fit, bounds):
    """Original full14 MM source OOF lambda grid for the K=1 expert map."""
    lambdas = (0., .001, .003, .01, .03, .1, .3)
    y, c = data["y"], data["c"]
    position = {int(row): i for i, row in enumerate(fit)}
    quality = np.empty((len(fit), 17), dtype=np.float32)
    cost = np.empty_like(quality)
    for fold in range(3):
        tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        at = np.array([position[int(i)] for i in va])
        quality[at] = y[tr].mean(0)
        cost[at] = c[tr].mean(0) / np.median(c[tr][c[tr] > 0])
    rows = []
    for lam in lambdas:
        action = (quality - lam * cost).argmax(axis=1)
        rows.append(dict(lambda_cost=lam, **tiny.metrics(y[fit], c[fit], action, bounds)))
    chosen = max(rows, key=lambda row: row["rank_score"])
    final_q = y[fit].mean(0)
    final_c = c[fit].mean(0) / np.median(c[fit][c[fit] > 0])
    return dict(lambda_cost=chosen["lambda_cost"], expert=int((final_q - chosen["lambda_cost"] * final_c).argmax()),
                source_oof=chosen, grid=rows)


def source(protocol):
    dest = where(protocol)
    if (dest / "SOURCE_SELECTION.json").exists():
        raise FileExistsError("source selection already exists")
    data, xq, xt, available, fit, names, domains, _ = tiny.source_inputs(protocol)
    tx = text_train(data) if protocol == "heldout11" else None
    keys = [(family, seed) for family, seeds in families(protocol) for seed in seeds]
    n = len(fit)
    position = {int(row): i for i, row in enumerate(fit)}
    oof = {key: {name: np.empty(n, dtype=np.float32 if name in ("gain", "delta") else np.int8)
                 for name in ("rank", "aggressive", "gain", "delta")} for key in keys}
    maxscore = np.empty(n, dtype=np.float32)
    coverage = np.zeros(n, dtype=np.int8)
    for fold in range(3):
        tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][tr]) & set(data["groups"][va]):
            raise RuntimeError("source fold dependency groups overlap")
        target = np.array([position[int(i)] for i in va])
        head = tiny.fit_domain(xt[tr], domains[tr], len(names))
        maxscore[target] = head.decision_function(xt[va]).max(axis=1)
        bank_states, query_states, profiles = {}, {}, {}
        for family, seed in keys:
            source_states, valid_states = fit_fold_states(family, seed, tr, va, data, xq, tx)
            key = (family, seed)
            bank_states[key], query_states[key] = source_states, valid_states
            profiles[key] = profile(data["y"][tr], data["c"][tr], source_states)
            say(stage="fold_states", protocol=protocol, fold=fold, family=family, seed=seed,
                support=profiles[key][3].tolist())
        local = {key: {name: np.empty(len(va), dtype=oof[key][name].dtype) for name in oof[key]} for key in keys}
        neighborhood_pass(xt[va], xt[tr], data["y"][tr], bank_states, query_states,
                          profiles, available[va], protocol == "full14", local)
        for key in keys:
            for name in oof[key]:
                oof[key][name][target] = local[key][name]
        coverage[target] += 1
        say(stage="source_fold_complete", protocol=protocol, fold=fold, train_n=len(tr), valid_n=len(va))
    if not (coverage == 1).all():
        raise RuntimeError("source OOF coverage incomplete")
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    selections, candidates, caps = {}, [], []
    for key in keys:
        family, seed = key
        rule, grid, cap_rows = choose_rule(data["y"][fit], data["c"][fit], oof[key],
                                           maxscore, available[fit], bounds)
        selections[f"{family}-{seed}"] = rule
        candidates += [dict(family=family, seed=seed, **r) for r in grid]
        caps += [dict(family=family, seed=seed, **r) for r in cap_rows]
        say(stage="source_selected", protocol=protocol, family=family, seed=seed,
            threshold=rule["accuracy"]["confidence_threshold"], cap=rule["cost_cap"]["cost_delta_cap"])
    singleton = singleton_full14(data, fit, bounds) if protocol == "full14" else None
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "SOURCE_CANDIDATES.csv", candidates)
    write_csv(dest / "SOURCE_CAP_CANDIDATES.csv", caps)
    write_json(dest / "SOURCE_SELECTION.json", dict(
        protocol=protocol, families={family: list(seeds) for family, seeds in families(protocol)},
        source_fit_n=len(fit), source_folds=3, candidate_count_per_seed=250,
        state_fit="same K=2 spherical MiniLM or MiniBatch Qwen K-means, refitted inside each grouped OOF fold",
        profile=dict(quality_kappa=64, cost_kappa=128, neighbor_temperature=.08,
                     rank="0.8 state + 0.2 global top64, state cost 0.06",
                     aggressive=("0.6 state + 0.4 same-state top32, state cost 0.025" if protocol == "full14"
                                 else "0.8 state + 0.2 same-state top32, global cost 0.025")),
        selection_rule="250 grouped source-fit OOF candidates; rank-safe: max accuracy with Rank >= rank baseline -0.02 and cost <=0.75; accuracy operating point: max accuracy with Rank >= baseline -0.3 and cost <=0.95; ties by Rank then lower cost; cost cap from switched delta quantiles .95/.975/.99/.995/1 maximizing Rank subject to accuracy >= uncapped and cost <= uncapped; guards from source OOF Ridge maxscore quantiles .1/.25",
        selected=selections, full14_k1=singleton,
        source_dev_test_outcomes_used=False, test_metric_selection=False,
        missing_tiny_policy="state profile only; switching and fallback disabled",
        candidates=record(dest / "SOURCE_CANDIDATES.csv"), caps=record(dest / "SOURCE_CAP_CANDIDATES.csv")))
    say(stage="source_complete", protocol=protocol, maps=len(keys))


def frozen_base(family, k, seed, protocol, xq_test, tx_test):
    if family == "T":
        path = text_model(k, seed)
        model = SphericalKMeansArtifact.load(path / "assigner.json")
        state = text_states(model, tx_test)
        table = np.load(path / "route.npz", allow_pickle=False)["table"]
        return state, table[state].astype(np.int8)
    if protocol == "full14" and k == 1:
        expert = read_json(where(protocol) / "SOURCE_SELECTION.json")["full14_k1"]["expert"]
        return np.zeros(len(xq_test), dtype=np.int8), np.full(len(xq_test), expert, dtype=np.int8)
    path = mm_model(k, seed, protocol)
    with np.load(path, allow_pickle=False) as model:
        state = mm_states(model["centers"], xq_test)
        table = model["selected_table"] if protocol == "full14" else model["table"]
        return state, table[state].astype(np.int8)


def compare_historical_base(family, k, seed, protocol, uids, actions):
    if protocol != "heldout11":
        return
    if family == "MM":
        path = ROOT / ("outputs/stateroute_multi_image_5seed_20260918/predictions.npz")
        name = f"StateRoute-MM-MultiImage-K{k}"
    else:
        path = ROOT / ("outputs/stateroute_k1_20260918/predictions.npz" if k == 1
                       else "outputs/stateroute_k_sweep_20260916/predictions.npz")
        name = f"StateRoute-T-K{k}"
    with np.load(path, allow_pickle=False) as old:
        match = np.flatnonzero((old["methods"] == name) & (old["seeds"] == seed))
        if len(match) != 1 or not np.array_equal(old["uids"], np.asarray(uids)):
            raise RuntimeError(f"historical base alignment failed: {family}, {k}, {seed}")
        if not np.array_equal(actions, old["actions"][int(match[0])]):
            raise RuntimeError(f"historical base actions mismatch: {family}, {k}, {seed}")


def predict(protocol):
    dest = where(protocol)
    if (dest / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("predictions already frozen")
    source_path = dest / "SOURCE_SELECTION.json"
    source_selected = read_json(source_path)
    if source_selected["protocol"] != protocol or source_selected["candidate_count_per_seed"] != 250:
        raise RuntimeError("source selection contract changed")
    from src.next_round_x.read_fence import install
    denied = install()
    data, xq_train, xt_train, _, fit, names, domains, _ = tiny.source_inputs(protocol)
    uids, views = tiny.test_uids(protocol)
    xq_test, overlay, _ = subject.qwen_features(uids, allow_multi=True)
    xt_test, available, _ = tiny.tiny_features(uids)
    if len(overlay) != 10 or int((~available).sum()) != 10:
        raise RuntimeError("cached test feature coverage changed")
    tx_test = text_test(uids) if protocol == "heldout11" else None
    tx_train = text_train(data) if protocol == "heldout11" else None
    head = tiny.fit_domain(xt_train[fit], domains[fit], len(names))
    maxscore = head.decision_function(xt_test).max(axis=1).astype(np.float32)
    keys = [(family, seed) for family, seeds in families(protocol) for seed in seeds]
    bank_states, query_states, profiles = {}, {}, {}
    baseline = {}
    for family, seed in keys:
        for k in (1, 2):
            _, action = frozen_base(family, k, seed, protocol, xq_test, tx_test)
            compare_historical_base(family, k, seed, protocol, uids, action)
            baseline[(family, seed, k)] = action
        if family == "T":
            model = SphericalKMeansArtifact.load(text_model(2, seed) / "assigner.json")
            train_state = np.array([model.cluster_ids.index(v) for v in model.assignments], dtype=np.int8)[fit]
            test_state = text_states(model, tx_test)
        else:
            with np.load(mm_model(2, seed, protocol), allow_pickle=False) as model:
                train_state = mm_states(model["centers"], xq_train[fit])
                test_state = mm_states(model["centers"], xq_test)
        bank_states[(family, seed)] = train_state
        query_states[(family, seed)] = test_state
        profiles[(family, seed)] = profile(data["y"][fit], data["c"][fit], train_state)
    local = {key: {name: np.empty(len(uids), dtype=np.float32 if name in ("gain", "delta") else np.int8)
                   for name in ("rank", "aggressive", "gain", "delta")} for key in keys}
    neighborhood_pass(xt_test, xt_train[fit], data["y"][fit], bank_states, query_states,
                      profiles, available, protocol == "full14", local)
    action_rows, families_out, seeds_out, row_out, switch_counts, guard_counts = [], [], [], [], [], []
    for family, seed in keys:
        rule = source_selected["selected"][f"{family}-{seed}"]
        rank, aggressive = local[(family, seed)]["rank"], local[(family, seed)]["aggressive"]
        gain, delta = local[(family, seed)]["gain"], local[(family, seed)]["delta"]
        chosen = rule["accuracy"]
        switch = ((maxscore >= chosen["confidence_threshold"])
                  & (gain - chosen["price_weight"] * delta > chosen["gain_margin"])
                  & (delta <= rule["cost_cap"]["cost_delta_cap"])
                  & (aggressive != rank) & available)
        switched = np.where(switch, aggressive, rank)
        guard10 = (maxscore < rule["guard10"]) & available
        guard25 = (maxscore < rule["guard25"]) & available
        actions = (baseline[(family, seed, 1)], baseline[(family, seed, 2)], aggressive,
                   switched, np.where(guard10, baseline[(family, seed, 2)], switched),
                   np.where(guard25, baseline[(family, seed, 2)], switched))
        for name, action in zip(ROWS, actions):
            action = np.asarray(action, dtype=np.int8)
            if action.shape != (6108,) or (action < 0).any() or (action >= 17).any():
                raise RuntimeError("invalid frozen expert actions")
            action_rows.append(action)
            families_out.append(family)
            seeds_out.append(seed)
            row_out.append(name)
            switch_counts.append(int(switch.sum()) if name in ROWS[3:] else 0)
            guard_counts.append(int(guard10.sum()) if name == "Guard10" else
                                int(guard25.sum()) if name == "Guard25" else 0)
        say(stage="predict_map", protocol=protocol, family=family, seed=seed,
            switch_n=int(switch.sum()), guard10_n=int(guard10.sum()), guard25_n=int(guard25.sum()))
    path = dest / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.asarray(uids), views=views, families=np.asarray(families_out),
                        seeds=np.asarray(seeds_out), rows=np.asarray(row_out), actions=np.stack(action_rows),
                        switch_count=np.asarray(switch_counts), guard_count=np.asarray(guard_counts))
    lock_path = dest / "MODEL_LOCK.json"
    write_json(lock_path, dict(protocol=protocol, source_selection=record(source_path),
                               implementation=record(Path(__file__)), source_only_selection=True,
                               test_outcomes_opened=0, families=source_selected["families"]))
    write_json(dest / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested learned-state five-seed refinement ladder",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(lock_path), labels_or_actual_cost_used=False,
        test_outcomes_opened=0, test_fit_performed=False, denied_attempts=denied))
    say(stage="predict_complete", protocol=protocol, routes=len(action_rows), n=len(uids))


def historical_targets():
    return {
        ("T", "K1", "11-ID"): (74.5740498034076, .762300131322058, 73.1368615839592),
        ("T", "K1", "3-OOD"): (67.5358539765319, .708518905809584, 67.0988918221902),
        ("MM", "K1", "11-ID"): (74.5740498034076, .762300131322058, 73.1368615839592),
        ("MM", "K1", "3-OOD"): (67.5358539765319, .708518905809584, 67.0988918221902),
        ("T", "K2", "11-ID"): (75.5401610185359, .655820363840188, 74.4211634279972),
        ("T", "K2", "3-OOD"): (74.1851368970013, .652864411055651, 73.2354789291852),
        ("MM", "K2", "11-ID"): (75.120763901891, .650297139350915, 74.0741921385776),
        ("MM", "K2", "3-OOD"): (73.6636245110821, .637214789120374, 72.8371742965125),
        ("MM", "K2", "14-ID"): (74.91486574983628, .6483267296248988, 73.90044528647852),
    }


def evaluate(protocol):
    dest = where(protocol)
    if (dest / "EVALUATION_DONE.json").exists():
        raise FileExistsError("evaluation already completed")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import evaluate_final_unlock
    path = dest / "PREDICTIONS.npz"
    receipt_path = dest / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(path) != receipt["prediction_sha256"] or digest(dest / "MODEL_LOCK.json") != receipt["model_lock"]["sha256"]:
        raise RuntimeError("frozen prediction binding changed")
    uids, views = tiny.test_uids(protocol)
    with np.load(path, allow_pickle=False) as pred:
        if pred["uids"].tolist() != uids or not np.array_equal(pred["views"], views):
            raise RuntimeError("frozen test alignment changed")
        actions = pred["actions"].copy()
        families_out, seeds_out, rows_out = pred["families"].copy(), pred["seeds"].copy(), pred["rows"].copy()
        switch_count, guard_count = pred["switch_count"].copy(), pred["guard_count"].copy()
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **subject.label_args())
    y, c = labels.matrix.y, labels.matrix.c
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    datasets = np.array([uid.split("/", 1)[0] for uid in uids])
    per_seed, per_dataset = [], []
    view_names = ("11-ID", "3-OOD", "14-ID-pooled-11-source") if protocol == "heldout11" else ("14-ID",)
    for j, (family, seed, row) in enumerate(zip(families_out, seeds_out, rows_out)):
        for view in view_names:
            mask = np.ones(len(uids), dtype=bool) if view == "14-ID-pooled-11-source" else views == view
            per_seed.append(dict(protocol=protocol, family=str(family), seed=int(seed), row=str(row),
                                 view=view, switch_n=int(switch_count[j]), guard_n=int(guard_count[j]),
                                 **tiny.metrics(y[mask], c[mask], actions[j, mask], bounds)))
        if protocol == "heldout11" and row in ("K2", "SwitchCostCap", "Guard25"):
            for dataset in ("MMMU_DEV_VAL", "MathVision_MINI", "ChartQA_TEST"):
                mask = datasets == dataset
                per_dataset.append(dict(protocol=protocol, family=str(family), seed=int(seed),
                                        row=str(row), dataset=dataset,
                                        **tiny.metrics(y[mask], c[mask], actions[j, mask], bounds)))
    summary = []
    for family, seeds in families(protocol):
        for row in ROWS:
            for view in view_names:
                selected = [r for r in per_seed if r["family"] == family and r["row"] == row and r["view"] == view]
                if len(selected) != 5 or {r["seed"] for r in selected} != set(seeds):
                    raise RuntimeError("incomplete five-seed view")
                result = dict(protocol=protocol, family=family, row=row, view=view, n=selected[0]["n"], seeds=5)
                for name in METRICS:
                    values = np.asarray([r[name] for r in selected])
                    result[name + "_mean"] = float(values.mean())
                    result[name + "_sd"] = float(values.std(ddof=1))
                summary.append(result)
    ds_summary = []
    for family, _ in families(protocol):
        for row in ("K2", "SwitchCostCap", "Guard25"):
            for dataset in ("MMMU_DEV_VAL", "MathVision_MINI", "ChartQA_TEST"):
                selected = [r for r in per_dataset if r["family"] == family and r["row"] == row and r["dataset"] == dataset]
                if not selected:
                    continue
                result = dict(family=family, row=row, dataset=dataset, n=selected[0]["n"], seeds=5)
                for name in METRICS:
                    values = np.asarray([r[name] for r in selected])
                    result[name + "_mean"] = float(values.mean())
                    result[name + "_sd"] = float(values.std(ddof=1))
                ds_summary.append(result)
    checks = []
    targets = historical_targets()
    for result in summary:
        key = (result["family"], result["row"], result["view"])
        if key in targets:
            observed = [result[name + "_mean"] for name in METRICS]
            target = targets[key]
            difference = [a - b for a, b in zip(observed, target)]
            checks.append(dict(kind="learned_base", family=key[0], row=key[1], view=key[2],
                               target=target, observed=observed, difference=difference,
                               match=all(abs(v) < 1e-6 for v in difference)))
    original = (ROOT / f"outputs/stateroute_mmmu_subject_four_rules_{'full14' if protocol == 'full14' else '11_3'}_20260928/PREDICTIONS.npz")
    with np.load(original, allow_pickle=False) as prior:
        if prior["uids"].tolist() != uids:
            raise RuntimeError("subject-state prediction UID mismatch")
        requested = (("SubjectAccuracy14", "14-ID", 75.08517043326412),) if protocol == "full14" else (
            ("SubjectRule3-CostCap-Guard10", "11-ID", 75.04963957142058),
            ("SubjectRule4-CostCap-Guard25", "3-OOD", 71.7525247030539))
        for method, view, target in requested:
            index = prior["methods"].tolist().index(method)
            mask = views == view
            observed = tiny.metrics(y[mask], c[mask], prior["actions"][index, mask], bounds)["rank_score"]
            checks.append(dict(kind="subject_frozen_action_rescore", method=method, view=view,
                               target_rank=target, observed_rank=observed,
                               difference=observed - target, match=abs(observed - target) < 1e-6))
    if not all(check["match"] for check in checks):
        raise RuntimeError(f"reproduction failed before reporting new numbers: {checks}")
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "per_seed.csv", per_seed)
    write_csv(dest / "summary.csv", summary)
    if per_dataset:
        write_csv(dest / "per_dataset.csv", per_dataset)
        write_csv(dest / "per_dataset_summary.csv", ds_summary)
    write_json(dest / "REPRODUCTION.json", dict(status="PASS", checks=checks,
                                                  subject_seed_dependence="none: frozen prototypes, profiles, Ridge and rules have no seed parameter"))
    write_json(dest / "EVALUATION_DONE.json", dict(protocol=protocol, reproduction="PASS",
                                                    prediction=record(path), provenance=provenance,
                                                    historical_test_exposure=True, test_metric_selection=False))
    say(stage="evaluate_complete", protocol=protocol, reproduction="PASS", rows=len(summary))


def report():
    for protocol in ("heldout11", "full14"):
        if not (where(protocol) / "EVALUATION_DONE.json").exists():
            raise RuntimeError(f"{protocol} evaluation incomplete")
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in ("per_seed", "summary", "per_dataset", "per_dataset_summary"):
        rows = []
        for protocol in ("heldout11", "full14"):
            path = where(protocol) / f"{name}.csv"
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    rows.extend(csv.DictReader(handle))
        if rows:
            write_csv(REPORT / f"{name}.csv", rows)
    say(stage="report_tables_complete", path=str(REPORT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("source", "predict", "evaluate", "report"), required=True)
    parser.add_argument("--protocol", choices=("heldout11", "full14"))
    args = parser.parse_args()
    if args.stage != "report" and not args.protocol:
        parser.error("--protocol is required for source/predict/evaluate")
    with threadpool_limits(limits=8):
        {"source": source, "predict": predict, "evaluate": evaluate, "report": lambda _: report()}[args.stage](args.protocol)


if __name__ == "__main__":
    main()
