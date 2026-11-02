#!/usr/bin/env python3
"""Complete E5 rungs 3-5 after the saved rung-2 acceptance gate passes."""
from __future__ import annotations

import json
import os
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from threadpoolctl import threadpool_limits

import e5
import e4
from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, read_json, scores, summarize, utc_now,
                    view_uids, write_csv, write_json)
from scripts import run_stateroute_tiny_knn_20260928 as tiny
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from scripts import run_stateroute_learned_ladder_20261024 as ladder
from scripts import run_stateroute_mmmu_full14_20260928 as mm14

DEST = OUTPUT / "E5"
KEYS = tuple((base, seed) for base in ("T", "MM") for seed in SEEDS)
FIELDS = ("rank", "aggressive", "gain", "delta")
BRANCH_COLUMNS = ("base", "rung", "thresholds", "view", "seed", "fraction_fallback",
                  "fraction_conservative", "fraction_aggressive")


def log(**message):
    print(json.dumps(message, sort_keys=True, default=str), flush=True)


def profile(y, c, state):
    count = np.bincount(state, minlength=2)
    global_y = y.mean(axis=0, dtype=np.float64)
    global_c = c.mean(axis=0, dtype=np.float64)
    sums_y = np.zeros((2, 17), dtype=np.float64)
    sums_c = np.zeros_like(sums_y)
    np.add.at(sums_y, state, y)
    np.add.at(sums_c, state, c)
    scale = float(np.median(c[c > 0]))
    quality = ((sums_y + e5.KAPPA * global_y) / (count[:, None] + e5.KAPPA)).astype(np.float32)
    state_cost = ((sums_c + e5.KAPPA * global_c) / (count[:, None] + e5.KAPPA) / scale).astype(np.float32)
    return quality, state_cost, (global_c / scale).astype(np.float32), count


def empty_local(n):
    return {key: {field: np.empty(n, dtype=np.float32 if field in ("gain", "delta") else np.int8)
                  for field in FIELDS} for key in KEYS}


def source_refit(data, xq, xt, available, names, domains, text, sha, lock):
    fit = np.flatnonzero(data["role"] == "fit")
    position = {int(row): i for i, row in enumerate(fit)}
    oof = empty_local(len(fit))
    oof_score = np.empty(len(fit), dtype=np.float32)
    coverage = np.zeros(len(fit), dtype=np.int8)
    for fold in range(3):
        path = DEST / f"refit-oof-fold-{fold}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                target = stored["target"].astype(np.int64)
                oof_score[target] = stored["maxscore"]
                for base, seed in KEYS:
                    for field in FIELDS:
                        oof[(base, seed)][field][target] = stored[f"{base}_{seed}_{field}"]
            coverage[target] += 1
            log(stage="refit_oof_fold_resumed", fold=fold, n=len(target))
            continue
        tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][tr]) & set(data["groups"][va]):
            raise RuntimeError("E5 refit OOF dependency groups overlap")
        target = np.array([position[int(i)] for i in va], dtype=np.int64)
        head = tiny.fit_domain(xt[tr], domains[tr], len(names))
        fold_score = head.decision_function(xt[va]).max(axis=1).astype(np.float32)
        bank_states, query_states, profiles = {}, {}, {}
        for base, seed in KEYS:
            if base == "T":
                src, dst = e5.fit_text_states(seed, text[tr], data["uids"][tr].tolist(), text[va], sha)
            else:
                src, dst = e5.fit_mm_states(seed, xq[tr], xq[va])
            bank_states[(base, seed)] = src
            query_states[(base, seed)] = dst
            profiles[(base, seed)] = profile(data["y"][tr], data["c"][tr], src)
        local = empty_local(len(va))
        ladder.neighborhood_pass(xt[va], xt[tr], data["y"][tr], bank_states,
                                 query_states, profiles, available[va], False, local)
        saved = dict(target=target, maxscore=fold_score)
        for base, seed in KEYS:
            for field in FIELDS:
                saved[f"{base}_{seed}_{field}"] = local[(base, seed)][field]
                oof[(base, seed)][field][target] = local[(base, seed)][field]
        np.savez_compressed(path, **saved)
        oof_score[target] = fold_score
        coverage[target] += 1
        log(stage="refit_oof_fold_complete", fold=fold, train=len(tr), valid=len(va))
    if not (coverage == 1).all() or not np.isfinite(oof_score).all():
        raise RuntimeError("E5 source OOF coverage incomplete")
    frozen = lock["rules"]
    price = frozen["accuracy"]["price_weight"]
    margin = frozen["accuracy"]["gain_margin"]
    thresholds = {}
    for base, seed in KEYS:
        local = oof[(base, seed)]
        tau35 = float(np.quantile(oof_score, frozen["accuracy"]["confidence_quantile"]))
        tau10 = float(np.quantile(oof_score, .1))
        tau25 = float(np.quantile(oof_score, .25))
        switch = ((oof_score >= tau35) &
                  (local["gain"] - price * local["delta"] > margin) &
                  (local["aggressive"] != local["rank"]))
        if not switch.any():
            raise RuntimeError(f"No source OOF switches for {base} seed {seed}")
        cap = float(np.quantile(local["delta"][switch], .995))
        thresholds[f"{base}-{seed}"] = dict(confidence_quantile=.35, confidence_threshold=tau35,
                                            gate_fpr10=tau10, gate_fpr25=tau25,
                                            cost_delta_cap_quantile=.995, cost_delta_cap=cap,
                                            price_weight=price, gain_margin=margin,
                                            source_fit_n=len(fit), source_pre_cap_switch_n=int(switch.sum()))
        log(stage="threshold_refit", base=base, seed=seed, tau35=tau35, cap=cap)
    write_json(DEST / "thresholds_refit.json", dict(
        status="source_only_complete", protocol="heldout11", folds=3,
        source_subset="official 11-source fit; dependency groups disjoint by fold",
        learned_state_profile_kappa=e5.KAPPA,
        quantiles=dict(confidence=.35, gate_fpr10=.1, gate_fpr25=.25, cost_cap=.995),
        selected=thresholds, test_outcomes_used=False, ood_outcomes_used=False))
    return thresholds


def branch(base, rung, threshold_name, view, seed, fallback, aggressive):
    fallback = np.asarray(fallback, dtype=bool)
    aggressive = np.asarray(aggressive, dtype=bool) & ~fallback
    return dict(base=base, rung=rung, thresholds=threshold_name, view=view, seed=seed,
                fraction_fallback=float(fallback.mean()),
                fraction_conservative=float((~fallback & ~aggressive).mean()),
                fraction_aggressive=float(aggressive.mean()))


def heldout_results(data, xq, xt, available, fit, names, domains, text, sha,
                    test_xq, test_xt, test_available, thresholds, lock, matrices, bounds):
    uids = view_uids()["11ID"] + view_uids()["3OOD"]
    head = tiny.fit_domain(xt[fit], domains[fit], len(names))
    maxscore = head.decision_function(test_xt).max(axis=1).astype(np.float32)
    bank_states, query_states, profiles, base2 = {}, {}, {}, {}
    lam = read_json(DEST / "base_source_selection.json")["mm_selected_lambda"]
    for base, seed in KEYS:
        if base == "T":
            src, dst = e5.fit_text_states(seed, text, data["uids"].tolist(),
                                           ladder.text_test(uids), sha)
        else:
            src, dst = e5.fit_mm_states(seed, xq, test_xq)
        q, state_cost, global_cost, _ = profile(data["y"], data["c"], src)
        if base == "T":
            table = e5.text_table(q, data["c"].mean(axis=0))
        else:
            table = (q - lam * global_cost).argmax(axis=1).astype(np.int8)
        base2[(base, seed)] = table[dst]
        bank_states[(base, seed)] = src
        query_states[(base, seed)] = dst
        profiles[(base, seed)] = q, state_cost, global_cost, np.bincount(src, minlength=2)
    local = empty_local(len(uids))
    ladder.neighborhood_pass(test_xt, xt, data["y"], bank_states, query_states,
                             profiles, test_available, False, local)
    all_rows, all_branch = [], []
    for base, seed in KEYS:
        path = DEST / f"refine-heldout-seed-{seed}-{base}.json"
        if path.exists():
            checkpoint = read_json(path)
            all_rows.extend(checkpoint["rows"])
            all_branch.extend(checkpoint["branch_fractions"])
            log(stage="heldout_seed_resumed", base=base, seed=seed)
            continue
        try:
            rank = local[(base, seed)]["rank"]
            aggressive_action = local[(base, seed)]["aggressive"]
            gain = local[(base, seed)]["gain"]
            delta = local[(base, seed)]["delta"]
            if not np.isin(rank, np.arange(17)).all():
                raise RuntimeError("invalid conservative actions")
            seed_rows, seed_branches = [], []
            for threshold_name in ("frozen", "refit"):
                if threshold_name == "frozen":
                    rules = lock["rules"]
                    tau35 = rules["accuracy"]["confidence_threshold"]
                    tau10 = rules["guard"]["0.1"]
                    tau25 = rules["guard"]["0.25"]
                    cap = rules["cap"]
                    price = rules["accuracy"]["price_weight"]
                    margin = rules["accuracy"]["gain_margin"]
                else:
                    t = thresholds[f"{base}-{seed}"]
                    tau35, tau10, tau25, cap = (t["confidence_threshold"], t["gate_fpr10"],
                                               t["gate_fpr25"], t["cost_delta_cap"])
                    price, margin = t["price_weight"], t["gain_margin"]
                switch = ((maxscore >= tau35) & (gain - price * delta > margin) &
                          (delta <= cap) & (aggressive_action != rank) & test_available)
                switched = np.where(switch, aggressive_action, rank)
                actions = {
                    "K2_local": (rank, np.zeros(len(rank), dtype=bool), np.zeros(len(rank), dtype=bool)),
                    "K2_local_switch": (switched, np.zeros(len(rank), dtype=bool), switch),
                }
                for low, label in ((tau10, "K2_local_switch_fb10"), (tau25, "K2_local_switch_fb25")):
                    fallback = (maxscore < low) & test_available
                    actions[label] = (np.where(fallback, base2[(base, seed)], switched), fallback, switch)
                for rung, (action, fallback, aggressive_branch) in actions.items():
                    for view, sl in (("11ID", slice(0, 5341)), ("3OOD", slice(5341, 6108))):
                        variant = f"{rung}__{threshold_name}"
                        seed_rows.append(e5.metric_row(base, variant, view, seed, action[sl], matrices, bounds,
                                          "11-source train profile, kappa=20; thresholds source-only"))
                        seed_branches.append(branch(base, rung, threshold_name, view, seed,
                                                    fallback[sl], aggressive_branch[sl]))
            write_json(path, dict(seed=seed, base=base, rows=seed_rows, branch_fractions=seed_branches))
            all_rows.extend(seed_rows)
            all_branch.extend(seed_branches)
            log(stage="heldout_seed_complete", base=base, seed=seed)
        except Exception:
            trace = DEST / f"refine-heldout-seed-{seed}-{base}-traceback.txt"
            trace.write_text(traceback.format_exc())
            raise
    return all_rows, all_branch


def full14_lambda(data, xq, bounds):
    fit = np.flatnonzero(data["role"] == "fit")
    position = {int(row): i for i, row in enumerate(fit)}
    quality = np.empty((len(fit), 17), dtype=np.float64)
    cost = np.empty_like(quality)
    for fold in range(3):
        tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][tr]) & set(data["groups"][va]):
            raise RuntimeError("14ID MM OOF dependency groups overlap")
        src, dst = e5.fit_mm_states(0, xq[tr], xq[va])
        prof, _, global_cost, _ = profile(data["y"][tr], data["c"][tr], src)
        target = np.array([position[int(i)] for i in va])
        quality[target] = prof[dst]
        cost[target] = global_cost
        log(stage="full14_mm_oof_fold", fold=fold, train=len(tr), valid=len(va))
    rows = []
    for lam in mm14.LAMBDAS:
        A, C, R = scores(data["y"][fit], data["c"][fit],
                         (quality - lam * cost).argmax(axis=1), bounds)
        rows.append(dict(lambda_cost=lam, A=A, C=C, R=R))
    chosen = max(rows, key=lambda row: row["R"])
    write_json(DEST / "full14_source_selection.json", dict(selected_lambda=chosen["lambda_cost"],
               oof=rows, kappa=e5.KAPPA, source_fit_n=len(fit), test_outcomes_used=False))
    return chosen["lambda_cost"]


def full14_results(matrices, bounds):
    views = view_uids()
    data, xq, _, overlay = mm14.source_train()
    if len(overlay) != 32:
        raise RuntimeError("full14 source Qwen overlay changed")
    text, test_text, sha = e4.text_features(data, views)
    test_xq, test_overlay, _ = subject.qwen_features(views["14ID"], allow_multi=True)
    if len(test_overlay) != 10:
        raise RuntimeError("full14 test Qwen overlay changed")
    xt, source_available, _ = tiny.tiny_features(data["uids"].tolist())
    test_xt, test_available, _ = tiny.tiny_features(views["14ID"])
    if int((~source_available).sum()) != 32 or int((~test_available).sum()) != 10:
        raise RuntimeError("full14 TinyLLaVA availability changed")
    selection_path = DEST / "full14_source_selection.json"
    lam = read_json(selection_path)["selected_lambda"] if selection_path.exists() else full14_lambda(data, xq, bounds)
    bank_states, query_states, profiles, base1, base2 = {}, {}, {}, {}, {}
    for base, seed in KEYS:
        if base == "T":
            src, dst = e5.fit_text_states(seed, text, data["uids"].tolist(), test_text, sha)
        else:
            src, dst = e5.fit_mm_states(seed, xq, test_xq)
        q, state_cost, global_cost, _ = profile(data["y"], data["c"], src)
        if base == "T":
            global_c = data["c"].mean(axis=0)
            table = e5.text_table(q, global_c)
            global_y = data["y"].mean(axis=0)
            singleton = int(e5.text_table(global_y[None, :], global_c)[0])
        else:
            table = (q - lam * global_cost).argmax(axis=1).astype(np.int8)
            singleton = int((data["y"].mean(axis=0) - lam * global_cost).argmax())
        base1[(base, seed)] = np.full(len(test_xq), singleton, dtype=np.int8)
        base2[(base, seed)] = table[dst]
        bank_states[(base, seed)] = src
        query_states[(base, seed)] = dst
        profiles[(base, seed)] = q, state_cost, global_cost, np.bincount(src, minlength=2)
        log(stage="full14_base_fit", base=base, seed=seed, table=table.tolist(), k1=singleton)
    local = empty_local(len(test_xq))
    ladder.neighborhood_pass(test_xt, xt, data["y"], bank_states, query_states,
                             profiles, test_available, True, local)
    all_rows, all_branch = [], []
    for base, seed in KEYS:
        path = DEST / f"refine-full14-seed-{seed}-{base}.json"
        if path.exists():
            value = read_json(path)
            all_rows.extend(value["rows"])
            all_branch.extend(value["branch_fractions"])
            log(stage="full14_seed_resumed", base=base, seed=seed)
            continue
        seed_rows, seed_branches = [], []
        actions = {"K1": base1[(base, seed)], "K2_fixed_map": base2[(base, seed)],
                   "K2_local": local[(base, seed)]["rank"]}
        for threshold_name in ("frozen", "refit"):
            for rung, action in actions.items():
                seed_rows.append(e5.metric_row(base, f"{rung}__{threshold_name}", "14ID", seed,
                                               action, matrices, bounds,
                                               "All test queries ID; switch/fallback gate omitted; thresholds inactive at rung <=3"))
                seed_branches.append(dict(base=base, rung=rung, thresholds=threshold_name,
                                          view="14ID", seed=seed, fraction_fallback=0.,
                                          fraction_conservative=1., fraction_aggressive=0.))
        write_json(path, dict(seed=seed, base=base, rows=seed_rows, branch_fractions=seed_branches))
        all_rows.extend(seed_rows)
        all_branch.extend(seed_branches)
        log(stage="full14_seed_complete", base=base, seed=seed)
    return all_rows, all_branch


def main():
    audit = read_json(DEST / "AUDIT.json")
    if audit.get("status") != "BASE_ACCEPTED" or not all(c["pass_check"] for c in audit["rung2_acceptance"]):
        raise RuntimeError("E5 rung-2 acceptance gate has not passed")
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    matrices, _, _ = eval_matrices(views)
    data, xq, xt, available, fit, names, domains, _ = tiny.source_inputs("heldout11")
    text = np.asarray(ladder.text_train(data), dtype=np.float32)
    sha = read_json(ladder.TEXT_METADATA)["feature_artifact_sha256"]
    held_uids = views["11ID"] + views["3OOD"]
    test_xq, overlay, _ = subject.qwen_features(held_uids, allow_multi=True)
    test_xt, test_available, _ = tiny.tiny_features(held_uids)
    if len(overlay) != 10 or int((~test_available).sum()) != 10:
        raise RuntimeError("heldout11 test feature overlay/availability changed")
    lock = read_json(ROOT / "outputs/stateroute_mmmu_subject_four_rules_11_3_20260928/MODEL_LOCK.json")
    thresholds = source_refit(data, xq, xt, available, names, domains, text, sha, lock)
    held_rows, held_branch = heldout_results(data, xq, xt, available, fit, names, domains,
                                             text, sha, test_xq, test_xt, test_available,
                                             thresholds, lock, matrices, bounds)
    full_rows, full_branch = full14_results(matrices, bounds)
    import csv
    with (DEST / "RESULTS.csv").open(newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    with (DEST / "branch_fractions.csv").open(newline="") as handle:
        base_branch = list(csv.DictReader(handle))
    # The base rows and new rows have disjoint (method, variant, view, seed) keys.
    rows = base_rows + held_rows + full_rows
    keys = [(r["method"], r["variant"], r["view"], int(r["seed"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate E5 method/variant/view/seed row")
    write_csv(DEST / "RESULTS.csv", RESULT_COLUMNS, rows)
    write_csv(DEST / "SUMMARY.csv", SUMMARY_COLUMNS, summarize(rows))
    write_csv(DEST / "branch_fractions.csv", BRANCH_COLUMNS, base_branch + held_branch + full_branch)
    audit["status"] = "COMPLETE"
    audit["end_time"] = utc_now()
    audit["command"] = "python -B scripts/journal_20261024/e5.py; python -B scripts/journal_20261024/e5_refine.py"
    audit["selection_rule"] += " Refit gate percentiles and delta cap: 3-fold group-isolated source-fit OOF on each learned-state base; frozen thresholds copied unchanged from subject-state MODEL_LOCK.json. 14ID gate omitted."
    audit["full14_mm_lambda"] = read_json(DEST / "full14_source_selection.json")["selected_lambda"]
    write_json(DEST / "AUDIT.json", audit)
    log(stage="complete", results=len(rows), summaries=len(summarize(rows)),
        branch_rows=len(base_branch + held_branch + full_branch))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
