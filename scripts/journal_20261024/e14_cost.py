#!/usr/bin/env python3
"""Manuscript Table 8: cost-map ablations at fixed K=2 states."""
from __future__ import annotations

import json
import os
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from threadpoolctl import threadpool_limits

import e4
import e5
import e12_matched as matched
import e13_shrinkage as shrink
from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from scripts import run_stateroute_mmmu_lambda_sweep_20260928 as sweep
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source

OUT = OUTPUT / "E14"
VARIANTS = ("full", "no_cost", "state_conditional_cost", "rule_swapped")


def state_cost(c, state):
    n = np.bincount(state, minlength=2).astype(np.float64)
    sums = np.zeros((2, 17), dtype=np.float64)
    np.add.at(sums, state, c)
    global_cost = c.mean(axis=0, dtype=np.float64)
    values = np.divide(sums, n[:, None], out=np.broadcast_to(global_cost, sums.shape).copy(),
                       where=n[:, None] > 0)
    return values, global_cost


def filter_table(q, costs, cap):
    table = []
    empty = 0
    for state, row in enumerate(q):
        eligible = np.flatnonzero(costs[state] <= cap + 1e-15)
        if not len(eligible):
            eligible = np.asarray([int(costs[state].argmin())])
            empty += 1
        table.append(min(eligible, key=lambda j: (-row[j], costs[state, j], j)))
    return np.asarray(table, dtype=np.int8), empty


def penalty_table(q, costs, scale, lam):
    return (q - lam * costs / scale).argmax(axis=1).astype(np.int8)


def source_t_lambda(protocol, data, train_t, sha, bounds):
    fit = np.flatnonzero(data["role"] == "fit")
    positions = {int(row): i for i, row in enumerate(fit)}
    quality = np.empty((len(fit), 17), dtype=np.float64)
    scaled_cost = np.empty_like(quality)
    for fold in range(3):
        tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][tr]) & set(data["groups"][va]):
            raise RuntimeError(f"{protocol} T OOF dependency groups overlap")
        if protocol == "full14":
            src, dst = e4.text_states(train_t[tr], train_t[va],
                                      data["uids"][tr].tolist(), 0, sha)
        else:
            src, dst = e5.fit_text_states(0, train_t[tr],
                                          data["uids"][tr].tolist(), train_t[va], sha)
        q, _, _ = shrink.profile(data["y"][tr], src, 20.)
        dest = np.asarray([positions[int(row)] for row in va], dtype=np.int64)
        quality[dest] = q[dst]
        c = data["c"][tr]
        scaled_cost[dest] = c.mean(axis=0, dtype=np.float64) / float(np.median(c[c > 0]))
        print(json.dumps(dict(stage="T_penalty_oof_fold", protocol=protocol,
                              fold=fold, n_train=len(tr), n_valid=len(va))), flush=True)
    rows = []
    for lam in sweep.LAMBDAS:
        action = (quality - lam * scaled_cost).argmax(axis=1)
        A, C, R = scores(data["y"][fit], data["c"][fit], action, bounds)
        rows.append(dict(protocol=protocol, lambda_cost=lam, A=A, C=C, R=R))
    selected = max(rows, key=lambda row: row["R"])
    return float(selected["lambda_cost"]), rows


def metric_row(base, variant, view, seed, action, matrices, bounds, note):
    A, C, R = scores(*matrices[view], action, bounds)
    return dict(experiment="E14", method=f"StateRoute-{base}", variant=variant,
                view=view, seed=seed, n_test=len(matrices[view][0]), A=A, C=C, R=R,
                notes=note)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    mm_lambdas = matched.selected_lambdas()
    audit = make_audit("E14", "python -B scripts/journal_20261024/e14_cost.py",
        "K=2 states fixed per seed. T uses E4 14-source fit subset and E5 "
        "11-source full train; kappa=20 and global-median eligibility. "
        "MM: kappa=64, frozen source-OOF global penalty. Conditional variant "
        "substitutes per-state source mean expert cost with the same global cap/scale. "
        "Swapped T penalty lambda is chosen by group-isolated 3-fold source-fit OOF. "
        "MM swapped rule uses the source global median cost; no 3OOD tuning.",
        {view: len(uids) for view, uids in views.items()})
    audit.update(c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu="cached features; CPU clustering", status="running")
    write_json(OUT / "AUDIT.json", audit)
    try:
        data = {protocol: load_source(protocol) for protocol in ("full14", "heldout11")}
        t = {protocol: shrink.text_features(protocol, data[protocol], views) for protocol in data}
        selected_t = {}
        oof_rows = []
        for protocol in data:
            selected_t[protocol], rows = source_t_lambda(
                protocol, data[protocol], t[protocol][0], t[protocol][2], bounds)
            oof_rows.extend(rows)
        write_json(OUT / "source_selection.json", dict(
            T_penalty_lambda=selected_t,
            MM_penalty_lambda={p: mm_lambdas[p, "KMeans2"] for p in data},
            T_oof_seed=0, T_oof_folds=3, group_isolated=True,
            conditional_cap="global mean cost median", test_selection=False))
        write_csv(OUT / "T_penalty_source_oof.csv",
                  ("protocol", "lambda_cost", "A", "C", "R"), oof_rows)
        x14, overlay, _ = subject.qwen_features(views["14ID"], allow_multi=True)
        if len(overlay) != 10:
            raise RuntimeError("test feature overlay changed")
        position = {uid: i for i, uid in enumerate(views["14ID"])}
        xheld = x14[[position[uid] for uid in views["11ID"] + views["3OOD"]]]
        mm_test = {"full14": x14, "heldout11": xheld}
        mm_train = {protocol: subject.qwen_features(data[protocol]["uids"].tolist(),
                    allow_multi=True)[0] for protocol in data}
        matrices, _, _ = eval_matrices(views)
        all_rows, details = [], []
        for seed in SEEDS:
            checkpoint = OUT / f"seed-{seed}.json"
            if checkpoint.is_file():
                old = read_json(checkpoint)
                all_rows.extend(old["rows"])
                details.extend(old["details"])
                continue
            seed_rows, seed_details = [], []
            try:
                for protocol in data:
                    source = data[protocol]
                    slices = (("14ID", slice(None)),) if protocol == "full14" else (
                        ("11ID", slice(0, 5341)), ("3OOD", slice(5341, 6108)))
                    train_t, test_t, sha = t[protocol]
                    if protocol == "full14":
                        fit_t = np.flatnonzero(source["role"] == "fit")
                        src_t, dst_t = e4.text_states(train_t[fit_t], test_t,
                                                      source["uids"][fit_t].tolist(), seed, sha)
                        profile_y_t, profile_c_t = source["y"][fit_t], source["c"][fit_t]
                    else:
                        src_t, dst_t = e5.fit_text_states(seed, train_t,
                                                         source["uids"].tolist(), test_t, sha)
                        profile_y_t, profile_c_t = source["y"], source["c"]
                    q_t, n_t, _ = shrink.profile(profile_y_t, src_t, 20.)
                    conditional_t, global_t = state_cost(profile_c_t, src_t)
                    cap_t = float(np.quantile(global_t, .5))
                    scale_t = float(np.median(profile_c_t[profile_c_t > 0]))
                    global_t2 = np.broadcast_to(global_t, (2, 17))
                    t_tables = {}
                    t_tables["full"], _ = filter_table(q_t, global_t2, cap_t)
                    t_tables["no_cost"] = q_t.argmax(axis=1).astype(np.int8)
                    t_tables["state_conditional_cost"], empty_t = filter_table(q_t, conditional_t, cap_t)
                    t_tables["rule_swapped"] = penalty_table(q_t, global_t2, scale_t, selected_t[protocol])
                    seed_details.append(dict(protocol=protocol, seed=seed, base="T",
                        state_support=n_t.astype(int).tolist(), conditional_no_eligible_states=empty_t,
                        tables={k: v.tolist() for k,v in t_tables.items()}))
                    for variant, table in t_tables.items():
                        action = table[dst_t]
                        for view, sl in slices:
                            seed_rows.append(metric_row("T", variant, view, seed, action[sl],
                                matrices, bounds, f"K2; source-only lambda={selected_t[protocol]} for swapped rule"))
                    print(json.dumps(dict(stage="T_complete", protocol=protocol, seed=seed)), flush=True)

                    fit = np.flatnonzero(source["role"] == "fit")
                    with np.load(matched.model_path(protocol, "KMeans2", seed), allow_pickle=False) as model:
                        centers = np.asarray(model["centers"], dtype=np.float32)
                        frozen_rates = np.asarray(model["rates"], dtype=np.float64)
                    src_mm = matched.assign(mm_train[protocol][fit], centers, "KMeans2")
                    dst_mm = matched.assign(mm_test[protocol], centers, "KMeans2")
                    q_mm, n_mm, _ = shrink.profile(source["y"][fit], src_mm, 64.)
                    if not np.allclose(q_mm, frozen_rates, atol=1e-6, rtol=0):
                        raise RuntimeError("MM frozen profile differs from reconstructed source-fit profile")
                    conditional_mm, global_mm = state_cost(source["c"][fit], src_mm)
                    cap_mm = float(np.quantile(global_mm, .5))
                    scale_mm = float(np.median(source["c"][fit][source["c"][fit] > 0]))
                    global_mm2 = np.broadcast_to(global_mm, (2, 17))
                    lam = mm_lambdas[protocol, "KMeans2"]
                    mm_tables = {
                        "full": penalty_table(q_mm, global_mm2, scale_mm, lam),
                        "no_cost": q_mm.argmax(axis=1).astype(np.int8),
                        "state_conditional_cost": penalty_table(q_mm, conditional_mm, scale_mm, lam),
                    }
                    mm_tables["rule_swapped"], _ = filter_table(q_mm, global_mm2, cap_mm)
                    seed_details.append(dict(protocol=protocol, seed=seed, base="MM",
                        state_support=n_mm.astype(int).tolist(), conditional_no_eligible_states=0,
                        tables={k: v.tolist() for k,v in mm_tables.items()}))
                    for variant, table in mm_tables.items():
                        action = table[dst_mm]
                        for view, sl in slices:
                            seed_rows.append(metric_row("MM", variant, view, seed, action[sl],
                                matrices, bounds, f"K2; source-only lambda={lam} for full/conditional rule"))
                    print(json.dumps(dict(stage="MM_complete", protocol=protocol, seed=seed)), flush=True)
                write_json(checkpoint, dict(seed=seed, rows=seed_rows, details=seed_details))
                all_rows.extend(seed_rows)
                details.extend(seed_details)
            except Exception as exc:
                trace = OUT / f"seed-{seed}-traceback.txt"
                trace.write_text(traceback.format_exc())
                audit["failures"].append(dict(id="E14", seed=seed, error=str(exc),
                                              traceback_path=str(trace.relative_to(ROOT))))
                write_json(OUT / "AUDIT.json", audit)
                print(json.dumps(dict(stage="seed_failed", seed=seed, error=str(exc)), flush=True))
        write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, all_rows)
        summary = summarize(all_rows)
        write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
        write_json(OUT / "map_details.json", details)
        anchors = {("T", "14ID"): (74.137, .04), ("T", "11ID"): (74.414, .04),
                   ("T", "3OOD"): (73.235, .01), ("MM", "14ID"): (73.900, .05),
                   ("MM", "11ID"): (74.074, .08), ("MM", "3OOD"): (72.837, .28)}
        checks = []
        for (base, view), (expected, tolerance) in anchors.items():
            found = [item for item in summary if item["method"] == f"StateRoute-{base}"
                     and item["variant"] == "full" and item["view"] == view]
            observed = found[0]["R_mean"] if found else None
            checks.append(dict(base=base, view=view, expected_R=expected, observed_R=observed,
                               tolerance=tolerance,
                               passed=bool(observed is not None and abs(observed - expected) <= tolerance)))
        audit["acceptance"] = checks
        audit["status"] = "complete" if all(item["passed"] for item in checks) and not audit["failures"] else "acceptance_failed"
        if audit["status"] != "complete":
            raise RuntimeError("E14 full-method reference or seed check failed")
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
