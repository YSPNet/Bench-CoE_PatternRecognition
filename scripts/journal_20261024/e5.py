#!/usr/bin/env python3
"""E5 base gate: fit unified K1/K2 bases before any local/switch/fallback rung."""
from __future__ import annotations

import json
import os
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from scripts import run_stateroute_learned_ladder_20261024 as ladder
from src.next_round_x.data import load_source
from src.next_round_x.models import normalized
from src.subject_routing.clustering.spherical_kmeans import SphericalKMeans, assign_frozen
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit

DEST = OUTPUT / "E5"
KAPPA = 20.0
MM_LAMBDAS = (0., .001, .003, .01, .03, .1, .3)
PUB = {
    ("T", "11ID"): ((75.54016101853585, .6558203638401883, 74.42116342799721),
                      (.010255056309777444, .00005087533075818954, .009264903927304011)),
    ("T", "3OOD"): ((74.1851368970013, .6528644110556513, 73.2354789291852), (0., 0., 0.)),
    ("MM", "11ID"): ((75.12076390189104, .6502971393509145, 74.07419213857766),
                       (.06539692640544262, .004765836905659926, .06473093334236653)),
    ("MM", "3OOD"): ((73.66362451108213, .6372147891203742, 72.83717429651246),
                       (.27657370841716794, .006748123505030968, .2714050629742314)),
}


def log(**message):
    print(json.dumps(message, sort_keys=True, default=str), flush=True)


def fit_text_states(seed, train_x, train_uids, test_x, sha):
    model = SphericalKMeans(2, seed=seed, n_init=10, max_iter=100, tolerance=1e-6).fit(
        train_x, train_uids, environment_id="ood11_3", fit_role="train_seen",
        feature_artifact_sha256=sha,
        audit=UIDAccessAudit.create(Operation.FIT, train_uids))
    train_state = np.array([model.cluster_ids.index(s) for s in model.assignments], dtype=np.int8)
    states = assign_frozen(model, test_x, feature_artifact_sha256=sha).states
    test_state = np.array([model.cluster_ids.index(s) for s in states], dtype=np.int8)
    return train_state, test_state


def fit_mm_states(seed, train_x, test_x):
    km = MiniBatchKMeans(n_clusters=2, random_state=seed, n_init=3, batch_size=1024, max_iter=100)
    train_state = km.fit_predict(normalized(train_x)).astype(np.int8)
    test_state = km.predict(normalized(test_x)).astype(np.int8)
    return train_state, test_state


def profile(y, c, state):
    count = np.bincount(state, minlength=2)
    global_y = y.mean(axis=0, dtype=np.float64)
    global_c = c.mean(axis=0, dtype=np.float64)
    sums = np.zeros((2, 17), dtype=np.float64)
    np.add.at(sums, state, y)
    quality = (sums + KAPPA * global_y) / (count[:, None] + KAPPA)
    scale = float(np.median(c[c > 0]))
    return quality, global_y, global_c, global_c / scale, count


def text_table(q, global_cost):
    cap = float(np.quantile(global_cost, .5))
    allowed = np.flatnonzero(global_cost <= cap + 1e-15)
    return np.array([min(allowed, key=lambda j: (-row[j], global_cost[j], j))
                     for row in q], dtype=np.int8)


def select_mm_lambda(data, x, bounds):
    fit = np.flatnonzero(data["role"] == "fit")
    position = {int(row): i for i, row in enumerate(fit)}
    pred = np.empty((len(fit), 17), dtype=np.float64)
    costs = np.empty_like(pred)
    for fold in range(3):
        tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][tr]) & set(data["groups"][va]):
            raise RuntimeError("MM learned-state OOF dependency groups overlap")
        src, dst = fit_mm_states(0, x[tr], x[va])
        quality, _, _, scaled_cost, _ = profile(data["y"][tr], data["c"][tr], src)
        dest = np.array([position[int(i)] for i in va])
        pred[dest] = quality[dst]
        costs[dest] = scaled_cost
        log(stage="mm_oof_fold", fold=fold, train=len(tr), valid=len(va))
    rows = []
    for lam in MM_LAMBDAS:
        A, C, R = scores(data["y"][fit], data["c"][fit], (pred - lam * costs).argmax(axis=1), bounds)
        rows.append(dict(lambda_cost=lam, A=A, C=C, R=R))
    chosen = max(rows, key=lambda row: row["R"])
    return chosen["lambda_cost"], rows


def metric_row(base, variant, view, seed, actions, matrices, bounds, note):
    y, c = matrices[view]
    A, C, R = scores(y, c, actions, bounds)
    return dict(experiment="E5", method=f"StateRoute-{base}", variant=variant,
                view=view, seed=seed, n_test=len(y), A=A, C=C, R=R, notes=note)


def check_acceptance(summary):
    checks = []
    for (base, view), (target, sd) in PUB.items():
        selected = [row for row in summary if row["method"] == f"StateRoute-{base}"
                    and row["variant"] == "K2_fixed_map__frozen" and row["view"] == view]
        observed = [selected[0][f"{name}_mean"] for name in ("A", "C", "R")] if selected else None
        within = [abs(a - b) <= max(s, 1e-6) for a, b, s in zip(observed, target, sd)] if observed else None
        checks.append(dict(base=base, view=view, target=target, published_sd=sd,
                           observed=observed, within_sd=within, pass_check=bool(within and all(within))))
    return checks


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit = make_audit("E5", "python -B scripts/journal_20261024/e5.py --stage base",
                       "Rungs 1-2 only. Both bases profile the full 11-source train with kappa=20; T uses source global-cost median eligibility, MM lambda uses group-isolated 3-fold source-fit OOF; no 3OOD tuning.",
                       {k: len(v) for k, v in views.items()})
    audit["c_min"], audit["c_max"] = bounds["cmin"], bounds["cmax"]
    audit["gpu"] = "cached features; CPU router fits"
    write_json(DEST / "AUDIT.json", audit)
    data = load_source("heldout11")
    train_x = np.asarray(data["x"], dtype=np.float32)
    tx = np.asarray(ladder.text_train(data), dtype=np.float32)
    held_uids = views["11ID"] + views["3OOD"]
    test_text = ladder.text_test(held_uids)
    test_mm, overlay, _ = subject.qwen_features(held_uids, allow_multi=True)
    if len(overlay) != 10:
        raise RuntimeError("heldout11 test Qwen multi-image overlay changed")
    text_sha = read_json(ladder.TEXT_METADATA)["feature_artifact_sha256"]
    mm_lam, mm_oof = select_mm_lambda(data, train_x, bounds)
    write_json(DEST / "base_source_selection.json", dict(profile_subset="all 18665 official 11-source train rows",
               kappa=KAPPA, text_eligibility_quantile=.5, mm_selected_lambda=mm_lam,
               mm_source_fit_oof=mm_oof, test_metric_selection=False))
    matrices, _, _ = eval_matrices(views)
    rows = []
    branch = []
    for seed in SEEDS:
        checkpoint = DEST / f"base-seed-{seed}.json"
        if checkpoint.exists():
            old = read_json(checkpoint)
            rows.extend(old["rows"])
            branch.extend(old["branch_fractions"])
            log(stage="seed_resumed", seed=seed)
            continue
        try:
            seed_rows = []
            seed_branches = []
            for base, source_x, test_x in (("T", tx, test_text), ("MM", train_x, test_mm)):
                if base == "T":
                    source_states, test_states = fit_text_states(seed, source_x,
                                                                  data["uids"].tolist(), test_x, text_sha)
                else:
                    source_states, test_states = fit_mm_states(seed, source_x, test_x)
                q, global_q, global_c, scaled_cost, support = profile(data["y"], data["c"], source_states)
                if base == "T":
                    k2_table = text_table(q, global_c)
                    k1_expert = int(text_table(global_q[None, :], global_c)[0])
                else:
                    k2_table = (q - mm_lam * scaled_cost).argmax(axis=1).astype(np.int8)
                    k1_expert = int((global_q - mm_lam * scaled_cost).argmax())
                actions = {"K1": np.full(len(held_uids), k1_expert, dtype=np.int8),
                           "K2_fixed_map": k2_table[test_states]}
                for rung, action in actions.items():
                    for view, sl in (("11ID", slice(0, 5341)), ("3OOD", slice(5341, 6108))):
                        for threshold in ("frozen", "refit"):
                            variant = f"{rung}__{threshold}"
                            seed_rows.append(metric_row(base, variant, view, seed, action[sl],
                                                        matrices, bounds,
                                                        "Full 11-source profile, kappa=20; thresholds inactive at rung <=2"))
                            seed_branches.append(dict(base=base, rung=rung, thresholds=threshold,
                                                      view=view, seed=seed, fraction_fallback=0.,
                                                      fraction_conservative=1., fraction_aggressive=0.))
                log(stage="base_fit", base=base, seed=seed, k1=k1_expert,
                    k2_table=k2_table.tolist(), support=support.tolist())
            write_json(checkpoint, dict(seed=seed, rows=seed_rows, branch_fractions=seed_branches))
            rows.extend(seed_rows)
            branch.extend(seed_branches)
            log(stage="seed_complete", seed=seed)
        except Exception as exc:
            path = DEST / f"base-seed-{seed}-traceback.txt"
            path.write_text(traceback.format_exc())
            audit["failures"].append(dict(id="E5.base", seed=seed, error=str(exc), traceback_path=str(path)))
            write_json(DEST / "AUDIT.json", audit)
            log(stage="seed_failed", seed=seed, error=str(exc))
    write_csv(DEST / "RESULTS.csv", RESULT_COLUMNS, rows)
    summary = summarize(rows)
    write_csv(DEST / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
    write_csv(DEST / "branch_fractions.csv",
              ("base", "rung", "thresholds", "view", "seed", "fraction_fallback",
               "fraction_conservative", "fraction_aggressive"), branch)
    checks = check_acceptance(summary)
    audit["rung2_acceptance"] = checks
    audit["end_time"] = utc_now()
    audit["status"] = "BASE_ACCEPTED" if all(c["pass_check"] for c in checks) else "STOP_RUNG2_ACCEPTANCE_FAILED"
    write_json(DEST / "AUDIT.json", audit)
    write_json(DEST / "thresholds_refit.json", dict(status="pending_rung2_acceptance" if
                 audit["status"] == "BASE_ACCEPTED" else "not_run_due_to_rung2_acceptance_failure",
                 reason="Refit thresholds are used only from rung 4; no threshold fit was run on 3OOD."))
    log(stage="rung2_acceptance", checks=checks)
    if audit["status"] != "BASE_ACCEPTED":
        raise RuntimeError("E5 rung-2 acceptance failed; stop before rungs 3-5 and E10")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
