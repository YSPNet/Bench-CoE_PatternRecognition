#!/usr/bin/env python3
"""Paired test-query bootstrap for v25's frozen spherical K=2 router."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import OUTPUT, ROOT, eval_matrices, read_json, scores, view_uids, write_csv, write_json
from routers.utils.rank_score import batch_rank_score
from v23_mm_spherical_compare import sha

OUT = OUTPUT / "V25_SPHERICAL_COMPLETION"
BOOTSTRAPS = 2000
COMPARATORS = (
    "Query-BERT-cost-aware", "Query-TinyLLaVA-cost-aware",
    "Subject-TinyLLaVA-cost-aware", "MLP", "VLC", "ZOOTER",
    "CosineCls", "RouterDC", "Strongest", "Cheapest")
COLUMNS = ("comparator", "view", "n_test", "n_bootstrap", "reference_R",
           "comparator_R", "delta_R", "ci_low", "ci_high", "p_two_sided",
           "delta_A", "delta_C", "action_difference_fraction")


def query_actions(directory: Path, view: str, uids: list[str]):
    root = directory / "heldout11" / "seed-0"
    complete = read_json(root / "COMPLETE.json")
    lam = complete["selected_lambda"]
    with np.load(root / f"final-lambda-{lam}.npz", allow_pickle=False) as saved:
        if saved[f"uid_{view}"].tolist() != uids:
            raise RuntimeError(f"query action UID mismatch: {root}/{view}")
        return saved[f"action_{view}"].astype(np.int8)


def e6_actions(method: str, view: str, uids: list[str]):
    root = OUTPUT / "E6/tasks/heldout11" / method / "seed-0"
    lam = read_json(root / "COMPLETE.json")["lambda_val"]
    with np.load(root / f"lambda-{lam}" / f"{view}.npz", allow_pickle=False) as saved:
        ids = saved["sample_ids"].tolist()
        if len(ids) != len(uids) or set(ids) != set(uids):
            raise RuntimeError(f"E6 action UID coverage mismatch: {method}/{view}")
        positions = {uid: i for i, uid in enumerate(ids)}
        return saved["actions"][[positions[uid] for uid in uids]].astype(np.int8)


def e3_fixed(view: str, method: str):
    root = OUTPUT / "E3"
    manifest = read_json(root / "seed-0.json")
    item = next((item for item in manifest["action_index"]
                 if item["method"] == method and item["variant"] == "source_train"
                 and item["view"] == view), None)
    if item is None:
        raise RuntimeError(f"E3 fixed expert unavailable: {method}/{view}")
    with np.load(root / "actions_seed-0.npz", allow_pickle=False) as saved:
        return saved[item["key"]].astype(np.int8)


def subject_actions(view: str):
    with np.load(OUTPUT / "E21/actions_seed-0.npz", allow_pickle=False) as saved:
        return saved[f"heldout11_seed0_lambda0.1_{view}"].astype(np.int8)


def rank(accuracy, cost, bounds):
    return 100 * batch_rank_score(accuracy, cost, bounds["cmin"],
                                  bounds["cmax"], beta=.1)


def main():
    lock = read_json(OUT / "ACTION_LOCK.json")
    if lock["status"] != "predictions_frozen_before_test_scoring":
        raise RuntimeError("spherical actions were not frozen before scoring")
    views = view_uids()
    matrices, _, _ = eval_matrices(views)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    main_file = OUT / "actions/K2_seed-0.npz"
    if sha(main_file) != lock["files"][str(main_file.relative_to(ROOT))]:
        raise RuntimeError("spherical action digest mismatch")
    output = []
    for view, section, rng_seed in (("11ID", slice(0, 5341), 25111),
                                    ("3OOD", slice(5341, None), 25303)):
        uids = views[view]
        y, c = matrices[view]
        with np.load(main_file, allow_pickle=False) as saved:
            reference = saved["main"][section].astype(np.int8)
        if reference.shape != (len(uids),):
            raise RuntimeError("reference action shape mismatch")
        comparators = {
            "Query-BERT-cost-aware": query_actions(OUTPUT / "E20", view, uids),
            "Query-TinyLLaVA-cost-aware": query_actions(
                OUTPUT / "V19_QUERY_COST_TINYLLAVA", view, uids),
            "Subject-TinyLLaVA-cost-aware": subject_actions(view),
            "MLP": e6_actions("mlp", view, uids),
            "VLC": e6_actions("vlc", view, uids),
            "ZOOTER": e6_actions("zooter", view, uids),
            "CosineCls": e6_actions("cosinecls", view, uids),
            "RouterDC": e6_actions("routerdc", view, uids),
            "Strongest": e3_fixed(view, "Strongest"),
            "Cheapest": e3_fixed(view, "Cheapest")}
        if set(comparators) != set(COMPARATORS):
            raise RuntimeError("comparator coverage changed")
        ref_scores = scores(y, c, reference, bounds)
        idx = np.arange(len(y))
        ref_y = np.asarray(y[idx, reference], dtype=np.float64)
        ref_c = np.asarray(c[idx, reference], dtype=np.float64)
        selected = {name: (np.asarray(y[idx, action], dtype=np.float64),
                           np.asarray(c[idx, action], dtype=np.float64))
                    for name, action in comparators.items()}
        rng = np.random.default_rng(rng_seed)
        differences = {name: [] for name in COMPARATORS}
        for start in range(0, BOOTSTRAPS, 100):
            sample = rng.integers(0, len(y), size=(min(100, BOOTSTRAPS-start), len(y)))
            ref_r = rank(ref_y[sample].mean(axis=1), ref_c[sample].mean(axis=1), bounds)
            for name, (cmp_y, cmp_c) in selected.items():
                cmp_r = rank(cmp_y[sample].mean(axis=1),
                             cmp_c[sample].mean(axis=1), bounds)
                differences[name].extend((ref_r - cmp_r).tolist())
        for name in COMPARATORS:
            action = comparators[name]
            cmp_scores = scores(y, c, action, bounds)
            draws = np.asarray(differences[name])
            low, high = np.quantile(draws, [.025, .975])
            p = min(1., 2 * min(((draws <= 0).sum()+1)/(BOOTSTRAPS+1),
                                 ((draws >= 0).sum()+1)/(BOOTSTRAPS+1)))
            output.append(dict(comparator=name, view=view, n_test=len(y),
                n_bootstrap=BOOTSTRAPS, reference_R=ref_scores[2],
                comparator_R=cmp_scores[2], delta_R=ref_scores[2]-cmp_scores[2],
                ci_low=float(low), ci_high=float(high), p_two_sided=float(p),
                delta_A=ref_scores[0]-cmp_scores[0],
                delta_C=ref_scores[1]-cmp_scores[1],
                action_difference_fraction=float(np.mean(reference != action))))
        print(json.dumps(dict(stage="bootstrap_view", view=view,
                              reference_R=ref_scores[2])), flush=True)
    write_csv(OUT / "PAIRED_BOOTSTRAP.csv", COLUMNS, output)
    write_json(OUT / "BOOTSTRAP_AUDIT.json", dict(status="complete", n=BOOTSTRAPS,
        views=["11ID", "3OOD"], reference="spherical K=2, seed 0",
        comparator_action_sources="E20, V19_QUERY_COST_TINYLLAVA, E21, E6, E3",
        source_selection="all model and cost knobs selected on source data",
        test_resampling="paired by UID, percentile 95% interval",
        p_value="two-sided sign probability with plus-one correction",
        action_lock_sha256=sha(OUT / "ACTION_LOCK.json")))
    print(json.dumps(dict(status="complete", rows=len(output))), flush=True)


if __name__ == "__main__":
    main()
