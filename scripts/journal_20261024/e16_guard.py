#!/usr/bin/env python3
"""Manuscript Sec. 7.6: measured coarse-fallback fractions by view and base."""
from __future__ import annotations

import hashlib
import json
import traceback

import numpy as np

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SUMMARY_COLUMNS,
                    make_audit, read_json, utc_now, view_uids,
                    write_csv, write_json)

OUT = OUTPUT / "E16"
SOURCE = ROOT / "outputs/bench_coe_query_subject_four_rules_20261024/PREDICTIONS.npz"
TRIAL_COLUMNS = ("router", "backbone", "seed", "view", "guard", "n",
                 "gate_count", "gate_fraction", "action_changed_count", "action_changed_fraction")
SUMMARY_SPECIAL_COLUMNS = ("router", "backbone", "view", "guard", "n_seeds", "n",
    "gate_fraction_mean", "gate_fraction_sd", "action_changed_fraction_mean",
    "action_changed_fraction_sd")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    audit = make_audit("E16", "python -B scripts/journal_20261024/e16_guard.py",
        "Frozen gate masks and on/off actions are replayed without test outcomes. "
        "Gate fraction counts threshold triggers; changed-action fraction excludes "
        "queries where fallback chose the same expert. The two are reported separately.",
        {view: len(uids) for view, uids in views.items()})
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit.update(c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu="none", status="running", source_sha256=sha(SOURCE))
    write_json(OUT / "AUDIT.json", audit)
    try:
        with np.load(SOURCE, allow_pickle=False) as source:
            ids = source["uids"].astype(str).tolist()
            if ids != views["11ID"] + views["3OOD"]:
                raise RuntimeError("frozen rule actions no longer align to held-out test UIDs")
            methods = source["methods"].astype(str).tolist()
            backbones = source["backbones"].astype(str).tolist()
            seeds = source["seeds"].astype(int).tolist()
            gates = source["k2_gates"]
            actions = source["heldout_actions"]
            regimes = source["regimes"].astype(str).tolist()
            if regimes != ["K2-on", "K2-off"] or actions.shape != (2, 3, 5, 2, 4, 6108):
                raise RuntimeError("frozen rule array schema changed")
            rows = []
            for mi, method in enumerate(methods):
                for bi, backbone in enumerate(backbones):
                    for si, seed in enumerate(seeds):
                        for guard, ri in (("gate_fpr10", 2), ("gate_fpr25", 3)):
                            gate = gates[mi, bi, si, ri]
                            changed = actions[mi, bi, si, 0, ri] != actions[mi, bi, si, 1, ri]
                            if np.any(changed & ~gate):
                                raise RuntimeError("fallback changed an action outside its gate")
                            for view, sl in (("11ID", slice(0, 5341)),
                                             ("3OOD", slice(5341, 6108))):
                                n = sl.stop - sl.start
                                gate_n = int(gate[sl].sum())
                                change_n = int(changed[sl].sum())
                                rows.append(dict(router=method, backbone=backbone, seed=seed,
                                    view=view, guard=guard, n=n, gate_count=gate_n,
                                    gate_fraction=gate_n/n, action_changed_count=change_n,
                                    action_changed_fraction=change_n/n))
        write_csv(OUT / "guard_fractions.csv", TRIAL_COLUMNS, rows)
        grouped = {}
        for row in rows:
            key = (row["router"], row["backbone"], row["view"], row["guard"])
            grouped.setdefault(key, []).append(row)
        summaries = []
        for (router, backbone, view, guard), values in sorted(grouped.items()):
            item = dict(router=router, backbone=backbone, view=view, guard=guard,
                        n_seeds=len(values), n=values[0]["n"])
            for metric in ("gate_fraction", "action_changed_fraction"):
                vec = np.asarray([row[metric] for row in values], dtype=float)
                item[metric + "_mean"] = float(vec.mean())
                item[metric + "_sd"] = float(vec.std(ddof=1))
            summaries.append(item)
        write_csv(OUT / "guard_fractions_SUMMARY.csv", SUMMARY_SPECIAL_COLUMNS, summaries)
        write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, [])
        write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, [])
        audit["acceptance"] = dict(expected_rows=120, observed_rows=len(rows),
            passed=len(rows) == 120 and all(item["n_seeds"] == 5 for item in summaries))
        audit["status"] = "complete" if audit["acceptance"]["passed"] else "acceptance_failed"
        if audit["status"] != "complete":
            raise RuntimeError("E16 guard fraction coverage check failed")
    except Exception as exc:
        trace = OUT / "traceback.txt"
        trace.write_text(traceback.format_exc())
        audit["failures"].append(dict(id="E16", seed=None, error=str(exc),
                                      traceback_path=str(trace.relative_to(ROOT))))
        if audit.get("status") == "running":
            audit["status"] = "failed"
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    main()
