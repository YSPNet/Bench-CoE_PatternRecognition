#!/usr/bin/env python3
"""E10: evaluate frozen Subject routes with and without MMMU in 3OOD."""
from __future__ import annotations

import csv
import json
import traceback

import numpy as np

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)

DEST = OUTPUT / "E10"
SOURCE = ROOT / "outputs/bench_coe_query_subject_four_rules_20261024/PREDICTIONS.npz"
EXTRA_COLUMNS = ("router", "backbone", "variant", "subset", "n", "A_mean", "A_sd",
                 "C_mean", "C_sd", "R_mean", "R_sd")
BACKBONES = {"bert": "BERT", "tinyllava": "TinyLLaVA", "qwen3vl": "Qwen3-VL"}


def log(**message):
    print(json.dumps(message, sort_keys=True, default=str), flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit = make_audit("E10", "python -B scripts/journal_20261024/e10.py",
                       "Read frozen Subject base and rule 4 K2-on actions only; evaluate full 3OOD and ChartQA+MathVision subset; no model fitting or 3OOD selection.",
                       {view: len(uids) for view, uids in views.items()})
    audit["c_min"], audit["c_max"] = bounds["cmin"], bounds["cmax"]
    audit["gpu"] = "none; frozen prediction evaluation"
    audit["prediction_source"] = str(SOURCE.relative_to(ROOT))
    write_json(DEST / "AUDIT.json", audit)
    matrices, _, _ = eval_matrices(views)
    ood_uids = views["3OOD"]
    keep = np.array([uid.split("/", 1)[0] != "MMMU_DEV_VAL" for uid in ood_uids], dtype=bool)
    if int(keep.sum()) != 561 or len(keep) != 767:
        raise RuntimeError("3OOD ChartQA+MathVision subset size changed")
    y, c = matrices["3OOD"]
    all_rows = []
    with np.load(SOURCE, allow_pickle=False) as pred:
        if pred["uids"].tolist() != views["11ID"] + ood_uids:
            raise RuntimeError("Subject frozen prediction UID order changed")
        if pred["methods"].tolist() != ["Query", "Subject"] or pred["regimes"].tolist()[0] != "K2-on":
            raise RuntimeError("Subject/K2-on frozen prediction layout changed")
        if pred["rules"].tolist()[3] != "4-CostCap-Guard25":
            raise RuntimeError("frozen rule 4 index changed")
        for seed in SEEDS:
            path = DEST / f"seed-{seed}.json"
            if path.exists():
                value = read_json(path)
                all_rows.extend(value["rows"])
                log(stage="seed_resumed", seed=seed)
                continue
            try:
                rows = []
                for backbone_index, backbone in enumerate(pred["backbones"].tolist()):
                    if backbone not in BACKBONES:
                        raise RuntimeError("unexpected frozen backbone")
                    actions = {
                        "base": pred["baseheld"][1, backbone_index, seed, 5341:],
                        "rule4_coarse_fallback": pred["heldout_actions"][1, backbone_index, seed, 0, 3, 5341:],
                    }
                    for variant, action in actions.items():
                        for subset, mask in (("3OOD_full", np.ones(767, dtype=bool)),
                                             ("3OOD_no_MMMU", keep)):
                            A, C, R = scores(y[mask], c[mask], action[mask], bounds)
                            rows.append(dict(experiment="E10", method=f"Subject-{BACKBONES[backbone]}",
                                             variant=variant, view=subset, seed=seed,
                                             n_test=int(mask.sum()), A=A, C=C, R=R,
                                             notes="Frozen 11-source Subject route; no 3OOD selection"))
                write_json(path, dict(seed=seed, rows=rows))
                all_rows.extend(rows)
                log(stage="seed_complete", seed=seed, rows=len(rows))
            except Exception as exc:
                trace = DEST / f"seed-{seed}-traceback.txt"
                trace.write_text(traceback.format_exc())
                audit["failures"].append(dict(id="E10", seed=seed, error=str(exc), traceback_path=str(trace)))
                write_json(DEST / "AUDIT.json", audit)
                log(stage="seed_failed", seed=seed, error=str(exc))
    write_csv(DEST / "RESULTS.csv", RESULT_COLUMNS, all_rows)
    summary = summarize(all_rows)
    write_csv(DEST / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
    extras = []
    for row in summary:
        method = row["method"]
        backbone = method.split("Subject-", 1)[1]
        extras.append(dict(router="Subject", backbone=backbone, variant=row["variant"],
                           subset=row["view"], n=767 if row["view"] == "3OOD_full" else 561,
                           **{name: row[name] for name in ("A_mean", "A_sd", "C_mean", "C_sd", "R_mean", "R_sd")}))
    write_csv(DEST / "subject_ood_clean_SUMMARY.csv", EXTRA_COLUMNS, extras)
    audit["end_time"] = utc_now()
    audit["status"] = "COMPLETE" if len(summary) == 12 and not audit["failures"] else "INCOMPLETE"
    audit["subset_sizes"] = dict(full=767, without_MMMU=561, removed_MMMU=206)
    write_json(DEST / "AUDIT.json", audit)
    log(stage="complete", status=audit["status"], results=len(all_rows), summaries=len(summary))
    if audit["status"] != "COMPLETE":
        raise RuntimeError("E10 incomplete; inspect AUDIT.json")


if __name__ == "__main__":
    main()
