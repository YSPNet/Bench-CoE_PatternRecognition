#!/usr/bin/env python3
"""E3: held-out dataset and seen-task-group breakdown from frozen actions."""
from __future__ import annotations

import json
import os
import traceback
from collections import defaultdict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from threadpoolctl import threadpool_limits

import e5
from common import (OUTPUT, PREVIEW, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from routers.baselines.cheapest_global import CheapestGlobal
from routers.baselines.oracle import Oracle
from routers.baselines.strongest_global import StrongestGlobal
from scripts import run_stateroute_learned_ladder_20261024 as ladder
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source

DEST = OUTPUT / "E3"
DATASET_COLUMNS = ("method", "variant", "seed", "dataset", "n", "A", "C", "R")
GROUP_COLUMNS = ("method", "variant", "seed", "task_group", "n", "A", "C", "R")
SUMMARY_EXTRA = ("method", "variant", "dataset", "n", "n_seeds",
                 "A_mean", "A_sd", "C_mean", "C_sd", "R_mean", "R_sd")
GROUP_SUMMARY = ("method", "variant", "task_group", "n", "n_seeds",
                 "A_mean", "A_sd", "C_mean", "C_sd", "R_mean", "R_sd")
BACKBONES = {"bert": "BERT", "tinyllava": "TinyLLaVA", "qwen3vl": "Qwen3-VL"}


def log(**items):
    print(json.dumps(items, sort_keys=True, default=str), flush=True)


def task_groups(uids):
    remaining = set(uids)
    found = {}
    for path in sorted((ROOT / "BENCHMARKS").glob("*/*_samples.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                uid = str(item["sample_id"])
                if uid in remaining:
                    found[uid] = str(item["task_type"])
                    remaining.remove(uid)
    if remaining:
        raise RuntimeError(f"11ID task-group metadata missing for {len(remaining)} UIDs")
    return np.array([found[uid] for uid in uids])


def selected_e6_actions(method, seed, view, ids, selection):
    protocol = "heldout11"
    task = OUTPUT / "E6/tasks" / protocol / method / f"seed-{seed}"
    frozen = read_json(task / "COMPLETE.json")
    lam = frozen["lambda_val"] if selection == "val" else frozen["lambda_best_test"][view]
    path = task / f"lambda-{lam}" / f"{view}.npz"
    with np.load(path, allow_pickle=False) as pred:
        saved_ids = pred["sample_ids"].tolist()
        actions = pred["actions"].astype(np.int8)
    if len(saved_ids) != len(ids) or set(saved_ids) != set(ids):
        raise RuntimeError(f"E6 {method}/{seed}/{view} test UID coverage changed")
    ordered = actions[[{uid: i for i, uid in enumerate(saved_ids)}[uid] for uid in ids]]
    return ordered, lam


def fixed_actions(data, matrices):
    y, c = data["y"], data["c"]
    strongest, cheapest = StrongestGlobal(), CheapestGlobal()
    strongest.fit(y, c, None)
    cheapest.fit(y, c, None)
    result = {}
    for view in ("11ID", "3OOD"):
        n = len(matrices[view][0])
        result[("Strongest", "source_train", view)] = np.full(n, strongest.best_j, dtype=np.int8)
        result[("Cheapest", "source_train", view)] = np.full(n, cheapest.cheapest_j, dtype=np.int8)
        result[("Oracle", "test_oracle", view)] = Oracle().predict(
            Y_test=matrices[view][0], C_test=matrices[view][1]).astype(np.int8)
    return result


def learned_actions(seed, data, text_train, text_test, mm_test, text_sha, mm_lambda):
    source_mm = np.asarray(data["x"], dtype=np.float32)
    result = {}
    for family, source_x, target_x in (("T", text_train, text_test),
                                        ("MM", source_mm, mm_test)):
        if family == "T":
            source_state, target_state = e5.fit_text_states(
                seed, source_x, data["uids"].tolist(), target_x, text_sha)
        else:
            source_state, target_state = e5.fit_mm_states(seed, source_x, target_x)
        q, global_q, global_c, scaled_c, _ = e5.profile(data["y"], data["c"], source_state)
        if family == "T":
            table = e5.text_table(q, global_c)
            one = int(e5.text_table(global_q[None, :], global_c)[0])
        else:
            table = (q - mm_lambda * scaled_c).argmax(axis=1).astype(np.int8)
            one = int((global_q - mm_lambda * scaled_c).argmax())
        for variant, action in (("K1", np.full(len(target_state), one, dtype=np.int8)),
                                ("K2_fixed_map", table[target_state])):
            result[(f"StateRoute-{family}", variant)] = action
    return result


def frozen_actions(seed, views):
    ids = views["11ID"] + views["3OOD"]
    result = {}
    state_path = ROOT / "outputs/stateroute_mmmu_subject_four_rules_11_3_20260928/PREDICTIONS.npz"
    with np.load(state_path, allow_pickle=False) as pred:
        if pred["uids"].tolist() != ids:
            raise RuntimeError("subject-state frozen UID order changed")
        if pred["methods"].tolist()[3] != "SubjectRule4-CostCap-Guard25":
            raise RuntimeError("subject-state rule 4 index changed")
        result[("Subject-state", "rule4_coarse_fallback")] = pred["actions"][3].copy()
    path = ROOT / "outputs/bench_coe_query_subject_four_rules_20261024/PREDICTIONS.npz"
    with np.load(path, allow_pickle=False) as pred:
        if pred["uids"].tolist() != ids or not set(SEEDS).issubset(set(pred["seeds"].tolist())):
            raise RuntimeError("Query/Subject frozen UID or seed order changed")
        if pred["methods"].tolist() != ["Query", "Subject"] or pred["regimes"].tolist()[0] != "K2-on":
            raise RuntimeError("Query/Subject frozen method or fallback order changed")
        if pred["rules"].tolist()[3] != "4-CostCap-Guard25":
            raise RuntimeError("Query/Subject rule 4 index changed")
        for family in range(2):
            for backbone, name in enumerate(pred["backbones"].tolist()):
                if name not in BACKBONES:
                    raise RuntimeError("Query/Subject backbone changed")
                method = f"{pred['methods'][family]}-{BACKBONES[name]}"
                result[(method, "base")] = pred["baseheld"][family, backbone, seed].copy()
                result[(method, "rule4_coarse_fallback")] = (
                    pred["heldout_actions"][family, backbone, seed, 0, 3].copy())
    return result


def summarize_extra(rows, dimension):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["variant"], row[dimension])].append(row)
    output = []
    for (method, variant, value), members in sorted(grouped.items()):
        if sorted(row["seed"] for row in members) != list(SEEDS):
            continue
        n = {row["n"] for row in members}
        if len(n) != 1:
            raise RuntimeError(f"{dimension} denominator differs across seeds")
        item = {"method": method, "variant": variant, dimension: value,
                "n": n.pop(), "n_seeds": len(members)}
        for name in ("A", "C", "R"):
            values = np.array([row[name] for row in members], dtype=np.float64)
            item[f"{name}_mean"] = float(values.mean())
            item[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        output.append(item)
    return output


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    e6_audit = read_json(OUTPUT / "E6/AUDIT.json")
    allowed = ("complete", "preview_complete") if PREVIEW else ("complete", "acceptance_needs_recheck")
    if e6_audit.get("status") not in allowed:
        raise RuntimeError("E3 extension requires completed E6 native actions")
    if e6_audit.get("status") == "acceptance_needs_recheck":
        recheck = read_json(OUTPUT / "E6/RECHECK_PRELIM.json")
        if not all(item["passed"] for item in recheck["checks"]):
            raise RuntimeError("E6 split/scorer/feature recheck failed")
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit = make_audit("E3", "python -B scripts/journal_20261024/e3.py",
        "Frozen rule actions and E6 source-dev-selected models; E6 best_test is separately labeled "
        "as a test-selected diagnostic. StateRoute K1/K2 use E5 source-only profile and OOF-selected "
        "MM lambda. Oracle is a declared test-outcome reference; no other 3OOD selection.",
        {view: len(ids) for view, ids in views.items()})
    audit["c_min"], audit["c_max"] = bounds["cmin"], bounds["cmax"]
    audit["gpu"] = "cached features; CPU state fits and frozen predictions"
    audit["task_group_source"] = "BENCHMARKS/<task_type>/*_samples.jsonl task_type"
    audit["E6_reproduction_status"] = e6_audit["status"]
    if e6_audit["status"] == "acceptance_needs_recheck":
        audit["E6_caveat"] = ("RouterDC 14-ID published reproduction remains outside the "
                             "two-point runbook window; E3 uses measured source-selected "
                             "E6 actions after the recorded split/scorer/feature recheck.")
    write_json(DEST / "AUDIT.json", audit)
    matrices, _, _ = eval_matrices(views)
    datasets = np.array([uid.split("/", 1)[0] for uid in views["3OOD"]])
    if dict(zip(*np.unique(datasets, return_counts=True))) != {
            "ChartQA_TEST": 510, "MMMU_DEV_VAL": 206, "MathVision_MINI": 51}:
        raise RuntimeError("held-out dataset composition changed")
    groups = task_groups(views["11ID"])
    data = load_source("heldout11")
    text_train = np.asarray(ladder.text_train(data), dtype=np.float32)
    ids = views["11ID"] + views["3OOD"]
    text_test = ladder.text_test(ids)
    mm_test, overlay, _ = subject.qwen_features(ids, allow_multi=True)
    if len(overlay) != 10:
        raise RuntimeError("held-out Qwen image overlay changed")
    text_sha = read_json(ladder.TEXT_METADATA)["feature_artifact_sha256"]
    mm_lambda = read_json(OUTPUT / "E5/base_source_selection.json")["mm_selected_lambda"]
    fixed = fixed_actions(data, matrices)
    standard, per_dataset, per_group = [], [], []
    checks = []
    for seed in SEEDS:
        checkpoint = DEST / f"seed-{seed}.json"
        if checkpoint.is_file():
            saved = read_json(checkpoint)
            standard.extend(saved["standard"])
            per_dataset.extend(saved["per_dataset"])
            per_group.extend(saved["per_group"])
            checks.extend(saved["checks"])
            log(stage="seed_resumed", seed=seed)
            continue
        try:
            methods = {}
            for (method, variant, view), action in fixed.items():
                methods.setdefault((method, variant), {})[view] = action
            combined = learned_actions(seed, data, text_train, text_test, mm_test,
                                       text_sha, mm_lambda)
            combined.update(frozen_actions(seed, views))
            for key, action in combined.items():
                if action.shape != (6108,):
                    raise RuntimeError(f"{key} held-out action shape changed")
                methods[key] = {"11ID": action[:5341], "3OOD": action[5341:]}
            for method in (("routerdc", "vlc", "mlp") if PREVIEW else
                           ("routerdc", "vlc", "mlp", "cosinecls", "zooter")):
                for selection in ("val", "best_test"):
                    variant = f"select={selection}"
                    methods[(method, variant)] = {}
                    for view in ("11ID", "3OOD"):
                        action, _ = selected_e6_actions(method, seed, view, views[view], selection)
                        methods[(method, variant)][view] = action
            action_index = []
            action_arrays = {}
            for index, ((method, variant), by_view) in enumerate(sorted(methods.items())):
                for view in ("11ID", "3OOD"):
                    key = f"action_{index:03d}_{view}"
                    action_arrays[key] = np.asarray(by_view[view], dtype=np.int8)
                    action_index.append(dict(key=key, method=method,
                                             variant=variant, view=view))
            np.savez_compressed(DEST / f"actions_seed-{seed}.npz", **action_arrays)
            one_standard, one_dataset, one_group, one_checks = [], [], [], []
            for (method, variant), by_view in methods.items():
                for view in ("11ID", "3OOD"):
                    action = np.asarray(by_view[view], dtype=np.int8)
                    y, c = matrices[view]
                    A, C, R = scores(y, c, action, bounds)
                    one_standard.append(dict(experiment="E3", method=method, variant=variant,
                        view=view, seed=seed, n_test=len(y), A=A, C=C, R=R,
                        notes="test-selected diagnostic" if variant == "select=best_test" else
                              "frozen/source-only selected actions; Oracle uses test outcomes"))
                    labels = datasets if view == "3OOD" else groups
                    sink = one_dataset if view == "3OOD" else one_group
                    dimension = "dataset" if view == "3OOD" else "task_group"
                    for label in sorted(set(labels)):
                        mask = labels == label
                        a, cost, rank = scores(y[mask], c[mask], action[mask], bounds)
                        sink.append(dict(method=method, variant=variant, seed=seed,
                                         **{dimension: str(label)}, n=int(mask.sum()),
                                         A=a, C=cost, R=rank))
                    if view == "3OOD":
                        matched = [row for row in one_dataset if row["method"] == method and
                                   row["variant"] == variant and row["seed"] == seed]
                        weighted_a = sum(row["A"] * row["n"] for row in matched) / len(y)
                        weighted_c = sum(row["C"] * row["n"] for row in matched) / len(y)
                        one_checks.append(dict(method=method, variant=variant, seed=seed,
                            weighted_A=weighted_a, aggregate_A=A,
                            weighted_C=weighted_c, aggregate_C=C,
                            passed=abs(weighted_a - A) <= .02 and abs(weighted_c - C) <= .02))
            write_json(checkpoint, dict(seed=seed, standard=one_standard,
                per_dataset=one_dataset, per_group=one_group, checks=one_checks,
                action_index=action_index))
            standard.extend(one_standard)
            per_dataset.extend(one_dataset)
            per_group.extend(one_group)
            checks.extend(one_checks)
            log(stage="seed_complete", seed=seed, methods=len(methods))
        except Exception as exc:
            trace = DEST / f"seed-{seed}-traceback.txt"
            trace.write_text(traceback.format_exc())
            audit["failures"].append(dict(id="E3", seed=seed, error=str(exc),
                                          traceback_path=str(trace.relative_to(ROOT))))
            write_json(DEST / "AUDIT.json", audit)
            log(stage="seed_failed", seed=seed, error=str(exc))
    write_csv(DEST / "RESULTS.csv", RESULT_COLUMNS, standard)
    write_csv(DEST / "SUMMARY.csv", SUMMARY_COLUMNS, summarize(standard))
    write_csv(DEST / "per_dataset_ood.csv", DATASET_COLUMNS, per_dataset)
    write_csv(DEST / "per_dataset_ood_SUMMARY.csv", SUMMARY_EXTRA,
              summarize_extra(per_dataset, "dataset"))
    write_csv(DEST / "per_group_id.csv", GROUP_COLUMNS, per_group)
    write_csv(DEST / "per_group_id_SUMMARY.csv", GROUP_SUMMARY,
              summarize_extra(per_group, "task_group"))
    audit["acceptance"] = checks
    audit["status"] = ("preview_complete" if PREVIEW else "complete") if (
        not audit["failures"] and checks and all(item["passed"] for item in checks)) else "acceptance_failed"
    audit["end_time"] = utc_now()
    write_json(DEST / "AUDIT.json", audit)
    if audit["status"] not in ("complete", "preview_complete"):
        raise RuntimeError("E3 acceptance failed")
    with (OUTPUT / "PROGRESS.md").open("a") as handle:
        handle.write("\n## E3 — complete, extended with E6\n\nPer-dataset 3OOD and "
                     "per-task-group 11ID results include E6 source-dev and diagnostic best-test "
                     "selections. Weighted dataset accuracy and cost reproduce aggregate 3OOD.\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
