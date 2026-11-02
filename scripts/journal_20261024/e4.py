#!/usr/bin/env python3
"""Run E4 under the 2026-10-24 journal runbook, with one checkpoint per seed."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from threadpoolctl import threadpool_limits

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from scripts import run_stateroute_mmmu_full14_20260928 as mm14
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source, manifests
from src.next_round_x.models import cluster_quality, normalized
from src.next_round_x.runtime import digest
from src.subject_routing.clustering.spherical_kmeans import SphericalKMeans, assign_frozen
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit

DEST = OUTPUT / "E4"
ARCHIVE = Path("/home/sm5/ys/ICASSP/archive/non_paper_research_20261011")
TEXT = ARCHIVE / "artifacts/subject_routing/id14/posthoc_v1"
MM = ARCHIVE / "outputs/stateroute_mmmu_full14_20260928"
Q_GRID = (0.25, 0.35, 0.5, 0.65, 0.75, 1.0)
RULE_VARIANTS = ("subject_rule1_gate_fpr10", "subject_rule2_gate_fpr10",
                 "subject_rule3_gate_fpr10", "subject_rule4_gate_fpr25")


def log(**message):
    print(json.dumps(message, sort_keys=True, default=str), flush=True)


def text_features(data, views):
    meta = read_json(TEXT / "features/metadata.json")
    current_split = manifests()[0]["environments"]["id14"]["roles"]["train"]
    if meta["roles"]["train"]["ordered_uid_sha256"] != current_split["ordered_uid_sha256"]:
        raise RuntimeError("archived MiniLM train UID order differs from current full14 source")
    train_path = TEXT / "features/minilm_train.npy"
    if digest(train_path) != meta["roles"]["train"]["file_sha256"]:
        raise RuntimeError("archived MiniLM train feature hash changed")
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    if train.shape != (len(data["uids"]), 384):
        raise RuntimeError("full14 MiniLM train feature dimensions changed")
    from scripts.run_stateroute_learned_ladder_20261024 import text_test
    test = text_test(views["14ID"])
    return np.asarray(train, dtype=np.float32), test, meta["feature_artifact_sha256"]


def text_states(x_train, x_valid, uids, seed, sha):
    model = SphericalKMeans(2, seed=seed, n_init=10, max_iter=100, tolerance=1e-6).fit(
        x_train, uids, environment_id="id14", fit_role="train",
        feature_artifact_sha256=sha,
        audit=UIDAccessAudit.create(Operation.FIT, uids))
    source = np.array([model.cluster_ids.index(s) for s in model.assignments], dtype=np.int8)
    assigned = assign_frozen(model, x_valid, feature_artifact_sha256=sha).states
    target = np.array([model.cluster_ids.index(s) for s in assigned], dtype=np.int8)
    return source, target


def quality(y, state):
    count = np.bincount(state, minlength=2)
    global_quality = y.mean(axis=0, dtype=np.float64)
    sums = np.zeros((2, y.shape[1]), dtype=np.float64)
    np.add.at(sums, state, y)
    return (sums + 20.0 * global_quality) / (count[:, None] + 20.0)


def text_table(q, y, c, state):
    cost = c.mean(axis=0, dtype=np.float64)
    cap = float(np.quantile(cost, q))
    eligible = cost <= cap + 1e-15
    estimate = quality(y, state)
    # Equal quality estimates prefer lower source mean cost and then canonical ID.
    table = np.array([min(np.flatnonzero(eligible), key=lambda j: (-estimate[s, j], cost[j], j))
                      for s in range(2)], dtype=np.int8)
    return table, cap


def fit_text(seed, data, fit, x_train, x_test, sha, bounds):
    position = {int(row): i for i, row in enumerate(fit)}
    oof_actions = {q: np.empty(len(fit), dtype=np.int8) for q in Q_GRID}
    for fold in range(3):
        tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][tr]) & set(data["groups"][va]):
            raise RuntimeError("text full14 OOF dependency groups overlap")
        src, dst = text_states(x_train[tr], x_train[va], data["uids"][tr].tolist(), seed, sha)
        target = np.array([position[int(i)] for i in va])
        for q in Q_GRID:
            table, _ = text_table(q, data["y"][tr], data["c"][tr], src)
            oof_actions[q][target] = table[dst]
        log(stage="text_oof_fold", seed=seed, fold=fold, train=len(tr), valid=len(va))
    oof = []
    for q in Q_GRID:
        A, C, R = scores(data["y"][fit], data["c"][fit], oof_actions[q], bounds)
        oof.append(dict(q=q, A=A, C=C, R=R))
    selected = max(oof, key=lambda row: (row["R"], row["A"], -row["C"], -row["q"]))
    src, dst = text_states(x_train[fit], x_test, data["uids"][fit].tolist(), seed, sha)
    table, cap = text_table(selected["q"], data["y"][fit], data["c"][fit], src)
    log(stage="text_fit", seed=seed, q=selected["q"], table=table.tolist())
    return table[dst], dict(selected_q=selected["q"], selected_cap=cap,
                            source_oof=oof, table=table.tolist(), state_support=np.bincount(src, minlength=2).tolist())


def mm_lambda(data, fit, bounds):
    with np.load(MM / "oof/StateRoute-MM-K2.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["uids"], data["uids"][fit]):
            raise RuntimeError("archived MM OOF source UID alignment changed")
        q = archive["quality"].copy()
        c = archive["scaled_cost"].copy()
    rows = []
    for lam in mm14.LAMBDAS:
        A, C, R = scores(data["y"][fit], data["c"][fit], (q - lam * c).argmax(axis=1), bounds)
        rows.append(dict(lambda_cost=lam, A=A, C=C, R=R))
    chosen = max(rows, key=lambda row: row["R"])
    return chosen["lambda_cost"], rows


def fit_mm(seed, data, fit, train_x, test_x, lam):
    _, km, rates = cluster_quality(train_x[fit], data["y"][fit], train_x[fit[:1]], 2, seed)
    scaled_cost = data["c"][fit].mean(axis=0) / np.median(data["c"][fit][data["c"][fit] > 0])
    table = (rates - lam * scaled_cost).argmax(axis=1).astype(np.int8)
    action = table[km.predict(normalized(test_x))]
    log(stage="mm_fit", seed=seed, lam=lam, table=table.tolist())
    return action, dict(selected_lambda=lam, table=table.tolist(), train_overlay=32)


def result(method, variant, view, seed, action, matrices, bounds, notes):
    y, c = matrices[view]
    A, C, R = scores(y, c, action, bounds)
    return dict(experiment="E4", method=method, variant=variant, view=view,
                seed=seed, n_test=len(y), A=A, C=C, R=R, notes=notes)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit = make_audit("E4", "python -B scripts/journal_20261024/e4.py",
                       "Subject rules: frozen MODEL_LOCK.json; MM lambda: source-fit group-isolated 3-fold OOF; T eligibility quantile: source-fit group-isolated 3-fold OOF Rank over preset q grid; no 3OOD selection",
                       {v: len(u) for v, u in views.items()})
    audit["c_min"], audit["c_max"] = bounds["cmin"], bounds["cmax"]
    audit["gpu"] = "cached features; CPU router fits"
    write_json(DEST / "AUDIT.json", audit)
    matrices, _, _ = eval_matrices(views)
    data, mm_train_x, fit, train_overlay = mm14.source_train()
    if len(fit) != 17871:
        raise RuntimeError("full14 source-fit size changed")
    if len(train_overlay) != 32:
        raise RuntimeError("full14 source multi-image overlay changed")
    x_train, x_test, text_sha = text_features(data, views)
    test_x, overlay, _ = subject.qwen_features(views["14ID"], allow_multi=True)
    if len(overlay) != 10:
        raise RuntimeError("MM test multi-image overlay changed")
    lam, mm_oof = mm_lambda(data, fit, bounds)
    write_json(DEST / "source_selection.json", dict(mm_lambda=lam, mm_oof=mm_oof,
                 text_q_grid=list(Q_GRID), mm_oof_source=str(MM / "oof/StateRoute-MM-K2.npz")))
    held_path = ROOT / "outputs/stateroute_mmmu_subject_four_rules_11_3_20260928/PREDICTIONS.npz"
    full_path = ROOT / "outputs/stateroute_mmmu_subject_four_rules_full14_20260928/PREDICTIONS.npz"
    with np.load(held_path, allow_pickle=False) as h, np.load(full_path, allow_pickle=False) as f:
        held_uids = views["11ID"] + views["3OOD"]
        if h["uids"].tolist() != held_uids or f["uids"].tolist() != views["14ID"]:
            raise RuntimeError("frozen subject prediction UID alignment changed")
        if h["methods"].tolist() != ["SubjectRule1-RankSafe-Guard10", "SubjectRule2-Accuracy-Guard10",
                                      "SubjectRule3-CostCap-Guard10", "SubjectRule4-CostCap-Guard25"]:
            raise RuntimeError("frozen subject rule order changed")
        h_actions, f_action = h["actions"].copy(), f["actions"][0].copy()
    lock = read_json(ROOT / "outputs/stateroute_mmmu_subject_four_rules_11_3_20260928/MODEL_LOCK.json")
    audit["frozen_subject_rules"] = lock["rules"]
    all_rows = []
    for seed in SEEDS:
        checkpoint = DEST / f"seed-{seed}.json"
        if checkpoint.exists():
            value = read_json(checkpoint)
            if value.get("mm", {}).get("train_overlay") != 32:
                mm_action, mm_info = fit_mm(seed, data, fit, mm_train_x, test_x, lam)
                value["rows"] = [r for r in value["rows"] if r["variant"] != "stateroute_MM_K2_14ID"]
                value["rows"].append(result("StateRoute-MM", "stateroute_MM_K2_14ID", "14ID", seed,
                                            mm_action, matrices, bounds,
                                            "New 14-source fit with 32-image overlay; lambda selected by source-fit 3-fold OOF"))
                value["mm"] = mm_info
                write_json(checkpoint, value)
                log(stage="seed_mm_overlay_corrected", seed=seed)
            all_rows.extend(value["rows"])
            log(stage="seed_resumed", seed=seed, rows=len(value["rows"]))
            continue
        try:
            rows = []
            for i, variant in enumerate(RULE_VARIANTS):
                for view, sl in (("11ID", slice(0, 5341)), ("3OOD", slice(5341, 6108))):
                    rows.append(result("Subject-state", variant, view, seed, h_actions[i, sl], matrices,
                                       bounds, "Frozen deterministic subject fit and thresholds; seed-independent replay"))
            rows.append(result("Subject-state", "subject_accuracy14", "14ID", seed, f_action,
                               matrices, bounds, "Frozen deterministic subject fit; seed-independent replay"))
            text_action, text_info = fit_text(seed, data, fit, x_train, x_test, text_sha, bounds)
            rows.append(result("StateRoute-T", "stateroute_T_K2_14ID", "14ID", seed, text_action,
                               matrices, bounds, "New 14-source fit; q selected by source-fit 3-fold OOF"))
            mm_action, mm_info = fit_mm(seed, data, fit, mm_train_x, test_x, lam)
            rows.append(result("StateRoute-MM", "stateroute_MM_K2_14ID", "14ID", seed, mm_action,
                               matrices, bounds, "New 14-source fit; lambda selected by source-fit 3-fold OOF"))
            write_json(checkpoint, dict(seed=seed, rows=rows, text=text_info, mm=mm_info))
            all_rows.extend(rows)
            log(stage="seed_complete", seed=seed, text_q=text_info["selected_q"], mm_lambda=lam)
        except Exception as exc:
            path = DEST / f"seed-{seed}-traceback.txt"
            path.write_text(traceback.format_exc())
            audit["failures"].append(dict(id="E4", seed=seed, error=str(exc), traceback_path=str(path)))
            write_json(DEST / "AUDIT.json", audit)
            log(stage="seed_failed", seed=seed, error=str(exc))
    write_csv(DEST / "RESULTS.csv", RESULT_COLUMNS, all_rows)
    summary = summarize(all_rows)
    write_csv(DEST / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
    checks = []
    expected = [("subject_rule4_gate_fpr25", "11ID", 0, (77.289, .8716, 75.052)),
                ("subject_rule4_gate_fpr25", "3OOD", 0, (72.751, .7063, 71.753)),
                ("subject_accuracy14", "14ID", 0, (78.389, 1.0984, 75.085))]
    for variant, view, seed, target in expected:
        matched = [r for r in all_rows if (r["variant"], r["view"], r["seed"]) == (variant, view, seed)]
        checks.append(dict(variant=variant, view=view, seed=seed, target=target,
                           observed=tuple(matched[0][k] for k in ("A", "C", "R")) if matched else None,
                           pass_check=bool(matched and all(abs(matched[0][k] - t) <= .005 for k, t in zip(("A", "C", "R"), target)))))
    mm_summary = [r for r in summary if r["variant"] == "stateroute_MM_K2_14ID"]
    checks.append(dict(variant="stateroute_MM_K2_14ID", view="14ID", seed="mean",
                       target_R=73.900, observed_R=mm_summary[0]["R_mean"] if mm_summary else None,
                       pass_check=bool(mm_summary and abs(mm_summary[0]["R_mean"] - 73.900) <= .044)))
    audit["acceptance"] = checks
    audit["end_time"] = utc_now()
    write_json(DEST / "AUDIT.json", audit)
    log(stage="acceptance", checks=checks)
    if not all(check["pass_check"] for check in checks):
        raise RuntimeError("E4 acceptance failed; stop before E5")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
