#!/usr/bin/env python3
"""Manuscript Table 7: K=2 shrinkage sweep and Proposition 1 diagnostic."""
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
from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from scripts import run_stateroute_learned_ladder_20261024 as ladder
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source

OUT = OUTPUT / "E13"
KAPPAS = (0., 5., 20., 64., 128., 512.)


def profile(y, state, kappa):
    n = np.bincount(state, minlength=2).astype(np.float64)
    successes = np.zeros((2, 17), dtype=np.float64)
    np.add.at(successes, state, y)
    global_q = y.mean(axis=0, dtype=np.float64)
    if kappa == 0:
        q = np.divide(successes, n[:, None], out=np.broadcast_to(global_q, (2, 17)).copy(),
                      where=n[:, None] > 0)
    else:
        q = (successes + kappa * global_q) / (n[:, None] + kappa)
    return q, n, global_q


def proposition_fraction(full_y, full_state, n_profile, q_global, kappa):
    if kappa == 0:
        return None
    q_full, n_full, _ = profile(full_y, full_state, 0)
    if np.any(n_full == 0) or np.any(n_profile == 0):
        raise RuntimeError("empty K2 state in Proposition 1 estimate")
    rhs = (2.0 / kappa + 1.0 / n_profile[:, None]) * q_full * (1.0 - q_full)
    condition = np.square(q_global[None, :] - q_full) < rhs
    weights = np.broadcast_to(n_profile[:, None], condition.shape)
    return float(weights[condition].sum() / weights.sum())


def text_features(protocol, data, views):
    if protocol == "full14":
        train, test, sha = e4.text_features(data, views)
    else:
        train = np.asarray(ladder.text_train(data), dtype=np.float32)
        test = ladder.text_test(views["11ID"] + views["3OOD"])
        sha = read_json(ladder.TEXT_METADATA)["feature_artifact_sha256"]
    return train, test, sha


def result(base, kappa, view, seed, action, matrices, bounds):
    A, C, R = scores(*matrices[view], action, bounds)
    return dict(experiment="E13", method=f"StateRoute-{base}",
                variant=f"K2/kappa={kappa:g}", view=view, seed=seed,
                n_test=len(matrices[view][0]), A=A, C=C, R=R,
                notes="T: E4/E5 source profile, median-cost filter; MM: source-fit, frozen OOF lambda")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    lambdas = matched.selected_lambdas()
    audit = make_audit("E13", "python -B scripts/journal_20261024/e13_shrinkage.py",
        "K=2 states and each encoder's published cost rule are fixed. "
        "T profiles the 14-source fit subset on 14ID, as in E4, and all official "
        "11-source train rows on 11ID/3OOD, as in E5. MM profiles source-fit. "
        "MM lambdas are frozen source-only OOF selections; T filter uses source median. "
        "Proposition 1 uses a plug-in Q estimate from the entire source train. "
        "Kappa=0 has no strict shrinkage condition and its fraction is undefined.",
        {view: len(uids) for view, uids in views.items()})
    audit.update(c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu="cached representations; CPU spherical k-means", status="running",
                 kappa_grid=list(KAPPAS))
    if (OUT / "AUDIT.json").is_file():
        prior = read_json(OUT / "AUDIT.json")
        audit["prior_attempts"] = prior.get("prior_attempts", []) + [dict(
            start_time=prior.get("start_time"), end_time=prior.get("end_time"),
            status=prior.get("status"), acceptance=prior.get("acceptance"),
            failures=prior.get("failures", []),
            correction="14ID T must use E4 source-fit subset, not full train")]
    write_json(OUT / "AUDIT.json", audit)
    try:
        data = {protocol: load_source(protocol) for protocol in ("full14", "heldout11")}
        t = {protocol: text_features(protocol, data[protocol], views) for protocol in data}
        x14, overlay, _ = subject.qwen_features(views["14ID"], allow_multi=True)
        if len(overlay) != 10:
            raise RuntimeError("test Qwen feature overlay changed")
        positions = {uid: i for i, uid in enumerate(views["14ID"])}
        xheld = x14[[positions[uid] for uid in views["11ID"] + views["3OOD"]]]
        mm_test = {"full14": x14, "heldout11": xheld}
        mm_train = {protocol: subject.qwen_features(data[protocol]["uids"].tolist(),
                   allow_multi=True)[0] for protocol in data}
        matrices, _, _ = eval_matrices(views)
        all_rows, all_fractions = [], []
        for seed in SEEDS:
            checkpoint = OUT / f"seed-{seed}-corrected.json"
            if checkpoint.is_file():
                old = read_json(checkpoint)
                all_rows.extend(old["rows"])
                all_fractions.extend(old["fractions"])
                print(json.dumps(dict(stage="seed_resumed", seed=seed)), flush=True)
                continue
            seed_rows, seed_fractions = [], []
            try:
                for protocol in ("full14", "heldout11"):
                    source = data[protocol]
                    train_t, test_t, sha = t[protocol]
                    if protocol == "full14":
                        fit_t = np.flatnonzero(source["role"] == "fit")
                        combined = np.concatenate([test_t, train_t], axis=0)
                        source_state_t, combined_state = e4.text_states(
                            train_t[fit_t], combined, source["uids"][fit_t].tolist(), seed, sha)
                        test_state_t = combined_state[:len(test_t)]
                        all_state_t = combined_state[len(test_t):]
                        profile_y_t, profile_c_t = source["y"][fit_t], source["c"][fit_t]
                        views_sl = (("14ID", slice(None)),)
                    else:
                        source_state_t, test_state_t = e5.fit_text_states(
                            seed, train_t, source["uids"].tolist(), test_t, sha)
                        all_state_t = source_state_t
                        profile_y_t, profile_c_t = source["y"], source["c"]
                        views_sl = (("11ID", slice(0, 5341)), ("3OOD", slice(5341, 6108)))
                    q_global_t = profile_y_t.mean(axis=0, dtype=np.float64)
                    c_global_t = profile_c_t.mean(axis=0, dtype=np.float64)
                    cap = float(np.quantile(c_global_t, .5))
                    eligible = np.flatnonzero(c_global_t <= cap + 1e-15)
                    for kappa in KAPPAS:
                        q, support, _ = profile(profile_y_t, source_state_t, kappa)
                        table = np.array([min(eligible, key=lambda j: (-row[j], c_global_t[j], j))
                                          for row in q], dtype=np.int8)
                        action = table[test_state_t]
                        for view, sl in views_sl:
                            seed_rows.append(result("T", kappa, view, seed, action[sl], matrices, bounds))
                        seed_fractions.append(dict(encoder="T", protocol=protocol, seed=seed,
                            kappa=kappa, fraction=proposition_fraction(source["y"], all_state_t,
                                                support, q_global_t, kappa),
                            source_support=support.astype(int).tolist()))
                    print(json.dumps(dict(stage="T_complete", protocol=protocol, seed=seed)), flush=True)

                    fit = np.flatnonzero(source["role"] == "fit")
                    path = matched.model_path(protocol, "KMeans2", seed)
                    with np.load(path, allow_pickle=False) as model:
                        centers = np.asarray(model["centers"], dtype=np.float32)
                        frozen_rates = np.asarray(model["rates"], dtype=np.float64)
                        frozen_support = np.asarray(model["support"], dtype=np.int64)
                    source_state_mm = matched.assign(mm_train[protocol][fit], centers, "KMeans2")
                    all_state_mm = matched.assign(mm_train[protocol], centers, "KMeans2")
                    test_state_mm = matched.assign(mm_test[protocol], centers, "KMeans2")
                    fit_y, fit_c = source["y"][fit], source["c"][fit]
                    q_global_mm = fit_y.mean(axis=0, dtype=np.float64)
                    q_check, n_check, _ = profile(fit_y, source_state_mm, 64.)
                    if not np.array_equal(n_check.astype(np.int64), frozen_support) or not np.allclose(q_check, frozen_rates, atol=1e-6, rtol=0):
                        raise RuntimeError(f"MM source-fit profile differs from frozen model: {path}")
                    scaled_cost = fit_c.mean(axis=0, dtype=np.float64) / float(np.median(fit_c[fit_c > 0]))
                    lam = lambdas[protocol, "KMeans2"]
                    for kappa in KAPPAS:
                        q, support, _ = profile(fit_y, source_state_mm, kappa)
                        table = (q - lam * scaled_cost).argmax(axis=1).astype(np.int8)
                        action = table[test_state_mm]
                        for view, sl in views_sl:
                            seed_rows.append(result("MM", kappa, view, seed, action[sl], matrices, bounds))
                        seed_fractions.append(dict(encoder="MM", protocol=protocol, seed=seed,
                            kappa=kappa, fraction=proposition_fraction(source["y"], all_state_mm,
                                                support, q_global_mm, kappa),
                            source_support=support.astype(int).tolist()))
                    print(json.dumps(dict(stage="MM_complete", protocol=protocol, seed=seed)), flush=True)
                write_json(checkpoint, dict(seed=seed, rows=seed_rows, fractions=seed_fractions))
                all_rows.extend(seed_rows)
                all_fractions.extend(seed_fractions)
            except Exception as exc:
                trace = OUT / f"seed-{seed}-traceback.txt"
                trace.write_text(traceback.format_exc())
                audit["failures"].append(dict(id="E13", seed=seed, error=str(exc),
                                              traceback_path=str(trace.relative_to(ROOT))))
                write_json(OUT / "AUDIT.json", audit)
                print(json.dumps(dict(stage="seed_failed", seed=seed, error=str(exc)), flush=True))
        write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, all_rows)
        summary = summarize(all_rows)
        write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
        write_csv(OUT / "prop1_fraction.csv",
                  ("encoder", "protocol", "seed", "kappa", "fraction", "source_support"),
                  [{**row, "source_support": json.dumps(row["source_support"])} for row in all_fractions])
        checks = []
        anchors = {("T", "14ID"): (20., 74.137, .04),
                   ("T", "11ID"): (20., 74.414, .04),
                   ("T", "3OOD"): (20., 73.235, .01),
                   ("MM", "14ID"): (64., 73.900, .05),
                   ("MM", "11ID"): (64., 74.074, .08),
                   ("MM", "3OOD"): (64., 72.837, .28)}
        for (base, view), (kappa, expected, tolerance) in anchors.items():
            found = [row for row in summary if row["method"] == f"StateRoute-{base}"
                     and row["variant"] == f"K2/kappa={kappa:g}" and row["view"] == view]
            observed = found[0]["R_mean"] if found else None
            checks.append(dict(base=base, view=view, kappa=kappa, expected_R=expected,
                observed_R=observed, tolerance=tolerance,
                passed=bool(observed is not None and abs(observed - expected) <= tolerance)))
        audit["acceptance"] = checks
        audit["status"] = "complete" if all(c["passed"] for c in checks) and not audit["failures"] else "acceptance_failed"
        if audit["status"] != "complete":
            raise RuntimeError("E13 reference or seed acceptance failed")
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
