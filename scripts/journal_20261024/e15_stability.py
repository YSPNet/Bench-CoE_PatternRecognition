#!/usr/bin/env python3
"""Manuscript Table 9: half-profile map stability for frozen partitions.

Five frozen partition seeds are crossed with ten disjoint random group halves.
The test features determine state assignment only; test outcomes are not read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from threadpoolctl import threadpool_limits

import e12_matched as matched
from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    make_audit, read_json, utc_now, view_uids, write_csv, write_json)
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source

OUT = OUTPUT / "E15"
PARTITIONS = ("KMeans2", "KMeans4", "KMeans12", "KMeans30", "MMMU30")
RULES = ("argmax_k0_l0", "penalty_k64_l003")
SPLITS = tuple(range(10))
LAMBDA = .03  # frozen K2 source-only OOF choice, held fixed across granularity
TRIAL_COLUMNS = ("partition", "K", "rule", "seed", "split", "n_source_fit",
                 "half_a_n", "half_b_n", "state_agreement", "route_agreement_11ID",
                 "route_agreement_3OOD", "empty_states_a", "empty_states_b")
SUMMARY_SPECIAL_COLUMNS = ("partition", "K", "rule", "n_trials",
    "state_agreement_mean", "state_agreement_sd", "route_agreement_11ID_mean",
    "route_agreement_11ID_sd", "route_agreement_3OOD_mean", "route_agreement_3OOD_sd")


def path_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def centers_path(partition, seed):
    if partition == "KMeans2":
        return ROOT / f"outputs/stateroute_k_sweep_20260916/models/MM/k2/seed-{seed}/model.npz"
    if partition == "KMeans4":
        return ROOT / f"outputs/stateroute_k_sweep_20260916/models/MM/k4/seed-{seed}/model.npz"
    if partition == "KMeans12":
        return ROOT / f"outputs/stateroute_k10_k12_20260916/models/MM/k12/seed-{seed}/model.npz"
    if partition == "KMeans30":
        return matched.K30 / "models" / f"seed-{seed}.npz"
    return subject.OUT / "PROTOTYPES.npz"


def group_halves(groups, seed, split):
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(10000 + 101 * seed + split)
    perm = rng.permutation(len(unique))
    allocation = np.zeros(len(unique), dtype=np.int8)
    sizes = [0, 0]
    for group in perm:
        side = int(sizes[1] < sizes[0])
        allocation[group] = side
        sizes[side] += int(counts[group])
    a = allocation[inverse] == 0
    b = ~a
    if a.sum() == 0 or b.sum() == 0 or set(groups[a]) & set(groups[b]):
        raise RuntimeError("source profiling halves are not group disjoint")
    return a, b


def map_from_half(y, c, state, K, rule):
    n = np.bincount(state, minlength=K).astype(np.float64)
    sums = np.zeros((K, 17), dtype=np.float64)
    np.add.at(sums, state, y)
    global_q = y.mean(axis=0, dtype=np.float64)
    if rule == "argmax_k0_l0":
        q = np.divide(sums, n[:, None], out=np.broadcast_to(global_q, (K,17)).copy(),
                      where=n[:, None] > 0)
        return q.argmax(axis=1).astype(np.int8), int((n == 0).sum())
    q = (sums + 64.0 * global_q) / (n[:, None] + 64.0)
    scaled = c.mean(axis=0, dtype=np.float64) / float(np.median(c[c > 0]))
    return (q - LAMBDA * scaled).argmax(axis=1).astype(np.int8), int((n == 0).sum())


def summary(rows):
    groups = {}
    for row in rows:
        key = row["partition"], row["K"], row["rule"]
        groups.setdefault(key, []).append(row)
    out = []
    for (partition, K, rule), values in sorted(groups.items()):
        item = dict(partition=partition, K=K, rule=rule, n_trials=len(values))
        for name in ("state_agreement", "route_agreement_11ID", "route_agreement_3OOD"):
            vec = np.asarray([v[name] for v in values], dtype=np.float64)
            item[name + "_mean"] = float(vec.mean())
            item[name + "_sd"] = float(vec.std(ddof=1))
        out.append(item)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    source = load_source("heldout11")
    fit = np.flatnonzero(source["role"] == "fit")
    if len(fit) != 16269:
        raise RuntimeError("11-source fit size changed")
    groups = source["groups"][fit]
    y, c = source["y"][fit], source["c"][fit]
    x = source["x"][fit]
    held_uids = views["11ID"] + views["3OOD"]
    x_test, overlay, _ = subject.qwen_features(held_uids, allow_multi=True)
    if len(overlay) != 10:
        raise RuntimeError("held-out test Qwen overlay changed")
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit = make_audit("E15", "python -B scripts/journal_20261024/e15_stability.py",
        "Frozen 11-source-fit partitions, seeds 0-4, crossed with ten group-disjoint "
        "random half-profiles per seed. Both argmax kappa=0 and a matched kappa=64 "
        "global penalty at source-selected K2 lambda=0.03 are reported. "
        "No test outcomes or costs are read. The per-query classifier row is a "
        "separate GPU experiment still required to complete manuscript Table 9.",
        {view: len(uids) for view, uids in views.items()})
    audit.update(c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu="cached Qwen3-VL features; CPU diagnostics", status="running",
                 split_repeats=list(SPLITS), lambda_cost=LAMBDA)
    write_json(OUT / "AUDIT.json", audit)
    try:
        model_hashes = {}
        rows = []
        for seed in SEEDS:
            checkpoint = OUT / f"seed-{seed}.json"
            if checkpoint.is_file():
                old = read_json(checkpoint)
                rows.extend(old["rows"])
                model_hashes.update(old["model_sha256"])
                continue
            seed_rows = []
            seed_hashes = {}
            for partition in PARTITIONS:
                path = centers_path(partition, seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path, allow_pickle=False) as model:
                    centers = np.asarray(model["centers"], dtype=np.float32)
                K = len(centers)
                if K != {"KMeans2": 2, "KMeans4": 4, "KMeans12": 12,
                         "KMeans30": 30, "MMMU30": 30}[partition]:
                    raise RuntimeError(f"frozen K mismatch: {path}")
                source_state = matched.assign(x, centers, partition)
                test_state = matched.assign(x_test, centers, partition)
                n_state = np.bincount(source_state, minlength=K)
                seed_hashes[f"{partition}/seed-{seed}"] = path_hash(path)
                for split in SPLITS:
                    a, b = group_halves(groups, seed, split)
                    for rule in RULES:
                        map_a, empty_a = map_from_half(y[a], c[a], source_state[a], K, rule)
                        map_b, empty_b = map_from_half(y[b], c[b], source_state[b], K, rule)
                        same = map_a == map_b
                        state_agree = float(n_state[same].sum() / n_state.sum())
                        route_agree = map_a[test_state] == map_b[test_state]
                        seed_rows.append(dict(partition=partition, K=K, rule=rule,
                            seed=seed, split=split, n_source_fit=len(fit), half_a_n=int(a.sum()),
                            half_b_n=int(b.sum()), state_agreement=state_agree,
                            route_agreement_11ID=float(route_agree[:5341].mean()),
                            route_agreement_3OOD=float(route_agree[5341:].mean()),
                            empty_states_a=empty_a, empty_states_b=empty_b))
                print(json.dumps(dict(stage="partition_complete", partition=partition,
                                      seed=seed)), flush=True)
            write_json(checkpoint, dict(seed=seed, rows=seed_rows, model_sha256=seed_hashes))
            rows.extend(seed_rows)
            model_hashes.update(seed_hashes)
        write_csv(OUT / "map_stability_trials.csv", TRIAL_COLUMNS, rows)
        values = summary(rows)
        write_csv(OUT / "map_stability_SUMMARY.csv", SUMMARY_SPECIAL_COLUMNS, values)
        # The runbook A/C/R schema is inapplicable to agreement diagnostics.
        # Keep its required filenames as empty, accurately typed tables.
        write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, [])
        write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, [])
        expected = len(PARTITIONS) * len(RULES) * len(SEEDS) * len(SPLITS)
        passed = len(rows) == expected and all(
            0 <= row[key] <= 1 for row in rows for key in
            ("state_agreement", "route_agreement_11ID", "route_agreement_3OOD"))
        audit["acceptance"] = dict(expected_rows=expected, observed_rows=len(rows),
                                    fractions_bounded=passed)
        audit["model_sha256"] = model_hashes
        audit["status"] = "awaiting_per_query_row" if passed else "acceptance_failed"
        if not passed:
            raise RuntimeError("E15 stability row coverage/fraction check failed")
    except Exception as exc:
        trace = OUT / "traceback.txt"
        trace.write_text(traceback.format_exc())
        audit["failures"].append(dict(id="E15", seed=None, error=str(exc),
                                      traceback_path=str(trace.relative_to(ROOT))))
        if audit.get("status") == "running":
            audit["status"] = "failed"
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
