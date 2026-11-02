#!/usr/bin/env python3
"""Manuscript Table 6: matched partitions under two identical map rules.

The frozen models all profile the same source-fit subset within a protocol.
Their stored kappa=64 rates and state supports also determine the unshrunk
rates exactly. Lambda choices were frozen by source-only OOF in the original
model locks; this script never selects a setting from test outcomes.
"""
from __future__ import annotations

import hashlib
import json
import os
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
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source
from src.next_round_x.models import normalized

OUT = OUTPUT / "E12"
ARCHIVE = Path("/home/sm5/ys/ICASSP/archive/non_paper_research_20261011")
FULL = ARCHIVE / "outputs/stateroute_mmmu_full14_20260928"
K30 = ARCHIVE / "outputs/stateroute_mmmu_k30_control_20260928"
HELD_K2 = ROOT / "outputs/stateroute_k_sweep_20260916/models/MM/k2"
PARTITIONS = ("MMMU30", "KMeans30", "KMeans2")


def model_path(protocol: str, partition: str, seed: int) -> Path:
    if protocol == "full14":
        name = {"MMMU30": "MMMU30-Cosine", "KMeans30": "StateRoute-MM-K30",
                "KMeans2": "StateRoute-MM-K2"}[partition]
        suffix = "model.npz" if partition == "MMMU30" else f"seed-{seed}.npz"
        return FULL / "models" / name / suffix
    if partition == "MMMU30":
        return subject.OUT / "MODEL.npz"
    if partition == "KMeans30":
        return K30 / "models" / f"seed-{seed}.npz"
    return HELD_K2 / f"seed-{seed}" / "model.npz"


def selected_lambdas() -> dict[tuple[str, str], float]:
    full = read_json(FULL / "SOURCE_SELECTION.json")["selected"]
    held_subject = read_json(subject.OUT / "SELECTION.json")["selected"]["lambda_cost"]
    held_k30 = read_json(K30 / "SELECTION.json")["oof_lambda"]["lambda_cost"]
    held_k2 = read_json(ROOT / "outputs/stateroute_k_sweep_20260916/MM_SOURCE_SELECTION.json")
    k2 = next(row["lambda_cost"] for row in held_k2["selections"] if row["k"] == 2)
    return {
        ("full14", "MMMU30"): float(full["MMMU30-Cosine"]["lambda_cost"]),
        ("full14", "KMeans30"): float(full["StateRoute-MM-K30"]["lambda_cost"]),
        ("full14", "KMeans2"): float(full["StateRoute-MM-K2"]["lambda_cost"]),
        ("heldout11", "MMMU30"): float(held_subject),
        ("heldout11", "KMeans30"): float(held_k30),
        ("heldout11", "KMeans2"): float(k2),
    }


def path_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign(x: np.ndarray, centers: np.ndarray, partition: str) -> np.ndarray:
    if partition == "MMMU30":
        state, _, _ = subject.assign_subjects(x, centers)
        return state.astype(np.int64)
    x = normalized(x)
    distance = (np.square(x).sum(axis=1)[:, None]
                + np.square(centers).sum(axis=1)[None, :]
                - 2 * x @ centers.T)
    return distance.argmin(axis=1)


def profiles(path: Path, global_q: np.ndarray, c: np.ndarray, partition: str):
    with np.load(path, allow_pickle=False) as model:
        centers = np.asarray(model["centers"], dtype=np.float32)
        rate_key = "quality" if "quality" in model.files else "rates"
        rates = np.asarray(model[rate_key], dtype=np.float64)
        support = np.asarray(model["support"], dtype=np.int64)
        stored = {key: np.asarray(model[key]) for key in
                  ("expert_table", "selected_table", "table_oof", "table") if key in model.files}
        stored_cost = np.asarray(model["scaled_cost"], dtype=np.float64) if "scaled_cost" in model.files else None
    if rates.shape != (len(support), 17) or centers.shape != (len(support), 2048):
        raise RuntimeError(f"invalid frozen profile: {path}")
    scaled_c = c.mean(axis=0, dtype=np.float64) / float(np.median(c[c > 0]))
    if stored_cost is not None and not np.allclose(stored_cost, scaled_c, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"stored source cost differs from reconstructed cost: {path}")
    raw = np.broadcast_to(global_q, rates.shape).copy()
    occupied = support > 0
    raw[occupied] = ((support[occupied, None] + 64.0) * rates[occupied]
                     - 64.0 * global_q) / support[occupied, None]
    if np.any(raw < -1e-6) or np.any(raw > 1 + 1e-6):
        raise RuntimeError(f"unshrunk profile is outside [0,1]: {path}")
    return centers, rates, np.clip(raw, 0, 1), scaled_c, support, stored


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    selected = selected_lambdas()
    audit = make_audit("E12", "python -B scripts/journal_20261024/e12_matched.py",
        "Within each protocol all partitions use the same source-fit Y/C subset. "
        "Plain map is argmax with kappa=0 and lambda=0; cost-aware map uses kappa=64 "
        "and each partition's previously frozen source-only 3-fold OOF lambda. "
        "The MMMU prototypes use official MMMU training inputs/labels, so the 3OOD "
        "MMMU slice carries the manuscript's declared input-side prototype caveat.",
        {view: len(uids) for view, uids in views.items()})
    audit.update(c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu="cached Qwen3-VL features; CPU map replay", status="running",
                 selected_lambdas={f"{p}/{k}": v for (p, k), v in selected.items()})
    if (OUT / "AUDIT.json").is_file():
        prior = read_json(OUT / "AUDIT.json")
        audit["prior_attempt_failures"] = prior.get("prior_attempt_failures", []) + [
            {"start_time": prior.get("start_time"), "end_time": prior.get("end_time"),
             "failures": prior.get("failures", [])}]
    write_json(OUT / "AUDIT.json", audit)
    try:
        x14, overlay, _ = subject.qwen_features(views["14ID"], allow_multi=True)
        held_uids = views["11ID"] + views["3OOD"]
        position = {uid: i for i, uid in enumerate(views["14ID"])}
        xheld = x14[[position[uid] for uid in held_uids]]
        if len(overlay) != 10 or x14.shape != (6108, 2048):
            raise RuntimeError("14ID Qwen feature/overlay contract changed")
        # First freeze all actions. Only then open test Y/C for scoring.
        actions = {}
        source_rows = []
        models = {}
        for protocol, x_test in (("full14", x14), ("heldout11", xheld)):
            data = load_source(protocol)
            fit = np.flatnonzero(data["role"] == "fit")
            expected = 17871 if protocol == "full14" else 16269
            if len(fit) != expected:
                raise RuntimeError(f"source-fit size changed for {protocol}")
            # Full14 has 32 native multi-image training features in the frozen
            # overlay. The source Y/C rows are complete and are all we need to
            # invert the archived profiles here.
            y, c = data["y"][fit], data["c"][fit]
            global_q = y.mean(axis=0, dtype=np.float64)
            for partition in PARTITIONS:
                for seed in SEEDS:
                    path = model_path(protocol, partition, seed)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    centers, shrunk, raw, scaled_c, support, stored = profiles(
                        path, global_q, c, partition)
                    states = assign(x_test, centers, partition)
                    plain = raw.argmax(axis=1).astype(np.int8)
                    costed = (shrunk - selected[protocol, partition] * scaled_c).argmax(axis=1).astype(np.int8)
                    if stored and not any(np.array_equal(costed, table) for table in stored.values()):
                        raise RuntimeError(f"frozen selected table mismatch: {path}")
                    for rule, table in (("argmax_k0_l0", plain), ("stateroute_k64", costed)):
                        actions[(protocol, partition, rule, seed)] = table[states]
                    source_rows.append(dict(protocol=protocol, partition=partition, seed=seed,
                        n_source_fit=len(fit), occupied_states=int((support > 0).sum()),
                        min_support=int(support.min()), selected_lambda=selected[protocol, partition],
                        model_sha256=path_hash(path), model_path=str(path)))
                    models[f"{protocol}/{partition}/seed-{seed}"] = path_hash(path)
                    print(json.dumps(dict(stage="actions_frozen", protocol=protocol,
                                          partition=partition, seed=seed)), flush=True)
        audit["model_sha256"] = models
        write_json(OUT / "ACTION_LOCK.json", dict(source_only_lambda=True,
            action_keys=[f"{p}/{k}/{r}/seed-{s}" for (p,k,r,s) in actions]))
        np.savez_compressed(OUT / "PREDICTIONS.npz",
                            uid_14ID=np.asarray(views["14ID"]), uid_heldout=np.asarray(held_uids),
                            **{f"{p}_{k}_{r}_seed{s}": value for (p,k,r,s), value in actions.items()})
        matrices, _, _ = eval_matrices(views)
        rows = []
        for (protocol, partition, rule, seed), action in actions.items():
            slices = (("14ID", slice(None)),) if protocol == "full14" else (
                ("11ID", slice(0, 5341)), ("3OOD", slice(5341, 6108)))
            for view, sl in slices:
                A, C, R = scores(*matrices[view], action[sl], bounds)
                rows.append(dict(experiment="E12", method=partition, variant=rule,
                    view=view, seed=seed, n_test=len(matrices[view][0]), A=A, C=C, R=R,
                    notes=f"source_fit; lambda={selected[protocol, partition] if rule == 'stateroute_k64' else 0}"))
        write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, rows)
        summary = summarize(rows)
        write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
        write_csv(OUT / "source_models.csv", tuple(source_rows[0]), source_rows)
        anchors = {"14ID": (74.915, .6483, 73.900),
                   "11ID": (75.121, .6503, 74.074),
                   "3OOD": (73.664, .6372, 72.837)}
        checks = []
        for view, expected in anchors.items():
            row = next(item for item in summary if item["method"] == "KMeans2"
                       and item["variant"] == "stateroute_k64" and item["view"] == view)
            observed = (row["A_mean"], row["C_mean"], row["R_mean"])
            tolerance = (.35, .02, .35) if view == "3OOD" else (.12, .02, .12)
            checks.append(dict(view=view, expected=expected, observed=observed,
                               tolerance=tolerance,
                               passed=all(abs(a-b) <= t for a,b,t in zip(observed, expected, tolerance))))
        audit["acceptance"] = checks
        audit["status"] = "complete" if all(item["passed"] for item in checks) else "acceptance_failed"
        if audit["status"] != "complete":
            raise RuntimeError("E12 K2 reference check failed")
    except Exception as exc:
        path = OUT / "traceback.txt"
        path.write_text(traceback.format_exc())
        audit["failures"].append(dict(id="E12", seed=None, error=str(exc),
                                      traceback_path=str(path.relative_to(ROOT))))
        if audit.get("status") == "running":
            audit["status"] = "failed"
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
