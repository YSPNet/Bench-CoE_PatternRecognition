#!/usr/bin/env python3
"""Optional v7 encoder robustness, changing only frozen representation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from src.next_round_x.data import load_source
from src.next_round_x.models import normalized
from src.subject_routing.clustering.spherical_kmeans import SphericalKMeans, assign_frozen
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit

OUT = OUTPUT / "V7_ENCODER_ROBUSTNESS_V2"
TEXT_FILE = ROOT / "EMBEDDINGS/text/bge-m3.parquet"
VISION_FILE = OUTPUT / "E6/siglip-large-patch16-384.parquet"
METHODS = {"BGE-M3": "StateRoute-T/BGE-M3",
           "BGE-M3+SigLIP": "StateRoute-MM/BGE-M3+SigLIP"}
SIZES = {"14ID": 6108, "11ID": 5341, "3OOD": 767}
SELECTED_COST_KNOB = {"full14": {"BGE-M3": .5, "BGE-M3+SigLIP": .02},
                      "heldout11": {"BGE-M3": .5, "BGE-M3+SigLIP": .03}}


def log(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_lookup(path, expected_dim):
    frame = pd.read_parquet(path)
    if len(frame) != 30540 or not frame.sample_id.is_unique:
        raise RuntimeError(f"feature UID coverage changed: {path}")
    matrix = np.stack(frame.embedding.to_numpy()).astype(np.float32)
    if matrix.shape != (30540, expected_dim) or not np.isfinite(matrix).all():
        raise RuntimeError(f"feature matrix invalid: {path}")
    return {uid: matrix[i] for i, uid in enumerate(frame.sample_id.tolist())}


def take(lookup, uids):
    try:
        matrix = np.stack([lookup[uid] for uid in uids]).astype(np.float32)
    except KeyError as exc:
        raise RuntimeError(f"frozen feature missing UID {exc}") from exc
    if not np.isfinite(matrix).all():
        raise RuntimeError("nonfinite frozen feature")
    return matrix


def fit_states(kind, seed, train_x, train_uids, test_x, artifact_sha, protocol):
    if kind == "BGE-M3":
        model = SphericalKMeans(2, seed=seed, n_init=10, max_iter=100,
                                tolerance=1e-6).fit(
            train_x, train_uids, environment_id=f"v7_robust_{protocol}",
            fit_role="train" if protocol == "full14" else "train_seen",
            feature_artifact_sha256=artifact_sha,
            audit=UIDAccessAudit.create(Operation.FIT, train_uids))
        src = np.asarray([model.cluster_ids.index(s) for s in model.assignments], dtype=np.int8)
        frozen = assign_frozen(model, test_x, feature_artifact_sha256=artifact_sha)
        dst = np.asarray([model.cluster_ids.index(s) for s in frozen.states], dtype=np.int8)
    else:
        model = MiniBatchKMeans(n_clusters=2, random_state=seed, n_init=3,
                                batch_size=1024, max_iter=100)
        src = model.fit_predict(train_x).astype(np.int8)
        dst = model.predict(test_x).astype(np.int8)
    if np.unique(src).size != 2 or src.shape != (len(train_uids),) or dst.shape != (len(test_x),):
        raise RuntimeError(f"state fit invalid: {kind}/{protocol}/seed-{seed}")
    return src, dst


def expert_map(kind, y, c, src, protocol):
    kappa = 20. if kind == "BGE-M3" else 64.
    count = np.bincount(src, minlength=2)
    global_q = y.mean(axis=0, dtype=np.float64)
    global_c = c.mean(axis=0, dtype=np.float64)
    successes = np.zeros((2, y.shape[1]), dtype=np.float64)
    np.add.at(successes, src, y)
    q = (successes + kappa * global_q) / (count[:, None] + kappa)
    knob = SELECTED_COST_KNOB[protocol][kind]
    if kind == "BGE-M3":
        cap = float(np.quantile(global_c, knob))
        eligible = np.flatnonzero(global_c <= cap + 1e-15)
        mapping = np.asarray([min(eligible, key=lambda j: (-row[j], global_c[j], j))
                              for row in q], dtype=np.int8)
    else:
        scaled = global_c / float(np.median(c[c > 0]))
        mapping = (q - knob * scaled).argmax(axis=1).astype(np.int8)
    return mapping, count.tolist()


def seed_rows(seed, source, test, matrices, bounds, artifact_sha):
    rows = []
    for protocol in ("full14", "heldout11"):
        data = source[protocol]
        target_uids = test[protocol]["uids"]
        for kind in METHODS:
            fit = (np.arange(len(data["uids"])) if
                   kind == "BGE-M3" and protocol == "heldout11" else
                   np.flatnonzero(data["role"] == "fit"))
            expected = (18665 if kind == "BGE-M3" and protocol == "heldout11"
                        else 17871 if protocol == "full14" else 16269)
            if len(fit) != expected:
                raise RuntimeError(f"source-fit mask changed: {protocol}/{kind}")
            uids = data["uids"][fit].tolist()
            src_x = source[protocol][kind][fit]
            dst_x = test[protocol][kind]
            src, dst = fit_states(kind, seed, src_x, uids, dst_x,
                                  artifact_sha[kind], protocol)
            mapping, support = expert_map(kind, data["y"][fit],
                                          data["c"][fit], src, protocol)
            action = mapping[dst]
            targets = (("14ID", slice(None)),) if protocol == "full14" else (
                ("11ID", slice(0, 5341)), ("3OOD", slice(5341, 6108)))
            if len(action) != len(target_uids):
                raise RuntimeError("test action length differs from UID order")
            for view, section in targets:
                A, C, R = scores(*matrices[view], action[section], bounds)
                rows.append(dict(experiment="V7_ENCODER_ROBUSTNESS_V2", method=METHODS[kind],
                                 variant="K2/frozen_original_source_cost_knob",
                                 view=view, seed=seed, n_test=SIZES[view],
                                 A=A, C=C, R=R,
                                 notes=f"{protocol}; kappa={20 if kind == 'BGE-M3' else 64}; "
                                       f"cost_knob={SELECTED_COST_KNOB[protocol][kind]}; "
                                       f"state_support={support}"))
            log(stage="cell_complete", seed=seed, protocol=protocol,
                encoder=kind, source_fit=len(fit), support=support,
                mapping=mapping.tolist())
    if len(rows) != 6:
        raise RuntimeError(f"seed {seed} has {len(rows)} rows, expected 6")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if any(seed not in range(5) for seed in seeds):
        raise RuntimeError("seeds must be 0–4")
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    views = view_uids()
    audit = make_audit("V7_ENCODER_ROBUSTNESS_V2",
                       "python -B scripts/journal_20261024/v7_encoder_robustness.py --seeds " +
                       " ".join(map(str, seeds)),
                       "Only frozen encoders change. Original source-CV-selected cost knobs "
                       "from E24 are fixed at q=.5, full14 lambda=.02 and 11-source lambda=.03. "
                       "K=2, profile subset and kappa match the v7 State-T/MM controls: "
                       "T uses 18,665 heldout11 train rows and 17,871 full14 source-fit "
                       "rows; MM uses 16,269 heldout11 and 17,871 full14 source-fit rows. "
                       "No 3OOD outcomes enter state fitting, profile or map selection.", SIZES)
    audit.update(status="running", c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu="none; cached BGE-M3 and SigLIP features; CPU fits with two BLAS threads",
                 seeds=seeds, feature_sha256={"bge_m3": sha256(TEXT_FILE),
                                         "siglip": sha256(VISION_FILE)},
                 cost_knobs=SELECTED_COST_KNOB,
                 supersedes="V7_ENCODER_ROBUSTNESS: first pass used 16,269 instead "
                            "of 18,665 heldout11 source rows for text BGE-M3")
    prior_path = OUT / "AUDIT.json"
    if prior_path.is_file():
        prior = read_json(prior_path)
        audit["prior_attempt_failures"] = prior.get("prior_attempt_failures", [])
        if prior.get("failures"):
            audit["prior_attempt_failures"].append({
                "start_time": prior.get("start_time"),
                "end_time": prior.get("end_time"),
                "failures": prior["failures"],
            })
    write_json(OUT / "AUDIT.json", audit)
    try:
        text = feature_lookup(TEXT_FILE, 1024)
        vision = feature_lookup(VISION_FILE, 1024)
        if set(text) != set(vision):
            raise RuntimeError("BGE/SigLIP feature UID sets differ")
        source = {protocol: load_source(protocol) for protocol in ("full14", "heldout11")}
        target = {"full14": {"uids": views["14ID"]},
                  "heldout11": {"uids": views["11ID"] + views["3OOD"]}}
        for protocol, data in source.items():
            ids = data["uids"].tolist()
            bge = normalized(take(text, ids))
            siglip = normalized(take(vision, ids))
            data["BGE-M3"] = bge
            data["BGE-M3+SigLIP"] = normalized(np.concatenate((bge, siglip), axis=1))
        for protocol, item in target.items():
            bge = normalized(take(text, item["uids"]))
            siglip = normalized(take(vision, item["uids"]))
            item["BGE-M3"] = bge
            item["BGE-M3+SigLIP"] = normalized(np.concatenate((bge, siglip), axis=1))
        artifact_sha = {"BGE-M3": audit["feature_sha256"]["bge_m3"],
                        "BGE-M3+SigLIP": hashlib.sha256(
                            (audit["feature_sha256"]["bge_m3"] +
                             audit["feature_sha256"]["siglip"]).encode()).hexdigest()}
        matrices, _, _ = eval_matrices(views)
        rows = []
        for seed in seeds:
            checkpoint = OUT / f"seed-{seed}.json"
            if checkpoint.is_file():
                part = read_json(checkpoint)["rows"]
                log(stage="seed_resumed", seed=seed)
            else:
                part = seed_rows(seed, source, target, matrices, bounds, artifact_sha)
                write_json(checkpoint, {"seed": seed, "rows": part})
                log(stage="seed_complete", seed=seed)
            rows.extend(part)
            write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, rows)
        if seeds == list(range(5)):
            summary = summarize(rows)
        else:
            summary = [dict(experiment=row["experiment"], method=row["method"],
                            variant=row["variant"], view=row["view"], n_seeds=1,
                            A_mean=row["A"], A_sd=None, C_mean=row["C"], C_sd=None,
                            R_mean=row["R"], R_sd=None, notes=row["notes"])
                       for row in rows]
        write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summary)
        if len(rows) != 6 * len(seeds) or len(summary) != 6:
            raise RuntimeError("robustness output coverage changed")
        audit["row_count"] = len(rows)
        audit["status"] = "complete" if seeds == list(range(5)) else "seed0_preview_complete"
        with (OUTPUT / "PROGRESS.md").open("a") as handle:
            handle.write(f"\n## V7 encoder robustness — {audit['status']}\n\n"
                         f"Frozen BGE-M3 text and BGE-M3+SigLIP multimodal features, seeds {seeds}, "
                         "K=2 and unchanged source-selected cost knobs. See "
                         "V7_ENCODER_ROBUSTNESS_V2/SUMMARY.csv. The first pass is retained "
                         "with its heldout11 text-profile subset discrepancy documented.\n")
    except Exception as exc:
        path = OUT / "traceback.txt"
        path.write_text(traceback.format_exc())
        audit["failures"].append(dict(id="V7_ENCODER_ROBUSTNESS_V2", seed=None,
                                      error=str(exc), traceback_path=str(path.relative_to(ROOT))))
        audit["status"] = "failed"
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
