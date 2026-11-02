#!/usr/bin/env python3
"""Held-out state-construction comparison: frozen text, image, and fused inputs.

The text row reproduces the v23 State-Bench-CoE-T control. All variants use
the same K=2 spherical clustering, full 11-source training rows, kappa=20,
and median-source-cost eligibility rule. Predictions are sealed before test
outcomes are read. This script does not edit the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from threadpoolctl import threadpool_limits

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SUMMARY_COLUMNS, eval_matrices,
                    make_audit, read_json, scores, summarize, utc_now, view_uids,
                    write_csv, write_json)
from v23_mm_spherical_compare import sha
from v7_encoder_robustness import feature_lookup, take
from scripts import run_stateroute_learned_ladder_20261024 as ladder
from src.next_round_x.data import load_source
from src.next_round_x.models import normalized
from src.subject_routing.clustering.spherical_kmeans import (
    SphericalKMeans, SphericalKMeansArtifact, assign_frozen)
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit

OUT = OUTPUT / "V23_STATE_MODALITY_COMPARE"
TEXT = ROOT / "EMBEDDINGS/text/bge-m3.parquet"
IMAGE = OUTPUT / "E6/siglip-large-patch16-384.parquet"
SEEDS = tuple(range(5))
VARIANTS = ("minilm_text_original", "bge_text_spherical",
            "siglip_image_spherical", "bge_siglip_spherical")
K = 2
KAPPA = 20.0
ELIGIBILITY_Q = 0.5


def log(**items) -> None:
    print(json.dumps(items, sort_keys=True), flush=True)


def load_inputs():
    views = view_uids()
    uids = views["11ID"] + views["3OOD"]
    source = load_source("heldout11")
    if len(source["uids"]) != 18665 or source["uids"].tolist() != __import__(
            "src.next_round_x.data", fromlist=["role_uids"]).role_uids("heldout11", "train"):
        raise RuntimeError("full 11-source text training scope changed")
    siglip_manifest = read_json(OUTPUT / "E6/SIGLIP_FEATURES.json")
    if siglip_manifest["features_sha256"] != sha(IMAGE) or siglip_manifest["n"] != 30540:
        raise RuntimeError("frozen SigLIP image artifact changed")
    text_lookup = feature_lookup(TEXT, 1024)
    image_lookup = feature_lookup(IMAGE, 1024)
    if set(text_lookup) != set(image_lookup) or set(uids) - set(text_lookup):
        raise RuntimeError("frozen modality features have unequal UID coverage")
    source_uids = source["uids"].tolist()
    bge_source = normalized(take(text_lookup, source_uids))
    bge_test = normalized(take(text_lookup, uids))
    image_source = normalized(take(image_lookup, source_uids))
    image_test = normalized(take(image_lookup, uids))
    features = {
        "minilm_text_original": (np.asarray(ladder.text_train(source), dtype=np.float32),
                                 ladder.text_test(uids)),
        "bge_text_spherical": (bge_source, bge_test),
        "siglip_image_spherical": (image_source, image_test),
        "bge_siglip_spherical": (
            normalized(np.concatenate((bge_source, image_source), axis=1)),
            normalized(np.concatenate((bge_test, image_test), axis=1))),
    }
    hashes = {"minilm_text_original": read_json(ladder.TEXT_METADATA)[
                  "feature_artifact_sha256"],
              "bge_text_spherical": sha(TEXT),
              "siglip_image_spherical": sha(IMAGE),
              "bge_siglip_spherical": __import__("hashlib").sha256(
                  (sha(TEXT) + sha(IMAGE)).encode()).hexdigest()}
    return source, views, uids, features, hashes


def fit_variant(variant: str, seed: int, source: dict,
                source_x: np.ndarray, test_x: np.ndarray,
                feature_hash: str) -> tuple[np.ndarray, dict]:
    path = OUT / "models" / f"seed-{seed}" / f"{variant}.json"
    uids = source["uids"].tolist()
    if path.is_file():
        artifact = SphericalKMeansArtifact.load(path)
        if (artifact.fit_uids != tuple(uids) or
                artifact.feature_artifact_sha256 != feature_hash or
                artifact.seed != seed or artifact.requested_k != K):
            raise RuntimeError(f"spherical artifact identity changed: {path}")
    else:
        artifact = SphericalKMeans(K, seed=seed, n_init=10, max_iter=100,
                                   tolerance=1e-6).fit(
            source_x, uids, environment_id="ood11_3", fit_role="train_seen",
            feature_artifact_sha256=feature_hash,
            audit=UIDAccessAudit.create(Operation.FIT, uids))
        artifact.save(path)
    states = np.asarray([artifact.cluster_ids.index(state) for state in
                         artifact.assignments], dtype=np.int8)
    frozen = assign_frozen(artifact, test_x,
                           feature_artifact_sha256=feature_hash)
    test_states = np.asarray([artifact.cluster_ids.index(state) for state in
                              frozen.states], dtype=np.int8)
    y = source["y"]
    c = source["c"]
    support = np.bincount(states, minlength=K)
    if np.any(support == 0):
        raise RuntimeError(f"empty {variant} source state")
    global_quality = y.mean(axis=0, dtype=np.float64)
    successes = np.zeros((K, y.shape[1]), dtype=np.float64)
    np.add.at(successes, states, y)
    quality = (successes + KAPPA * global_quality) / (support[:, None] + KAPPA)
    global_cost = c.mean(axis=0, dtype=np.float64)
    cap = float(np.quantile(global_cost, ELIGIBILITY_Q))
    eligible = np.flatnonzero(global_cost <= cap + 1e-15)
    table = np.asarray([min(eligible,
                            key=lambda j: (-quality[state, j], global_cost[j], j))
                        for state in range(K)], dtype=np.int8)
    details = dict(seed=seed, variant=variant, source_n=len(source_x),
                   source_support=support.tolist(), test_support=np.bincount(
                       test_states, minlength=K).tolist(), expert_table=table.tolist(),
                   cap=cap, eligible_experts=eligible.tolist(),
                   center_norms=np.linalg.norm(artifact.centers, axis=1).tolist(),
                   converged=artifact.converged,
                   objective=artifact.objective_cosine_distance_sum,
                   model_sha256=sha(path), feature_sha256=feature_hash)
    return table[test_states], details


def fit_seed(seed: int, source: dict, uids: list[str],
             features: dict, hashes: dict) -> None:
    dest = OUT / "models" / f"seed-{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    prediction = dest / "PREDICTIONS.npz"
    metadata = dest / "COMPLETE.json"
    if prediction.is_file() and metadata.is_file():
        saved = read_json(metadata)
        if saved.get("protocol_version") == 2:
            with np.load(prediction, allow_pickle=False) as archive:
                if archive["uids"].tolist() != uids or set(VARIANTS) - set(archive.files):
                    raise RuntimeError("resumed modality prediction identity changed")
            if saved["predictions_sha256"] != sha(prediction):
                raise RuntimeError("resumed modality prediction digest changed")
            log(stage="seed_reused", seed=seed)
            return
    actions = {}
    details = {}
    for variant in VARIANTS:
        action, info = fit_variant(variant, seed, source, *features[variant],
                                   hashes[variant])
        actions[variant] = action
        details[variant] = info
        log(stage="modality_fit", seed=seed, variant=variant,
            support=info["source_support"], table=info["expert_table"])
    np.savez_compressed(prediction, uids=np.asarray(uids),
                        **{name: value.astype(np.int8) for name, value in actions.items()})
    write_json(metadata, dict(protocol_version=2, seed=seed, status="complete",
                              source_n=len(source["uids"]),
                              test_n=len(uids), k=K, kappa=KAPPA,
                              eligibility_q=ELIGIBILITY_Q,
                              new_variants=details, predictions_sha256=sha(prediction)))
    log(stage="seed_complete", seed=seed)


def evaluate(views: dict, uids: list[str], bounds: dict) -> None:
    frozen = {}
    for seed in SEEDS:
        path = OUT / "models" / f"seed-{seed}" / "PREDICTIONS.npz"
        metadata = read_json(path.parent / "COMPLETE.json")
        if metadata["status"] != "complete" or sha(path) != metadata["predictions_sha256"]:
            raise RuntimeError(f"modality seed {seed} predictions are not sealed")
        with np.load(path, allow_pickle=False) as archive:
            if archive["uids"].tolist() != uids:
                raise RuntimeError("frozen modality UID order changed")
            frozen[seed] = {name: archive[name] for name in VARIANTS}
    write_json(OUT / "ACTION_LOCK.json", dict(
        status="predictions_frozen_before_test_scoring", seeds=list(SEEDS),
        variants=list(VARIANTS), source_scope="all_18665_heldout11_source_train_rows",
        prediction_sha256={str(seed): sha(OUT / "models" / f"seed-{seed}" /
                                           "PREDICTIONS.npz") for seed in SEEDS}))
    matrices, _, _ = eval_matrices(views)
    rows = []
    for seed in SEEDS:
        for variant, actions in frozen[seed].items():
            for view, section in (("11ID", slice(0, 5341)),
                                  ("3OOD", slice(5341, 6108))):
                A, C, R = scores(*matrices[view], actions[section], bounds)
                rows.append(dict(experiment="V23_STATE_MODALITY_COMPARE",
                                 method="State-Bench-CoE", variant=variant,
                                 view=view, seed=seed, n_test=len(matrices[view][0]),
                                 A=A, C=C, R=R,
                                 notes="K=2; spherical; full source n=18665; kappa=20; q=0.5"))
    with (OUTPUT / "V7_MAIN_FIVE_SEED/RESULTS.csv").open(newline="") as handle:
        original = {(int(row["seed"]), row["view"]): row for row in csv.DictReader(handle)
                    if row["method"] == "StateRoute-T" and row["variant"] == "source_cv"
                    and row["view"] in ("11ID", "3OOD")}
    with (OUTPUT / "V7_ENCODER_ROBUSTNESS_V2/RESULTS.csv").open(newline="") as handle:
        bge = {(int(row["seed"]), row["view"]): row for row in csv.DictReader(handle)
               if row["method"] == "StateRoute-T/BGE-M3" and
               row["view"] in ("11ID", "3OOD")}
    if len(original) != 10 or len(bge) != 10:
        raise RuntimeError("published text comparison coverage changed")
    reference_checks = []
    for row in rows:
        if row["variant"] not in ("minilm_text_original", "bge_text_spherical"):
            continue
        reference = original if row["variant"] == "minilm_text_original" else bge
        differences = {key: abs(row[key] - float(reference[row["seed"], row["view"]][key]))
                       for key in ("A", "C", "R")}
        reference_checks.append(dict(seed=row["seed"], view=row["view"],
                                     variant=row["variant"], differences=differences,
                                     passed=max(differences.values()) < 1e-8))
    if not all(check["passed"] for check in reference_checks):
        raise RuntimeError("archived text controls failed to reproduce")
    write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, rows)
    write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summarize(rows))
    write_json(OUT / "COMPARISON.json", dict(status="complete",
        reference_checks=reference_checks, seeds=list(SEEDS),
        important_limit="MiniLM, BGE-M3, SigLIP, and Qwen3-VL are different "
                        "encoders; BGE-M3/SigLIP pair shares the same source, "
                        "K=2 spherical fitter, and budget rule, whereas Qwen "
                        "MM uses a smaller source-fit subset and a penalty rule."))
    log(stage="evaluation_complete", rows=len(rows), checks=len(reference_checks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    seeds = tuple(sorted(set(args.seeds)))
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise RuntimeError("seeds must be a nonempty subset of 0–4")
    OUT.mkdir(parents=True, exist_ok=True)
    source, views, uids, features, hashes = load_inputs()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit = make_audit("V23_STATE_MODALITY_COMPARE", str(Path(__file__).relative_to(ROOT)),
        "Reproduced MiniLM text control; new BGE-M3 text, SigLIP image, and "
        "equal-weight BGE-M3+SigLIP spherical K=2 state variants. "
        "All use 18,665 11-source train rows, kappa=20, source median-cost "
        "eligibility q=0.5. No test outcomes enter fitting or selection.",
        {view: len(views[view]) for view in ("11ID", "3OOD")})
    audit.update(status="fitting", seeds_requested=list(seeds), gpu="none; cached embeddings",
        feature_sha256=hashes, source_n=len(source["uids"]),
        test_n=len(uids), k=K, kappa=KAPPA, eligibility_q=ELIGIBILITY_Q)
    write_json(OUT / "AUDIT.json", audit)
    try:
        for seed in seeds:
            fit_seed(seed, source, uids, features, hashes)
        if args.evaluate:
            if seeds != SEEDS:
                raise RuntimeError("test scoring requires all five sealed seeds")
            evaluate(views, uids, bounds)
            audit["status"] = "complete"
        else:
            audit["status"] = "predictions_partial" if seeds != SEEDS else "predictions_frozen"
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = str(exc)
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
