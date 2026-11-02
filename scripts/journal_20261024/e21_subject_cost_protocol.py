#!/usr/bin/env python3
"""V5 cost protocol: frozen Subject-Bench-CoE classifier, source-selected map cost.

The classifier is the already trained subject classifier.  Only the expert
map changes.  The map profile is cross-fitted by the repository's group folds,
and test predictions are frozen before test outcomes are read.
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from routers.bench_coe_query.config import load_contract
from routers.bench_coe_query.heads import AuditedBenchCoEHead
from scripts import run_bench_coe_subject_mmmu as subject
from src.next_round_x.data import load_source

SOURCE = ROOT / "outputs/bench_coe_subject_mmmu30"
LAMBDA_GRID = (0.0, .001, .003, .01, .02, .03, .05, .1, .3, 1.0)
BACKBONE = os.environ.get("SUBJECT_COST_BACKBONE", "tinyllava")
METHOD = {"bert": "Subject-BERT", "tinyllava": "Subject-TinyLLaVA",
          "qwen3vl": "Subject-Qwen3-VL"}[BACKBONE]
EXPERIMENT = "E21" if BACKBONE == "tinyllava" else f"V19_SUBJECT_COST_{BACKBONE.upper()}"
OUT = OUTPUT / EXPERIMENT
CURVE_COLUMNS = ("method", "protocol", "seed", "lambda", "view", "A", "C", "R")
SELECT_COLUMNS = ("protocol", "seed", "view", "selection", "lambda", "source_oof_R", "test_R")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def source_classifier_states(protocol: str, seed: int, data, features, available):
    name = "id14" if protocol == "full14" else "ood11_3"
    run = SOURCE / name / BACKBONE / f"seed-{seed}"
    trained = read_json(run / "TRAINED.json")
    checkpoint = run / "best.pt"
    if digest(checkpoint) != trained["checkpoint_sha256"]:
        raise RuntimeError(f"frozen subject classifier changed: {checkpoint}")
    uids = data["uids"].tolist()
    registry_ids = subject.rows_and_metadata()[0]
    registry_pos = {uid: i for i, uid in enumerate(registry_ids)}
    positions = np.asarray([registry_pos[uid] for uid in uids], dtype=np.int64)
    states = np.full(len(uids), -1, dtype=np.int16)
    contract = load_contract()
    if BACKBONE == "bert":
        path = Path(str(contract.values["models"]["bert"]["path"]))
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), num_labels=30, ignore_mismatched_sizes=True, local_files_only=True)
        text = subject.bert_texts(registry_ids)
        encoded = tokenizer([text[pos] for pos in positions], truncation=True,
                            padding="max_length",
                            max_length=int(contract.values["models"]["bert"]["max_length"]),
                            return_tensors="pt")
        valid = np.arange(len(uids))
        loader = DataLoader(TensorDataset(encoded["input_ids"], encoded["attention_mask"]),
                            batch_size=64, shuffle=False)
    else:
        cfg = contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"]
        model = AuditedBenchCoEHead(features.shape[1], 30, float(cfg["dropout"]))
        valid = np.flatnonzero(available[positions])
        loader = DataLoader(TensorDataset(torch.from_numpy(features[positions[valid]])),
                            batch_size=256, shuffle=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    device = torch.device("cuda" if BACKBONE == "bert" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    got = []
    with torch.no_grad():
        for batch in loader:
            if BACKBONE == "bert":
                logits = model(input_ids=batch[0].to(device),
                               attention_mask=batch[1].to(device)).logits
            else:
                logits = model(batch[0].to(device))
            got.append(logits.argmax(-1).cpu().numpy())
    states[valid] = np.concatenate(got).astype(np.int16)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # The frozen TinyLLaVA cache intentionally excludes native multi-image
    # samples.  The original classifier uses its map-global fallback there.
    # Preserve that input-side missingness in both OOF and test replay.
    return states, trained["checkpoint_sha256"]


def map_from_source(y, c, prototype_state, lam):
    support = np.bincount(prototype_state, minlength=30)
    sums = np.zeros((30, 17), dtype=np.float64)
    np.add.at(sums, prototype_state, y)
    global_quality = y.mean(axis=0, dtype=np.float64)
    quality = np.broadcast_to(global_quality, sums.shape).copy()
    occupied = support > 0
    quality[occupied] = sums[occupied] / support[occupied, None]
    global_cost = c.mean(axis=0, dtype=np.float64)
    cost_scale = float(np.median(c[c > 0]))
    table = (quality - lam * global_cost[None, :] / cost_scale).argmax(axis=1).astype(np.int8)
    fallback = int((global_quality - lam * global_cost / cost_scale).argmax())
    return table, fallback, support


def oof_selection(data, prototype_state, classifier_state, bounds):
    fit = np.flatnonzero(data["role"] == "fit")
    order = {int(row): i for i, row in enumerate(fit)}
    action = {lam: np.full(len(fit), -1, dtype=np.int16) for lam in LAMBDA_GRID}
    folds = []
    for fold in range(3):
        train = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][train]) & set(data["groups"][valid]):
            raise RuntimeError("group overlap in subject source OOF")
        target = np.asarray([order[int(row)] for row in valid], dtype=np.int64)
        for lam in LAMBDA_GRID:
            table, fallback, _ = map_from_source(data["y"][train], data["c"][train],
                                                 prototype_state[train], lam)
            states = classifier_state[valid]
            action[lam][target] = np.where(states >= 0, table[np.maximum(states, 0)], fallback)
        folds.append(dict(fold=fold, train_n=len(train), valid_n=len(valid)))
    candidates = []
    for lam in LAMBDA_GRID:
        if np.any(action[lam] < 0):
            raise RuntimeError("incomplete subject OOF actions")
        A, C, R = scores(data["y"][fit], data["c"][fit], action[lam], bounds)
        candidates.append(dict(lambda_cost=lam, A=A, C=C, R=R))
    selected = max(candidates, key=lambda item: (item["R"], item["A"], -item["C"], -item["lambda_cost"]))
    return selected, candidates, folds


def main():
    if BACKBONE == "bert" and not torch.cuda.is_available():
        raise RuntimeError("BERT subject replay requires an available CUDA device")
    OUT.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit = make_audit(EXPERIMENT, "python -B scripts/journal_20261024/e21_subject_cost_protocol.py",
        f"Fixed {BACKBONE} subject classifier. Map lambda from preset grid selected by "
        "group-isolated three-fold OOF on source-fit Y/C; classifier predictions on "
        "source inputs are fixed, and are never trained on held-out expert outcomes. "
        "Test-selected lambda is a labelled supplement diagnostic only.",
        {view: len(uids) for view, uids in views.items()})
    audit.update(status="running", c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu=("CUDA replay of frozen BERT subject classifier" if BACKBONE == "bert"
                      else "CPU replay of frozen subject heads"), lambda_grid=list(LAMBDA_GRID),
                 backbone=BACKBONE)
    if (OUT / "AUDIT.json").is_file():
        previous = read_json(OUT / "AUDIT.json")
        audit["prior_attempts"] = previous.get("prior_attempts", []) + [dict(
            start_time=previous.get("start_time"), status=previous.get("status"),
            failures=previous.get("failures", []))]
    write_json(OUT / "AUDIT.json", audit)
    all_rows, all_curves, all_select = [], [], []
    try:
        registry_ids, _, _ = subject.rows_and_metadata()
        with np.load(SOURCE / "assignments.npz", allow_pickle=False) as archive:
            if archive["uids"].tolist() != list(registry_ids):
                raise RuntimeError("prototype assignment UID order changed")
            all_proto = archive["states"].astype(np.int8)
        if BACKBONE == "bert":
            features, available, feature_hash = None, None, None
        else:
            features, available, feature_hash = subject.feature_matrix(BACKBONE, registry_ids)
        lookup = {uid: i for i, uid in enumerate(registry_ids)}
        source = {name: load_source(name) for name in ("full14", "heldout11")}
        actions = {}
        selection_metadata = []
        for seed in SEEDS:
            checkpoint = OUT / f"seed-{seed}.json"
            if checkpoint.is_file():
                saved = read_json(checkpoint)
                all_rows.extend(saved["rows"])
                all_curves.extend(saved["curves"])
                all_select.extend(saved["selections"])
                continue
            seed_rows, seed_curves, seed_select = [], [], []
            try:
                for protocol, data in source.items():
                    name = "id14" if protocol == "full14" else "ood11_3"
                    uids = data["uids"].tolist()
                    proto = all_proto[np.asarray([lookup[uid] for uid in uids])]
                    classifier, classifier_hash = source_classifier_states(
                        protocol, seed, data, features, available)
                    chosen, oof, folds = oof_selection(data, proto, classifier, bounds)
                    with np.load(SOURCE / name / "expert_map.npz", allow_pickle=False) as base:
                        original = base["table"].astype(np.int8)
                        stored_rows = base["train_rows"]
                    if not np.array_equal(data["uids"], np.asarray(registry_ids)[stored_rows]):
                        raise RuntimeError("frozen subject map source train differs")
                    zero_table, _, _ = map_from_source(data["y"], data["c"], proto, 0.0)
                    if not np.array_equal(zero_table, original):
                        raise RuntimeError("lambda-zero map does not reproduce frozen subject map")
                    targets = (("14ID",) if protocol == "full14" else ("11ID", "3OOD"))
                    for lam in LAMBDA_GRID:
                        table, fallback, support = map_from_source(data["y"], data["c"], proto, lam)
                        for view in targets:
                            path = SOURCE / name / BACKBONE / f"seed-{seed}" / (
                                "predictions_14-ID.npz" if view == "14ID" else
                                f"predictions_{view.replace('ID', '-ID').replace('OOD', '-OOD')}.npz")
                            # Existing filenames are 14-ID, 11-ID and 3-OOD.
                            if view == "11ID":
                                path = path.with_name("predictions_11-ID.npz")
                            elif view == "3OOD":
                                path = path.with_name("predictions_3-OOD.npz")
                            with np.load(path, allow_pickle=False) as pred:
                                if pred["uids"].tolist() != views[view]:
                                    raise RuntimeError(f"subject test UID order changed: {view}")
                                state = pred["predicted_subject"].astype(np.int64)
                                base_action = pred["selected_expert"].astype(np.int8)
                            action = np.where(state >= 0, table[np.maximum(state, 0)], fallback).astype(np.int8)
                            if lam == 0.0 and not np.array_equal(action, base_action):
                                raise RuntimeError(f"subject baseline action mismatch: {protocol}/{view}")
                            actions[f"{protocol}_seed{seed}_lambda{lam}_{view}"] = action
                    selection_metadata.append(dict(protocol=protocol, seed=seed,
                        source_oof=oof, selected_lambda=chosen["lambda_cost"], folds=folds,
                        classifier_sha256=classifier_hash, feature_sha256=feature_hash))
                # Freeze every lambda action before opening test Y/C.
                np.savez_compressed(OUT / f"actions_seed-{seed}.npz",
                    **{view: np.asarray(views[view]) for view in views}, **actions)
                matrices, _, _ = eval_matrices(views)
                for protocol, data in source.items():
                    chosen = next(item for item in selection_metadata if item["protocol"] == protocol
                                  and item["seed"] == seed)
                    source_lambda = chosen["selected_lambda"]
                    targets = (("14ID",) if protocol == "full14" else ("11ID", "3OOD"))
                    for view in targets:
                        candidates = []
                        for lam in LAMBDA_GRID:
                            action = actions[f"{protocol}_seed{seed}_lambda{lam}_{view}"]
                            A, C, R = scores(*matrices[view], action, bounds)
                            candidates.append(dict(lambda_cost=lam, A=A, C=C, R=R))
                            seed_curves.append(dict(method=METHOD, protocol=protocol,
                                seed=seed, **{"lambda": lam}, view=view, A=A, C=C, R=R))
                        best = max(candidates, key=lambda item: item["R"])
                        for selection, lam in (("source_cv", source_lambda),
                                               ("best_test", best["lambda_cost"])):
                            item = next(row for row in candidates if row["lambda_cost"] == lam)
                            seed_rows.append(dict(experiment=EXPERIMENT, method=METHOD,
                                variant=f"lambda={lam}/select={selection}", view=view, seed=seed,
                                n_test=len(views[view]), A=item["A"], C=item["C"], R=item["R"],
                                notes="Cost-penalized unshrunk subject map; best_test diagnostic only"))
                            seed_select.append(dict(protocol=protocol, seed=seed, view=view,
                                selection=selection, **{"lambda": lam},
                                source_oof_R=next(row["R"] for row in chosen["source_oof"]
                                                  if row["lambda_cost"] == lam), test_R=item["R"]))
                write_json(checkpoint, dict(seed=seed, rows=seed_rows, curves=seed_curves,
                                            selections=seed_select))
                all_rows.extend(seed_rows)
                all_curves.extend(seed_curves)
                all_select.extend(seed_select)
                print(json.dumps(dict(stage="seed_complete", seed=seed, rows=len(seed_rows))), flush=True)
            except Exception as exc:
                trace = OUT / f"seed-{seed}-traceback.txt"
                trace.write_text(traceback.format_exc())
                audit["failures"].append(dict(id=EXPERIMENT, seed=seed, error=str(exc),
                    traceback_path=str(trace.relative_to(ROOT))))
                write_json(OUT / "AUDIT.json", audit)
                raise
        write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, all_rows)
        write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summarize(all_rows))
        write_csv(OUT / "curves.csv", CURVE_COLUMNS, all_curves)
        write_csv(OUT / "selections.csv", SELECT_COLUMNS, all_select)
        write_json(OUT / "source_cv.json", selection_metadata)
        audit["status"] = "preview_complete" if SEEDS == (0,) else "complete"
        audit["source_cv_selection"] = selection_metadata
        with (OUTPUT / "PROGRESS.md").open("a") as handle:
            handle.write(f"\n## {EXPERIMENT} — {audit['status']}\n\nCost-aware {METHOD} map, "
                         f"seeds {list(SEEDS)}, source group-OOF and test-selected diagnostic.\n")
    except Exception:
        audit["status"] = "failed"
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(OUT / "AUDIT.json", audit)


if __name__ == "__main__":
    main()
