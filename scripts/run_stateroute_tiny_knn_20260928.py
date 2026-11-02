#!/usr/bin/env python3
"""Source-only TinyLLaVA domain and local-neighbor expert router.

Run each protocol with --stage train, freeze, predict, evaluate in order.
Only official source-fit outcomes enter the model and rule selection. Test
actions are hash-bound before test outcomes are materialized.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from routers.utils.rank_score import rank_score
from scripts import run_stateroute_domain_ridge_full14_20260928 as prior14
from scripts import run_stateroute_domain_ridge_11_3_20260928 as prior11
from scripts import run_stateroute_mmmu_full14_20260928 as full14
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source, role_uids
from src.next_round_x.models import normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

METHODS = ("TinyKNN-SourceRank", "TinyKNN-SourceAccuracy")
KAPPAS = (64., 256.)
MODES = ("all", "same")
WEIGHTS = ("uniform", "soft")
KS = (16, 32, 64)
BLENDS = (0., .2, .4, .6)
COST_MODES = ("domain", "global")
LAMBDAS = (.02, .025, .03, .035, .04, .045, .05, .06, .07, .08, .1)
NEIGHBOR_TEMPERATURE = .08
N_EXPERTS = 17
TINY_POINTER = ROOT / "outputs/bench_coe_query/features/tinyllava/LATEST_FULL.json"


def paths(protocol):
    if protocol not in ("full14", "heldout11"):
        raise ValueError(protocol)
    stem = "full14" if protocol == "full14" else "11_3"
    return ROOT / f"outputs/stateroute_tiny_knn_{stem}_20260928", ROOT / f"reports/stateroute_tiny_knn_{stem}_20260928"


def tiny_features(uids):
    pointer = read_json(TINY_POINTER)
    if pointer["input_content"]["model_name"] != "tinyllava":
        raise RuntimeError("TinyLLaVA feature pointer changed")
    merged = ROOT / pointer["cache_root"] / "merged"
    manifest_path = merged / "manifest.json"
    manifest = read_json(manifest_path)
    for filename, key in (("features.npy", "features_sha256"), ("index.jsonl", "index_sha256")):
        if digest(merged / filename) != manifest[key]:
            raise RuntimeError(f"TinyLLaVA feature cache changed: {filename}")
    index = {}
    with (merged / "index.jsonl").open(encoding="utf-8") as handle:
        for compact_row, line in enumerate(handle):
            item = json.loads(line)
            uid = item["sample_id"]
            if uid in index:
                raise RuntimeError("duplicate TinyLLaVA feature UID")
            # row_index refers to the 30,540-row extraction plan and includes
            # unavailable multi-image inputs; features.npy is compacted.
            index[uid] = compact_row
    cache = np.load(merged / "features.npy", mmap_mode="r", allow_pickle=False)
    if cache.shape != (manifest["rows"], 5120) or len(index) != len(cache):
        raise RuntimeError("TinyLLaVA feature shape changed")
    output = np.zeros((len(uids), 5120), dtype=np.float32)
    available = np.array([uid in index for uid in uids], dtype=bool)
    present = np.flatnonzero(available)
    output[present] = cache[[index[uids[i]] for i in present]]
    if not np.isfinite(output).all() or (np.linalg.norm(output[present], axis=1) < 1e-8).any():
        raise RuntimeError("invalid TinyLLaVA feature")
    return normalized(output), available, [TINY_POINTER, manifest_path, merged / "features.npy", merged / "index.jsonl"]


def source_inputs(protocol):
    if protocol == "full14":
        data, xq, fit, overlay = full14.source_train()
        if len(overlay) != 32:
            raise RuntimeError("full14 Qwen multi-image source overlay changed")
    else:
        data = load_source("heldout11")
        if data["uids"].tolist() != role_uids("heldout11", "train") or len(data["uids"]) != 18665:
            raise RuntimeError("11-source UID contract changed")
        if not data["available"].all():
            raise RuntimeError("11-source Qwen features incomplete")
        xq = np.asarray(data["x"])
        fit = np.flatnonzero(data["role"] == "fit")
        if len(fit) != 16269:
            raise RuntimeError("11-source fit count changed")
    names = sorted(set(data["dataset"].tolist()))
    if len(names) != (14 if protocol == "full14" else 11):
        raise RuntimeError("source dataset count changed")
    tiny, available, tiny_records = tiny_features(data["uids"].tolist())
    if int((~available).sum()) != (32 if protocol == "full14" else 0):
        raise RuntimeError("TinyLLaVA source coverage changed")
    lookup = {name: i for i, name in enumerate(names)}
    domains = np.array([lookup[name] for name in data["dataset"]], dtype=np.int8)
    return data, normalized(xq), tiny, available, fit, names, domains, tiny_records


def fit_domain(x, labels, n_domains):
    model = RidgeClassifier(alpha=.1, solver="lsqr", class_weight="balanced", tol=1e-4)
    model.fit(x, labels)
    if not np.array_equal(model.classes_, np.arange(n_domains)):
        raise RuntimeError("domain classifier lost a source class")
    return model


def infer_domain(x, coef, intercept):
    return (x @ coef.T + intercept).argmax(axis=1).astype(np.int8)


def profile(protocol, y, c, d, kappa):
    if protocol == "full14":
        return prior14.profile(y, c, d, quality_kappa=kappa)
    return prior11.profile(y, c, d, kappa)


def nearest_quality(query, bank, bank_y, bank_domain, predicted):
    """Return both global and predicted-domain top-k quality estimates."""
    result = {(mode, weighting, k): np.empty((len(query), N_EXPERTS), dtype=np.float32)
              for mode in MODES for weighting in WEIGHTS for k in KS}
    bank = np.asarray(bank, dtype=np.float32)
    labels = np.asarray(bank_y, dtype=np.float32)
    if bank.shape[0] < max(KS) or labels.shape != (len(bank), N_EXPERTS):
        raise RuntimeError("invalid frozen neighbor bank")
    for start in range(0, len(query), 128):
        end = min(len(query), start + 128)
        similarities = query[start:end] @ bank.T
        for mode in MODES:
            scores = similarities if mode == "all" else np.where(
                bank_domain[None, :] == predicted[start:end, None], similarities, -2.)
            indices = np.argpartition(scores, -max(KS), axis=1)[:, -max(KS):]
            values = np.take_along_axis(scores, indices, axis=1)
            order = np.argsort(-values, axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
            values = np.take_along_axis(values, order, axis=1)
            if (values[:, -1] <= -1.5).any():
                raise RuntimeError("predicted source domain has fewer than 64 neighbors")
            neighbors = labels[indices]
            for k in KS:
                result[mode, "uniform", k][start:end] = neighbors[:, :k].mean(axis=1)
                weights = np.exp((values[:, :k] - values[:, :1]) / NEIGHBOR_TEMPERATURE)
                result[mode, "soft", k][start:end] = (
                    (neighbors[:, :k] * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1)[:, None])
    return result


def metrics(y, c, action, bounds):
    selected = np.arange(len(action))
    accuracy = float(np.asarray(y[selected, action], dtype=np.float64).mean())
    usd_per_question = float(np.asarray(c[selected, action], dtype=np.float64).mean())
    return dict(n=len(action), accuracy_percent=100 * accuracy,
                cost_usd_per_10k=10000 * usd_per_question,
                rank_score=100 * rank_score(accuracy, usd_per_question, bounds["cmin"], bounds["cmax"]))


def route_actions(model, bank, rules, xq, xt, available):
    predicted = infer_domain(xt, model["tiny_coef"], model["tiny_intercept"])
    missing = ~available
    if missing.any():
        predicted[missing] = infer_domain(xq[missing], model["qwen_coef"], model["qwen_intercept"])
    near = nearest_quality(xt, bank, model["bank_y"], model["bank_domain"], predicted)
    actions = []
    for method_index, method in enumerate(METHODS):
        rule = rules[method]
        quality = model["quality"][method_index, predicted]
        if rule["blend"]:
            neighborhood = near[rule["neighbor_mode"], rule["neighbor_weight"], rule["neighbors"]]
            quality = (1 - rule["blend"]) * quality + rule["blend"] * neighborhood
            quality[missing] = model["quality"][method_index, predicted[missing]]
        cost = (model["domain_cost"][method_index, predicted] if rule["cost_mode"] == "domain"
                else np.broadcast_to(model["global_cost"][method_index], quality.shape))
        actions.append((quality - rule["lambda_cost"] * cost).argmax(axis=1).astype(np.int8))
    return np.stack(actions), predicted


def train(protocol):
    out, _ = paths(protocol)
    if (out / "SOURCE_SELECTION.json").exists():
        raise FileExistsError("TinyKNN source selection already exists")
    data, xq, xt, available, fit, names, domains, tiny_records = source_inputs(protocol)
    y, c = data["y"], data["c"]
    source_y, source_c = y[fit], c[fit]
    positions = {int(index): i for i, index in enumerate(fit)}
    n = len(fit)
    quality_oof = {kappa: np.empty((n, N_EXPERTS), dtype=np.float32) for kappa in KAPPAS}
    cost_domain_oof = np.empty((n, N_EXPERTS), dtype=np.float32)
    cost_global_oof = np.empty_like(cost_domain_oof)
    neighbor_oof = {(mode, weight, k): np.empty((n, N_EXPERTS), dtype=np.float32)
                    for mode in MODES for weight in WEIGHTS for k in KS}
    inferred_oof = np.empty(n, dtype=np.int8)
    coverage = np.zeros(n, dtype=np.int8)
    folds = []
    for fold in range(3):
        train_rows = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid_rows = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][train_rows]) & set(data["groups"][valid_rows]):
            raise RuntimeError("source dependency groups overlap")
        target = np.array([positions[int(row)] for row in valid_rows])
        tiny_head = fit_domain(xt[train_rows], domains[train_rows], len(names))
        qwen_head = fit_domain(xq[train_rows], domains[train_rows], len(names))
        predicted = infer_domain(xt[valid_rows], tiny_head.coef_, np.asarray(tiny_head.intercept_))
        missing = ~available[valid_rows]
        if missing.any():
            predicted[missing] = infer_domain(xq[valid_rows][missing], qwen_head.coef_,
                                               np.asarray(qwen_head.intercept_))
        inferred_oof[target] = predicted
        for kappa in KAPPAS:
            quality, domain_cost, global_cost, _ = profile(
                protocol, y[train_rows], c[train_rows], domains[train_rows], kappa)
            quality_oof[kappa][target] = quality[predicted]
            cost_domain_oof[target] = domain_cost[predicted]
            cost_global_oof[target] = global_cost
        neighbors = nearest_quality(xt[valid_rows], xt[train_rows], y[train_rows],
                                    domains[train_rows], predicted)
        for key, value in neighbors.items():
            neighbor_oof[key][target] = value
        coverage[target] += 1
        folds.append(dict(fold=fold, train_n=len(train_rows), valid_n=len(valid_rows), group_overlap_n=0))
        print(json.dumps(dict(stage="source_fold", protocol=protocol, **folds[-1])), flush=True)
    if not (coverage == 1).all():
        raise RuntimeError("incomplete source OOF coverage")
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    candidates = []
    missing_fit = ~available[fit]
    for kappa, base_quality in quality_oof.items():
        for mode in MODES:
            for weight in WEIGHTS:
                for k in KS:
                    near = neighbor_oof[mode, weight, k]
                    for blend in BLENDS:
                        quality = (1 - blend) * base_quality + blend * near
                        quality[missing_fit] = base_quality[missing_fit]
                        for cost_mode, cost in (("domain", cost_domain_oof), ("global", cost_global_oof)):
                            for lam in LAMBDAS:
                                action = (quality - lam * cost).argmax(axis=1)
                                candidates.append(dict(quality_kappa=kappa, neighbor_mode=mode,
                                                       neighbor_weight=weight, neighbors=k, blend=blend,
                                                       cost_mode=cost_mode, lambda_cost=lam,
                                                       **metrics(source_y, source_c, action, bounds)))
    rank_winner = max(candidates, key=lambda row: row["rank_score"])
    if protocol == "full14":
        eligible = [row for row in candidates if row["rank_score"] > 74.59 and row["cost_usd_per_10k"] <= 1.04]
        accuracy_criterion = "source OOF Rank > 74.59 and cost <= 1.04 USD/10k"
    else:
        rank_floor = rank_winner["rank_score"] - .5
        eligible = [row for row in candidates if row["rank_score"] >= rank_floor and
                    row["cost_usd_per_10k"] <= 1.04]
        accuracy_criterion = "source OOF Rank within 0.5 of best and cost <= 1.04 USD/10k"
    if not eligible:
        raise RuntimeError("no source accuracy candidate meets the guard")
    accuracy_winner = max(eligible, key=lambda row: (row["accuracy_percent"], row["rank_score"],
                                                     -row["cost_usd_per_10k"]))
    selected = {METHODS[0]: rank_winner, METHODS[1]: accuracy_winner}
    tiny_head = fit_domain(xt[fit], domains[fit], len(names))
    qwen_head = fit_domain(xq[fit], domains[fit], len(names))
    final_profiles = [profile(protocol, source_y, source_c, domains[fit], selected[method]["quality_kappa"])
                      for method in METHODS]
    out.mkdir(parents=True, exist_ok=True)
    bank_path = out / "BANK_FEATURES.npy"
    np.save(bank_path, xt[fit], allow_pickle=False)
    model_path = out / "MODEL.npz"
    np.savez_compressed(model_path, dataset_names=np.array(names),
                        tiny_coef=np.asarray(tiny_head.coef_), tiny_intercept=np.asarray(tiny_head.intercept_),
                        qwen_coef=np.asarray(qwen_head.coef_), qwen_intercept=np.asarray(qwen_head.intercept_),
                        quality=np.stack([p[0] for p in final_profiles]).astype(np.float32),
                        domain_cost=np.stack([p[1] for p in final_profiles]).astype(np.float32),
                        global_cost=np.stack([p[2] for p in final_profiles]).astype(np.float32),
                        bank_y=source_y.astype(np.uint8), bank_domain=domains[fit],
                        bank_uids=data["uids"][fit])
    diagnostic = []
    with np.load(model_path, allow_pickle=False) as model:
        bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        for role in ("qcal", "cert"):
            rows = np.flatnonzero(data["role"] == role)
            actions, inferred = route_actions(model, bank, selected, xq[rows], xt[rows], available[rows])
            for method, action in zip(METHODS, actions):
                diagnostic.append(dict(method=method, role=role,
                                       domain_prediction_accuracy=float((inferred == domains[rows]).mean()),
                                       **metrics(y[rows], c[rows], action, bounds)))
    write_csv(out / "SOURCE_CANDIDATES.csv", candidates)
    write_csv(out / "SOURCE_HOLDOUT_DIAGNOSTIC.csv", diagnostic)
    write_json(out / "SOURCE_SELECTION.json", dict(
        protocol=protocol, source_train_n=len(data["uids"]), source_fit_n=n,
        source_qcal_n=int((data["role"] == "qcal").sum()),
        source_cert_n=int((data["role"] == "cert").sum()),
        source_folds=folds, source_domain_inference_accuracy=float((inferred_oof == domains[fit]).mean()),
        candidate_grid=dict(quality_kappas=list(KAPPAS), neighbor_modes=list(MODES),
                            neighbor_weights=list(WEIGHTS), neighbors=list(KS), blends=list(BLENDS),
                            cost_modes=list(COST_MODES), lambdas=list(LAMBDAS),
                            ridge_alpha=.1, neighbor_temperature=NEIGHBOR_TEMPERATURE),
        selection=dict(rank="maximum source-fit grouped OOF Rank", accuracy=accuracy_criterion),
        selected=selected, source_holdout_diagnostic_not_used_for_selection=diagnostic,
        tiny_source_missing_n=int((~available).sum()), qwen_fallback_on_tiny_missing=True,
        source_dev_or_test_outcomes_used=False, source_dataset_names_used_for_training=True,
        test_dataset_names_used_for_actions=False,
        model=record(model_path), bank=record(bank_path),
        candidates=record(out / "SOURCE_CANDIDATES.csv"),
        holdout_diagnostic=record(out / "SOURCE_HOLDOUT_DIAGNOSTIC.csv"),
        tiny_features=[record(path) for path in tiny_records]))
    print(json.dumps(dict(stage="train", protocol=protocol, selected=selected, diagnostic=diagnostic)), flush=True)


def freeze(protocol):
    out, _ = paths(protocol)
    if (out / "MODEL_LOCK.json").exists():
        raise FileExistsError("TinyKNN model already frozen")
    selection = read_json(out / "SOURCE_SELECTION.json")
    if selection["protocol"] != protocol:
        raise RuntimeError("source selection protocol changed")
    subject.require_records([selection[key] for key in ("model", "bank", "candidates", "holdout_diagnostic")])
    subject.require_records(selection["tiny_features"])
    files = [Path(__file__), out / "SOURCE_SELECTION.json", out / "MODEL.npz",
             out / "BANK_FEATURES.npy", out / "SOURCE_CANDIDATES.csv",
             out / "SOURCE_HOLDOUT_DIAGNOSTIC.csv", TINY_POINTER,
             subject.POINTER, subject.OVERLAY_MANIFEST,
             ROOT / f"outputs/next_round_x/source/{protocol}/MANIFEST.json",
             ROOT / "data/matrices/cost_bounds.json"]
    files += [ROOT / entry["path"] for entry in selection["tiny_features"]]
    files = list(dict.fromkeys(files))
    write_json(out / "MODEL_LOCK.json", dict(
        status="TINY_KNN_SOURCE_RULES_FROZEN_BEFORE_TEST_ACTIONS", protocol=protocol,
        selected=selection["selected"], source_only_rule_selection=True,
        no_test_dataset_label_at_inference=True,
        test_exposure="historical test outcomes were visible in earlier work; exploratory replay",
        cost_scope="cached expert USD proxy; excludes TinyLLaVA and Qwen3-VL feature extraction",
        files=[record(path) for path in files]))
    print(json.dumps(dict(stage="freeze", protocol=protocol, files=len(files))), flush=True)


def verify_lock(protocol):
    out, _ = paths(protocol)
    lock = read_json(out / "MODEL_LOCK.json")
    if lock["protocol"] != protocol:
        raise RuntimeError("model lock protocol changed")
    subject.require_records(lock["files"])
    return lock


def test_uids(protocol):
    if protocol == "full14":
        uids = role_uids("full14", "test")
        views = np.array(["14-ID"] * len(uids))
    else:
        seen = role_uids("heldout11", "id")
        ood = role_uids("heldout11", "ood")
        uids = seen + ood
        views = np.array(["11-ID"] * len(seen) + ["3-OOD"] * len(ood))
        if len(seen) != 5341 or len(ood) != 767:
            raise RuntimeError("heldout11 test view counts changed")
    if len(uids) != 6108 or len(set(uids)) != len(uids):
        raise RuntimeError("test UID count/order changed")
    return uids, views


def predict(protocol):
    out, _ = paths(protocol)
    if (out / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("TinyKNN test actions already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = verify_lock(protocol)
    uids, views = test_uids(protocol)
    xt, available, _ = tiny_features(uids)
    if int((~available).sum()) != 10:
        raise RuntimeError("TinyLLaVA test multi-image coverage changed")
    xq, overlay, _ = subject.qwen_features(uids, allow_multi=True)
    if len(overlay) != 10:
        raise RuntimeError("Qwen test multi-image overlay coverage changed")
    with np.load(out / "MODEL.npz", allow_pickle=False) as model:
        bank = np.load(out / "BANK_FEATURES.npy", mmap_mode="r", allow_pickle=False)
        if bank.shape != (17871 if protocol == "full14" else 16269, 5120):
            raise RuntimeError("frozen neighbor bank shape changed")
        actions, inferred = route_actions(model, bank, lock["selected"], normalized(xq), xt, available)
        if len(model["dataset_names"]) != (14 if protocol == "full14" else 11):
            raise RuntimeError("frozen source dataset classes changed")
    if actions.shape != (2, 6108) or (actions < 0).any() or (actions >= N_EXPERTS).any():
        raise RuntimeError("invalid TinyKNN test actions")
    path = out / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), views=views, methods=np.array(METHODS),
                        actions=actions, inferred_domain=inferred, tiny_available=available)
    write_json(out / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested further optimization with source-only rule selection",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(out / "MODEL_LOCK.json"),
        all_test_rows_have_qwen_features=True, tiny_missing_n=int((~available).sum()),
        qwen_fallback_on_tiny_missing=True, test_dataset_name_used_for_action=False,
        labels_or_actual_cost_used=False, test_outcomes_opened=0, test_fit_performed=False,
        denied_attempts=denied))
    print(json.dumps(dict(stage="predict", protocol=protocol, n=len(uids),
                          tiny_missing=int((~available).sum()),
                          experts=[int(len(set(action))) for action in actions])), flush=True)


def prior_test_actions(protocol, uids):
    if protocol == "full14":
        paths_and_labels = ((prior14.OUT, "Previous-Qwen-SourceRank"),
                            (ROOT / "outputs/stateroute_domain_accuracy_full14_20260928",
                             "Previous-Qwen-SourceAccuracy"))
        result = {}
        for directory, label in paths_and_labels:
            path = directory / "PREDICTIONS.npz"
            receipt = read_json(directory / "PREDICTION_RECEIPT.json")
            if digest(path) != receipt["prediction_sha256"]:
                raise RuntimeError("previous full14 test actions changed")
            with np.load(path, allow_pickle=False) as pred:
                if pred["uids"].tolist() != uids:
                    raise RuntimeError("previous full14 test UID order changed")
                result[label] = pred["actions"].astype(np.int8)
        return result
    directory = prior11.OUT
    path = directory / "PREDICTIONS.npz"
    receipt = read_json(directory / "PREDICTION_RECEIPT.json")
    if digest(path) != receipt["prediction_sha256"]:
        raise RuntimeError("previous heldout11 test actions changed")
    with np.load(path, allow_pickle=False) as pred:
        if pred["uids"].tolist() != uids or pred["methods"].tolist() != list(prior11.METHODS):
            raise RuntimeError("previous heldout11 test UID/method order changed")
        return {"Previous-Qwen-SourceRank": pred["actions"][0].astype(np.int8),
                "Previous-Qwen-SourceAccuracy": pred["actions"][1].astype(np.int8)}


def evaluate(protocol):
    out, report = paths(protocol)
    if (report / "RESULTS.json").exists():
        raise FileExistsError("TinyKNN test already evaluated")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock
    lock = verify_lock(protocol)
    path = out / "PREDICTIONS.npz"
    receipt_path = out / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(path) != receipt["prediction_sha256"] or digest(out / "MODEL_LOCK.json") != receipt["model_lock"]["sha256"]:
        raise RuntimeError("frozen TinyKNN prediction/model binding changed")
    uids, expected_views = test_uids(protocol)
    with np.load(path, allow_pickle=False) as pred:
        if pred["uids"].tolist() != uids or pred["methods"].tolist() != list(METHODS):
            raise RuntimeError("frozen TinyKNN prediction UID/method order changed")
        views = pred["views"].copy()
        actions = pred["actions"].astype(np.int64)
        inferred = pred["inferred_domain"].astype(np.int64)
        available = pred["tiny_available"].copy()
    if not np.array_equal(views, expected_views) or actions.shape != (2, 6108) or int((~available).sum()) != 10:
        raise RuntimeError("frozen TinyKNN test view/shape changed")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **subject.label_args())
    write_json(out / "OUTCOME_PROVENANCE.json", dict(materialization=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    if y.shape != (6108, N_EXPERTS) or c.shape != y.shape:
        raise RuntimeError("test expert outcome shape changed")
    datasets = np.array([uid.split("/", 1)[0] for uid in uids])
    prior = prior_test_actions(protocol, uids)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    with np.load(out / "MODEL.npz", allow_pickle=False) as model:
        names = model["dataset_names"].tolist()
    view_names = ("14-ID",) if protocol == "full14" else ("11-ID", "3-OOD")
    summary, per_dataset = [], []
    all_routes = {METHODS[0]: actions[0], METHODS[1]: actions[1], **prior}
    for method, action in all_routes.items():
        for view in view_names:
            mask = views == view
            score = metrics(y[mask], c[mask], action[mask], bounds)
            macro = np.mean([metrics(y[mask & (datasets == name)], c[mask & (datasets == name)],
                                     action[mask & (datasets == name)], bounds)["accuracy_percent"]
                             for name in sorted(set(datasets[mask]))])
            inferred_accuracy = (float(100 * np.mean(inferred[mask] == np.array(
                [names.index(name) for name in datasets[mask]]))) if view != "3-OOD" else None)
            summary.append(dict(method=method, view=view, macro_dataset_accuracy_percent=float(macro),
                                inferred_domain_accuracy_percent=inferred_accuracy if method in METHODS else None,
                                **score))
        if method in METHODS:
            for name in sorted(set(datasets)):
                mask = datasets == name
                per_dataset.append(dict(method=method, dataset=name, view=str(views[np.flatnonzero(mask)[0]]),
                                        **metrics(y[mask], c[mask], action[mask], bounds)))
    deltas = []
    for method, previous in zip(METHODS, ("Previous-Qwen-SourceRank", "Previous-Qwen-SourceAccuracy")):
        for view in view_names:
            new = next(row for row in summary if row["method"] == method and row["view"] == view)
            old = next(row for row in summary if row["method"] == previous and row["view"] == view)
            deltas.append(dict(method=method, view=view, compared_to=previous,
                               accuracy_pp=new["accuracy_percent"] - old["accuracy_percent"],
                               cost_usd_per_10k=new["cost_usd_per_10k"] - old["cost_usd_per_10k"],
                               rank_score=new["rank_score"] - old["rank_score"]))
    report.mkdir(parents=True, exist_ok=True)
    write_csv(report / "summary.csv", summary)
    write_csv(report / "per_dataset.csv", per_dataset)
    write_csv(report / "paired_deltas.csv", deltas)
    write_json(report / "RESULTS.json", dict(
        protocol=protocol, source_selection=record(out / "SOURCE_SELECTION.json"),
        model_lock=record(out / "MODEL_LOCK.json"), prediction=record(path),
        source_fit_n=17871 if protocol == "full14" else 16269,
        test_n=6108, test_metric_selection=False, summary=summary,
        paired_deltas=deltas, tiny_missing_test_n=int((~available).sum()),
        historical_test_exposure="exploratory replay; previous results visible",
        cost_scope=lock["cost_scope"]))
    lines = [f"# TinyLLaVA 局部近邻专家路由：{protocol}", "",
             "平衡 Ridge 根据 TinyLLaVA 冻结特征推断已知数据集；源域专家画像与同类/全域近邻正确率按源域 OOF 选择的权重混合。TinyLLaVA 缺失的 10 条多图测试题使用 Qwen 数据集分类器及源域画像。", "",
             "训练只使用官方 source-fit，三折按依赖组隔离；qcal/cert 只做选参后的诊断。全部测试动作冻结并记录哈希后才读取结果。", "",
             "| 方法 | 视图 | Accuracy % | 专家费用 USD/万题 | Rank |", "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['view']} | {row['accuracy_percent']:.3f} | "
                     f"{row['cost_usd_per_10k']:.4f} | {row['rank_score']:.3f} |")
    lines += ["", "同题旧方法差值见 `paired_deltas.csv`；完整数值和冻结绑定见 `RESULTS.json`。",
              "专家费用仅包含缓存 USD proxy，不含 TinyLLaVA/Qwen 特征提取。历史测试结果已经曝光，本轮是探索性回放；11→3 的 OOD 专家结果不参与拟合和选参。"]
    (report / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(stage="evaluate", protocol=protocol, summary=summary, deltas=deltas)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("full14", "heldout11"), required=True)
    parser.add_argument("--stage", choices=("train", "freeze", "predict", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=12):
        {"train": train, "freeze": freeze, "predict": predict, "evaluate": evaluate}[args.stage](args.protocol)


if __name__ == "__main__":
    main()
