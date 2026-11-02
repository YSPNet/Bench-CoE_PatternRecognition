#!/usr/bin/env python3
"""Replace StateRoute-MM KMeans states with 30 MMMU subject prototypes.

Run, in order:
  python scripts/run_stateroute_mmmu_subject_20260928.py --stage prototypes
  python scripts/run_stateroute_mmmu_subject_20260928.py --stage train
  python scripts/run_stateroute_mmmu_subject_20260928.py --stage freeze
  python scripts/run_stateroute_mmmu_subject_20260928.py --stage predict
  python scripts/run_stateroute_mmmu_subject_20260928.py --stage evaluate

MMMU official train inputs and native subject labels define prototypes. Only
heldout11 source-fit expert Y/C trains the expert map. The 11-ID test Y/C are
opened only after the test actions are frozen. No OOD expert outcome is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from routers.bench_coe_query.assets import TSVRowIndex
from routers.utils.rank_score import rank_score
from scripts.gemma_subject.run_pilot import SUBJECTS, native_subject_from_uid
from src.next_round_x.data import FeatureCache, load_source, manifests, role_uids
from src.next_round_x.models import normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

OUT = ROOT / "outputs/stateroute_mmmu_subject_20260928"
REPORT = ROOT / "reports/stateroute_mmmu_subject_20260928"
POINTER = ROOT / "outputs/bench_coe_query/features/qwen3vl/LATEST_FULL.json"
OVERLAY = ROOT / "outputs/stateroute_multi_image_20260918/multi_image_features.npz"
OVERLAY_MANIFEST = ROOT / "outputs/stateroute_multi_image_20260918/FEATURE_MANIFEST.json"
TSV = ROOT / "vlm_router_data/TSV_images/MMMU_DEV_VAL.tsv"
MODEL_PATH = "/home/sm5/ys/Project/SegR/models/Qwen3-VL-2B-Instruct"
LAMBDAS = (0., .001, .003, .01, .03, .1, .3)
KAPPA = 64.
N_SUBJECTS = 30


def say(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def require_records(items):
    for item in items:
        path = ROOT / item["path"]
        if not item["exists"] or not path.is_file() or digest(path) != item["sha256"]:
            raise RuntimeError(f"frozen artifact changed: {path}")


def subject_labels(uids):
    index = TSVRowIndex(TSV, "id")
    labels = []
    for uid in uids:
        subject = native_subject_from_uid(str(uid))
        _, row = index.row_by_key(str(uid).split("/", 1)[1])
        if row.get("l2-category") != subject:
            raise RuntimeError(f"UID/TSV subject mismatch: {uid}")
        labels.append(subject)
    return np.array(labels)


def qwen_features(uids, *, allow_multi):
    """Read only hash-bound features from the requested Qwen3-VL model."""
    pointer = read_json(POINTER)
    if pointer["input_content"]["model_configured_path"] != MODEL_PATH:
        raise RuntimeError("feature cache uses a different Qwen3-VL model")
    cache = FeatureCache()
    x, available = cache.rows(uids)
    missing = [i for i, ok in enumerate(available) if not ok]
    overlay_used = []
    if missing and allow_multi:
        manifest = read_json(OVERLAY_MANIFEST)
        require_records([manifest["feature_file"], record(OVERLAY_MANIFEST)])
        if manifest["model"] != "Qwen3-VL-2B-Instruct" or manifest["old_cache_root"] != pointer["cache_root"]:
            raise RuntimeError("multi-image overlay does not match the frozen cache")
        with np.load(OVERLAY, allow_pickle=False) as supplement:
            positions = {str(uid): i for i, uid in enumerate(supplement["uids"])}
            if len(positions) != len(supplement["uids"]):
                raise RuntimeError("duplicate overlay UID")
            for i in missing:
                uid = str(uids[i])
                if uid not in positions:
                    raise RuntimeError(f"Qwen3-VL feature missing: {uid}")
                x[i] = supplement["features"][positions[uid]]
                available[i] = True
                overlay_used.append(uid)
    if not available.all() or x.shape != (len(uids), 2048) or not np.isfinite(x).all():
        raise RuntimeError("incomplete/nonfinite Qwen3-VL features")
    if (np.linalg.norm(x, axis=1) < 1e-8).any():
        raise RuntimeError("zero Qwen3-VL feature vector")
    return x, overlay_used, cache


def build_centroids(features, labels, subject_names=SUBJECTS):
    """Spherical mean: unit-normalize each sample, average, then normalize."""
    x = normalized(features)
    names = list(subject_names)
    if len(names) != len(set(names)):
        raise ValueError("duplicate subject")
    centers = []
    support = []
    for subject in names:
        rows = x[np.asarray(labels) == subject]
        if not len(rows):
            raise ValueError(f"empty MMMU subject: {subject}")
        centers.append(rows.mean(axis=0))
        support.append(len(rows))
    result = normalized(np.asarray(centers, dtype=np.float32))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite subject center")
    return result, np.asarray(support, dtype=np.int64)


def assign_subjects(features, centers, *, batch=2048):
    x = normalized(features)
    if x.shape[1] != centers.shape[1] or not np.isfinite(x).all():
        raise ValueError("invalid query features")
    states = np.empty(len(x), dtype=np.int64)
    similarities = np.empty(len(x), dtype=np.float32)
    margins = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch):
        sim = x[start:start + batch] @ centers.T
        best = sim.argmax(axis=1)
        states[start:start + batch] = best
        similarities[start:start + batch] = sim[np.arange(len(best)), best]
        margins[start:start + batch] = sim.max(axis=1) - np.partition(sim, -2, axis=1)[:, -2]
    return states, similarities, margins


def quality_profile(y, states, *, n_subjects=N_SUBJECTS, kappa=KAPPA):
    if y.ndim != 2 or len(y) != len(states):
        raise ValueError("state/outcome alignment")
    if len(y) == 0 or (np.asarray(states) < 0).any() or (np.asarray(states) >= n_subjects).any():
        raise ValueError("invalid state support")
    count = np.bincount(states, minlength=n_subjects)
    sums = np.zeros((n_subjects, y.shape[1]), dtype=np.float64)
    np.add.at(sums, states, y)
    rates = (sums + kappa * y.mean(axis=0)) / (count[:, None] + kappa)
    return rates, count


def metrics(y, c, actions):
    selected = np.arange(len(actions))
    accuracy = float(np.asarray(y[selected, actions], dtype=np.float64).mean())
    cost = float(np.asarray(c[selected, actions], dtype=np.float64).mean())
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    return dict(n=len(actions), accuracy_percent=100 * accuracy,
                cost_usd_per_10k=10000 * cost,
                rank_score=100 * rank_score(accuracy, cost, bounds["cmin"], bounds["cmax"]))


def prototypes():
    if (OUT / "PROTOTYPES.npz").exists():
        raise FileExistsError("prototypes already exist")
    train = role_uids("full14", "train")
    uids = [uid for uid in train if uid.startswith("MMMU_DEV_VAL/")]
    if len(uids) != 735 or len(set(uids)) != len(uids):
        raise RuntimeError("MMMU official train split changed")
    labels = subject_labels(uids)
    x, overlay_used, cache = qwen_features(uids, allow_multi=True)
    if len(overlay_used) != 32 or set(labels) != set(SUBJECTS):
        raise RuntimeError("MMMU train subject/feature coverage changed")
    centers, support = build_centroids(x, labels)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "PROTOTYPES.npz"
    np.savez_compressed(path, uids=np.array(uids), subjects=np.array(SUBJECTS),
                        native_labels=labels, centers=centers, support=support)
    write_json(OUT / "PROTOTYPES.json", dict(
        method="Qwen3-VL frozen multimodal feature; per-query L2 normalization; per-subject mean; centroid L2 normalization; maximum cosine",
        input_role="MMMU_DEV_VAL official train only", input_n=len(uids), subject_n=len(SUBJECTS),
        support=dict(zip(SUBJECTS, map(int, support))), multi_image_overlay_n=len(overlay_used),
        multi_image_overlay_uids=overlay_used, model_path=MODEL_PATH,
        labels="native MMMU TSV l2-category, checked against UID",
        mmmu_expert_y_or_c_read=False, mmmu_dev_or_test_used_for_centers=False,
        sources=[record(p) for p in (POINTER, cache.root / "manifest.json", cache.root / "features.npy",
                                     cache.root / "index.jsonl", OVERLAY, OVERLAY_MANIFEST, TSV,
                                     ROOT / "artifacts/subject_routing/manifests/splits.json")],
        artifact=record(path)))
    say(stage="prototypes", n=len(uids), subjects=len(SUBJECTS), multi_image=len(overlay_used))


def train():
    if (OUT / "SELECTION.json").exists():
        raise FileExistsError("source selection already exists")
    proto = read_json(OUT / "PROTOTYPES.json")
    require_records(proto["sources"] + [proto["artifact"]])
    with np.load(OUT / "PROTOTYPES.npz", allow_pickle=False) as p:
        centers = np.asarray(p["centers"], dtype=np.float32)
        if p["subjects"].tolist() != SUBJECTS or centers.shape != (30, 2048):
            raise RuntimeError("prototype contract changed")
    data = load_source("heldout11")
    if data["uids"].tolist() != role_uids("heldout11", "train"):
        raise RuntimeError("source train UID order changed")
    if set(data["dataset"]) & {"MMMU_DEV_VAL", "ChartQA_TEST", "MathVision_MINI"}:
        raise RuntimeError("source expert profile includes an original OOD dataset")
    fit = np.flatnonzero(data["role"] == "fit")
    if len(fit) != 16269 or not data["available"][fit].all():
        raise RuntimeError("source fit feature coverage changed")
    states, sim, margin = assign_subjects(data["x"][fit], centers)
    fit_rows = {int(row): i for i, row in enumerate(fit)}
    oof_quality = np.empty((len(fit), data["y"].shape[1]), dtype=np.float64)
    oof_cost = np.empty_like(oof_quality)
    oof_seen = np.zeros(len(fit), dtype=np.int8)
    folds = []
    for fold in range(3):
        fit_train = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        fit_valid = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][fit_train]) & set(data["groups"][fit_valid]):
            raise RuntimeError("OOF dependency group overlap")
        train_positions = np.array([fit_rows[int(i)] for i in fit_train])
        valid_positions = np.array([fit_rows[int(i)] for i in fit_valid])
        rates, counts = quality_profile(data["y"][fit_train], states[train_positions])
        oof_quality[valid_positions] = rates[states[valid_positions]]
        c_train = data["c"][fit_train]
        oof_cost[valid_positions] = c_train.mean(axis=0) / float(np.median(c_train[c_train > 0]))
        oof_seen[valid_positions] += 1
        folds.append(dict(fold=fold, train_n=len(fit_train), valid_n=len(fit_valid),
                          zero_support_states=int((counts == 0).sum())))
    if not np.all(oof_seen == 1) or not np.isfinite(oof_quality).all() or not np.isfinite(oof_cost).all():
        raise RuntimeError("OOF predictions are incomplete or duplicated")
    rows = []
    for lam in LAMBDAS:
        actions = (oof_quality - lam * oof_cost).argmax(axis=1)
        score = metrics(data["y"][fit], data["c"][fit], actions)
        # Match the original StateRoute-MM selection calculation (float32 Y mean).
        exact_source_accuracy = float(data["y"][fit][np.arange(len(fit)), actions].mean())
        bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
        selection_rank = 100 * rank_score(exact_source_accuracy, score["cost_usd_per_10k"] / 10000,
                                          bounds["cmin"], bounds["cmax"])
        rows.append(dict(lambda_cost=lam, selection_rank=selection_rank, **score))
    winner = max(rows, key=lambda row: row["selection_rank"])
    rates, support = quality_profile(data["y"][fit], states)
    c_fit = data["c"][fit]
    scaled_cost = c_fit.mean(axis=0) / float(np.median(c_fit[c_fit > 0]))
    table = (rates - winner["lambda_cost"] * scaled_cost).argmax(axis=1)
    np.savez_compressed(OUT / "MODEL.npz", subjects=np.array(SUBJECTS), centers=centers,
                        quality=rates, scaled_cost=scaled_cost, support=support,
                        expert_table=table)
    np.savez_compressed(OUT / "OOF.npz", uids=data["uids"][fit], states=states,
                        quality=oof_quality.astype(np.float32), scaled_cost=oof_cost.astype(np.float32))
    write_csv(OUT / "OOF_RESULTS.csv", rows)
    _, pool = manifests()
    write_json(OUT / "SELECTION.json", dict(
        protocol="heldout11", selection="max pooled 3-fold source-fit OOF Rank Score; first lambda on ties",
        selection_lambda_grid=list(LAMBDAS), selected=winner, kappa=KAPPA,
        fit_role="fit within official 11-source train", fit_n=len(fit), folds=folds,
        source_fit_state_counts=dict(zip(SUBJECTS, map(int, support))),
        cosine_similarity_mean=float(sim.mean()), cosine_margin_mean=float(margin.mean()),
        expert_order=pool["canonical_order"],
        selected_experts={subject: pool["canonical_order"][int(table[j])] for j, subject in enumerate(SUBJECTS)},
        all_wrong_fit_rows=int((data["y"][fit].sum(axis=1) == 0).sum()),
        mmmu_expert_y_or_c_read=False, id_test_y_or_c_read=False,
        model=record(OUT / "MODEL.npz"), oof=record(OUT / "OOF.npz")))
    say(stage="train", fit_n=len(fit), lambda_cost=winner["lambda_cost"],
        oof_accuracy=winner["accuracy_percent"], oof_rank=winner["rank_score"],
        used_states=int((support > 0).sum()))


def freeze():
    if (OUT / "MODEL_LOCK.json").exists():
        raise FileExistsError("model is already locked")
    selection = read_json(OUT / "SELECTION.json")
    proto = read_json(OUT / "PROTOTYPES.json")
    require_records(proto["sources"] + [proto["artifact"], selection["model"], selection["oof"]])
    records = [record(p) for p in (
        Path(__file__), ROOT / "src/next_round_x/data.py", ROOT / "src/next_round_x/models.py",
        ROOT / "routers/utils/rank_score.py", OUT / "PROTOTYPES.json", OUT / "PROTOTYPES.npz",
        OUT / "SELECTION.json", OUT / "MODEL.npz", OUT / "OOF.npz", OUT / "OOF_RESULTS.csv",
        ROOT / "outputs/next_round_x/source/heldout11/MANIFEST.json",
        ROOT / "artifacts/subject_routing/manifests/splits.json",
        ROOT / "artifacts/subject_routing/manifests/model_pool.json",
        ROOT / "data/matrices/cost_bounds.json")]
    write_json(OUT / "MODEL_LOCK.json", dict(
        status="FROZEN_BEFORE_NEW_11_ID_TEST_ACTIONS_AND_OUTCOMES", files=records,
        model_path=MODEL_PATH, source_fit_n=16269, prototype_n=735, subject_n=30,
        test_selection=False, original_ood_expert_outcomes_used=False,
        exposure="historical 11-ID test results already exposed; exploratory comparison",
        cost_scope="cached expert USD proxy only; excludes Qwen3-VL feature extraction overhead"))
    say(stage="freeze", files=len(records))


def predict():
    if (OUT / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("predictions already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = read_json(OUT / "MODEL_LOCK.json")
    require_records(lock["files"])
    uids = role_uids("heldout11", "id")
    if len(uids) != 5341 or len(set(uids)) != len(uids):
        raise RuntimeError("11-ID test split changed")
    x, overlay_used, _ = qwen_features(uids, allow_multi=False)
    if overlay_used:
        raise RuntimeError("unexpected ID multi-image overlay")
    with np.load(OUT / "MODEL.npz", allow_pickle=False) as model:
        states, sim, margin = assign_subjects(x, model["centers"])
        actions = model["expert_table"][states]
    if actions.shape != (5341,) or (actions < 0).any() or (actions >= 17).any():
        raise RuntimeError("invalid expert actions")
    path = OUT / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), states=states, actions=actions,
                        cosine_similarity=sim, cosine_margin=margin,
                        subjects=np.array(SUBJECTS))
    write_json(OUT / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested MMMU subject-state 11-ID experiment",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(OUT / "MODEL_LOCK.json"), labels_or_actual_cost_used=False,
        test_outcomes_opened=0, test_fit_performed=False, denied_attempts=denied))
    say(stage="predict", n=len(uids), states=int(len(set(states))), distinct_experts=int(len(set(actions))))


def label_args():
    from scripts.subject_routing_v2.run_b00_query_teacher import _common_label_arguments
    split, pool = manifests()
    return _common_label_arguments(
        experts=pool["canonical_order"], model_pool=pool, split_manifest=split,
        matrix_audit=read_json(ROOT / "artifacts/subject_routing/audits/P02_matrix_semantics.json"))


def baseline_actions(path, method, uids):
    with np.load(path, allow_pickle=False) as prediction:
        if prediction["uids"][:len(uids)].tolist() != uids:
            raise RuntimeError(f"baseline UID order changed: {path}")
        rows = np.flatnonzero(prediction["methods"] == method)
        if len(rows) != 5 or set(map(int, prediction["seeds"][rows])) != set(range(5)):
            raise RuntimeError(f"baseline five-seed contract changed: {method}")
        return np.asarray(prediction["actions"][rows, :len(uids)], dtype=np.int64)


def evaluate():
    if (REPORT / "RESULTS.json").exists():
        raise FileExistsError("test evaluation already exists")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock

    lock = read_json(OUT / "MODEL_LOCK.json")
    require_records(lock["files"])
    path = OUT / "PREDICTIONS.npz"
    receipt_path = OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(path) != receipt["prediction_sha256"] or receipt["model_lock"]["sha256"] != digest(OUT / "MODEL_LOCK.json"):
        raise RuntimeError("frozen prediction/model binding changed")
    with np.load(path, allow_pickle=False) as prediction:
        uids = prediction["uids"].tolist()
        states = prediction["states"].copy()
        actions = prediction["actions"].copy()
        sim = prediction["cosine_similarity"].copy()
        margin = prediction["cosine_margin"].copy()
    if uids != role_uids("heldout11", "id"):
        raise RuntimeError("test UID order changed")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **label_args())
    write_json(OUT / "OUTCOME_PROVENANCE.json", dict(materialization=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    result = metrics(y, c, actions)
    comparisons = []
    baseline_paths = {
        "StateRoute-MM-K2": ROOT / "outputs/stateroute_k_sweep_20260916/predictions.npz",
        "StateRoute-MM-K12": ROOT / "outputs/stateroute_k10_k12_20260916/predictions.npz",
    }
    baseline_by_method = {}
    for method, baseline_path in baseline_paths.items():
        baseline = baseline_actions(baseline_path, method, uids)
        baseline_by_method[method] = baseline
        values = [metrics(y, c, a) for a in baseline]
        average = {key: float(np.mean([v[key] for v in values]))
                   for key in ("accuracy_percent", "cost_usd_per_10k", "rank_score")}
        comparisons.append(dict(method=method, seeds=5, baseline_mean=average,
                                baseline_sd={key: float(np.std([v[key] for v in values], ddof=1))
                                             for key in average},
                                delta_new_minus_baseline_mean={key: result[key] - average[key] for key in average},
                                baseline_prediction=record(baseline_path)))
    datasets = np.array([uid.split("/", 1)[0] for uid in uids])
    per_dataset = []
    for dataset in sorted(set(datasets)):
        take = datasets == dataset
        row = dict(dataset=dataset, **metrics(y[take], c[take], actions[take]))
        for method, baseline in baseline_by_method.items():
            prefix = method.replace("StateRoute-", "").replace("-", "_")
            baseline_values = [metrics(y[take], c[take], a[take]) for a in baseline]
            for key in ("accuracy_percent", "cost_usd_per_10k", "rank_score"):
                mean = float(np.mean([v[key] for v in baseline_values]))
                row[f"{prefix}_{key}_mean"] = mean
                row[f"delta_vs_{prefix}_{key}"] = row[key] - mean
        per_dataset.append(row)
    with np.load(OUT / "MODEL.npz", allow_pickle=False) as model:
        table = model["expert_table"].copy()
        support = model["support"].copy()
    _, pool = manifests()
    per_subject = [dict(subject=subject, mmmu_train_n=int(read_json(OUT / "PROTOTYPES.json")["support"][subject]),
                        source_fit_n=int(support[j]), id_test_n=int((states == j).sum()),
                        expert=pool["canonical_order"][int(table[j])],
                        mean_max_cosine=float(sim[states == j].mean()) if (states == j).any() else None,
                        mean_margin=float(margin[states == j].mean()) if (states == j).any() else None)
                   for j, subject in enumerate(SUBJECTS)]
    # Input-only diagnostic on the official MMMU dev role. It never affects the map.
    mmmu_dev = [uid for uid in role_uids("full14", "dev") if uid.startswith("MMMU_DEV_VAL/")]
    dev_x, dev_overlay, _ = qwen_features(mmmu_dev, allow_multi=True)
    with np.load(OUT / "PROTOTYPES.npz", allow_pickle=False) as proto:
        dev_states, _, _ = assign_subjects(dev_x, proto["centers"])
    dev_gold = subject_labels(mmmu_dev)
    dev_correct = np.array(SUBJECTS)[dev_states] == dev_gold
    dev_diagnostic = dict(n=len(mmmu_dev), correct=int(dev_correct.sum()),
                          subject_accuracy_percent=float(100 * dev_correct.mean()),
                          multi_image_overlay_n=len(dev_overlay), role="MMMU official dev; semantic diagnostic only",
                          expert_y_or_c_read=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    write_csv(REPORT / "per_subject.csv", per_subject)
    np.savez_compressed(OUT / "EVALUATION_TRACES.npz", uids=np.array(uids), states=states,
                        actions=actions, correct=y[np.arange(len(uids)), actions],
                        cost=c[np.arange(len(uids)), actions])
    write_json(REPORT / "RESULTS.json", dict(
        method="StateRoute-MM-MMMU30-Cosine", view="11-ID", deterministic=True, seeds=1,
        main=result, baselines=comparisons, per_dataset=per_dataset, per_subject=per_subject,
        mmmu_dev_subject_diagnostic=dev_diagnostic, prediction=record(path),
        model_lock=record(OUT / "MODEL_LOCK.json"), all_wrong_rows_retained=True,
        original_ood_expert_outcomes_used=False,
        cost_scope="cached expert USD proxy only; Qwen3-VL encoding cost excluded",
        interpretation="exploratory replay on previously exposed 11-ID test outcomes"))
    lines = ["# MMMU 学科原型替换 StateRoute-MM 状态：11-ID 实验", "",
             "使用 MMMU 官方 train 735 题构造 30 个学科原型；复用指定 Qwen3-VL-2B-Instruct 的冻结图文特征及 32 条多图补齐特征。",
             "11 个源数据集的 fit 16,269 题训练专家画像，三折 OOF 选择费用系数。测试动作先冻结，再读取 5,341 条 11-ID 测试题的专家正确性和费用。",
             "", "| 路由 | 准确率 | 专家费用 USD/万题 | Rank Score |", "|---|---:|---:|---:|"]
    lines.append(f"| MMMU30-Cosine | {result['accuracy_percent']:.3f}% | {result['cost_usd_per_10k']:.4f} | {result['rank_score']:.3f} |")
    for row in comparisons:
        x = row["baseline_mean"]
        lines.append(f"| {row['method']}（5 种子均值） | {x['accuracy_percent']:.3f}% | {x['cost_usd_per_10k']:.4f} | {x['rank_score']:.3f} |")
    lines += ["", f"MMMU dev 学科识别诊断：{dev_diagnostic['correct']}/{dev_diagnostic['n']} = {dev_diagnostic['subject_accuracy_percent']:.2f}%。",
              "", "费用仅包含缓存中的专家 USD proxy，未计入 Qwen3-VL 特征提取开销。MMMU train 的输入及原生学科标签用于定义状态，因此这不是原始 11→3 完全留出 MMMU 的 OOD 协议；本报告仅评价 11-ID。历史测试结果已暴露，属于探索性复跑。",
              "", "逐数据集结果见 `per_dataset.csv`；学科占用和专家映射见 `per_subject.csv`；完整数值及绑定见 `RESULTS.json`。"]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(stage="evaluate", main=result, baselines=[(x["method"], x["baseline_mean"]) for x in comparisons],
        mmmu_dev=dev_diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("prototypes", "train", "freeze", "predict", "evaluate"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"prototypes": prototypes, "train": train, "freeze": freeze,
         "predict": predict, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
