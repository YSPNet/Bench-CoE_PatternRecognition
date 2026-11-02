#!/usr/bin/env python3
"""Matched 30-state StateRoute-MM KMeans control for the MMMU prototype run.

Run --stage train, freeze, predict, evaluate in separate processes. The
source-fit OOF lambda and the MMMU prototype's source-selected lambda are both
evaluated. Neither choice consults 11-ID test outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from routers.utils.rank_score import rank_score
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import FeatureCache, load_source, role_uids
from src.next_round_x.models import cluster_quality, normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

OUT = ROOT / "outputs/stateroute_mmmu_k30_control_20260928"
REPORT = ROOT / "reports/stateroute_mmmu_k30_control_20260928"
SEEDS = (0, 1, 2, 3, 4)
K = 30


def say(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def train():
    if (OUT / "SELECTION.json").exists():
        raise FileExistsError("K30 source selection already exists")
    data = load_source("heldout11")
    if data["uids"].tolist() != role_uids("heldout11", "train"):
        raise RuntimeError("source UID order changed")
    fit = np.flatnonzero(data["role"] == "fit")
    if len(fit) != 16269 or not data["available"][fit].all():
        raise RuntimeError("source fit coverage changed")
    position = {int(index): row for row, index in enumerate(fit)}
    oof_quality = np.empty((len(fit), 17), dtype=np.float64)
    oof_cost = np.empty_like(oof_quality)
    seen = np.zeros(len(fit), dtype=np.int8)
    for fold in range(3):
        train_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][train_indices]) & set(data["groups"][valid_indices]):
            raise RuntimeError("K30 OOF dependency group overlap")
        target = np.array([position[int(i)] for i in valid_indices])
        oof_quality[target], _, _ = cluster_quality(data["x"][train_indices], data["y"][train_indices],
                                                     data["x"][valid_indices], K, 0)
        costs = data["c"][train_indices]
        oof_cost[target] = costs.mean(axis=0) / float(np.median(costs[costs > 0]))
        seen[target] += 1
        say(stage="oof", fold=fold, train_n=len(train_indices), valid_n=len(valid_indices))
    if not (seen == 1).all() or not np.isfinite(oof_quality).all():
        raise RuntimeError("K30 OOF coverage incomplete")
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    rows = []
    for lam in subject.LAMBDAS:
        actions = (oof_quality - lam * oof_cost).argmax(axis=1)
        score = subject.metrics(data["y"][fit], data["c"][fit], actions)
        source_acc = float(data["y"][fit][np.arange(len(fit)), actions].mean())
        selection_rank = 100 * rank_score(source_acc, score["cost_usd_per_10k"] / 10000,
                                          bounds["cmin"], bounds["cmax"])
        rows.append(dict(lambda_cost=lam, selection_rank=selection_rank, **score))
    best = max(rows, key=lambda row: row["selection_rank"])
    fixed_lambda = read_json(subject.OUT / "SELECTION.json")["selected"]["lambda_cost"]
    mean_c = data["c"][fit].mean(axis=0)
    cost_scale = float(np.median(data["c"][fit][data["c"][fit] > 0]))
    model_files = []
    for seed in SEEDS:
        _, km, rates = cluster_quality(data["x"][fit], data["y"][fit], data["x"][fit[:1]], K, seed)
        support = np.bincount(km.predict(normalized(data["x"][fit])), minlength=K)
        center = km.cluster_centers_.astype(np.float32)
        scale_cost = mean_c / cost_scale
        dest = OUT / "models" / f"seed-{seed}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, centers=center, rates=rates, support=support,
                            table_oof=(rates - best["lambda_cost"] * scale_cost).argmax(axis=1),
                            table_fixed=(rates - fixed_lambda * scale_cost).argmax(axis=1))
        model_files.append(record(dest))
        say(stage="fit", seed=seed, occupied_states=int((support > 0).sum()))
    write_csv(OUT / "OOF_RESULTS.csv", rows)
    write_json(OUT / "SELECTION.json", dict(
        control="StateRoute-MM-K30", source_fit_n=len(fit), seeds=list(SEEDS), k=K,
        kmeans=dict(n_init=3, batch_size=1024, max_iter=100, distance="squared Euclidean on L2-normalized embeddings"),
        profile_kappa=64, oof_fold_seed=0, oof_lambda=best, fixed_lambda=fixed_lambda,
        fixed_lambda_source="MMMU30 prototype method source-fit OOF selection",
        selection="maximum pooled source-fit 3-fold OOF Rank; first lambda on ties",
        models=model_files, id_test_outcomes_read=False))
    say(stage="train", oof_lambda=best["lambda_cost"], fixed_lambda=fixed_lambda,
        oof_rank=best["rank_score"])


def freeze():
    if (OUT / "MODEL_LOCK.json").exists():
        raise FileExistsError("K30 control already locked")
    selection = read_json(OUT / "SELECTION.json")
    subject.require_records(selection["models"])
    files = [record(p) for p in (
        Path(__file__), Path(subject.__file__), ROOT / "src/next_round_x/models.py",
        ROOT / "src/next_round_x/data.py", ROOT / "routers/utils/rank_score.py",
        OUT / "SELECTION.json", OUT / "OOF_RESULTS.csv",
        subject.OUT / "SELECTION.json", subject.OUT / "PROTOTYPES.json",
        ROOT / "outputs/next_round_x/source/heldout11/MANIFEST.json",
        ROOT / "artifacts/subject_routing/manifests/splits.json",
        ROOT / "data/matrices/cost_bounds.json")]
    files += selection["models"]
    write_json(OUT / "MODEL_LOCK.json", dict(
        status="FROZEN_BEFORE_MATCHED_K30_CONTROL_TEST_ACTIONS", files=files,
        source_only_lambda_selection=True, test_fit_performed=False,
        cost_scope="cached expert USD proxy; excludes Qwen3-VL feature extraction"))
    say(stage="freeze", files=len(files))


def assign_kmeans(x, centers):
    nx = normalized(x)
    distances = np.square(nx).sum(1)[:, None] + np.square(centers).sum(1)[None, :] - 2 * nx @ centers.T
    return distances.argmin(axis=1)


def predict():
    if (OUT / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("K30 predictions already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    uids = role_uids("heldout11", "id")
    if len(uids) != 5341:
        raise RuntimeError("11-ID test size changed")
    x, available = FeatureCache().rows(uids)
    if not available.all():
        raise RuntimeError("ID Qwen feature missing")
    states, actions_oof, actions_fixed = [], [], []
    for seed in SEEDS:
        with np.load(OUT / "models" / f"seed-{seed}.npz", allow_pickle=False) as model:
            assigned = assign_kmeans(x, model["centers"])
            states.append(assigned)
            actions_oof.append(model["table_oof"][assigned])
            actions_fixed.append(model["table_fixed"][assigned])
    path = OUT / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), seeds=np.array(SEEDS), states=np.stack(states),
                        actions_oof=np.stack(actions_oof), actions_fixed=np.stack(actions_fixed))
    write_json(OUT / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested MMMU subject-state 11-ID validation; matched K30 control",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(OUT / "MODEL_LOCK.json"), labels_or_actual_cost_used=False,
        test_outcomes_opened=0, test_fit_performed=False, denied_attempts=denied))
    say(stage="predict", rows=len(uids), seeds=len(SEEDS), maps=2)


def evaluate():
    if (REPORT / "RESULTS.json").exists():
        raise FileExistsError("K30 control already evaluated")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    path = OUT / "PREDICTIONS.npz"
    receipt_path = OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if receipt["prediction_sha256"] != digest(path):
        raise RuntimeError("K30 prediction changed")
    with np.load(path, allow_pickle=False) as prediction:
        uids = prediction["uids"].tolist()
        arrays = {key: prediction[key].copy() for key in ("actions_oof", "actions_fixed")}
    if uids != role_uids("heldout11", "id"):
        raise RuntimeError("K30 test UID order changed")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **subject.label_args())
    write_json(OUT / "OUTCOME_PROVENANCE.json", dict(materialization=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    dataset = np.array([uid.split("/", 1)[0] for uid in uids])
    per_seed, per_dataset, summary = [], [], []
    subject_result = read_json(subject.REPORT / "RESULTS.json")["main"]
    for key, title in (("actions_oof", "K30 source-OOF lambda"),
                       ("actions_fixed", "K30 MMMU30-matched lambda")):
        values = []
        for seed, actions in zip(SEEDS, arrays[key]):
            score = subject.metrics(y, c, actions)
            values.append(score)
            per_seed.append(dict(method=title, seed=seed, **score))
        metrics = ("accuracy_percent", "cost_usd_per_10k", "rank_score")
        means = {metric: float(np.mean([v[metric] for v in values])) for metric in metrics}
        sds = {metric: float(np.std([v[metric] for v in values], ddof=1)) for metric in metrics}
        summary.append(dict(method=title, seeds=len(SEEDS), n=len(uids), mean=means, sd=sds,
                            delta_mmmu30_minus_control={metric: subject_result[metric] - means[metric]
                                                        for metric in metrics}))
        for name in sorted(set(dataset)):
            take = dataset == name
            scores = [subject.metrics(y[take], c[take], actions[take]) for actions in arrays[key]]
            per_dataset.append(dict(method=title, dataset=name, n=int(take.sum()),
                                    **{metric + "_mean": float(np.mean([row[metric] for row in scores]))
                                       for metric in metrics}))
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "per_seed.csv", per_seed)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    write_json(REPORT / "RESULTS.json", dict(
        control="matched StateRoute-MM K30", view="11-ID", subject_result=subject_result,
        summary=summary, per_seed=per_seed, per_dataset=per_dataset,
        prediction=record(path), source_selection=record(OUT / "SELECTION.json"),
        interpretation="exploratory exposed-test replay; both lambdas source-selected before K30 test predictions",
        cost_scope="cached expert USD proxy; Qwen3-VL extraction excluded"))
    lines = ["# StateRoute-MM K30 等状态数对照", "",
             "同样使用 30 个状态、同一 Qwen3-VL 特征、同一 11-ID source-fit 专家正确性与费用、64 样本收缩及三折源域选择。KMeans 种子为 0–4。",
             "", "| 路由 | 准确率 | 专家费用 USD/万题 | Rank Score |", "|---|---:|---:|---:|"]
    lines.append(f"| MMMU30-Cosine | {subject_result['accuracy_percent']:.3f}% | {subject_result['cost_usd_per_10k']:.4f} | {subject_result['rank_score']:.3f} |")
    for item in summary:
        value = item["mean"]
        lines.append(f"| {item['method']}（5 种子均值） | {value['accuracy_percent']:.3f}% | {value['cost_usd_per_10k']:.4f} | {value['rank_score']:.3f} |")
    lines += ["", "source-OOF λ 由 K30 自己的三折源域结果选出；matched λ 固定为 MMMU30 在源域选出的 λ。两者均未用 11-ID 测试指标选参。",
              "费用均为缓存中的专家 USD proxy，未计入 Qwen3-VL 特征提取；旧测试结果已公开，属于探索性对照。",
              "逐种子与逐数据集数值见 `per_seed.csv` 和 `per_dataset.csv`。"]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(stage="evaluate", summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "freeze", "predict", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"train": train, "freeze": freeze, "predict": predict, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
