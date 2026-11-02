#!/usr/bin/env python3
"""Frozen-model lambda sensitivity replay for four StateRoute-MM variants.

Run --stage freeze, predict, evaluate in separate processes. The lambda grid
is fixed before test actions are materialized. This is a post-hoc sensitivity
analysis on the previously exposed 11-ID test set, not test-based selection.
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

from scripts import run_stateroute_mmmu_subject_20260928 as subject
from scripts import run_stateroute_mmmu_k30_control_20260928 as k30
from src.next_round_x.data import FeatureCache, role_uids
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

OUT = ROOT / "outputs/stateroute_mmmu_lambda_sweep_20260928"
REPORT = ROOT / "reports/stateroute_mmmu_lambda_sweep_20260928"
LAMBDAS = (0., .001, .003, .01, .02, .03, .05, .1, .2, .3)
METHODS = ("MMMU30-Cosine", "StateRoute-MM-K30", "StateRoute-MM-K2", "StateRoute-MM-K12")
SEEDS = {METHODS[0]: (-1,), METHODS[1]: tuple(range(5)),
         METHODS[2]: tuple(range(5)), METHODS[3]: tuple(range(5))}
MODEL_ROOTS = {
    METHODS[0]: subject.OUT / "MODEL.npz",
    METHODS[1]: k30.OUT / "models",
    METHODS[2]: ROOT / "outputs/stateroute_k_sweep_20260916/models/MM/k2",
    METHODS[3]: ROOT / "outputs/stateroute_k10_k12_20260916/models/MM/k12",
}


def say(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def model_path(method, seed):
    if method == METHODS[0]:
        return MODEL_ROOTS[method]
    if method == METHODS[1]:
        return MODEL_ROOTS[method] / f"seed-{seed}.npz"
    return MODEL_ROOTS[method] / f"seed-{seed}" / "model.npz"


def original_lambdas():
    subject_lambda = float(read_json(subject.OUT / "SELECTION.json")["selected"]["lambda_cost"])
    k30_selection = read_json(k30.OUT / "SELECTION.json")
    old_k2 = read_json(ROOT / "outputs/stateroute_k_sweep_20260916/MM_SOURCE_SELECTION.json")
    old_k12 = read_json(ROOT / "outputs/stateroute_k10_k12_20260916/MM_SOURCE_SELECTION.json")
    result = {
        METHODS[0]: subject_lambda,
        METHODS[1]: float(k30_selection["oof_lambda"]["lambda_cost"]),
        METHODS[2]: float(next(row["lambda_cost"] for row in old_k2["selections"] if row["k"] == 2)),
        METHODS[3]: float(next(row["lambda_cost"] for row in old_k12["selections"] if row["k"] == 12)),
    }
    if result != {METHODS[0]: .1, METHODS[1]: .1, METHODS[2]: .03, METHODS[3]: .03}:
        raise RuntimeError("source-selected lambda binding changed")
    return result


def freeze():
    if (OUT / "MODEL_LOCK.json").exists():
        raise FileExistsError("lambda sweep is already frozen")
    subject.require_records(read_json(subject.OUT / "MODEL_LOCK.json")["files"])
    subject.require_records(read_json(k30.OUT / "MODEL_LOCK.json")["files"])
    cache = FeatureCache()
    pointer = read_json(subject.POINTER)
    if pointer["input_content"]["model_configured_path"] != subject.MODEL_PATH:
        raise RuntimeError("Qwen3-VL model path differs from subject experiment")
    with np.load(subject.OUT / "MODEL.npz", allow_pickle=False) as model:
        subject_scaled_cost = model["scaled_cost"].copy()
    old_stats = [ROOT / "outputs/stateroute_k_sweep_20260916/models/MM/stats.json",
                 ROOT / "outputs/stateroute_k10_k12_20260916/models/MM/stats.json"]
    for path in old_stats:
        stats = read_json(path)
        old_scaled_cost = np.asarray(stats["cost_mean"], dtype=np.float64) / float(stats["cost_scale"])
        if not np.array_equal(subject_scaled_cost, old_scaled_cost):
            raise RuntimeError("source cost scales differ between methods")
    source = [Path(__file__), Path(subject.__file__), Path(k30.__file__),
              subject.OUT / "MODEL_LOCK.json", k30.OUT / "MODEL_LOCK.json",
              subject.OUT / "SELECTION.json", k30.OUT / "SELECTION.json",
              ROOT / "outputs/stateroute_k_sweep_20260916/MM_SOURCE_SELECTION.json",
              ROOT / "outputs/stateroute_k10_k12_20260916/MM_SOURCE_SELECTION.json",
              *old_stats,
              subject.POINTER, cache.root / "manifest.json", cache.root / "features.npy",
              cache.root / "index.jsonl",
              ROOT / "artifacts/subject_routing/manifests/splits.json",
              ROOT / "artifacts/subject_routing/manifests/model_pool.json",
              ROOT / "data/matrices/cost_bounds.json"]
    source += [model_path(method, seed) for method in METHODS for seed in SEEDS[method]]
    write_json(OUT / "MODEL_LOCK.json", dict(
        status="FROZEN_LAMBDA_GRID_BEFORE_SENSITIVITY_TEST_ACTIONS",
        lambda_grid=list(LAMBDAS), methods=list(METHODS), seeds={key: list(value) for key, value in SEEDS.items()},
        original_source_selected_lambda=original_lambdas(),
        state_centers_and_quality_profiles_frozen=True,
        no_refit_or_new_expert_calls=True, test_metric_selection=False,
        test_exposure="historical 11-ID test metrics exposed; exploratory sensitivity replay",
        cost_scope="cached expert USD proxy; excludes Qwen3-VL feature extraction",
        files=[record(path) for path in source]))
    say(stage="freeze", lambdas=len(LAMBDAS), models=sum(len(v) for v in SEEDS.values()), files=len(source))


def original_actions(method, seed, uids):
    if method == METHODS[0]:
        path = subject.OUT / "PREDICTIONS.npz"
        with np.load(path, allow_pickle=False) as previous:
            if previous["uids"].tolist() != uids:
                raise RuntimeError("original MMMU30 UID mismatch")
            return previous["actions"].copy()
    if method == METHODS[1]:
        path = k30.OUT / "PREDICTIONS.npz"
        with np.load(path, allow_pickle=False) as previous:
            if previous["uids"].tolist() != uids:
                raise RuntimeError("original K30 UID mismatch")
            return previous["actions_oof"][seed].copy()
    path = ROOT / ("outputs/stateroute_k_sweep_20260916/predictions.npz" if method == METHODS[2]
                   else "outputs/stateroute_k10_k12_20260916/predictions.npz")
    with np.load(path, allow_pickle=False) as previous:
        if previous["uids"][:len(uids)].tolist() != uids:
            raise RuntimeError(f"original {method} UID mismatch")
        matching = np.flatnonzero((previous["methods"] == method) & (previous["seeds"] == seed))
        if len(matching) != 1:
            raise RuntimeError(f"original {method} seed binding missing")
        return previous["actions"][int(matching[0]), :len(uids)].copy()


def predict():
    if (OUT / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("lambda sweep predictions are already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    if tuple(lock["lambda_grid"]) != LAMBDAS or tuple(lock["methods"]) != METHODS:
        raise RuntimeError("frozen lambda grid or methods changed")
    uids = role_uids("heldout11", "id")
    if len(uids) != 5341 or len(set(uids)) != len(uids):
        raise RuntimeError("11-ID test UID contract changed")
    x, available = FeatureCache().rows(uids)
    if not available.all():
        raise RuntimeError("11-ID frozen features incomplete")
    with np.load(subject.OUT / "MODEL.npz", allow_pickle=False) as original_subject_model:
        scaled_cost = original_subject_model["scaled_cost"].copy()
    action_rows, state_rows, method_rows, seed_rows, lambda_rows = [], [], [], [], []
    checked = 0
    for method in METHODS:
        for seed in SEEDS[method]:
            with np.load(model_path(method, seed), allow_pickle=False) as model:
                centers = model["centers"]
                quality = model["quality" if method == METHODS[0] else "rates"]
                if method == METHODS[0]:
                    states, _, _ = subject.assign_subjects(x, centers)
                else:
                    states = k30.assign_kmeans(x, centers)
                if quality.shape != (len(centers), 17) or scaled_cost.shape != (17,):
                    raise RuntimeError("quality/cost shape mismatch")
                for lam in LAMBDAS:
                    table = (quality - lam * scaled_cost).argmax(axis=1)
                    actions = table[states]
                    if lam == lock["original_source_selected_lambda"][method]:
                        expected = original_actions(method, seed, uids)
                        if not np.array_equal(actions, expected):
                            raise RuntimeError(f"lambda replay fails original action equivalence: {method}, {seed}")
                        checked += 1
                    action_rows.append(actions.astype(np.int8))
                    state_rows.append(states.astype(np.int8))
                    method_rows.append(method)
                    seed_rows.append(seed)
                    lambda_rows.append(lam)
    path = OUT / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), actions=np.stack(action_rows), states=np.stack(state_rows),
                        methods=np.array(method_rows), seeds=np.array(seed_rows), lambdas=np.array(lambda_rows))
    write_json(OUT / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested lambda sensitivity experiment",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(OUT / "MODEL_LOCK.json"),
        selected_lambda_routes_exactly_reproduce_prior_actions=checked,
        labels_or_actual_cost_used=False, test_outcomes_opened=0,
        test_fit_performed=False, denied_attempts=denied))
    say(stage="predict", routes=len(action_rows), rows=len(uids), original_routes_verified=checked)


def evaluate():
    if (REPORT / "RESULTS.json").exists():
        raise FileExistsError("lambda sweep already evaluated")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    path = OUT / "PREDICTIONS.npz"
    receipt_path = OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if receipt["prediction_sha256"] != digest(path) or receipt["model_lock"]["sha256"] != digest(OUT / "MODEL_LOCK.json"):
        raise RuntimeError("lambda sweep prediction or model lock changed")
    with np.load(path, allow_pickle=False) as prediction:
        uids = prediction["uids"].tolist()
        actions = prediction["actions"].astype(np.int64)
        methods = prediction["methods"].tolist()
        seeds = prediction["seeds"].tolist()
        lambdas = prediction["lambdas"].tolist()
    if uids != role_uids("heldout11", "id"):
        raise RuntimeError("lambda sweep test UID order changed")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **subject.label_args())
    write_json(OUT / "OUTCOME_PROVENANCE.json", dict(materialization=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    datasets = np.array([uid.split("/", 1)[0] for uid in uids])
    per_seed = []
    for method, seed, lam, action in zip(methods, seeds, lambdas, actions):
        per_seed.append(dict(method=method, seed=seed, lambda_cost=lam,
                             **subject.metrics(y, c, action), distinct_experts=int(len(set(action)))))
    summary, per_dataset = [], []
    metrics = ("accuracy_percent", "cost_usd_per_10k", "rank_score", "distinct_experts")
    for method in METHODS:
        for lam in LAMBDAS:
            take = [i for i, (m, l) in enumerate(zip(methods, lambdas)) if m == method and l == lam]
            if len(take) != len(SEEDS[method]):
                raise RuntimeError("incomplete method/lambda seed matrix")
            group = [per_seed[i] for i in take]
            row = dict(method=method, lambda_cost=lam, selected_in_original_source_oof=(lam == lock["original_source_selected_lambda"][method]),
                       n=5341, seeds=len(take))
            for key in metrics:
                values = np.array([r[key] for r in group], dtype=np.float64)
                row[key + "_mean"] = float(values.mean())
                row[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            summary.append(row)
            for dataset in sorted(set(datasets)):
                mask = datasets == dataset
                scores = [subject.metrics(y[mask], c[mask], actions[i][mask]) for i in take]
                per_dataset.append(dict(method=method, lambda_cost=lam, dataset=dataset,
                                        n=int(mask.sum()), seeds=len(take),
                                        **{key + "_mean": float(np.mean([r[key] for r in scores]))
                                           for key in metrics[:3]}))
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "per_seed.csv", per_seed)
    write_csv(REPORT / "summary.csv", summary)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    correct = y[np.arange(len(uids))[None, :], actions]
    cost = c[np.arange(len(uids))[None, :], actions]
    np.savez_compressed(OUT / "EVALUATION_TRACES.npz", uids=np.array(uids), methods=np.array(methods),
                        seeds=np.array(seeds), lambdas=np.array(lambdas), correct=correct, cost=cost)
    write_json(REPORT / "RESULTS.json", dict(
        view="11-ID", n=5341, methods=list(METHODS), lambda_grid=list(LAMBDAS),
        source_selected_lambda=lock["original_source_selected_lambda"],
        summary=summary, prediction=record(path), model_lock=record(OUT / "MODEL_LOCK.json"),
        all_wrong_rows_retained=True, test_metric_selection=False,
        interpretation="post-hoc lambda sensitivity on exposed 11-ID test, not a new test-selected policy",
        cost_scope=lock["cost_scope"]))
    lines = ["# StateRoute-MM 学科状态与聚类状态：λ 敏感性分析", "",
             "固定先前训练得到的状态和专家正确率画像，仅重算专家映射 `argmax_j(q[state,j] − λ·normalized_cost[j])`。",
             "MMMU30 为确定性路由，其余 K30/K2/K12 各为 5 种子均值。全程使用相同 11-ID test 5,341 题、17 专家和 Qwen3-VL 冻结特征。",
             "", "| 方法 | λ | 准确率 % | 专家费用 USD/万题 | Rank Score | 不同专家数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        star = " *" if row["selected_in_original_source_oof"] else ""
        lines.append(f"| {row['method']} | {row['lambda_cost']:.3f}{star} | {row['accuracy_percent_mean']:.3f} | "
                     f"{row['cost_usd_per_10k_mean']:.4f} | {row['rank_score_mean']:.3f} | "
                     f"{row['distinct_experts_mean']:.1f} |")
    lines += ["", "`*` 是此前用源域 OOF 选出的 λ：MMMU30/K30 为 0.1，K2/K12 为 0.03。其他行只是固定模型的测试集敏感性诊断，不能拿测试最高 Rank 反向选择新部署 λ。",
              "费用只含缓存中的专家 USD proxy，不含 Qwen3-VL 特征提取。历史 11-ID 测试结果已暴露，本表是探索性复跑。",
              "", "逐种子结果见 `per_seed.csv`，逐数据集结果见 `per_dataset.csv`；完整绑定和精确数值见 `RESULTS.json`。"]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(stage="evaluate", summary_rows=len(summary), per_seed_rows=len(per_seed), per_dataset_rows=len(per_dataset))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "predict", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"freeze": freeze, "predict": predict, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
