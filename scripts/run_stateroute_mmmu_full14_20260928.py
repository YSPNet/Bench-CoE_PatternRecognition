#!/usr/bin/env python3
"""Full 14-ID MMMU-subject StateRoute-MM and matched KMeans lambda sweep.

Run --stage train, freeze, predict, evaluate in separate processes. MMMU train
inputs and native subjects define the final 30 prototypes. In OOF folds, each
validation dependency group is removed from its fold's prototypes. All 47 MMMU
multi-image rows use the existing hash-bound native multi-image feature overlay.
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
from scripts import run_stateroute_mmmu_k30_control_20260928 as kmeans_reference
from scripts import run_stateroute_mmmu_lambda_sweep_20260928 as prior_sweep
from src.next_round_x.data import load_source, role_uids
from src.next_round_x.models import cluster_quality, normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

OUT = ROOT / "outputs/stateroute_mmmu_full14_20260928"
REPORT = ROOT / "reports/stateroute_mmmu_full14_20260928"
METHODS = ("MMMU30-Cosine", "StateRoute-MM-K30", "StateRoute-MM-K2", "StateRoute-MM-K12")
SEEDS = {METHODS[0]: (-1,), METHODS[1]: tuple(range(5)),
         METHODS[2]: tuple(range(5)), METHODS[3]: tuple(range(5))}
KS = {METHODS[1]: 30, METHODS[2]: 2, METHODS[3]: 12}
LAMBDAS = prior_sweep.LAMBDAS
DIM = 2048
EXPERTS = 17


def say(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def model_path(method, seed):
    folder = OUT / "models" / method
    return folder / ("model.npz" if method == METHODS[0] else f"seed-{seed}.npz")


def source_train():
    data = load_source("full14")
    uids = data["uids"].tolist()
    if uids != role_uids("full14", "train") or len(uids) != 21378:
        raise RuntimeError("full14 official train UID order/count changed")
    if len({uid.split("/", 1)[0] for uid in uids}) != 14:
        raise RuntimeError("full14 does not contain 14 source datasets")
    x, overlay, _ = subject.qwen_features(uids, allow_multi=True)
    originally_missing = {uid for uid, available in zip(uids, data["available"]) if not available}
    if len(overlay) != 32 or set(overlay) != originally_missing:
        raise RuntimeError("full14 train multi-image overlay coverage changed")
    fit = np.flatnonzero(data["role"] == "fit")
    if len(fit) != 17871 or data["y"].shape != (len(uids), EXPERTS) or data["c"].shape != (len(uids), EXPERTS):
        raise RuntimeError("full14 source-fit outcome shape changed")
    return data, x, fit, overlay


def prototype_contract(data, x):
    manifest = read_json(subject.OUT / "PROTOTYPES.json")
    subject.require_records(manifest["sources"] + [manifest["artifact"]])
    with np.load(subject.OUT / "PROTOTYPES.npz", allow_pickle=False) as proto:
        proto_uids = proto["uids"].tolist()
        labels = proto["native_labels"].copy()
        subjects = proto["subjects"].tolist()
        centers = proto["centers"].copy()
    if len(proto_uids) != 735 or subjects != subject.SUBJECTS or centers.shape != (30, DIM):
        raise RuntimeError("MMMU train prototype contract changed")
    if set(proto_uids) != {str(uid) for uid in data["uids"] if str(uid).startswith("MMMU_DEV_VAL/")}:
        raise RuntimeError("MMMU train prototypes differ from full14 official source train")
    positions = {str(uid): i for i, uid in enumerate(data["uids"])}
    proto_positions = np.array([positions[uid] for uid in proto_uids], dtype=np.int64)
    recomputed, support = subject.build_centroids(x[proto_positions], labels)
    if not np.allclose(recomputed, centers, rtol=0, atol=1e-6) or support.sum() != 735:
        raise RuntimeError("final MMMU centers differ from frozen feature/label binding")
    return proto_positions, labels, centers


def select_lambda(oof_quality, oof_cost, y, c):
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    rows = []
    for lam in LAMBDAS:
        actions = (oof_quality - lam * oof_cost).argmax(axis=1)
        result = subject.metrics(y, c, actions)
        source_accuracy = float(y[np.arange(len(actions)), actions].mean())
        exact_rank = 100 * rank_score(source_accuracy, result["cost_usd_per_10k"] / 10000,
                                      bounds["cmin"], bounds["cmax"])
        rows.append(dict(lambda_cost=lam, selection_rank=exact_rank, **result))
    return rows, max(rows, key=lambda row: row["selection_rank"])


def train():
    if (OUT / "SOURCE_SELECTION.json").exists():
        raise FileExistsError("full14 source selection already exists")
    data, x, fit, overlay = source_train()
    proto_positions, proto_labels, centers = prototype_contract(data, x)
    position = {int(index): row for row, index in enumerate(fit)}
    oof = {method: np.empty((len(fit), EXPERTS), dtype=np.float64) for method in METHODS}
    oof_cost = np.empty((len(fit), EXPERTS), dtype=np.float64)
    seen = np.zeros(len(fit), dtype=np.int8)
    fold_notes = []
    for fold in range(3):
        train_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        train_groups = set(data["groups"][train_indices])
        valid_groups = set(data["groups"][valid_indices])
        if train_groups & valid_groups:
            raise RuntimeError("full14 source-fit OOF groups overlap")
        train_positions = np.array([position[int(i)] for i in train_indices])
        valid_positions = np.array([position[int(i)] for i in valid_indices])
        keep_proto = np.array([data["groups"][index] not in valid_groups for index in proto_positions])
        fold_proto_positions = proto_positions[keep_proto]
        if set(data["groups"][fold_proto_positions]) & valid_groups:
            raise RuntimeError("MMMU fold prototype saw validation dependency group")
        fold_centers, fold_subject_support = subject.build_centroids(x[fold_proto_positions], proto_labels[keep_proto])
        source_states, _, _ = subject.assign_subjects(x[train_indices], fold_centers)
        valid_states, _, _ = subject.assign_subjects(x[valid_indices], fold_centers)
        quality, _ = subject.quality_profile(data["y"][train_indices], source_states)
        oof[METHODS[0]][valid_positions] = quality[valid_states]
        c_train = data["c"][train_indices]
        oof_cost[valid_positions] = c_train.mean(axis=0) / float(np.median(c_train[c_train > 0]))
        for method in METHODS[1:]:
            predicted, _, _ = cluster_quality(x[train_indices], data["y"][train_indices],
                                               x[valid_indices], KS[method], 0)
            oof[method][valid_positions] = predicted
        seen[valid_positions] += 1
        note = dict(fold=fold, fit_train_n=len(train_indices), fit_valid_n=len(valid_indices),
                    prototype_train_n=len(fold_proto_positions), prototype_removed_valid_group_n=int((~keep_proto).sum()),
                    minimum_mmmu_subject_prototype_support=int(fold_subject_support.min()),
                    prototype_validation_group_overlap_n=0)
        fold_notes.append(note)
        say(stage="oof", **note)
    if not (seen == 1).all() or not np.isfinite(oof_cost).all():
        raise RuntimeError("full14 source-fit OOF coverage incomplete")
    for method in METHODS:
        if not np.isfinite(oof[method]).all():
            raise RuntimeError(f"nonfinite full14 OOF quality: {method}")
    selection = {}
    oof_rows = []
    for method in METHODS:
        rows, winner = select_lambda(oof[method], oof_cost, data["y"][fit], data["c"][fit])
        selection[method] = winner
        oof_rows.extend(dict(method=method, **row) for row in rows)
        path = OUT / "oof" / (method + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, uids=data["uids"][fit], quality=oof[method].astype(np.float32),
                            scaled_cost=oof_cost.astype(np.float32))
        say(stage="select", method=method, lambda_cost=winner["lambda_cost"],
            source_oof_rank=winner["rank_score"])
    write_csv(OUT / "OOF_RESULTS.csv", oof_rows)
    c_fit = data["c"][fit]
    scaled_cost = c_fit.mean(axis=0) / float(np.median(c_fit[c_fit > 0]))
    subject_states, _, _ = subject.assign_subjects(x[fit], centers)
    rates, support = subject.quality_profile(data["y"][fit], subject_states)
    path = model_path(METHODS[0], -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centers=centers, rates=rates, scaled_cost=scaled_cost, support=support,
                        subjects=np.array(subject.SUBJECTS),
                        selected_table=(rates - selection[METHODS[0]]["lambda_cost"] * scaled_cost).argmax(axis=1))
    model_records = {METHODS[0]: [record(path)]}
    for method in METHODS[1:]:
        model_records[method] = []
        for seed in SEEDS[method]:
            _, km, fitted_rates = cluster_quality(x[fit], data["y"][fit], x[fit[:1]], KS[method], seed)
            counts = np.bincount(km.predict(normalized(x[fit])), minlength=KS[method])
            path = model_path(method, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, centers=km.cluster_centers_, rates=fitted_rates,
                                scaled_cost=scaled_cost, support=counts,
                                selected_table=(fitted_rates - selection[method]["lambda_cost"] * scaled_cost).argmax(axis=1))
            model_records[method].append(record(path))
            say(stage="fit", method=method, seed=seed, support_min=int(counts.min()))
    write_json(OUT / "SOURCE_SELECTION.json", dict(
        protocol="full14-ID", source_train_n=len(data["uids"]), source_fit_n=len(fit),
        source_train_datasets=sorted(set(data["dataset"].tolist())),
        mmmu_train_prototype_n=len(proto_positions), train_multi_image_overlay_n=len(overlay),
        fold_prototype_contract="remove all MMMU prototype rows in each validation dependency group",
        folds=fold_notes, lambda_grid=list(LAMBDAS), selected=selection,
        kmeans=dict(n_init=3, batch_size=1024, max_iter=100, oof_seed=0, final_seeds=list(range(5))),
        shrinkage_kappa=64, expert_cost_normalization="fit mean per expert / median positive fit cost",
        qcal_cert_outcomes_used=False, source_dev_outcomes_used=False, test_outcomes_used=False,
        subject_source_fit_counts=dict(zip(subject.SUBJECTS, map(int, support))),
        model_records=model_records,
        oof_records=[record(OUT / "oof" / (method + ".npz")) for method in METHODS],
        oof_results=record(OUT / "OOF_RESULTS.csv")))
    say(stage="train", fit_n=len(fit), models=sum(len(v) for v in model_records.values()),
        selected_lambdas={key: value["lambda_cost"] for key, value in selection.items()})


def freeze():
    if (OUT / "MODEL_LOCK.json").exists():
        raise FileExistsError("full14 model already frozen")
    selection = read_json(OUT / "SOURCE_SELECTION.json")
    records = [entry for values in selection["model_records"].values() for entry in values]
    records += selection["oof_records"] + [selection["oof_results"]]
    subject.require_records(records)
    pointer = read_json(subject.POINTER)
    if pointer["input_content"]["model_configured_path"] != subject.MODEL_PATH:
        raise RuntimeError("Qwen3-VL model path binding changed")
    cache_root = ROOT / pointer["cache_root"] / "merged"
    files = [Path(__file__), Path(subject.__file__), Path(kmeans_reference.__file__),
             ROOT / "src/next_round_x/data.py", ROOT / "src/next_round_x/models.py",
             ROOT / "routers/utils/rank_score.py", OUT / "SOURCE_SELECTION.json",
             subject.OUT / "PROTOTYPES.npz", subject.OUT / "PROTOTYPES.json",
             subject.POINTER, cache_root / "manifest.json", cache_root / "features.npy",
             cache_root / "index.jsonl", subject.OVERLAY, subject.OVERLAY_MANIFEST,
             ROOT / "outputs/next_round_x/source/full14/MANIFEST.json",
             ROOT / "artifacts/subject_routing/manifests/splits.json",
             ROOT / "artifacts/subject_routing/manifests/model_pool.json",
             ROOT / "data/matrices/cost_bounds.json"]
    files += [ROOT / item["path"] for item in records]
    write_json(OUT / "MODEL_LOCK.json", dict(
        status="FULL14_MODELS_AND_LAMBDAS_FROZEN_BEFORE_TEST_ACTIONS", protocol="full14-ID",
        lambda_grid=list(LAMBDAS), selected_source_oof_lambda={key: value["lambda_cost"] for key, value in selection["selected"].items()},
        methods=list(METHODS), seeds={key: list(value) for key, value in SEEDS.items()},
        all_47_mmmu_multi_image_rows_supported=True, source_dev_test_outcomes_used=False,
        test_exposure="historical full14 test outcomes have appeared elsewhere; exploratory replay",
        cost_scope="cached expert USD proxy; excludes Qwen3-VL feature extraction",
        files=[record(path) for path in files]))
    say(stage="freeze", files=len(files))


def predict():
    if (OUT / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("full14 predictions already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    if tuple(lock["lambda_grid"]) != LAMBDAS or tuple(lock["methods"]) != METHODS:
        raise RuntimeError("full14 model lock grid/method changed")
    uids = role_uids("full14", "test")
    if len(uids) != 6108 or len(set(uids)) != len(uids):
        raise RuntimeError("full14 official test UID order/count changed")
    x, overlay, _ = subject.qwen_features(uids, allow_multi=True)
    if len(overlay) != 10:
        raise RuntimeError("full14 test multi-image coverage changed")
    methods, seeds, lambdas, action_rows, state_rows = [], [], [], [], []
    for method in METHODS:
        for seed in SEEDS[method]:
            with np.load(model_path(method, seed), allow_pickle=False) as model:
                center = model["centers"]
                rate = model["rates"]
                cost = model["scaled_cost"]
                if method == METHODS[0]:
                    states, _, _ = subject.assign_subjects(x, center)
                else:
                    states = kmeans_reference.assign_kmeans(x, center)
                for lam in LAMBDAS:
                    table = (rate - lam * cost).argmax(axis=1)
                    actions = table[states]
                    if lam == lock["selected_source_oof_lambda"][method] and not np.array_equal(table, model["selected_table"]):
                        raise RuntimeError(f"selected source route changed: {method}, seed={seed}")
                    if actions.shape != (6108,) or (actions < 0).any() or (actions >= EXPERTS).any():
                        raise RuntimeError("invalid full14 action row")
                    methods.append(method)
                    seeds.append(seed)
                    lambdas.append(lam)
                    action_rows.append(actions.astype(np.int8))
                    state_rows.append(states.astype(np.int8))
    path = OUT / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), methods=np.array(methods), seeds=np.array(seeds),
                        lambdas=np.array(lambdas), actions=np.stack(action_rows), states=np.stack(state_rows))
    write_json(OUT / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested full 14-dataset ID MMMU-subject experiment",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(OUT / "MODEL_LOCK.json"),
        all_test_rows_have_qwen_features=True, test_multi_image_overlay_n=len(overlay),
        labels_or_actual_cost_used=False, test_outcomes_opened=0,
        test_fit_performed=False, denied_attempts=denied))
    say(stage="predict", routes=len(action_rows), test_n=len(uids), mmmu_test_multi_image_n=len(overlay))


def evaluate():
    if (REPORT / "RESULTS.json").exists():
        raise FileExistsError("full14 test already evaluated")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    path = OUT / "PREDICTIONS.npz"
    receipt_path = OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(path) != receipt["prediction_sha256"] or digest(OUT / "MODEL_LOCK.json") != receipt["model_lock"]["sha256"]:
        raise RuntimeError("full14 frozen prediction/model binding changed")
    with np.load(path, allow_pickle=False) as prediction:
        uids = prediction["uids"].tolist()
        methods = prediction["methods"].tolist()
        seeds = prediction["seeds"].tolist()
        lambdas = prediction["lambdas"].tolist()
        actions = prediction["actions"].astype(np.int64)
        states = prediction["states"].astype(np.int64)
    if uids != role_uids("full14", "test"):
        raise RuntimeError("full14 test UID order changed")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **subject.label_args())
    write_json(OUT / "OUTCOME_PROVENANCE.json", dict(materialization=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    datasets = np.array([uid.split("/", 1)[0] for uid in uids])
    names = sorted(set(datasets))
    if len(names) != 14 or y.shape != (6108, EXPERTS):
        raise RuntimeError("full14 test dataset/outcome contract changed")
    per_seed, per_dataset, summary = [], [], []
    metrics = ("accuracy_percent", "cost_usd_per_10k", "rank_score")
    for method, seed, lam, action, state in zip(methods, seeds, lambdas, actions, states):
        per_seed.append(dict(method=method, seed=seed, lambda_cost=lam, **subject.metrics(y, c, action),
                             distinct_experts=int(len(set(action))), occupied_states=int(len(set(state)))))
    for method in METHODS:
        for lam in LAMBDAS:
            take = [i for i, (m, l) in enumerate(zip(methods, lambdas)) if m == method and l == lam]
            if len(take) != len(SEEDS[method]):
                raise RuntimeError("incomplete full14 method/lambda/seed matrix")
            scores = [per_seed[i] for i in take]
            macro = [float(np.mean([subject.metrics(y[datasets == name], c[datasets == name], actions[i][datasets == name])["accuracy_percent"]
                                    for name in names])) for i in take]
            row = dict(method=method, lambda_cost=lam,
                       selected_in_source_oof=(lam == lock["selected_source_oof_lambda"][method]),
                       n=len(uids), seeds=len(take),
                       macro_dataset_accuracy_percent_mean=float(np.mean(macro)),
                       macro_dataset_accuracy_percent_sd=float(np.std(macro, ddof=1)) if len(macro) > 1 else None)
            for key in (*metrics, "distinct_experts", "occupied_states"):
                value = np.array([score[key] for score in scores], dtype=np.float64)
                row[key + "_mean"] = float(value.mean())
                row[key + "_sd"] = float(value.std(ddof=1)) if len(value) > 1 else None
            summary.append(row)
            for name in names:
                mask = datasets == name
                values = [subject.metrics(y[mask], c[mask], actions[i][mask]) for i in take]
                per_dataset.append(dict(method=method, lambda_cost=lam, dataset=name,
                                        n=int(mask.sum()), seeds=len(take),
                                        **{key + "_mean": float(np.mean([score[key] for score in values])) for key in metrics}))
    with np.load(model_path(METHODS[0], -1), allow_pickle=False) as model:
        subject_support = model["support"].copy()
    subject_selected = next(i for i, (m, l) in enumerate(zip(methods, lambdas))
                            if m == METHODS[0] and l == lock["selected_source_oof_lambda"][METHODS[0]])
    id_subject_counts = np.bincount(states[subject_selected], minlength=30)
    mmmu_test = [uid for uid in uids if uid.startswith("MMMU_DEV_VAL/")]
    mmmu_mask = datasets == "MMMU_DEV_VAL"
    if len(mmmu_test) != 206:
        raise RuntimeError("MMMU test denominator changed")
    native_subject = subject.subject_labels(mmmu_test)
    correct_subject = np.array(subject.SUBJECTS)[states[subject_selected][mmmu_mask]] == native_subject
    diagnostic = dict(mmmu_test_n=len(mmmu_test), native_subject_correct=int(correct_subject.sum()),
                      native_subject_accuracy_percent=float(100 * correct_subject.mean()),
                      source_fit_occupied_subjects=int((subject_support > 0).sum()),
                      test_occupied_subjects=int((id_subject_counts > 0).sum()),
                      test_mmmu_multi_image_n=receipt["test_multi_image_overlay_n"])
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "per_seed.csv", per_seed)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    write_csv(REPORT / "summary.csv", summary)
    write_csv(REPORT / "selected_source_oof.csv", [row for row in summary if row["selected_in_source_oof"]])
    write_csv(REPORT / "mmmu30_test_subject_counts.csv", [dict(subject=name, source_fit_n=int(subject_support[i]),
                                                                test_n=int(id_subject_counts[i]))
                                                           for i, name in enumerate(subject.SUBJECTS)])
    np.savez_compressed(OUT / "EVALUATION_TRACES.npz", uids=np.array(uids), methods=np.array(methods),
                        seeds=np.array(seeds), lambdas=np.array(lambdas),
                        correct=y[np.arange(len(uids))[None, :], actions],
                        cost=c[np.arange(len(uids))[None, :], actions])
    write_json(REPORT / "RESULTS.json", dict(
        protocol="full14-ID", n=len(uids), dataset_n=len(names), methods=list(METHODS),
        lambda_grid=list(LAMBDAS), selected_source_oof_lambda=lock["selected_source_oof_lambda"],
        mmmu_native_subject_diagnostic=diagnostic, summary=summary,
        prediction=record(path), model_lock=record(OUT / "MODEL_LOCK.json"),
        test_metric_selection=False, all_wrong_rows_retained=True,
        cost_scope=lock["cost_scope"], interpretation="exploratory full14-ID replay"))
    lines = ["# 14-ID：MMMU 学科状态与 StateRoute-MM 聚类状态", "",
             "全 14 个数据集官方 train/dev/test 分别为 21,378/3,054/6,108；仅在官方 train 的 source-fit 17,871 题上学习专家画像和三折选择 λ。",
             "MMMU 官方 train 的 735 题定义 30 个学科原型。三折 OOF 中排除验证 dependency group 的 MMMU 原型题，最终原型使用全部 735 题。",
             "MMMU train/dev/test 的 32/5/10 条多图题使用同模型原生多图特征；dev 专家结果未用于选参。K2/K12/K30 是同特征、同源域设置下重训的五种子 KMeans 对照。",
             "", "## 源域 OOF 选定 λ 的 14-ID 测试结果", "",
             "| 方法 | λ | 准确率 % | 14 数据集宏平均准确率 % | 专家费用 USD/万题 | Rank Score |", "|---|---:|---:|---:|---:|---:|"]
    selected_rows = [row for row in summary if row["selected_in_source_oof"]]
    for row in selected_rows:
        lines.append(f"| {row['method']} | {row['lambda_cost']:.3f} | {row['accuracy_percent_mean']:.3f} | "
                     f"{row['macro_dataset_accuracy_percent_mean']:.3f} | {row['cost_usd_per_10k_mean']:.4f} | "
                     f"{row['rank_score_mean']:.3f} |")
    lines += ["", "## 固定 λ 网格敏感性", "",
              "下表每一行仅变动费用权重 λ；学科/聚类状态和专家画像保持冻结。星号是该方法在 14-ID source-fit OOF 上选出的 λ。",
              "", "| 方法 | λ | 准确率 % | 专家费用 USD/万题 | Rank Score |", "|---|---:|---:|---:|---:|"]
    for row in summary:
        star = " *" if row["selected_in_source_oof"] else ""
        lines.append(f"| {row['method']} | {row['lambda_cost']:.3f}{star} | {row['accuracy_percent_mean']:.3f} | "
                     f"{row['cost_usd_per_10k_mean']:.4f} | {row['rank_score_mean']:.3f} |")
    lines += ["", f"MMMU test 原生学科识别诊断：{diagnostic['native_subject_correct']}/{diagnostic['mmmu_test_n']} = "
              f"{diagnostic['native_subject_accuracy_percent']:.2f}%。这只衡量状态是否接近原生学科，不参与专家路由选参。",
              "", "费用只包含缓存中的专家 USD proxy，不包含 Qwen3-VL 特征提取。历史完整测试集结果已经在其他研究中出现，本次为探索性分析；不可用测试最高 Rank 反向选取新 λ。",
              "逐数据集见 `per_dataset.csv`，逐种子见 `per_seed.csv`，逐学科测试占用见 `mmmu30_test_subject_counts.csv`。"]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(stage="evaluate", selected=[dict(method=row["method"], lambda_cost=row["lambda_cost"],
                                      accuracy=row["accuracy_percent_mean"], cost=row["cost_usd_per_10k_mean"],
                                      rank=row["rank_score_mean"]) for row in selected_rows],
        mmmu_subject=diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "freeze", "predict", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"train": train, "freeze": freeze, "predict": predict, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
