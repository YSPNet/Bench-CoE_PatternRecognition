#!/usr/bin/env python3
"""Retrain query-inferred-domain router on 11 sources; evaluate 11-ID/3-OOD.

The classifier has exactly 11 known classes. All hyperparameters and expert
profiles are selected using source-fit group-held-out OOF only. Both ID and OOD
test actions are frozen in one file before either view's outcomes are opened.
Unseen OOD queries are necessarily assigned to a known class; this is the
direct-transfer result, without an OOD-specific fallback or test tuning.
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
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from routers.utils.rank_score import rank_score
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import load_source, role_uids
from src.next_round_x.models import normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

OUT = ROOT / "outputs/stateroute_domain_ridge_11_3_20260928"
REPORT = ROOT / "reports/stateroute_domain_ridge_11_3_20260928"
N_DOMAINS = 11
N_EXPERTS = 17
ALPHAS = (.1, 1., 10.)
QUALITY_KAPPAS = (64., 256., 512.)
COST_KAPPA = 128.
COST_MODES = ("global", "domain")
LAMBDAS = (.02, .025, .03, .035, .04, .045, .05, .06, .07, .08, .1)
METHODS = ("SourceRank", "SourceAccuracyWithin0.5Rank")


def say(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def domain_index(names, values):
    lookup = {name: index for index, name in enumerate(names)}
    if len(lookup) != N_DOMAINS or not set(values) <= set(lookup):
        raise RuntimeError("11-source dataset classes changed")
    return np.array([lookup[value] for value in values], dtype=np.int64)


def profile(y, c, domains, quality_kappa):
    if y.shape != c.shape or y.shape[1] != N_EXPERTS or len(domains) != len(y):
        raise ValueError("quality, cost, and domain alignment")
    count = np.bincount(domains, minlength=N_DOMAINS)
    if len(count) != N_DOMAINS or (count == 0).any():
        raise RuntimeError("source dataset has no training examples")
    global_y = y.mean(0, dtype=np.float64)
    global_c = c.mean(0, dtype=np.float64)
    median_cost = float(np.median(c[c > 0]))
    sy = np.zeros((N_DOMAINS, N_EXPERTS), dtype=np.float64)
    sc = np.zeros_like(sy)
    np.add.at(sy, domains, y)
    np.add.at(sc, domains, c)
    quality = (sy + quality_kappa * global_y) / (count[:, None] + quality_kappa)
    domain_cost = (sc + COST_KAPPA * global_c) / (count[:, None] + COST_KAPPA) / median_cost
    global_cost = global_c / median_cost
    return quality, domain_cost, global_cost, count


def fit_domain(x, labels, alpha):
    classifier = RidgeClassifier(alpha=alpha, solver="lsqr", class_weight="balanced", tol=1e-4)
    classifier.fit(normalized(x), labels)
    if not np.array_equal(classifier.classes_, np.arange(N_DOMAINS)):
        raise RuntimeError("not all source datasets appear in training")
    return classifier


def infer_domain(x, coef, intercept):
    unit = normalized(x)
    if coef.shape != (N_DOMAINS, unit.shape[1]) or intercept.shape != (N_DOMAINS,):
        raise ValueError("frozen domain classifier shape changed")
    return (unit @ coef.T + intercept).argmax(axis=1)


def selection_metrics(y, c, actions):
    result = subject.metrics(y, c, actions)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    selection_rank = 100 * rank_score(float(y[np.arange(len(y)), actions].mean()),
                                      result["cost_usd_per_10k"] / 10000,
                                      bounds["cmin"], bounds["cmax"])
    return dict(selection_rank=float(selection_rank), **result)


def source_data():
    data = load_source("heldout11")
    if data["uids"].tolist() != role_uids("heldout11", "train") or len(data["uids"]) != 18665:
        raise RuntimeError("11-source official train UID contract changed")
    names = sorted(set(data["dataset"].tolist()))
    if len(names) != N_DOMAINS or set(names) & {"ChartQA_TEST", "MMMU_DEV_VAL", "MathVision_MINI"}:
        raise RuntimeError("source contains an OOD dataset")
    fit = np.flatnonzero(data["role"] == "fit")
    if len(fit) != 16269 or not data["available"][fit].all():
        raise RuntimeError("11-source fit feature coverage changed")
    return data, names, domain_index(names, data["dataset"]), fit


def train():
    if (OUT / "SOURCE_SELECTION.json").exists():
        raise FileExistsError("11-source domain selection already exists")
    data, names, domains, fit = source_data()
    x, y, c = data["x"], data["y"], data["c"]
    rows_by_source = {int(index): row for row, index in enumerate(fit)}
    oof_domain = {alpha: np.empty(len(fit), dtype=np.int8) for alpha in ALPHAS}
    oof_quality = {(alpha, kappa): np.empty((len(fit), N_EXPERTS), dtype=np.float32)
                   for alpha in ALPHAS for kappa in QUALITY_KAPPAS}
    oof_cost = {(alpha, mode): np.empty((len(fit), N_EXPERTS), dtype=np.float32)
                for alpha in ALPHAS for mode in COST_MODES}
    seen = np.zeros(len(fit), dtype=np.int8)
    folds = []
    for fold in range(3):
        train_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][train_indices]) & set(data["groups"][valid_indices]):
            raise RuntimeError("source OOF dependency groups overlap")
        target = np.array([rows_by_source[int(index)] for index in valid_indices], dtype=np.int64)
        fitted = {kappa: profile(y[train_indices], c[train_indices], domains[train_indices], kappa)
                  for kappa in QUALITY_KAPPAS}
        for alpha in ALPHAS:
            clf = fit_domain(x[train_indices], domains[train_indices], alpha)
            predicted = infer_domain(x[valid_indices], clf.coef_, np.asarray(clf.intercept_))
            if not np.array_equal(predicted, clf.predict(normalized(x[valid_indices]))):
                raise RuntimeError("NumPy/sklearn inference mismatch")
            oof_domain[alpha][target] = predicted
            for kappa in QUALITY_KAPPAS:
                quality, domain_cost, global_cost, _ = fitted[kappa]
                oof_quality[(alpha, kappa)][target] = quality[predicted]
                oof_cost[(alpha, "domain")][target] = domain_cost[predicted]
                oof_cost[(alpha, "global")][target] = global_cost
        seen[target] += 1
        folds.append(dict(fold=fold, train_n=len(train_indices), valid_n=len(valid_indices), group_overlap_n=0))
        say(stage="source_fold", **folds[-1])
    if not (seen == 1).all():
        raise RuntimeError("11-source OOF coverage incomplete")
    candidates = []
    for alpha in ALPHAS:
        for kappa in QUALITY_KAPPAS:
            q = oof_quality[(alpha, kappa)].astype(np.float64)
            for mode in COST_MODES:
                cost = oof_cost[(alpha, mode)].astype(np.float64)
                for lam in LAMBDAS:
                    action = (q - lam * cost).argmax(axis=1)
                    candidates.append(dict(alpha=alpha, quality_kappa=kappa, cost_mode=mode, lambda_cost=lam,
                                           domain_prediction_accuracy=float((oof_domain[alpha] == domains[fit]).mean()),
                                           **selection_metrics(y[fit], c[fit], action)))
    rank_winner = max(candidates, key=lambda candidate: candidate["selection_rank"])
    rank_floor = rank_winner["rank_score"] - .5
    eligible = [item for item in candidates if item["rank_score"] >= rank_floor and
                item["cost_usd_per_10k"] <= 1.04]
    if not eligible:
        raise RuntimeError("no source accuracy working point satisfies predefined guard")
    accuracy_winner = max(eligible, key=lambda item: (item["accuracy_percent"],
                                                         item["rank_score"], -item["cost_usd_per_10k"]))
    selected = {METHODS[0]: rank_winner, METHODS[1]: accuracy_winner}
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "SOURCE_CANDIDATES.csv", candidates)
    holdout = []
    model_records = {}
    for method, winner in selected.items():
        clf = fit_domain(x[fit], domains[fit], winner["alpha"])
        quality, domain_cost, global_cost, support = profile(
            y[fit], c[fit], domains[fit], winner["quality_kappa"])
        scaled = domain_cost if winner["cost_mode"] == "domain" else np.broadcast_to(global_cost, quality.shape)
        table = (quality - winner["lambda_cost"] * scaled).argmax(axis=1)
        path = OUT / "models" / (method + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dataset_names=np.array(names), coef=np.asarray(clf.coef_),
                            intercept=np.asarray(clf.intercept_), quality=quality,
                            domain_scaled_cost=domain_cost, global_scaled_cost=global_cost,
                            expert_table=table, support=support, alpha=np.array(winner["alpha"]),
                            quality_kappa=np.array(winner["quality_kappa"]),
                            cost_mode=np.array(winner["cost_mode"]), lambda_cost=np.array(winner["lambda_cost"]))
        model_records[method] = record(path)
        for role in ("qcal", "cert"):
            indices = np.flatnonzero(data["role"] == role)
            inferred = infer_domain(x[indices], clf.coef_, np.asarray(clf.intercept_))
            holdout.append(dict(method=method, role=role,
                                domain_prediction_accuracy=float((inferred == domains[indices]).mean()),
                                **subject.metrics(y[indices], c[indices], table[inferred])))
    write_csv(OUT / "SOURCE_HOLDOUT_DIAGNOSTIC.csv", holdout)
    write_json(OUT / "SOURCE_SELECTION.json", dict(
        protocol="heldout11-to-3-OOD", source_train_n=len(data["uids"]), source_fit_n=len(fit),
        source_datasets=names, ood_datasets=["ChartQA_TEST", "MMMU_DEV_VAL", "MathVision_MINI"],
        method="Qwen feature -> 11-class balanced Ridge domain head -> source-domain expert profile",
        source_fold_protocol=folds, alphas=list(ALPHAS), quality_kappas=list(QUALITY_KAPPAS),
        cost_kappa=COST_KAPPA, cost_modes=list(COST_MODES), lambda_grid=list(LAMBDAS),
        rank_selection="maximum source-fit group OOF Rank",
        accuracy_selection="maximum source-fit OOF accuracy among OOF Rank within 0.5 of best and source cost <=1.04 USD/10k",
        accuracy_rank_floor=rank_floor, selected=selected,
        holdout_diagnostic_not_used_for_selection=holdout,
        source_or_ood_test_outcomes_used=False, ood_source_expert_outcomes_used=False,
        prior_full14_study_exposed_ood_datasets=True,
        candidates=record(OUT / "SOURCE_CANDIDATES.csv"),
        source_holdout=record(OUT / "SOURCE_HOLDOUT_DIAGNOSTIC.csv"),
        models=model_records))
    say(stage="train", selected=selected, holdout=holdout)


def freeze():
    if (OUT / "MODEL_LOCK.json").exists():
        raise FileExistsError("11/3 domain models already frozen")
    selection = read_json(OUT / "SOURCE_SELECTION.json")
    subject.require_records([selection["candidates"], selection["source_holdout"], *selection["models"].values()])
    pointer = read_json(subject.POINTER)
    if pointer["input_content"]["model_configured_path"] != subject.MODEL_PATH:
        raise RuntimeError("Qwen feature model path changed")
    cache = ROOT / pointer["cache_root"] / "merged"
    files = [Path(__file__), Path(subject.__file__), ROOT / "src/next_round_x/data.py",
             ROOT / "src/next_round_x/models.py", ROOT / "routers/utils/rank_score.py",
             OUT / "SOURCE_SELECTION.json", OUT / "SOURCE_CANDIDATES.csv",
             OUT / "SOURCE_HOLDOUT_DIAGNOSTIC.csv", ROOT / "outputs/next_round_x/source/heldout11/MANIFEST.json",
             ROOT / "artifacts/subject_routing/manifests/splits.json",
             ROOT / "artifacts/subject_routing/manifests/model_pool.json",
             ROOT / "data/matrices/cost_bounds.json", subject.POINTER,
             cache / "manifest.json", cache / "features.npy", cache / "index.jsonl",
             subject.OVERLAY, subject.OVERLAY_MANIFEST]
    files += [OUT / "models" / (method + ".npz") for method in METHODS]
    write_json(OUT / "MODEL_LOCK.json", dict(
        status="11_SOURCE_DOMAIN_ROUTES_FROZEN_BEFORE_ID_OOD_ACTIONS",
        protocol="heldout11-to-3-OOD", methods=list(METHODS), selected=selection["selected"],
        source_only_parameter_selection=True, no_ood_expert_outcomes_in_fit=True,
        test_exposure="historical 11-ID/3-OOD results and full14 source outcomes already exposed; exploratory replay",
        cost_scope="cached expert USD proxy; excludes Qwen3-VL feature extraction",
        files=[record(path) for path in files]))
    say(stage="freeze", files=len(files))


def verify_lock():
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    if tuple(lock["methods"]) != METHODS:
        raise RuntimeError("frozen 11/3 methods changed")
    return lock


def predict():
    if (OUT / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("11/3 domain test predictions already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = verify_lock()
    id_uids = role_uids("heldout11", "id")
    ood_uids = role_uids("heldout11", "ood")
    uids = id_uids + ood_uids
    if len(id_uids) != 5341 or len(ood_uids) != 767 or len(set(uids)) != 6108:
        raise RuntimeError("11-ID/3-OOD official test UID contract changed")
    x, overlay, _ = subject.qwen_features(uids, allow_multi=True)
    if len(overlay) != 10 or any(uid in id_uids for uid in overlay):
        raise RuntimeError("OOD test multi-image feature coverage changed")
    actions, inferred = [], []
    for method in METHODS:
        with np.load(OUT / "models" / (method + ".npz"), allow_pickle=False) as model:
            domain_id = infer_domain(x, model["coef"], model["intercept"])
            expected = (model["quality"] - float(model["lambda_cost"]) *
                        (model["domain_scaled_cost"] if str(model["cost_mode"]) == "domain" else model["global_scaled_cost"])).argmax(axis=1)
            if not np.array_equal(expected, model["expert_table"]):
                raise RuntimeError("source expert table changed")
            expert = model["expert_table"][domain_id]
        if expert.shape != (6108,) or (expert < 0).any() or (expert >= N_EXPERTS).any():
            raise RuntimeError("invalid ID/OOD actions")
        actions.append(expert.astype(np.int8))
        inferred.append(domain_id.astype(np.int8))
    path = OUT / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), methods=np.array(METHODS),
                        views=np.array(["11-ID"] * len(id_uids) + ["3-OOD"] * len(ood_uids)),
                        actions=np.stack(actions), inferred_domain=np.stack(inferred))
    write_json(OUT / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested 11-ID and 3-OOD results for source-only domain router",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(OUT / "MODEL_LOCK.json"),
        id_test_n=len(id_uids), ood_test_n=len(ood_uids), test_multi_image_overlay_n=len(overlay),
        true_test_dataset_name_used_for_actions=False, labels_or_actual_cost_used=False,
        test_outcomes_opened=0, test_fit_performed=False, denied_attempts=denied))
    say(stage="predict", n=len(uids), id_n=len(id_uids), ood_n=len(ood_uids),
        distinct_experts=[int(len(set(action))) for action in actions])


def evaluate():
    if (REPORT / "RESULTS.json").exists():
        raise FileExistsError("11/3 domain result already evaluated")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock
    lock = verify_lock()
    path = OUT / "PREDICTIONS.npz"
    receipt_path = OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(path) != receipt["prediction_sha256"] or digest(OUT / "MODEL_LOCK.json") != receipt["model_lock"]["sha256"]:
        raise RuntimeError("frozen 11/3 prediction/model binding changed")
    with np.load(path, allow_pickle=False) as pred:
        uids = pred["uids"].tolist()
        methods = pred["methods"].tolist()
        views = pred["views"].copy()
        actions = pred["actions"].astype(np.int64)
        inferred = pred["inferred_domain"].astype(np.int64)
    if uids != role_uids("heldout11", "id") + role_uids("heldout11", "ood") or methods != list(METHODS):
        raise RuntimeError("11/3 test UID or method order changed")
    if actions.shape != (2, 6108) or int((views == "11-ID").sum()) != 5341:
        raise RuntimeError("11/3 prediction shape changed")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **subject.label_args())
    write_json(OUT / "OUTCOME_PROVENANCE.json", dict(materialization=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    dataset = np.array([uid.split("/", 1)[0] for uid in uids])
    if y.shape != (6108, N_EXPERTS) or len(set(dataset[views == "11-ID"])) != 11 or len(set(dataset[views == "3-OOD"])) != 3:
        raise RuntimeError("11/3 scoped test outcome contract changed")
    keys = ("accuracy_percent", "cost_usd_per_10k", "rank_score")
    summary, per_dataset, ood_assignment = [], [], []
    for method, action, domain_id in zip(METHODS, actions, inferred):
        with np.load(OUT / "models" / (method + ".npz"), allow_pickle=False) as model:
            names = model["dataset_names"].tolist()
        id_mask = views == "11-ID"
        id_true = domain_index(names, dataset[id_mask])
        id_inference_accuracy = float((domain_id[id_mask] == id_true).mean())
        for view in ("11-ID", "3-OOD"):
            take = views == view
            value = subject.metrics(y[take], c[take], action[take])
            macro = float(np.mean([subject.metrics(y[take & (dataset == name)], c[take & (dataset == name)],
                                                  action[take & (dataset == name)])["accuracy_percent"]
                                   for name in sorted(set(dataset[take]))]))
            summary.append(dict(method=method, view=view, lambda_cost=lock["selected"][method]["lambda_cost"],
                                macro_dataset_accuracy_percent=macro,
                                id_domain_inference_accuracy_percent=100 * id_inference_accuracy if view == "11-ID" else None,
                                **value))
        for name in sorted(set(dataset)):
            mask = dataset == name
            per_dataset.append(dict(method=method, dataset=name, view=str(views[np.flatnonzero(mask)[0]]),
                                    **subject.metrics(y[mask], c[mask], action[mask])))
            if str(views[np.flatnonzero(mask)[0]]) == "3-OOD":
                support = np.bincount(domain_id[mask], minlength=N_DOMAINS)
                ood_assignment.extend(dict(method=method, ood_dataset=name, inferred_source_dataset=source,
                                           n=int(support[i])) for i, source in enumerate(names) if support[i])
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "summary.csv", summary)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    write_csv(REPORT / "ood_inferred_source_domains.csv", ood_assignment)
    write_json(REPORT / "RESULTS.json", dict(
        protocol="heldout11-to-3-OOD", source_fit_n=16269, id_test_n=5341, ood_test_n=767,
        summary=summary, source_selection=record(OUT / "SOURCE_SELECTION.json"),
        prediction=record(path), model_lock=record(OUT / "MODEL_LOCK.json"),
        no_ood_expert_outcomes_in_fit=True, test_metric_selection=False,
        historical_exposure="earlier full14 study saw all three OOD datasets as ID; this is an exploratory 11-source-only retrain, not a pristine blind OOD test",
        cost_scope=lock["cost_scope"]))
    lines = ["# 11-ID / 3-OOD：Qwen 推断已知数据集专家画像", "",
             "在 11 个源数据集的官方 train 中只用 source-fit 16,269 题训练 11 类 Qwen 数据集分类器及专家画像；三折 dependency-group OOF 选择两种工作点。3 个 OOD 数据集的专家正确率与费用没有参与拟合或选参。",
             "OOD 题被强制归入一个已知源类别，直接迁移该源类别的专家动作；本实验没有事后 OOD 回退或测试调参。两个测试视图共 6,108 题，预测动作一次冻结后再读取结果。",
             "", "| 工作点 | 视图 | λ | 准确率 % | 专家费用 USD/万题 | Rank |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['view']} | {row['lambda_cost']:.3f} | "
                     f"{row['accuracy_percent']:.3f} | {row['cost_usd_per_10k']:.4f} | {row['rank_score']:.3f} |")
    lines += ["", "逐数据集见 `per_dataset.csv`；OOD 被归入哪些源类别见 `ood_inferred_source_domains.csv`。",
              "历史 full14 研究曾把这三个 OOD 数据集用于 ID 训练并曝光其结果；本轮 11-source 拟合与选参没有读入这些结果，但不能称为全新盲测。",
              "费用只包含缓存专家 USD proxy，不含 Qwen3-VL 特征提取；此 11→3 结果不能由先前 full14 测试切片替代。"]
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
