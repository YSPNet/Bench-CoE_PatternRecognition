#!/usr/bin/env python3
"""Full14 source-only domain-inference expert router on frozen Qwen3-VL features.

The 14 official training dataset names supervise a linear domain classifier.
At inference, only the query feature is used to infer the domain; UID prefixes
are used solely for ordering and evaluation. Expert quality and cost profiles
are learned from source-fit outcomes. Run train, freeze, predict, evaluate in
separate processes so no test outcome is available when actions are chosen.
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
from scripts import run_stateroute_mmmu_full14_20260928 as full14
from scripts import run_stateroute_mmmu_subject_20260928 as subject
from src.next_round_x.data import role_uids
from src.next_round_x.models import normalized
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

OUT = ROOT / "outputs/stateroute_domain_ridge_full14_20260928"
REPORT = ROOT / "reports/stateroute_domain_ridge_full14_20260928"
ALPHAS = (.1, 1., 10.)
QUALITY_KAPPAS = (64., 256., 512.)
COST_KAPPA = 128.
COST_MODES = ("global", "domain")
LAMBDAS = (.02, .025, .03, .035, .04, .045, .05, .06, .07, .08, .1)
N_DOMAINS = 14
EXPERTS = 17


def say(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def domain_index(names, all_names):
    lookup = {name: i for i, name in enumerate(names)}
    if len(lookup) != len(names) or not set(all_names) <= set(lookup):
        raise RuntimeError("source dataset names changed")
    return np.array([lookup[name] for name in all_names], dtype=np.int64)


def profile(y, c, d, *, quality_kappa, cost_kappa=COST_KAPPA):
    if len(y) != len(c) or len(y) != len(d) or y.shape[1] != EXPERTS:
        raise ValueError("source quality/cost/domain alignment")
    count = np.bincount(d, minlength=N_DOMAINS)
    if len(count) != N_DOMAINS or (count == 0).any():
        raise RuntimeError("a domain has no source training rows")
    global_quality = y.mean(axis=0, dtype=np.float64)
    global_cost = c.mean(axis=0, dtype=np.float64)
    median_positive = float(np.median(c[c > 0]))
    quality_sum = np.zeros((N_DOMAINS, EXPERTS), dtype=np.float64)
    cost_sum = np.zeros_like(quality_sum)
    np.add.at(quality_sum, d, y)
    np.add.at(cost_sum, d, c)
    quality = (quality_sum + quality_kappa * global_quality) / (count[:, None] + quality_kappa)
    domain_cost = (cost_sum + cost_kappa * global_cost) / (count[:, None] + cost_kappa) / median_positive
    global_scaled_cost = global_cost / median_positive
    return quality, domain_cost, global_scaled_cost, count


def fit_classifier(x, labels, alpha):
    model = RidgeClassifier(alpha=alpha, solver="lsqr", class_weight="balanced", tol=1e-4)
    model.fit(normalized(x), labels)
    if not np.array_equal(model.classes_, np.arange(N_DOMAINS)) or model.coef_.shape != (N_DOMAINS, x.shape[1]):
        raise RuntimeError("domain classifier class/dimension contract changed")
    return model


def infer_domain(x, coef, intercept):
    unit = normalized(x)
    if coef.shape != (N_DOMAINS, unit.shape[1]) or intercept.shape != (N_DOMAINS,):
        raise RuntimeError("invalid frozen domain classifier")
    return (unit @ coef.T + intercept).argmax(axis=1)


def source_candidate_metrics(y, c, actions):
    values = subject.metrics(y, c, actions)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    selection_rank = 100 * rank_score(float(y[np.arange(len(y)), actions].mean()),
                                      values["cost_usd_per_10k"] / 10000,
                                      bounds["cmin"], bounds["cmax"])
    return dict(selection_rank=float(selection_rank), **values)


def train():
    if (OUT / "SOURCE_SELECTION.json").exists():
        raise FileExistsError("domain router source selection already exists")
    data, x, fit, overlay = full14.source_train()
    names = sorted(set(data["dataset"].tolist()))
    if len(names) != N_DOMAINS:
        raise RuntimeError("full14 source dataset count changed")
    d = domain_index(names, data["dataset"])
    y, c = data["y"], data["c"]
    row = {int(index): i for i, index in enumerate(fit)}
    oof_domain = {alpha: np.empty(len(fit), dtype=np.int8) for alpha in ALPHAS}
    oof_quality = {(alpha, kappa): np.empty((len(fit), EXPERTS), dtype=np.float32)
                   for alpha in ALPHAS for kappa in QUALITY_KAPPAS}
    oof_cost = {(alpha, mode): np.empty((len(fit), EXPERTS), dtype=np.float32)
                for alpha in ALPHAS for mode in COST_MODES}
    seen = np.zeros(len(fit), dtype=np.int8)
    fold_notes = []
    for fold in range(3):
        train_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
        valid_indices = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
        if set(data["groups"][train_indices]) & set(data["groups"][valid_indices]):
            raise RuntimeError("source OOF dependency groups overlap")
        target = np.array([row[int(index)] for index in valid_indices], dtype=np.int64)
        fitted = {kappa: profile(y[train_indices], c[train_indices], d[train_indices],
                                 quality_kappa=kappa) for kappa in QUALITY_KAPPAS}
        for alpha in ALPHAS:
            clf = fit_classifier(x[train_indices], d[train_indices], alpha)
            predicted = infer_domain(x[valid_indices], clf.coef_, np.asarray(clf.intercept_))
            if not np.array_equal(predicted, clf.predict(normalized(x[valid_indices]))):
                raise RuntimeError("NumPy and sklearn domain predictions differ")
            oof_domain[alpha][target] = predicted
            for kappa in QUALITY_KAPPAS:
                quality, domain_cost, global_cost, _ = fitted[kappa]
                oof_quality[(alpha, kappa)][target] = quality[predicted]
                oof_cost[(alpha, "domain")][target] = domain_cost[predicted]
                oof_cost[(alpha, "global")][target] = global_cost
        seen[target] += 1
        note = dict(fold=fold, train_n=len(train_indices), valid_n=len(valid_indices), group_overlap_n=0)
        fold_notes.append(note)
        say(stage="oof_fold", **note)
    if not (seen == 1).all():
        raise RuntimeError("source OOF predictions incomplete")
    rows = []
    source_y, source_c = y[fit], c[fit]
    for alpha in ALPHAS:
        for kappa in QUALITY_KAPPAS:
            q = oof_quality[(alpha, kappa)].astype(np.float64)
            for mode in COST_MODES:
                cost = oof_cost[(alpha, mode)].astype(np.float64)
                for lam in LAMBDAS:
                    actions = (q - lam * cost).argmax(axis=1)
                    values = source_candidate_metrics(source_y, source_c, actions)
                    rows.append(dict(alpha=alpha, quality_kappa=kappa, cost_mode=mode, lambda_cost=lam,
                                     domain_prediction_accuracy=float((oof_domain[alpha] == d[fit]).mean()),
                                     **values))
    # Selection is entirely within official train/source-fit group-held-out OOF.
    winner = max(rows, key=lambda item: item["selection_rank"])
    alpha = winner["alpha"]
    kappa = winner["quality_kappa"]
    mode = winner["cost_mode"]
    lam = winner["lambda_cost"]
    clf = fit_classifier(x[fit], d[fit], alpha)
    quality, domain_cost, global_cost, count = profile(y[fit], c[fit], d[fit], quality_kappa=kappa)
    selected_cost = domain_cost if mode == "domain" else np.broadcast_to(global_cost, quality.shape)
    table = (quality - lam * selected_cost).argmax(axis=1)
    # Source qcal/cert are read only after the selected rule and final model exist.
    validation = []
    for role in ("qcal", "cert"):
        indices = np.flatnonzero(data["role"] == role)
        predicted = infer_domain(x[indices], clf.coef_, np.asarray(clf.intercept_))
        if not np.array_equal(predicted, clf.predict(normalized(x[indices]))):
            raise RuntimeError("frozen NumPy/sklearn classifier mismatch")
        values = subject.metrics(y[indices], c[indices], table[predicted])
        validation.append(dict(role=role, domain_prediction_accuracy=float((predicted == d[indices]).mean()), **values))
    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / "MODEL.npz"
    np.savez_compressed(model_path, dataset_names=np.array(names), coef=np.asarray(clf.coef_),
                        intercept=np.asarray(clf.intercept_), quality=quality, domain_scaled_cost=domain_cost,
                        global_scaled_cost=global_cost, expert_table=table, support=count,
                        alpha=np.array(alpha), quality_kappa=np.array(kappa), cost_mode=np.array(mode),
                        lambda_cost=np.array(lam))
    write_csv(OUT / "SOURCE_CANDIDATES.csv", rows)
    write_csv(OUT / "SOURCE_HOLDOUT_DIAGNOSTIC.csv", validation)
    write_json(OUT / "SOURCE_SELECTION.json", dict(
        protocol="full14-ID", source_fit_n=len(fit), source_train_n=len(data["uids"]),
        source_qcal_n=int((data["role"] == "qcal").sum()), source_cert_n=int((data["role"] == "cert").sum()),
        method="query-only Qwen3-VL feature -> balanced RidgeClassifier dataset head -> dataset expert quality/cost table",
        test_domain_label_used=False, source_dataset_label_used_for_classifier_and_profile=True,
        source_fit_folds=fold_notes, classifier_alphas=list(ALPHAS), quality_kappas=list(QUALITY_KAPPAS),
        cost_kappa=COST_KAPPA, cost_modes=list(COST_MODES), lambda_grid=list(LAMBDAS),
        selected=winner, source_holdout_diagnostic_not_used_for_selection=validation,
        source_train_multi_image_overlay_n=len(overlay), source_dev_or_test_outcomes_used=False,
        model=record(model_path), candidates=record(OUT / "SOURCE_CANDIDATES.csv"),
        holdout_diagnostic=record(OUT / "SOURCE_HOLDOUT_DIAGNOSTIC.csv")))
    say(stage="train", selected=winner, holdout=validation)


def freeze():
    if (OUT / "MODEL_LOCK.json").exists():
        raise FileExistsError("domain router model already frozen")
    selection = read_json(OUT / "SOURCE_SELECTION.json")
    subject.require_records([selection[key] for key in ("model", "candidates", "holdout_diagnostic")])
    prior_lock = read_json(full14.OUT / "MODEL_LOCK.json")
    subject.require_records(prior_lock["files"])
    files = [Path(__file__), OUT / "SOURCE_SELECTION.json", OUT / "MODEL.npz",
             OUT / "SOURCE_CANDIDATES.csv", OUT / "SOURCE_HOLDOUT_DIAGNOSTIC.csv",
             full14.OUT / "MODEL_LOCK.json", ROOT / "outputs/next_round_x/source/full14/MANIFEST.json",
             ROOT / "data/matrices/cost_bounds.json"]
    write_json(OUT / "MODEL_LOCK.json", dict(
        status="DOMAIN_ROUTER_AND_SELECTION_FROZEN_BEFORE_TEST_ACTIONS",
        protocol="full14-ID", selected=selection["selected"],
        no_test_dataset_label_at_inference=True,
        test_exposure="historical full14 test outcomes have appeared elsewhere; exploratory replay",
        cost_scope="cached expert USD proxy; excludes Qwen3-VL feature extraction",
        files=[record(path) for path in files]))
    say(stage="freeze", files=len(files))


def verify_lock():
    lock = read_json(OUT / "MODEL_LOCK.json")
    subject.require_records(lock["files"])
    prior_lock = read_json(full14.OUT / "MODEL_LOCK.json")
    subject.require_records(prior_lock["files"])
    return lock


def predict():
    if (OUT / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("domain router predictions already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = verify_lock()
    uids = role_uids("full14", "test")
    if len(uids) != 6108 or len(set(uids)) != len(uids):
        raise RuntimeError("full14 official test UID contract changed")
    x, overlay, _ = subject.qwen_features(uids, allow_multi=True)
    if len(overlay) != 10:
        raise RuntimeError("full14 test multi-image feature coverage changed")
    with np.load(OUT / "MODEL.npz", allow_pickle=False) as model:
        predicted = infer_domain(x, model["coef"], model["intercept"])
        table = model["expert_table"]
        expected = (model["quality"] - float(model["lambda_cost"]) *
                    (model["domain_scaled_cost"] if str(model["cost_mode"]) == "domain" else model["global_scaled_cost"])).argmax(axis=1)
        if not np.array_equal(table, expected) or len(model["dataset_names"]) != N_DOMAINS:
            raise RuntimeError("frozen expert table changed")
        actions = table[predicted]
    if actions.shape != (6108,) or (actions < 0).any() or (actions >= EXPERTS).any():
        raise RuntimeError("invalid test actions")
    path = OUT / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), actions=actions.astype(np.int8),
                        inferred_domain=predicted.astype(np.int8),
                        lambda_cost=np.array(lock["selected"]["lambda_cost"]))
    write_json(OUT / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested source-only improvement over full14 paper baselines",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(OUT / "MODEL_LOCK.json"),
        all_test_rows_have_qwen_features=True, test_multi_image_overlay_n=len(overlay),
        test_dataset_name_used_for_action=False, labels_or_actual_cost_used=False,
        test_outcomes_opened=0, test_fit_performed=False, denied_attempts=denied))
    say(stage="predict", n=len(uids), inferred_domains=int(len(set(predicted))),
        distinct_experts=int(len(set(actions))))


def prior_actions(uids):
    path = full14.OUT / "PREDICTIONS.npz"
    receipt = read_json(full14.OUT / "PREDICTION_RECEIPT.json")
    if digest(path) != receipt["prediction_sha256"]:
        raise RuntimeError("previous full14 test predictions changed")
    with np.load(path, allow_pickle=False) as data:
        if data["uids"].tolist() != uids:
            raise RuntimeError("previous full14 test UID order changed")
        methods, seeds, lambdas = data["methods"], data["seeds"], data["lambdas"]
        subject_rows = np.flatnonzero((methods == full14.METHODS[0]) & (lambdas == .05))
        k30_rows = [np.flatnonzero((methods == full14.METHODS[1]) & (seeds == seed) & (lambdas == .05))
                    for seed in range(5)]
        if len(subject_rows) != 1 or any(len(indices) != 1 for indices in k30_rows):
            raise RuntimeError("previous full14 comparison action grid changed")
        return data["actions"][subject_rows[0]].copy(), np.stack([data["actions"][indices[0]] for indices in k30_rows])


def evaluate():
    if (REPORT / "RESULTS.json").exists():
        raise FileExistsError("domain router test already evaluated")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock
    lock = verify_lock()
    path, receipt_path = OUT / "PREDICTIONS.npz", OUT / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(path) != receipt["prediction_sha256"] or digest(OUT / "MODEL_LOCK.json") != receipt["model_lock"]["sha256"]:
        raise RuntimeError("frozen test prediction/model binding changed")
    with np.load(path, allow_pickle=False) as prediction:
        uids = prediction["uids"].tolist()
        actions = prediction["actions"].astype(np.int64)
        inferred = prediction["inferred_domain"].astype(np.int64)
    if uids != role_uids("full14", "test") or len(actions) != 6108:
        raise RuntimeError("full14 test UID order/count changed")
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
    if len(names) != N_DOMAINS or y.shape != (6108, EXPERTS):
        raise RuntimeError("full14 test outcome contract changed")
    with np.load(OUT / "MODEL.npz", allow_pickle=False) as model:
        if model["dataset_names"].tolist() != names:
            raise RuntimeError("full14 dataset name order changed")
    inferred_correct = inferred == domain_index(names, datasets)
    previous_subject, previous_k30 = prior_actions(uids)
    routes = {"MMMU30-Cosine": [previous_subject],
              "StateRoute-MM-K30": list(previous_k30),
              "Qwen-InferredDomain-Ridge": [actions]}
    keys = ("accuracy_percent", "cost_usd_per_10k", "rank_score")
    summary, per_dataset, per_seed = [], [], []
    for method, route_rows in routes.items():
        scores = [subject.metrics(y, c, route) for route in route_rows]
        macro = [np.mean([subject.metrics(y[datasets == name], c[datasets == name], route[datasets == name])["accuracy_percent"]
                          for name in names]) for route in route_rows]
        entry = dict(method=method, seeds=len(route_rows), n=len(uids),
                     macro_dataset_accuracy_percent_mean=float(np.mean(macro)))
        for key in keys:
            values = np.array([score[key] for score in scores])
            entry[key + "_mean"] = float(values.mean())
            entry[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(entry)
        for seed, score in enumerate(scores):
            per_seed.append(dict(method=method, seed=-1 if len(route_rows) == 1 else seed, **score))
        for name in names:
            mask = datasets == name
            local = [subject.metrics(y[mask], c[mask], route[mask]) for route in route_rows]
            per_dataset.append(dict(method=method, dataset=name, n=int(mask.sum()),
                                    inferred_domain_accuracy=float(inferred_correct[mask].mean()) if method == "Qwen-InferredDomain-Ridge" else None,
                                    **{key + "_mean": float(np.mean([score[key] for score in local])) for key in keys}))
    new = next(row for row in summary if row["method"] == "Qwen-InferredDomain-Ridge")
    k30 = next(row for row in summary if row["method"] == "StateRoute-MM-K30")
    paper = {"CosineCls": dict(accuracy_percent=70.24, cost_usd_per_10k=.61, rank_score=69.90),
             "RouterDC": dict(accuracy_percent=77.52, cost_usd_per_10k=1.04, rank_score=74.59),
             "ZOOTER": dict(accuracy_percent=74.65, cost_usd_per_10k=.93, rank_score=72.58),
             "VLC": dict(accuracy_percent=78.09, cost_usd_per_10k=1.23, rank_score=74.33)}
    comparison = [dict(method=method, **values,
                       new_rank_minus_paper_mean=new["rank_score_mean"] - values["rank_score"])
                  for method, values in paper.items()]
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "summary.csv", summary)
    write_csv(REPORT / "per_seed.csv", per_seed)
    write_csv(REPORT / "per_dataset.csv", per_dataset)
    write_csv(REPORT / "paper_descriptive_comparison.csv", comparison)
    write_json(REPORT / "RESULTS.json", dict(
        protocol="full14-ID", selected_source_rule=lock["selected"], n=len(uids), dataset_n=len(names),
        inferred_dataset_accuracy_percent=float(100 * inferred_correct.mean()), summary=summary,
        paper_table2_rank_threshold=74.59, exceeds_paper_routerdc_mean_rank=bool(new["rank_score_mean"] > 74.59),
        comparison_to_k30=dict(accuracy_percent_delta=new["accuracy_percent_mean"] - k30["accuracy_percent_mean"],
                               cost_usd_per_10k_delta=new["cost_usd_per_10k_mean"] - k30["cost_usd_per_10k_mean"],
                               rank_score_delta=new["rank_score_mean"] - k30["rank_score_mean"]),
        paper_comparison_scope="descriptive only: published means have different training/seed/selection details and no per-sample predictions",
        historical_test_exposure="exploratory replay", cost_scope=lock["cost_scope"],
        prediction=record(path), model_lock=record(OUT / "MODEL_LOCK.json")))
    selection = read_json(OUT / "SOURCE_SELECTION.json")
    lines = ["# 14-ID：Qwen 特征推断数据集后按专家画像路由", "",
             "官方 train 的数据集名称只用于训练 Qwen 冻结特征上的线性数据集分类器、及每数据集专家正确率/费用画像。测试动作只读取 query 特征，不读取真实数据集名称或专家结果。",
             "17,871 道 source-fit 题上用三折 dependency-group OOF 选择 Ridge 正则、质量平滑、费用模式和 λ；2,518 道 qcal 与 989 道 cert 仅做选择后的诊断。冻结模型及全部 6,108 条测试动作后解锁测试结果。",
             "", "## 源域规则与诊断", "",
             f"选中：Ridge α={selection['selected']['alpha']}, 质量平滑 κ={selection['selected']['quality_kappa']}, "
             f"费用模式={selection['selected']['cost_mode']}, λ={selection['selected']['lambda_cost']}。",
             f"源域 OOF Rank={selection['selected']['rank_score']:.3f}。",
             "", "## 14-ID 测试结果", "",
             "| 方法 | 准确率 % | 14 集宏平均准确率 % | 专家费用 USD/万题 | Rank Score |",
             "|---|---:|---:|---:|---:|"]
    for entry in summary:
        lines.append(f"| {entry['method']} | {entry['accuracy_percent_mean']:.3f} | "
                     f"{entry['macro_dataset_accuracy_percent_mean']:.3f} | "
                     f"{entry['cost_usd_per_10k_mean']:.4f} | {entry['rank_score_mean']:.3f} |")
    lines += ["", f"测试数据集推断准确率 {100 * inferred_correct.mean():.2f}%。",
              f"相对 K30 五种子均值：准确率 {new['accuracy_percent_mean'] - k30['accuracy_percent_mean']:+.3f} pp，"
              f"费用 {new['cost_usd_per_10k_mean'] - k30['cost_usd_per_10k_mean']:+.4f} USD/万题，"
              f"Rank {new['rank_score_mean'] - k30['rank_score_mean']:+.3f}。",
              f"与论文 Table 2 RouterDC 的均值 74.59 相比，Rank 差值 {new['rank_score_mean'] - 74.59:+.3f}。",
              "", "论文原表只有方法均值和 SD，未取得相同题目的逐题预测；该数值比较不构成复现或配对显著性证明。此方法为确定性单次拟合，也不是论文五次训练的同口径均值。",
              "历史测试结果已有暴露，本轮为探索性回放。专家费用不含 Qwen3-VL 特征提取，未测路由吞吐量。",
              "源域全部候选、保留集诊断、测试逐数据集结果分别见输出目录 `SOURCE_CANDIDATES.csv`、`SOURCE_HOLDOUT_DIAGNOSTIC.csv` 与本目录 `per_dataset.csv`。"]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(stage="evaluate", new=new, k30=k30, inferred_dataset_accuracy_percent=100 * inferred_correct.mean(),
        exceeds_routerdc_mean_rank=new["rank_score_mean"] > 74.59)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "freeze", "predict", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"train": train, "freeze": freeze, "predict": predict, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
