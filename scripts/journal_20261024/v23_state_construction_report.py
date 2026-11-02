#!/usr/bin/env python3
"""Audit and report the v23 state-construction experiments without editing v23."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from common import (OUTPUT, ROOT, eval_matrices, read_json, scores,
                    view_uids, write_csv, write_json)
from v23_mm_spherical_compare import sha

MM = OUTPUT / "V23_MM_SPHERICAL_COMPARE"
MODALITY = OUTPUT / "V23_STATE_MODALITY_COMPARE"
OUT = OUTPUT / "V23_STATE_CONSTRUCTION_REPORT"
SEEDS = tuple(range(5))
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261031
COMPARISONS = (
    ("MM: 球面 - 原欧氏", "MM", "euclidean_original", "spherical_fixed_lambda"),
    ("MM: 原质心余弦分配 - 原欧氏", "MM", "euclidean_original", "cosine_original_centroids"),
    ("图像 SigLIP - 原文本 MiniLM", "modality", "minilm_text_original", "siglip_image_spherical"),
    ("图像 SigLIP - 文本 BGE-M3", "modality", "bge_text_spherical", "siglip_image_spherical"),
    ("BGE-M3+SigLIP - 文本 BGE-M3", "modality", "bge_text_spherical", "bge_siglip_spherical"),
)
PAIRED_COLUMNS = ("comparison", "view", "n_test", "n_seeds", "bootstrap_reps",
                  "delta_A", "A_ci_low", "A_ci_high", "delta_C", "C_ci_low",
                  "C_ci_high", "delta_R", "R_ci_low", "R_ci_high",
                  "changed_actions_mean")
DATASET_COLUMNS = ("dataset", "n_test", "method", "variant", "A_mean", "A_sd",
                   "C_mean", "C_sd", "R_mean", "R_sd")


def rank_array(accuracy: np.ndarray, cost: np.ndarray, bounds: dict) -> np.ndarray:
    clipped = np.clip(cost, bounds["cmin"], bounds["cmax"])
    normalized_cost = (np.log2(bounds["cmax"]) - np.log2(clipped)) / (
        np.log2(bounds["cmax"]) - np.log2(bounds["cmin"]) + 1e-12)
    normalized_cost = np.clip(normalized_cost, 0, 1)
    accuracy = np.clip(accuracy, 0, 1)
    return np.clip(1.1 * accuracy * normalized_cost /
                   (.1 * accuracy + normalized_cost + 1e-12), 0, 1)


def load_actions(root: Path, variants: set[str], uids: list[str]) -> dict[str, np.ndarray]:
    lock = read_json(root / "ACTION_LOCK.json")
    if lock["status"] != "predictions_frozen_before_test_scoring":
        raise RuntimeError(f"predictions are not sealed: {root}")
    records = {variant: [] for variant in variants}
    for seed in SEEDS:
        path = root / "models" / f"seed-{seed}" / "PREDICTIONS.npz"
        metadata = read_json(path.parent / "COMPLETE.json")
        if sha(path) != metadata["predictions_sha256"]:
            raise RuntimeError(f"prediction digest changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if archive["uids"].tolist() != uids:
                raise RuntimeError(f"test UID order changed: {path}")
            for variant in variants:
                records[variant].append(np.asarray(archive[variant], dtype=np.int8))
    return {variant: np.stack(rows) for variant, rows in records.items()}


def selected(y: np.ndarray, c: np.ndarray,
             actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))[None, :]
    return y[idx, actions].astype(np.float64), c[idx, actions].astype(np.float64)


def paired(label: str, view: str, y: np.ndarray, c: np.ndarray,
           before: np.ndarray, after: np.ndarray, bounds: dict) -> dict:
    base_y, base_c = selected(y, c, before)
    alt_y, alt_c = selected(y, c, after)
    exact_a = 100 * float((alt_y.mean(axis=1) - base_y.mean(axis=1)).mean())
    exact_c = 10000 * float((alt_c.mean(axis=1) - base_c.mean(axis=1)).mean())
    exact_r = 100 * float((rank_array(alt_y.mean(axis=1), alt_c.mean(axis=1), bounds) -
                            rank_array(base_y.mean(axis=1), base_c.mean(axis=1), bounds)).mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if view == "11ID" else 1) +
                                sum(ord(character) for character in label))
    replicates = {"A": [], "C": [], "R": []}
    for start in range(0, BOOTSTRAPS, 100):
        batch = min(100, BOOTSTRAPS - start)
        indices = rng.integers(0, len(y), size=(batch, len(y)))
        ay = alt_y[:, indices].mean(axis=2)
        ac = alt_c[:, indices].mean(axis=2)
        by = base_y[:, indices].mean(axis=2)
        bc = base_c[:, indices].mean(axis=2)
        replicates["A"].append(100 * (ay - by).mean(axis=0))
        replicates["C"].append(10000 * (ac - bc).mean(axis=0))
        replicates["R"].append(100 * (rank_array(ay, ac, bounds) -
                                       rank_array(by, bc, bounds)).mean(axis=0))
    quantiles = {key: np.quantile(np.concatenate(value), (.025, .975)).tolist()
                 for key, value in replicates.items()}
    return dict(comparison=label, view=view, n_test=len(y), n_seeds=len(SEEDS),
                bootstrap_reps=BOOTSTRAPS, delta_A=exact_a,
                A_ci_low=quantiles["A"][0], A_ci_high=quantiles["A"][1],
                delta_C=exact_c, C_ci_low=quantiles["C"][0],
                C_ci_high=quantiles["C"][1], delta_R=exact_r,
                R_ci_low=quantiles["R"][0], R_ci_high=quantiles["R"][1],
                changed_actions_mean=float(np.mean(np.mean(before != after, axis=1))))


def check_summaries(root: Path, actions: dict, matrices: dict, bounds: dict) -> list[dict]:
    with (root / "SUMMARY.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        variant, view = row["variant"], row["view"]
        sl = slice(0, 5341) if view == "11ID" else slice(5341, 6108)
        metrics = np.asarray([scores(*matrices[view], action[sl], bounds)
                              for action in actions[variant]])
        for index, key in enumerate(("A", "C", "R")):
            if abs(float(row[key + "_mean"]) - metrics[:, index].mean()) > 1e-8:
                raise RuntimeError(f"stored summary differs from frozen actions: {root}/{variant}/{view}/{key}")
    return rows


def dataset_rows(uids: list[str], matrices: dict, bounds: dict,
                 modality: dict, mm: dict) -> list[dict]:
    names = sorted(set(uid.split("/", 1)[0] for uid in uids[5341:]))
    output = []
    for dataset in names:
        indices = np.flatnonzero(np.asarray([uid.split("/", 1)[0] == dataset
                                             for uid in uids[5341:]]))
        for method, table, variants in (
                ("T/I/BGE", modality, ("minilm_text_original",
                                        "bge_text_spherical",
                                        "siglip_image_spherical",
                                        "bge_siglip_spherical")),
                ("MM", mm, ("euclidean_original", "spherical_fixed_lambda"))):
            for variant in variants:
                metrics = np.asarray([scores(matrices["3OOD"][0][indices],
                                             matrices["3OOD"][1][indices],
                                             action[5341:][indices], bounds)
                                      for action in table[variant]])
                row = dict(dataset=dataset, n_test=len(indices), method=method,
                           variant=variant)
                for col, pos in (("A", 0), ("C", 1), ("R", 2)):
                    row[col + "_mean"] = float(metrics[:, pos].mean())
                    row[col + "_sd"] = float(metrics[:, pos].std(ddof=1))
                output.append(row)
    return output


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def report(mm_rows: list[dict], modality_rows: list[dict], paired_rows: list[dict],
           datasets: list[dict], docx_sha: str) -> str:
    by = {(item["variant"], item["view"]): item for item in mm_rows + modality_rows}
    lines = ["# v23 状态构建实验：图文聚类几何与单模态消融", "",
             "实验日期：2026-10-31。以下为探索性补充实验，v23 Word/PDF 未修改。", "",
             "## 协议", "",
             "- 11-ID 为 5,341 条；3-OOD 为 767 条。均使用 11-source 拟合，且相同路由器直接应用到两个测试视图。",
             "- MM：冻结 Qwen3-VL-2B 2048 维特征，16,269 条 source-fit，K=2，κ=64，λ=0.03；对照已有小批量欧氏聚类、只把原质心改为余弦分配，以及新球面 k-means。新方法的三折源数据选参在五个种子上也都选中 λ=0.03。",
             "- T/I：MiniLM 文本、BGE-M3 文本、SigLIP-L/16 图像、等权拼接的 BGE-M3+SigLIP，均用 18,665 条源训练样本、K=2 球面 k-means、κ=20、全局源成本中位数资格 q=0.5；所有输入编码器冻结。SigLIP 多图特征先平均再归一化。",
             "- A 为准确率百分比；C 为每 10,000 次查询的专家调用成本（USD，不含编码器运行成本）；R 为 Rank Score。数值是 5 种子平均值 ± 样本标准差。", "",
             "## 整体结果", "",
             "| 构建方式 | 11-ID A / C / R | 3-OOD A / C / R |", "| --- | ---: | ---: |"]
    order = [
        ("原 MM 欧氏小批量", "euclidean_original"),
        ("原 MM 质心余弦分配", "cosine_original_centroids"),
        ("新 MM 球面 k-means", "spherical_fixed_lambda"),
        ("原 T MiniLM 文本", "minilm_text_original"),
        ("BGE-M3 文本", "bge_text_spherical"),
        ("SigLIP 图像", "siglip_image_spherical"),
        ("BGE-M3+SigLIP 图文", "bge_siglip_spherical"),
    ]
    for title, variant in order:
        cells = []
        for view in ("11ID", "3OOD"):
            item = by[variant, view]
            cells.append(" / ".join(f"{float(item[k + '_mean']):.3f} ± {float(item[k + '_sd']):.3f}"
                                    for k in ("A", "C", "R")))
        lines.append(f"| {title} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "跨 MM 与 T/I 行的直接差异同时包含不同编码器、源训练子集、收缩强度和成本规则；应先看各自内部对照。", "",
                  "## 配对差异", "",
                  "同一测试 UID 联合重采样 2,000 次，保持五种子的相关性；区间为 95% 分位区间。Δ 为新方法减对照方法。", "",
                  "| 对比 | 视图 | ΔA | ΔC | ΔR [95% 区间] | 动作变化率 |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for item in paired_rows:
        lines.append(f"| {item['comparison']} | {item['view']} | {fmt(item['delta_A'])} | "
                     f"{fmt(item['delta_C'])} | {fmt(item['delta_R'])} "
                     f"[{fmt(item['R_ci_low'])}, {fmt(item['R_ci_high'])}] | "
                     f"{100*item['changed_actions_mean']:.1f}% |")
    lines.extend(["", "## 3-OOD 数据集分解", "",
                  "下表为五种子平均 Rank Score。", "",
                  "| 数据集 | n | MiniLM 文本 | BGE 文本 | SigLIP 图像 | BGE+SigLIP | MM 原欧氏 | MM 球面 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    lookup = {(row["dataset"], row["variant"]): row for row in datasets}
    for dataset in sorted(set(row["dataset"] for row in datasets)):
        sample = next(row for row in datasets if row["dataset"] == dataset)
        values = [float(lookup[dataset, variant]["R_mean"]) for variant in
                  ("minilm_text_original", "bge_text_spherical",
                   "siglip_image_spherical", "bge_siglip_spherical",
                   "euclidean_original", "spherical_fixed_lambda")]
        lines.append(f"| {dataset} | {sample['n_test']} | " +
                     " | ".join(fmt(value) for value in values) + " |")
    lines.extend(["", "## 审计和解释", "",
                  "- 原 MM 五种子 11-ID/3-OOD 的 10 个 A/C/R 单元格与 V7 冻结结果逐格完全一致；MiniLM 与 BGE-M3 文本的 20 个单元格也逐格完全一致。",
                  "- v23 当前对 MM 的描述写着‘表征未做角度归一化’，但被冻结的原实现先对输入做 L2 归一化，再用非单位范数质心做欧氏分配。新增球面方法将质心也单位化，并用余弦分配。",
                  "- v23 另称 MM 两状态支持数为 11,830 和 6,835，合计 18,665；当前冻结 MM source-fit 模型 seed 0 的支持数是 11,797 和 4,472，合计 16,269。这两处方法文字与当前表值所对应的实现需要在下一版核对。",
                  "- 五个种子的球面解可能收敛到同一个两状态划分；种子标准差为 0 不代表测试抽样不确定性为 0。配对区间用于显示该不确定性。",
                  "- 图像单模态在 OOD 的损失不能单独归因于‘图像信息不足’：SigLIP 与 MiniLM/BGE-M3 是不同编码器；BGE 文本、SigLIP 图像及其拼接使用同一拟合与决策规则，提供较直接的输入消融。",
                  f"- v23 Word 的 SHA-256：`{docx_sha}`。", "",
                  "原始数据：`../V23_MM_SPHERICAL_COMPARE/RESULTS.csv`、`../V23_STATE_MODALITY_COMPARE/RESULTS.csv`；本目录包含配对区间及逐数据集结果。", ""])
    return "\n".join(lines)


def main() -> None:
    if (read_json(MM / "AUDIT.json")["status"] != "complete" or
            read_json(MODALITY / "AUDIT.json")["status"] != "complete"):
        raise RuntimeError("both state-construction experiments must be complete")
    OUT.mkdir(parents=True, exist_ok=True)
    views = view_uids()
    uids = views["11ID"] + views["3OOD"]
    mm = load_actions(MM, {"euclidean_original", "cosine_original_centroids",
                           "spherical_fixed_lambda", "spherical_source_cv"}, uids)
    modality = load_actions(MODALITY, {"minilm_text_original", "bge_text_spherical",
                                        "siglip_image_spherical", "bge_siglip_spherical"},
                            uids)
    matrices, _, _ = eval_matrices(views)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    mm_rows = check_summaries(MM, mm, matrices, bounds)
    modality_rows = check_summaries(MODALITY, modality, matrices, bounds)
    paired_rows = []
    for title, family, base, alternate in COMPARISONS:
        table = mm if family == "MM" else modality
        for view, section in (("11ID", slice(0, 5341)),
                              ("3OOD", slice(5341, 6108))):
            paired_rows.append(paired(title, view, *matrices[view],
                                      table[base][:, section],
                                      table[alternate][:, section], bounds))
    write_csv(OUT / "PAIRED_BOOTSTRAP.csv", PAIRED_COLUMNS, paired_rows)
    datasets = dataset_rows(uids, matrices, bounds, modality, mm)
    write_csv(OUT / "OOD_BY_DATASET.csv", DATASET_COLUMNS, datasets)
    docx_sha = sha(ROOT / "Bench-CoE_PatternRecognition_manuscript_v23.docx")
    (OUT / "REPORT.md").write_text(report(mm_rows, modality_rows,
                                          paired_rows, datasets, docx_sha))
    write_json(OUT / "AUDIT.json", dict(status="complete", bootstraps=BOOTSTRAPS,
        bootstrap_seed=BOOTSTRAP_SEED, scope="paired_test_UID_resampling_across_5_seeds",
        mm_action_lock_sha256=sha(MM / "ACTION_LOCK.json"),
        modality_action_lock_sha256=sha(MODALITY / "ACTION_LOCK.json"),
        manuscript_docx_sha256=docx_sha,
        manuscript_pdf_sha256=sha(ROOT / "Bench-CoE_PatternRecognition_manuscript_v23.pdf")))
    print(f"Wrote {OUT / 'REPORT.md'}; paired comparisons={len(paired_rows)}")


if __name__ == "__main__":
    main()
