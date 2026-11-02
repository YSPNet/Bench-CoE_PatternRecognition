#!/usr/bin/env python3
"""Seed-0 11-ID subject-routed outcomes grouped by the predicted MMMU subject.

The grouped labels come from the frozen TinyLLaVA subject head.  Both actions
are frozen before test outcomes are opened.  The plotted support cutoff is a
display rule, not a router-selection rule; the CSV retains all 30 subjects.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/journal_20261024"))
from common import eval_matrices, scores, view_uids  # noqa: E402

OUT = ROOT / "outputs/journal_20261024/V10_SUBJECT_RADAR"
PRED = ROOT / "outputs/bench_coe_subject_mmmu30/ood11_3/tinyllava/seed-0/predictions_11-ID.npz"
ASSIGN = ROOT / "outputs/bench_coe_subject_mmmu30/assignments.npz"
ACTIONS = ROOT / "outputs/journal_20261024/E21/actions_seed-0.npz"
E21 = ROOT / "outputs/journal_20261024/E21/RESULTS.csv"
BOUNDS = ROOT / "data/matrices/cost_bounds.json"
MIN_PLOT_N = 20


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def lookup_e21() -> dict[str, str]:
    matches = [row for row in read_csv(E21)
               if row["method"] == "Subject-TinyLLaVA" and row["view"] == "11ID"
               and row["seed"] == "0" and row["variant"].endswith("/select=source_cv")]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one seed-0 source-selected E21 11-ID row")
    return matches[0]


def gather() -> tuple[list[dict], dict]:
    views = view_uids()
    ids = views["11ID"]
    selected = lookup_e21()
    lam = selected["variant"].split("/")[0].removeprefix("lambda=")
    action_key = f"heldout11_seed0_lambda{lam}_11ID"
    with np.load(PRED, allow_pickle=False) as archive:
        if archive["uids"].tolist() != ids:
            raise RuntimeError("frozen subject prediction UID order changed")
        subject = archive["predicted_subject"].astype(np.int64)
        base = archive["selected_expert"].astype(np.int64)
    with np.load(ASSIGN, allow_pickle=False) as archive:
        names = archive["subjects"].tolist()
    with np.load(ACTIONS, allow_pickle=False) as archive:
        if archive["11ID"].tolist() != ids or action_key not in archive:
            raise RuntimeError("E21 source-selected action alignment or lambda changed")
        aware = archive[action_key].astype(np.int64)
    if len(names) != 30 or subject.shape != (5341,) or np.any((subject < 0) | (subject >= 30)):
        raise RuntimeError("11-ID subject-head coverage changed")
    bounds = json.loads(BOUNDS.read_text())
    y, c = eval_matrices(views)[0]["11ID"]
    base_overall = scores(y, c, base, bounds)
    aware_overall = scores(y, c, aware, bounds)
    for got, expected in zip(aware_overall, (selected["A"], selected["C"], selected["R"])):
        if abs(got - float(expected)) > 1e-8:
            raise RuntimeError("grouping actions do not reproduce source-selected E21")
    rows = []
    for index, name in enumerate(names):
        mask = subject == index
        n = int(mask.sum())
        metrics = (scores(y[mask], c[mask], base[mask], bounds),
                   scores(y[mask], c[mask], aware[mask], bounds)) if n else (None, None)
        for variant, action, values in (("base_cost_free", base, metrics[0]),
                                        ("cost_aware_source_cv", aware, metrics[1])):
            rows.append(dict(view="11ID", seed=0, backbone="TinyLLaVA",
                             predicted_subject=name, subject_index=index, n=n,
                             variant=variant, A=values[0] if values else "",
                             C=values[1] if values else "",
                             R=values[2] if values else "",
                             plotted=n >= MIN_PLOT_N,
                             selected_experts=";".join(str(i) for i in np.unique(action[mask]))))
    if len(rows) != 60 or sum(row["n"] for row in rows if row["variant"] == "base_cost_free") != 5341:
        raise RuntimeError("per-subject coverage is incomplete")
    # The weighted group means must reproduce the full 11-ID evaluation.
    for variant, overall in (("base_cost_free", base_overall),
                             ("cost_aware_source_cv", aware_overall)):
        group = [row for row in rows if row["variant"] == variant and row["n"]]
        for metric, expected in zip(("A", "C"), overall[:2]):
            weighted = sum(row["n"] * row[metric] for row in group) / 5341
            if abs(weighted - expected) > 1e-9:
                raise RuntimeError(f"weighted per-subject {metric} failed to replay")
    audit = dict(status="complete", view="11ID", n_test=5341, seed=0,
                 classifier="frozen Subject-Bench-CoE TinyLLaVA MMMU-30 head",
                 grouping="predicted_subject from seed-0 11-ID predictions; not the true subject",
                 variants=["base_cost_free", "cost_aware_source_cv"],
                 source_selected_lambda=lam,
                 metric_definition="repository common.scores, Rank Score with full-sample bounds",
                 overall={"base_cost_free": dict(zip(("A", "C", "R"), base_overall)),
                          "cost_aware_source_cv": dict(zip(("A", "C", "R"), aware_overall))},
                 source_hashes={str(path.relative_to(ROOT)): digest(path)
                                for path in (PRED, ASSIGN, ACTIONS, E21, BOUNDS)},
                 plotted_min_n=MIN_PLOT_N,
                 plotted_subject_count=sum(int(row["plotted"]) for row in rows
                                           if row["variant"] == "base_cost_free"),
                 plotted_query_count=sum(row["n"] for row in rows
                                         if row["variant"] == "base_cost_free" and row["plotted"]),
                 no_test_selection="The >=20 support cutoff is only for figure legibility; all 30 subjects are in CSV.",
                 generated_utc=datetime.now(timezone.utc).isoformat())
    return rows, audit


def figure(rows: list[dict]) -> None:
    names = [row["predicted_subject"] for row in rows
             if row["variant"] == "base_cost_free" and row["plotted"]]
    labels = [name.replace("_and_", " & ").replace("_", " ") for name in names]
    n = len(names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle_closed = np.r_[angles, angles[0]]
    palette = {"base_cost_free": "#8d9c60", "cost_aware_source_cv": "#d77e57"}
    title = {"A": "Accuracy (%)", "R": "Rank Score"}
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(15, 7.7))
    for ax, metric in zip(axes, ("A", "R")):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks((20, 40, 60, 80, 100))
        ax.set_yticklabels(("20", "40", "60", "80", "100"), fontsize=7, color="#666666")
        ax.set_rlabel_position(83)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=7.1)
        ax.tick_params(axis="x", pad=12)
        ax.grid(alpha=0.32, linewidth=0.6)
        ax.spines["polar"].set_color("#777777")
        for variant, label in (("base_cost_free", "Cost-free subject map"),
                               ("cost_aware_source_cv", "Cost-aware source-selected map")):
            lookup = {row["predicted_subject"]: float(row[metric]) for row in rows
                      if row["variant"] == variant and row["plotted"]}
            values = np.array([lookup[name] for name in names])
            ax.plot(angle_closed, np.r_[values, values[0]], color=palette[variant],
                    linewidth=1.8, label=label)
            ax.fill(angle_closed, np.r_[values, values[0]], color=palette[variant], alpha=0.09)
            ax.scatter(angles, values, s=13, color=palette[variant], zorder=4)
        ax.set_title(title[metric], fontsize=12, pad=20)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=10)
    fig.suptitle("11-ID outcomes by predicted MMMU subject (seed 0)", y=1.025, fontsize=14)
    fig.text(0.5, 0.01,
             "Shown: predicted subjects with at least 20 queries; all 30 subjects and counts are in the companion CSV.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.10, wspace=0.32)
    fig.savefig(OUT / "subject_radar_seed0.pdf", bbox_inches="tight")
    fig.savefig(OUT / "subject_radar_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audit = gather()
    path = OUT / "per_subject_seed0.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    figure(rows)
    audit["outputs"] = {path.name: digest(path) for path in
                        (OUT / "per_subject_seed0.csv", OUT / "subject_radar_seed0.pdf",
                         OUT / "subject_radar_seed0.png")}
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    (OUT / "README.md").write_text(
        "# v10 subject-level per-subject radar\n\n"
        "Seed-0 11-ID predictions from the frozen Subject-Bench-CoE TinyLLaVA head "
        "were grouped by its *predicted* MMMU subject. The source-CV-selected "
        f"cost-aware map uses λ={audit['source_selected_lambda']}; the cost-free map "
        "uses the same classifier. All 30 subject outcomes, counts and expert "
        "choices are in `per_subject_seed0.csv`. The two-panel radar displays "
        f"{audit['plotted_subject_count']} groups with n≥20, covering "
        f"{audit['plotted_query_count']}/5341 queries. Small or empty groups are "
        "excluded from the image solely for legibility; their data remain in the CSV. "
        "The subgroup values are descriptive and one-seed, without uncertainty "
        "intervals. The weighted subgroup A and C reproduce E21's full 11-ID point.\n")
    with (ROOT / "outputs/journal_20261024/PROGRESS.md").open("a") as stream:
        stream.write("\n## V10 per-subject radar — complete\n\n"
                     "Frozen seed-0 Subject-TinyLLaVA 11-ID predictions were grouped by "
                     "predicted MMMU subject. The 30-subject CSV and support-filtered "
                     "accuracy/Rank radar are in V10_SUBJECT_RADAR; weighted A/C replays E21.\n")
    print(json.dumps({"status": "complete", "rows": len(rows),
                      "plotted_subjects": audit["plotted_subject_count"],
                      "plotted_queries": audit["plotted_query_count"]}))


if __name__ == "__main__":
    main()
