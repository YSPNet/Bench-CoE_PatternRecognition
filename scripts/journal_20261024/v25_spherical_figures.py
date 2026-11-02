#!/usr/bin/env python3
"""Regenerate the v25 figures whose plotted State results changed."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from common import OUTPUT, ROOT, eval_matrices, read_json, scores, view_uids, write_json
from v25_spherical_bootstrap import e3_fixed, e6_actions, query_actions, subject_actions

OUT = OUTPUT / "V25_SPHERICAL_COMPLETION"
FIG = OUT / "FIGURES"
BLUE, SKY, AMBER, TEAL, GREY = "#2569ab", "#72b4e4", "#bd8e35", "#3f9d94", "#9ca8b4"


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make(w, h):
    fig = plt.figure(figsize=(w / 200, h / 200), dpi=200, facecolor="white")
    return fig


def save(fig, stem, w, h):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{stem}.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    from PIL import Image
    if Image.open(path).size != (w, h):
        raise RuntimeError(f"figure dimensions changed: {path}")
    return path


def plot_front(matrices, views, bounds):
    w, h = 2244, 665
    fig = make(w, h)
    ax = fig.add_axes([.085, .18, .88, .74])
    curves = {}
    rows = [r for r in csv_rows(OUT / "COST_CURVE.csv") if int(r["seed"]) == 0 and r["view"] == "3OOD"]
    curves["State, spherical"] = [(float(r["C"]), float(r["A"]), float(r["R"]),
                                    float(r["lambda_cost"])) for r in rows]
    uids = views["3OOD"]
    y, c = matrices["3OOD"]
    mlp = []
    root = OUTPUT / "E6/tasks/heldout11/mlp/seed-0"
    for lamdir in sorted(root.glob("lambda-*")):
        path = lamdir / "3OOD.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["sample_ids"].tolist()
            order = {uid: i for i, uid in enumerate(ids)}
            action = saved["actions"][[order[uid] for uid in uids]]
        a, cost, rank = scores(y, c, action, bounds)
        mlp.append((cost, a, rank, lamdir.name))
    curves["MLP, local"] = mlp
    query = []
    root = OUTPUT / "V19_QUERY_COST_TINYLLAVA/heldout11/seed-0"
    for path in sorted(root.glob("final-lambda-*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            if saved["uid_3OOD"].tolist() != uids:
                raise RuntimeError("query curve UID order differs")
            action = saved["action_3OOD"]
        a, cost, rank = scores(y, c, action, bounds)
        query.append((cost, a, rank, path.stem))
    curves["Query, TinyLLaVA"] = query
    styles = {"State, spherical": (BLUE, "o"), "MLP, local": (GREY, "o"),
              "Query, TinyLLaVA": (AMBER, "o")}
    for name, points in curves.items():
        color, marker = styles[name]
        ordered = sorted(points)
        ax.plot([p[0] for p in ordered], [p[1] for p in ordered], "-", color=color,
                lw=2, marker=marker, ms=5, label=name, alpha=.95)
    for name, action, color in (("Query", query_actions(OUTPUT / "V19_QUERY_COST_TINYLLAVA",
                                    "3OOD", uids), AMBER),
                                ("Subject", subject_actions("3OOD"), TEAL)):
        cost_free = None
        if name == "Query":
            # E3 contains the cost-free TinyLLaVA action under its base variant.
            manifest = read_json(OUTPUT / "E3/seed-0.json")
            item = next(x for x in manifest["action_index"] if x["method"] == "Query-TinyLLaVA"
                        and x["variant"] == "base" and x["view"] == "3OOD")
            with np.load(OUTPUT / "E3/actions_seed-0.npz", allow_pickle=False) as saved:
                cost_free = saved[item["key"]]
        else:
            manifest = read_json(OUTPUT / "E3/seed-0.json")
            item = next(x for x in manifest["action_index"] if x["method"] == "Subject-TinyLLaVA"
                        and x["variant"] == "base" and x["view"] == "3OOD")
            with np.load(OUTPUT / "E3/actions_seed-0.npz", allow_pickle=False) as saved:
                cost_free = saved[item["key"]]
        aa, cc, _ = scores(y, c, action, bounds)
        af, cf, _ = scores(y, c, cost_free, bounds)
        ax.annotate("", xy=(cc, aa), xytext=(cf, af),
                    arrowprops=dict(arrowstyle="->", lw=2, color=color))
        ax.scatter([cc], [aa], marker="*", s=140, color=color, zorder=5)
        ax.text(cc * 1.015, aa + .5, name, color=color, fontsize=9)
    main = next(p for p in curves["State, spherical"] if p[3] == .03)
    ax.scatter([main[0]], [main[1]], marker="*", s=170, color=BLUE, zorder=6)
    ax.set_xscale("log")
    ax.set_ylim(57, 82)
    ax.set_xlabel("Expert cost (USD per 10K queries, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("3-OOD accuracy–cost operating points", loc="left", weight="bold", fontsize=12)
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    save(fig, "Fig4_spherical_front", w, h)
    write_json(OUT / "FRONT_POINTS.json", {name: [dict(C=p[0], A=p[1], R=p[2],
                                                       knob=p[3]) for p in points]
                                             for name, points in curves.items()})


def plot_k_sweep():
    w, h = 2250, 1380
    rows = [r for r in csv_rows(OUT / "SUMMARY.csv") if r["variant"].endswith("/main")]
    lookup = {(r["view"], int(r["variant"].split("/")[0][1:])): r for r in rows}
    ks = [1, 2, 4, 6, 8, 10, 12]
    fig, axes = plt.subplots(2, 3, figsize=(w/200,h/200), dpi=200,
                             gridspec_kw=dict(wspace=.32, hspace=.34))
    for ri, view in enumerate(("11ID", "3OOD")):
        for ci, (metric, title) in enumerate((("A", "Accuracy (%)"),
                                               ("C", "Expert cost (USD per 10K)"),
                                               ("R", "Rank Score"))):
            ax = axes[ri, ci]
            values = [float(lookup[view, k][f"{metric}_mean"]) for k in ks]
            sd = [float(lookup[view, k][f"{metric}_sd"]) for k in ks]
            ax.axvspan(1.55, 2.45, color=SKY, alpha=.20, zorder=0)
            ax.errorbar(ks, values, yerr=sd, color=BLUE if ri else SKY,
                        marker="o", lw=2.2, ms=6, capsize=3)
            ax.set_xticks(ks)
            ax.grid(alpha=.2)
            short = ("Accuracy", "Cost", "Rank Score")[ci]
            ax.set_title(f"({'abcdef'[ri*3+ci]}) {view}: {short}", loc="left",
                         fontsize=11, weight="bold")
            if ri == 1:
                ax.set_xlabel("Number of states K")
            ax.set_ylabel(title)
    fig.text(.5, .02, "Five final fits; λ selected on source folds for each K. Error bars: between-seed SD.",
             ha="center", fontsize=9, color="#4a5660")
    save(fig, "Fig6_spherical_k_sweep", w, h)


def local_e3(method, variant, view):
    root = OUTPUT / "E3"
    manifest = read_json(root / "seed-0.json")
    item = next(x for x in manifest["action_index"] if x["method"] == method
                and x["variant"] == variant and x["view"] == view)
    with np.load(root / "actions_seed-0.npz", allow_pickle=False) as saved:
        return saved[item["key"]].astype(np.int8)


def plot_cases(matrices, views, bounds):
    w, h = 2244, 1170
    held = views["3OOD"]
    y, c = matrices["3OOD"]
    with np.load(OUT / "actions/K2_seed-0.npz", allow_pickle=False) as saved:
        state = saved["main"][5341:]
    methods = [
        ("State, spherical", state),
        ("RouterDC", e6_actions("routerdc", "3OOD", held)),
        ("VLC", e6_actions("vlc", "3OOD", held)),
        ("Query aware", query_actions(OUTPUT / "V19_QUERY_COST_TINYLLAVA", "3OOD", held)),
        ("Query free", local_e3("Query-TinyLLaVA", "base", "3OOD")),
        ("Subject aware", subject_actions("3OOD")),
        ("Subject free", local_e3("Subject-TinyLLaVA", "base", "3OOD")),
    ]
    datasets = ("MMMU", "MathVision", "ChartQA")
    indices = [np.flatnonzero(np.asarray([uid.split("/",1)[0].startswith(dataset)
                                          for uid in held])) for dataset in datasets]
    ranks = np.zeros((len(methods), 4))
    accuracy = np.zeros((len(methods), 3))
    shares = np.zeros((len(methods), 3))
    for row, (_, action) in enumerate(methods):
        ranks[row, 3] = scores(y, c, action, bounds)[2]
        for col, ix in enumerate(indices):
            a, _, r = scores(y[ix], c[ix], action[ix], bounds)
            ranks[row, col], accuracy[row, col] = r, a
            shares[row, col] = 100 * np.isin(action[ix], [0, 1, 3, 12]).mean()
    fig = make(w, h)
    ax = fig.add_axes([.16,.15,.38,.73]); bx = fig.add_axes([.64,.15,.32,.73])
    ax.imshow(ranks, cmap="Blues", vmin=0, vmax=85, aspect="auto")
    ax.set_xticks(range(4), list(datasets)+["Pooled"], fontsize=9)
    ax.set_yticks(range(len(methods)), [m[0] for m in methods], fontsize=9)
    ax.set_title("(a) Rank Score by OOD dataset", loc="left", fontsize=12, weight="bold")
    for i in range(len(methods)):
        for j in range(4):
            color = "white" if ranks[i,j] > 45 else "#243b4b"
            ax.text(j,i,f"{ranks[i,j]:.1f}",ha="center",va="center",color=color,fontsize=9,
                    weight="bold" if ranks[i,j] >= ranks[:,j].max()-1e-9 else "normal")
            if ranks[i,j] >= ranks[:,j].max()-1e-9:
                ax.add_patch(Rectangle((j-.49,i-.49),.98,.98,fill=False,edgecolor=AMBER,lw=1.8))
    bx.set_xlim(0,3); bx.set_ylim(len(methods),0)
    bx.set_xticks(np.arange(3)+.5,datasets,fontsize=9)
    bx.set_yticks([])
    bx.set_title("(b) Accuracy / API-or-70B share (%)", loc="left",fontsize=12,weight="bold")
    cmap=plt.cm.Blues
    for i in range(len(methods)):
        for j in range(3):
            x,y0=j,i
            av,sv=accuracy[i,j],shares[i,j]
            bx.add_patch(Polygon([(x,y0),(x+1,y0),(x,y0+1)],closed=True,
                                 facecolor=cmap(.22+.6*av/100),edgecolor="white",lw=1))
            bx.add_patch(Polygon([(x+1,y0),(x+1,y0+1),(x,y0+1)],closed=True,
                                 facecolor=plt.cm.OrRd(.08+.8*sv/100),edgecolor="white",lw=1))
            bx.text(x+.23,y0+.27,f"{av:.0f}",ha="center",va="center",fontsize=9,weight="bold")
            bx.text(x+.77,y0+.74,f"{sv:.0f}",ha="center",va="center",fontsize=9,weight="bold")
    for axx in (ax,bx):
        for spine in axx.spines.values(): spine.set_visible(False)
    fig.text(.8,.085,"Upper left: accuracy    Lower right: API or 70B share",ha="center",fontsize=9)
    save(fig,"Fig10_spherical_cases",w,h)
    write_json(OUT / "FIG10_VALUES.json", dict(methods=[m[0] for m in methods],
        datasets=list(datasets)+["Pooled"], ranks=ranks.tolist(),
        accuracy=accuracy.tolist(), api_or_70b_share=shares.tolist()))


LABELS = {
    "Query-BERT-cost-aware":"Query-BERT (aware)",
    "Query-TinyLLaVA-cost-aware":"Query-TinyLLaVA (aware)",
    "Subject-TinyLLaVA-cost-aware":"Subject-TinyLLaVA (aware)",
}


def forest(ax, rows, title):
    names = [r["comparator"] for r in rows]
    x = np.asarray([float(r["delta_R"]) for r in rows])
    lo = np.asarray([float(r["ci_low"]) for r in rows])
    hi = np.asarray([float(r["ci_high"]) for r in rows])
    yy = np.arange(len(rows))
    ax.axvline(0,color=GREY,lw=1.3,ls="--")
    for i,(a,b,v) in enumerate(zip(lo,hi,x)):
        sig=a>0 or b<0
        color=BLUE if sig else GREY
        ax.plot([a,b],[i,i],color=color,lw=2)
        ax.plot([a,a],[i-.12,i+.12],color=color,lw=1.5)
        ax.plot([b,b],[i-.12,i+.12],color=color,lw=1.5)
        ax.scatter([v],[i],color=color if sig else "white",edgecolor=color,s=35,zorder=3)
        ax.text(max(b,v)+.25,i,f"{v:+.2f} [{a:+.2f}, {b:+.2f}]",
                va="center",fontsize=7.8,color=color)
    ax.set_yticks(yy,[LABELS.get(n,n) for n in names],fontsize=8.5)
    ax.set_ylim(len(rows)-.4,-.6)
    ax.set_xlim(min(-4.,float(lo.min())-1),max(30.,float(hi.max())+8))
    ax.grid(axis="x",alpha=.2)
    ax.set_xlabel("Δ Rank Score: spherical State-Bench-CoE minus comparator")
    ax.set_title(title,loc="left",fontsize=12,weight="bold")


def plot_bootstrap():
    rows = csv_rows(OUT / "PAIRED_BOOTSTRAP.csv")
    ordered = [r for r in rows if r["view"]=="3OOD"]
    fig=make(2244,868); ax=fig.add_axes([.25,.13,.72,.80]);
    forest(ax,ordered,"3-OOD paired Rank Score differences (2,000 resamples)")
    save(fig,"Fig12_spherical_paired_ood",2244,868)
    fig=make(1980,1380)
    ax1=fig.add_axes([.24,.55,.72,.40]); ax2=fig.add_axes([.24,.07,.72,.40])
    forest(ax1,[r for r in rows if r["view"]=="11ID"],"(a) 11-ID")
    forest(ax2,ordered,"(b) 3-OOD")
    save(fig,"FigA1_spherical_paired_all",1980,1380)


def plot_variants():
    w,h=2100,690
    summaries={(r["view"],r["variant"]):r for r in csv_rows(OUT / "SUMMARY.csv")}
    fig,axes=plt.subplots(1,2,figsize=(w/200,h/200),dpi=200,gridspec_kw=dict(wspace=.24))
    fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.85,wspace=.22)
    fixed={
        "11ID":((69.29,73.83),(69.73,73.83)),
        "3OOD":((68.54,63.32),(51.00,70.58)),
    }
    for ax,view,title in zip(axes,("11ID","3OOD"),("(a) 11-ID","(b) 3-OOD")):
        state=(float(summaries[view,"K2/no_cost"]["R_mean"]),
               float(summaries[view,"K2/main"]["R_mean"]))
        data=list(fixed[view])+[state]
        for i,((before,after),label,color) in enumerate(zip(data,("Query","Subject","State"),
                                                       (AMBER,TEAL,BLUE))):
            ax.axhspan(i-.39,i+.39,color=color,alpha=.07 if i==2 else 0)
            ax.plot([before,after],[i,i],color=color,lw=2)
            ax.scatter([before],[i],facecolors="white",edgecolors=color,s=85,lw=2,zorder=4)
            ax.scatter([after],[i],color=color,s=90,zorder=4)
            ax.text(max(before,after)+.45,i-.1,f"{after-before:+.2f}",color=color,
                    fontsize=10,weight="bold")
        ax.set_yticks(range(3),("Query","Subject","State"),fontsize=10)
        ax.set_xlim(44,83);ax.set_ylim(2.6,-.6)
        ax.set_xlabel("Rank Score")
        ax.set_title(title,loc="left",weight="bold",fontsize=13)
        ax.grid(axis="x",alpha=.2)
    fig.text(.5,.035,"Open marker: cost-free       Filled marker: cost-aware       Label: change in Rank Score",
             ha="center",fontsize=9)
    save(fig,"Fig13_spherical_cost_dumbbell",w,h)


def main():
    if read_json(OUT / "AUDIT.json")["status"] != "complete":
        raise RuntimeError("spherical completion experiment is incomplete")
    if read_json(OUT / "BOOTSTRAP_AUDIT.json")["status"] != "complete":
        raise RuntimeError("paired bootstrap is incomplete")
    views=view_uids(); matrices,_,_=eval_matrices(views)
    bounds=read_json(ROOT / "data/matrices/cost_bounds.json")
    plot_front(matrices,views,bounds)
    plot_k_sweep()
    plot_cases(matrices,views,bounds)
    plot_bootstrap()
    plot_variants()
    print(json.dumps(dict(status="complete",figures=sorted(p.name for p in FIG.glob('*.png')))),flush=True)


if __name__ == "__main__":
    main()
