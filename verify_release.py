#!/usr/bin/env python3
"""Check release hashes and rescore the main frozen spherical actions."""
from __future__ import annotations
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from routers.utils.rank_score import rank_score

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    with (ROOT / "MANIFEST.csv").open(newline="") as stream:
        entries = list(csv.DictReader(stream))
    for row in entries:
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha(path) != row["sha256"]:
            raise RuntimeError(f"Archive mismatch: {row['path']}")
    with (ROOT / "data/registry/sample_index.pkl").open("rb") as stream:
        uids = pickle.load(stream)
    if len(uids) != 30540 or len(set(uids)) != 30540:
        raise RuntimeError("Sample registry differs from paper protocol")
    positions = {uid: i for i, uid in enumerate(uids)}
    with np.load(ROOT / "data/matrices/Y.npz", allow_pickle=False) as archive:
        y = archive["Y"]
    c = np.load(ROOT / "data/matrices/C.npy", allow_pickle=False)
    bounds = json.loads((ROOT / "data/matrices/cost_bounds.json").read_text())
    out = ROOT / "outputs/journal_20261024/V25_SPHERICAL_COMPLETION"
    rows = []
    for seed in range(5):
        with np.load(out / f"actions/K2_seed-{seed}.npz", allow_pickle=False) as archive:
            test_uids = archive["uids"].tolist()
            actions = archive["main"]
        if len(test_uids) != 6108:
            raise RuntimeError("Unexpected held-out test size")
        for view, section in (("11ID", slice(0, 5341)), ("3OOD", slice(5341, None))):
            ids = np.asarray([positions[uid] for uid in test_uids[section]])
            selected = actions[section]
            yy = y[ids, selected].astype(float).mean()
            cc = c[ids, selected].astype(float).mean()
            rr = rank_score(yy, cc, bounds["cmin"], bounds["cmax"], beta=0.1)
            rows.append((seed, view, 100*yy, 10000*cc, 100*rr))
    with (out / "SUMMARY.csv").open(newline="") as stream:
        published = {row["view"]: row for row in csv.DictReader(stream)
                     if row["variant"] == "K2/main"}
    for view in ("11ID", "3OOD"):
        values = np.asarray([(a, c, r) for _, v, a, c, r in rows if v == view])
        for index, metric in enumerate(("A", "C", "R")):
            reported = float(published[view][f"{metric}_mean"])
            if abs(float(values[:, index].mean()) - reported) > 1e-9:
                raise RuntimeError(f"Rescore differs: {view} {metric}")
    print(json.dumps({"status": "ok", "files": len(entries),
                      "11ID_R": float(np.mean([r for _,v,_,_,r in rows if v == "11ID"])),
                      "3OOD_R": float(np.mean([r for _,v,_,_,r in rows if v == "3OOD"]))}, indent=2))

if __name__ == "__main__":
    main()
