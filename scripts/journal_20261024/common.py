"""Shared, read-only input and metric helpers for the 2026-10-24 journal run."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from routers.bench_coe_query.registry import load_registry_list
from routers.utils.rank_score import rank_score
from src.next_round_x.data import role_uids

OUTPUT = Path(os.environ.get("JOURNAL_OUTPUT_DIR", str(ROOT / "outputs/journal_20261024")))
SEEDS = tuple(int(value) for value in os.environ.get("JOURNAL_SEEDS", "0,1,2,3,4").split(","))
PREVIEW = os.environ.get("JOURNAL_SEED0_PREVIEW") == "1"
VIEWS = ("14ID", "11ID", "3OOD")
RESULT_COLUMNS = ("experiment", "method", "variant", "view", "seed", "n_test", "A", "C", "R", "notes")
SUMMARY_COLUMNS = ("experiment", "method", "variant", "view", "n_seeds",
                   "A_mean", "A_sd", "C_mean", "C_sd", "R_mean", "R_sd", "notes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def view_uids() -> dict[str, list[str]]:
    views = {"14ID": role_uids("full14", "test"),
             "11ID": role_uids("heldout11", "id"),
             "3OOD": role_uids("heldout11", "ood")}
    if tuple(len(views[view]) for view in VIEWS) != (6108, 5341, 767):
        raise RuntimeError("split sizes changed")
    if set(views["14ID"]) != set(views["11ID"]) | set(views["3OOD"]):
        raise RuntimeError("held-out split does not partition the 14ID test set")
    if set(views["11ID"]) & set(views["3OOD"]):
        raise RuntimeError("11ID and 3OOD overlap")
    return views


def eval_matrices(views: dict[str, list[str]]):
    registry = load_registry_list(ROOT / "data/registry/sample_index.pkl", name="sample")
    index = {uid: i for i, uid in enumerate(registry)}
    if len(index) != 30540:
        raise RuntimeError("benchmark registry size or uniqueness changed")
    with np.load(ROOT / "data/matrices/Y.npz", allow_pickle=False) as archive:
        all_y = archive["Y"]
    all_c = np.load(ROOT / "data/matrices/C.npy", mmap_mode="r", allow_pickle=False)
    if all_y.shape != all_c.shape or all_y.shape != (30540, 17):
        raise RuntimeError("outcome matrices changed shape")
    result = {}
    for view, uids in views.items():
        if not all(uid in index for uid in uids):
            raise RuntimeError(f"{view} includes UIDs absent from registry")
        positions = np.fromiter((index[uid] for uid in uids), dtype=np.int64, count=len(uids))
        result[view] = (np.asarray(all_y[positions]), np.asarray(all_c[positions], dtype=np.float64))
    return result, all_y, all_c


def scores(y: np.ndarray, c: np.ndarray, actions: np.ndarray, bounds: dict) -> tuple[float, float, float]:
    actions = np.asarray(actions, dtype=np.int64)
    if y.shape != c.shape or actions.shape != (len(y),) or np.any(actions < 0) or np.any(actions >= y.shape[1]):
        raise RuntimeError("invalid actions or outcome matrix alignment")
    selected = np.arange(len(y))
    accuracy = float(np.asarray(y[selected, actions], dtype=np.float64).mean())
    mean_cost = float(np.asarray(c[selected, actions], dtype=np.float64).mean())
    return 100 * accuracy, 10000 * mean_cost, 100 * rank_score(
        accuracy, mean_cost, bounds["cmin"], bounds["cmax"], beta=0.1)


def summarize(results: list[dict]) -> list[dict]:
    groups = {}
    for row in results:
        key = row["experiment"], row["method"], row["variant"], row["view"]
        groups.setdefault(key, []).append(row)
    summaries = []
    for (experiment, method, variant, view), rows in sorted(groups.items()):
        if sorted(int(row["seed"]) for row in rows) != list(SEEDS):
            continue
        item = dict(experiment=experiment, method=method, variant=variant,
                    view=view, n_seeds=len(rows))
        for metric in ("A", "C", "R"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        item["notes"] = rows[0]["notes"]
        summaries.append(item)
    return summaries


def make_audit(experiment: str, command: str, selection_rule: str, sizes: dict[str, int]) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return dict(experiment=experiment, git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                git_commit_note=None if commit.returncode == 0 else "Repository has no .git metadata",
                start_time=utc_now(), end_time=None, command=command,
                python_env=dict(python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform()),
                gpu="none", split_sizes=sizes, c_min=None, c_max=None,
                selection_rule=selection_rule, seeds=list(SEEDS), failures=[])
