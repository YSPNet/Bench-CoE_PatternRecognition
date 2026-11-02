"""One restartable E6 method/protocol/seed task using native VL-RouterBench routers."""
from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUT, ROOT, read_json, scores, write_json
from routers.table_a2.adapters import create_router, fit_classifier, load_router, predictions
from routers.table_a2.config import load_config
from routers.table_a2.data import Dataset
from routers.table_a2.inputs import benchmark_samples, forbid_synthetic_fallback_images

OUT = OUTPUT / "E6"
ARCHIVE = Path("/home/sm5/ys/ICASSP/archive/non_paper_research_20261011")
ARCHIVE_RUNS = ARCHIVE / "outputs/table_a2_factory_v2_20260914/runs"
LAMBDAS = ("0", "10", "100", "1000", "10000", "inf")
THROUGHPUT_METHODS = frozenset(("routerdc", "vlc", "mlp"))


def log(**fields):
    print(json.dumps(fields, sort_keys=True, default=str), flush=True)


def split_rows(data: Dataset, protocol: str):
    if protocol == "heldout11":
        return {"train": data.roles["train_seen"], "dev": data.roles["dev_seen"],
                "11ID": data.roles["id_test"], "3OOD": data.roles["ood_test"]}
    index = {sid: i for i, sid in enumerate(data.meta.sample_id)}
    result = {}
    for role, view in (("train", "train"), ("dev", "dev"), ("test", "14ID")):
        ids = [json.loads(line)["sample_id"] for line in (ROOT / "SPLITS" / f"{role}.jsonl").read_text().splitlines()]
        result[view] = np.asarray([index[sid] for sid in ids], dtype=np.int64)
    if tuple(map(len, (result["train"], result["dev"], result["14ID"]))) != (21378, 3054, 6108):
        raise RuntimeError("full14 official split changed")
    return result


def end_to_end_meta(data: Dataset) -> pd.DataFrame:
    samples = benchmark_samples(data)
    images = read_json(OUT / "IMAGE_PATHS.json")
    if set(images) != set(data.meta.sample_id):
        raise RuntimeError("E6 strict image cache incomplete")
    frame = data.meta.copy()
    frame["text"] = [samples[sid]["prompt"] for sid in frame.sample_id]
    frame["assets"] = [[{"type": "image", "path": images[sid]["path"],
                          "uri": images[sid]["path"]}] for sid in frame.sample_id]
    return frame


def feature_arrays(data: Dataset):
    paths = (ROOT / "EMBEDDINGS/text/bge-m3.parquet", OUT / "siglip-large-patch16-384.parquet")
    ids = data.meta.sample_id.tolist()
    arrays = []
    for path, dim in zip(paths, (1024, 1024)):
        frame = pd.read_parquet(path)
        if len(frame) != len(ids) or not frame.sample_id.is_unique or set(frame.sample_id) != set(ids):
            raise RuntimeError(f"E6 feature coverage mismatch: {path}")
        lookup = dict(zip(frame.sample_id, frame.embedding))
        x = np.asarray([lookup[sid] for sid in ids], dtype=np.float32)
        if x.shape != (len(ids), dim) or not np.isfinite(x).all():
            raise RuntimeError(f"E6 feature dimensions or values invalid: {path}")
        arrays.append(x)
    return tuple(arrays)


def archived_dev(method, seed, lam, ids):
    base = ARCHIVE_RUNS / method / f"seed_{seed}" / f"lambda_{lam}"
    with np.load(base / "dev_predictions.npz", allow_pickle=False) as saved:
        if saved["sample_ids"].tolist() != ids:
            raise RuntimeError("archived source-dev predictions differ from E6 split")
        choice = saved["columns"].astype(np.int64)
    return choice, base / "model.pkl"


def native_train(method, cfg, lam, seed, data, rows, matrices, meta, features, output):
    y, c = matrices["train"]
    yd, cd = matrices["dev"]
    tr, dv = rows["train"], rows["dev"]
    router = create_router(method, cfg, lam, seed)
    if method == "mlp":
        text, vision = features
        fit_classifier(router, method, cfg, y, c, text[tr], vision[tr], yd, cd,
                       text[dv], vision[dv], data.models, data.cost_bounds, output)
    else:
        kwargs = dict(model_mapping=dict(enumerate(data.models)), costs=c.mean(axis=0),
                      cmin=data.cost_bounds[0], cmax=data.cost_bounds[1],
                      rank_score_beta=cfg["rank_score_beta"],
                      Y_dev=yd, C_dev=cd, meta_dev=meta.iloc[dv].copy(),
                      monitor_output_dir=output / "training_monitor")
        with forbid_synthetic_fallback_images():
            router.fit(y, c, meta.iloc[tr].copy(), **kwargs)
    return router


def predict_choice(router, method, meta, rows, features, cfg, strict):
    text, vision = features if features is not None else (None, None)
    if strict:
        with forbid_synthetic_fallback_images():
            choice = predictions(router, method, meta.iloc[rows].copy(),
                                 None if text is None else text[rows],
                                 None if vision is None else vision[rows],
                                 cfg["training"]["batch_size"])
    else:
        choice = predictions(router, method, meta.iloc[rows].copy(),
                             None if text is None else text[rows],
                             None if vision is None else vision[rows],
                             cfg["training"]["batch_size"])
    choice = np.asarray(choice, dtype=np.int64)
    if choice.shape != (len(rows),) or np.any(choice < 0) or np.any(choice >= len(router.model_mapping)):
        raise RuntimeError("invalid native router output")
    return choice


def save_prediction(path, ids, choice):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sample_ids=np.asarray(ids), actions=np.asarray(choice, dtype=np.int8))


def run(protocol: str, method: str, seed: int, reuse_archive: bool):
    import torch

    cfg = load_config(ROOT / "configs/table_a2.yaml")
    cfg["features"]["vision_encoder"] = "google/siglip-large-patch16-384"
    cfg["runtime"].update(device="cuda", offline=True, verbose=0)
    task = OUT / "tasks" / protocol / method / f"seed-{seed}"
    task.mkdir(parents=True, exist_ok=True)
    complete = task / "COMPLETE.json"
    if complete.is_file():
        log(stage="task_resumed", protocol=protocol, method=method, seed=seed)
        return
    data = Dataset.load(ROOT, cfg)
    rows = split_rows(data, protocol)
    matrices = {role: data.matrices(index) for role, index in rows.items()}
    bounds = {"cmin": data.cost_bounds[0], "cmax": data.cost_bounds[1]}
    meta = data.meta if method == "mlp" else end_to_end_meta(data)
    features = feature_arrays(data) if method == "mlp" else None
    if reuse_archive and not (protocol == "heldout11" and method in ("routerdc", "vlc")):
        raise RuntimeError("archive reuse is allowed only for exact historical 11-source end-to-end fits")
    best_dev, best_checkpoint = -float("inf"), None
    all_results = []
    for lam in LAMBDAS:
        candidate = task / f"lambda-{lam}"
        candidate.mkdir(parents=True, exist_ok=True)
        checkpoint = candidate / "CANDIDATE.json"
        if checkpoint.is_file():
            saved = read_json(checkpoint)
            all_results.append(saved)
            if protocol == "full14" and seed == 0 and method in THROUGHPUT_METHODS and saved["val"][2] > best_dev:
                best_dev = saved["val"][2]
                model_path = task / "selected_val_model.pkl"
                if model_path.is_file():
                    best_checkpoint = str(model_path.relative_to(ROOT))
            log(stage="candidate_resumed", protocol=protocol, method=method, seed=seed, lam=lam)
            continue
        router = None
        try:
            if reuse_archive:
                dev_choice, model_file = archived_dev(method, seed, lam,
                                                      data.meta.sample_id.iloc[rows["dev"]].tolist())
                router = load_router(method, model_file, cfg)
            else:
                router = native_train(method, cfg, lam, seed, data, rows, matrices, meta, features, candidate)
                dev_choice = predict_choice(router, method, meta, rows["dev"], features, cfg, method != "mlp")
            val = scores(*matrices["dev"], dev_choice, bounds)
            predictions_by_view = {}
            metrics_by_view = {}
            for view in ("14ID",) if protocol == "full14" else ("11ID", "3OOD"):
                choice = predict_choice(router, method, meta, rows[view], features, cfg, method != "mlp")
                ids = data.meta.sample_id.iloc[rows[view]].tolist()
                path = candidate / f"{view}.npz"
                save_prediction(path, ids, choice)
                predictions_by_view[view] = str(path.relative_to(ROOT))
                metrics_by_view[view] = scores(*matrices[view], choice, bounds)
            item = {"protocol": protocol, "method": method, "seed": seed, "lambda": lam,
                    "source": "historical_native_checkpoint" if reuse_archive else "new_native_fit",
                    "val": val, "test": metrics_by_view, "predictions": predictions_by_view}
            if protocol == "full14" and seed == 0 and method in THROUGHPUT_METHODS and val[2] > best_dev and not reuse_archive:
                model_path = task / "selected_val_model.pkl"
                if model_path.exists():
                    model_path.unlink()
                weights = task / "selected_val_model_model.pt"
                if weights.exists():
                    weights.unlink()
                router.save(str(model_path))
                best_checkpoint = str(model_path.relative_to(ROOT))
                best_dev = val[2]
                write_json(task / "SELECTED_MODEL.json", {"lambda": lam, "val_R": best_dev,
                            "model": best_checkpoint})
            write_json(checkpoint, item)
            # Native TrainingMonitor writes a full ~1.2 GB model per lambda.
            # The test actions are already frozen above; keep only the seed-0
            # source-selected checkpoint needed for E9 throughput.
            if method != "mlp":
                for obsolete in (candidate / "training_monitor/best_model.pkl",
                                 candidate / "training_monitor/best_model_model.pt"):
                    if obsolete.is_file():
                        obsolete.unlink()
            all_results.append(item)
            log(stage="candidate_complete", protocol=protocol, method=method, seed=seed,
                lam=lam, val_R=round(val[2], 4), test_R={v: round(m[2], 4) for v, m in metrics_by_view.items()})
        except Exception:
            (candidate / "traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            del router
            gc.collect()
            torch.cuda.empty_cache()
    if len(all_results) != len(LAMBDAS):
        raise RuntimeError("E6 candidate sweep incomplete")
    chosen = max(all_results, key=lambda item: item["val"][2])
    if protocol == "full14" and seed == 0 and method in THROUGHPUT_METHODS:
        selected = read_json(task / "SELECTED_MODEL.json")
        if selected["lambda"] != chosen["lambda"] or not (task / "selected_val_model.pkl").is_file():
            raise RuntimeError("seed-0 source-selected model checkpoint is incomplete")
        best_checkpoint = selected["model"]
    best_test = {view: max(all_results, key=lambda item: item["test"][view][2])["lambda"]
                 for view in (("14ID",) if protocol == "full14" else ("11ID", "3OOD"))}
    write_json(complete, {"protocol": protocol, "method": method, "seed": seed,
                          "lambda_val": chosen["lambda"], "lambda_best_test": best_test,
                          "selected_val_model": best_checkpoint,
                          "candidates": [str((task / f"lambda-{lam}" / "CANDIDATE.json").relative_to(ROOT))
                                         for lam in LAMBDAS]})
    log(stage="task_complete", protocol=protocol, method=method, seed=seed,
        lambda_val=chosen["lambda"], lambda_best_test=best_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("full14", "heldout11"), required=True)
    parser.add_argument("--method", choices=("routerdc", "vlc", "mlp", "cosinecls", "zooter"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--reuse-archive", action="store_true")
    args = parser.parse_args()
    run(args.protocol, args.method, args.seed, args.reuse_archive)
