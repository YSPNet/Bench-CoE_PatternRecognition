#!/usr/bin/env python3
"""MMMU-prototype Subject Bench-CoE on the frozen VL-RouterBench views.

The Qwen3-VL cosine prototype labels are input-only targets. A separate
30-class classifier is trained for each backbone/protocol. Expert maps are
fitted only from the protocol's official training outcomes. Prediction files
are written before the separate ``evaluate`` command opens test outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from routers.bench_coe_query.config import load_contract, sha256_file
from routers.bench_coe_query.heads import AuditedBenchCoEHead
from routers.bench_coe_query.registry import load_registry_list
from routers.bench_coe_query.reproducibility import seed_everything
from routers.bench_coe_query.bert_router import _optimizer, _linear_warmup_decay_scheduler
from routers.utils.rank_score import rank_score
from scripts import run_stateroute_mmmu_subject_20260928 as subject

OUT = ROOT / "outputs/bench_coe_subject_mmmu30"
PROTO = ROOT / "outputs/stateroute_mmmu_subject_20260928/PROTOTYPES.npz"
HELD_OUT = {"ChartQA_TEST", "MMMU_DEV_VAL", "MathVision_MINI"}
BACKBONES = ("bert", "tinyllava", "qwen3vl")
PROTOCOLS = ("id14", "ood11_3")
SEEDS = tuple(range(5))


def sha(path: Path) -> str:
    return sha256_file(path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, obj: dict) -> None:
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def rows_and_metadata() -> tuple[tuple[str, ...], pd.DataFrame, dict[str, np.ndarray]]:
    """Read only frozen input registries and splits; no expert matrix."""
    contract = load_contract()
    for section in ("splits", "registry"):
        for name, item in contract.values["data"][section].items():
            path = ROOT / str(item["path"])
            if sha(path) != str(item["sha256"]):
                raise RuntimeError(f"frozen {section}.{name} changed: {path}")
    ids = load_registry_list(ROOT / "data/registry/sample_index.pkl", name="sample")
    meta = pd.read_parquet(ROOT / "data/registry/meta.parquet")
    if tuple(meta.sample_id.astype(str)) != ids or len(ids) != 30540:
        raise RuntimeError("sample registry/meta mismatch")
    index = {uid: i for i, uid in enumerate(ids)}
    roles: dict[str, np.ndarray] = {}
    union: set[str] = set()
    for split in ("train", "dev", "test"):
        path = ROOT / f"SPLITS/{split}.jsonl"
        names = [str(json.loads(line)["sample_id"]) for line in path.read_text().splitlines()]
        if len(names) != int(contract.values["data"]["splits"][split]["count"]):
            raise RuntimeError(f"wrong {split} count")
        if len(names) != len(set(names)) or union.intersection(names):
            raise RuntimeError(f"duplicated/overlapping {split} UIDs")
        union.update(names)
        roles[split] = np.asarray([index[uid] for uid in names], dtype=np.int64)
        if any(str(meta.iloc[i]["split"]) != split for i in roles[split]):
            raise RuntimeError(f"metadata disagrees with {split}")
    if union != set(ids):
        raise RuntimeError("official splits do not cover registry")
    return ids, meta, roles


def protocol_rows(meta: pd.DataFrame, roles: dict[str, np.ndarray], protocol: str) -> dict[str, np.ndarray]:
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    datasets = meta.dataset.astype(str).to_numpy()
    if protocol == "id14":
        return {"train": roles["train"], "dev": roles["dev"], "14-ID": roles["test"]}
    seen = ~np.isin(datasets, tuple(HELD_OUT))
    result = {
        "train": roles["train"][seen[roles["train"]]],
        "dev": roles["dev"][seen[roles["dev"]]],
        "11-ID": roles["test"][seen[roles["test"]]],
        "3-OOD": roles["test"][~seen[roles["test"]]],
    }
    if tuple(map(len, result.values())) != (18665, 2680, 5341, 767):
        raise RuntimeError("held-out view counts changed")
    return result


def prepare() -> None:
    ids, _, _ = rows_and_metadata()
    if sha(PROTO) != "aa8de7786ea4d95e3770d3a3765bbccb52560c1d4f3598979e2409a7ad94db64":
        raise RuntimeError("frozen MMMU prototype changed")
    with np.load(PROTO, allow_pickle=False) as proto:
        centers = np.asarray(proto["centers"], dtype=np.float32)
        names = proto["subjects"].astype(str).tolist()
        if centers.shape != (30, 2048) or len(set(names)) != 30 or len(proto["uids"]) != 735:
            raise RuntimeError("invalid prototype contents")
    features, overlay, _ = subject.qwen_features(list(ids), allow_multi=True)
    if len(overlay) != 47:
        raise RuntimeError(f"expected 47 native multi-image features, got {len(overlay)}")
    states, similarity, margin = subject.assign_subjects(features, centers)
    if len(states) != len(ids) or not np.isfinite(similarity).all():
        raise RuntimeError("incomplete subject assignments")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "assignments.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if not np.array_equal(old["states"], states) or not np.array_equal(old["uids"], np.asarray(ids)):
                raise RuntimeError("existing subject assignments disagree")
    else:
        np.savez_compressed(path, uids=np.asarray(ids), states=states.astype(np.int8),
                            similarity=similarity, margin=margin, subjects=np.asarray(names))
    write_json(OUT / "ASSIGNMENT_MANIFEST.json", {
        "method": "Qwen3-VL L2-normalized cosine nearest MMMU native subject centroid",
        "prototype_sha256": sha(PROTO), "assignments_sha256": sha(path),
        "mmmu_train_prototype_count": 735, "subject_count": 30,
        "assigned_count": len(ids), "multi_image_overlay_count": len(overlay),
        "target_information": "inputs and MMMU train native labels only; no expert outcome",
        "ood_caveat": "MMMU train inputs/labels enter centroids; 3-OOD is MMMU-assisted",
    })
    print(json.dumps({"stage": "prepare", "rows": len(ids), "subject_counts": np.bincount(states, minlength=30).tolist()}))


def assignments(ids: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    manifest = json.loads((OUT / "ASSIGNMENT_MANIFEST.json").read_text())
    path = OUT / "assignments.npz"
    if sha(path) != manifest["assignments_sha256"] or sha(PROTO) != manifest["prototype_sha256"]:
        raise RuntimeError("subject assignment/prototype hash changed")
    with np.load(path, allow_pickle=False) as arr:
        if tuple(arr["uids"].astype(str)) != ids:
            raise RuntimeError("subject assignment UID order changed")
        states = arr["states"].astype(np.int64)
        names = arr["subjects"].astype(str).tolist()
    return states, names


def fit_map(protocol: str) -> None:
    ids, meta, roles = rows_and_metadata()
    states, names = assignments(ids)
    rows = protocol_rows(meta, roles, protocol)["train"]
    contract = load_contract()
    # Use the previously scoped source-only outcome bundle. In particular the
    # 11-source map process never opens the held-out datasets' Y/C matrix rows.
    from src.next_round_x.data import load_source
    source = load_source("full14" if protocol == "id14" else "heldout11")
    requested_uids = np.asarray(ids)[rows]
    if len(source["uids"]) != len(rows) or not np.array_equal(source["uids"], requested_uids):
        raise RuntimeError("scoped source outcomes do not match official train order")
    y = np.asarray(source["y"])
    if y.shape != (len(rows), 17) or not np.isin(y, (0, 1)).all():
        raise RuntimeError("invalid scoped source expert outcomes")
    source_manifest = ROOT / "outputs/next_round_x/source" / ("full14" if protocol == "id14" else "heldout11") / "MANIFEST.json"
    global_acc = y.mean(axis=0)
    global_best = int(global_acc.argmax())
    counts = np.bincount(states[rows], minlength=30)
    sums = np.zeros((30, 17), dtype=np.int64)
    np.add.at(sums, states[rows], y)
    table = np.empty(30, dtype=np.int64)
    for s in range(30):
        table[s] = int(sums[s].argmax()) if counts[s] else global_best
    dest = OUT / protocol
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "expert_map.npz", table=table, counts=counts,
                        correct_sums=sums, global_best=np.asarray(global_best),
                        train_rows=rows, subjects=np.asarray(names))
    write_json(dest / "EXPERT_MAP.json", {
        "protocol": protocol, "source_train_count": len(rows),
        "mapping": "argmax train subject expert accuracy; ties use canonical expert index",
        "empty_subject_fallback": "train global accuracy best expert",
        "source_datasets": sorted(set(meta.dataset.iloc[rows].astype(str))),
        "expert_order": list(contract.model_order),
        "support": dict(zip(names, counts.astype(int).tolist())),
        "selected_experts": {name: contract.model_order[int(table[i])] for i, name in enumerate(names)},
        "map_sha256": sha(dest / "expert_map.npz"),
        "assignments_sha256": sha(OUT / "assignments.npz"),
        "source_outcomes_manifest_sha256": sha(source_manifest),
    })
    print(json.dumps({"stage": "map", "protocol": protocol, "train": len(rows), "empty_states": int((counts == 0).sum()), "experts_used": len(set(table))}))


def feature_matrix(backbone: str, ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, str]:
    pointer_path = ROOT / f"outputs/bench_coe_query/features/{backbone}/LATEST_FULL.json"
    pointer = json.loads(pointer_path.read_text())
    base = ROOT / pointer["cache_root"] / "merged"
    manifest = json.loads((base / "manifest.json").read_text())
    if sha(base / "features.npy") != manifest["features_sha256"] or sha(base / "index.jsonl") != manifest["index_sha256"]:
        raise RuntimeError(f"{backbone} feature cache hash mismatch")
    index = [json.loads(line) for line in (base / "index.jsonl").read_text().splitlines()]
    if len(index) != 30493 or len({x["row_index"] for x in index}) != len(index):
        raise RuntimeError(f"{backbone} feature cache coverage changed")
    values = np.load(base / "features.npy", mmap_mode="r", allow_pickle=False)
    result = np.zeros((len(ids), values.shape[1]), dtype=np.float32)
    available = np.zeros(len(ids), dtype=bool)
    for position, record in enumerate(index):
        row = int(record["row_index"])
        if ids[row] != str(record["sample_id"]):
            raise RuntimeError(f"{backbone} feature UID mismatch")
        result[row] = values[position]
        available[row] = True
    if available.sum() != 30493 or not np.isfinite(result).all():
        raise RuntimeError(f"{backbone} feature matrix invalid")
    return result, available, manifest["features_sha256"]


def bert_texts(ids: tuple[str, ...]) -> list[str]:
    paths = sorted((ROOT / "BENCHMARKS").glob("*/*_samples.jsonl"))
    if len(paths) != 14:
        raise RuntimeError("expected 14 benchmark input files")
    texts: dict[str, str] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            uid = str(obj["sample_id"])
            if uid in texts:
                raise RuntimeError(f"duplicate benchmark UID {uid}")
            texts[uid] = str(obj["prompt"])
    if set(texts) != set(ids):
        raise RuntimeError("benchmark inputs differ from registry")
    return [texts[uid] for uid in ids]


def _best_path(protocol: str, backbone: str, seed: int) -> Path:
    return OUT / protocol / backbone / f"seed-{seed}" / "best.pt"


def train(protocol: str, backbone: str, seed: int) -> None:
    if backbone not in BACKBONES or seed not in SEEDS:
        raise ValueError("invalid backbone/seed")
    ids, meta, roles = rows_and_metadata()
    labels, _ = assignments(ids)
    views = protocol_rows(meta, roles, protocol)
    train_rows, dev_rows = views["train"], views["dev"]
    contract = load_contract()
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; full Subject Bench-CoE training requires a GPU")
    device = torch.device("cuda")
    print(json.dumps({"stage": "train_start", "protocol": protocol,
                      "backbone": backbone, "seed": seed, "device": str(device),
                      "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    run_dir = _best_path(protocol, backbone, seed).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "TRAINED.json").exists():
        raise RuntimeError(f"already trained: {run_dir}")
    source_hash = sha(Path(__file__))
    if backbone == "bert":
        cfg = contract.values["profiles"]["vlrb_adapted_v1"]["bert"]
        texts = bert_texts(ids)
        path = Path(str(contract.values["models"]["bert"]["path"]))
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), num_labels=30, ignore_mismatched_sizes=True, local_files_only=True
        ).to(device)
        def batches(rows: np.ndarray, shuffle: bool):
            encoded = tokenizer([texts[i] for i in rows], truncation=True, padding="max_length",
                                max_length=int(contract.values["models"]["bert"]["max_length"]), return_tensors="pt")
            ds = TensorDataset(encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels[rows], dtype=torch.long))
            return DataLoader(ds, batch_size=int(cfg["batch_size_per_gpu"]), shuffle=shuffle,
                              generator=torch.Generator().manual_seed(seed) if shuffle else None)
        train_loader, dev_loader = batches(train_rows, True), batches(dev_rows, False)
        optimizer = _optimizer(model, float(cfg["learning_rate"]), float(cfg["weight_decay"]))
        epochs = int(cfg["epochs"])
        scheduler = _linear_warmup_decay_scheduler(optimizer, warmup_steps=int(epochs * len(train_loader) * float(cfg["warmup_ratio"])), total_steps=epochs * len(train_loader))
        amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        feature_hash = None
    else:
        cfg = contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"]
        x, available, feature_hash = feature_matrix(backbone, ids)
        model = AuditedBenchCoEHead(x.shape[1], 30, float(cfg["dropout"])).to(device)
        def batches(rows: np.ndarray, shuffle: bool):
            selected = rows[available[rows]]
            ds = TensorDataset(torch.from_numpy(x[selected]), torch.tensor(labels[selected], dtype=torch.long))
            return DataLoader(ds, batch_size=int(cfg["batch_size_per_gpu"]), shuffle=shuffle,
                              generator=torch.Generator().manual_seed(seed) if shuffle else None)
        train_loader, dev_loader = batches(train_rows, True), batches(dev_rows, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        epochs = int(cfg["epochs"])
        scheduler = None
        amp = False
        scaler = None
    best = float("inf")
    bad = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            *inputs, target = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(input_ids=inputs[0], attention_mask=inputs[1]).logits if backbone == "bert" else model(inputs[0])
                loss = torch.nn.functional.cross_entropy(logits, target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach().cpu()) * len(target)
            n += len(target)
        model.eval()
        dev_total, dev_n, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                *inputs, target = [item.to(device) for item in batch]
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(input_ids=inputs[0], attention_mask=inputs[1]).logits if backbone == "bert" else model(inputs[0])
                    loss = torch.nn.functional.cross_entropy(logits, target)
                dev_total += float(loss.cpu()) * len(target)
                dev_n += len(target)
                correct += int((logits.argmax(-1) == target).sum().item())
        dev_loss = dev_total / dev_n
        history.append({"epoch": epoch, "train_subject_loss": total / n, "dev_subject_loss": dev_loss,
                        "dev_subject_accuracy": correct / dev_n})
        print(json.dumps({"protocol": protocol, "backbone": backbone, "seed": seed, **history[-1]}), flush=True)
        if dev_loss < best - 1e-9:
            best, bad = dev_loss, 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "dev_subject_loss": best}, _best_path(protocol, backbone, seed))
        else:
            bad += 1
        if epoch >= int(cfg["min_epochs"]) and bad >= int(cfg["patience"]):
            break
    with (run_dir / "history.jsonl").open("w") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(run_dir / "TRAINED.json", {"protocol": protocol, "backbone": backbone, "seed": seed,
        "device": str(device),
        "best_dev_subject_loss": best, "best_epoch": min(history, key=lambda r: r["dev_subject_loss"])["epoch"],
        "checkpoint_sha256": sha(_best_path(protocol, backbone, seed)), "source_sha256": source_hash,
        "assignment_sha256": sha(OUT / "assignments.npz"), "feature_sha256": feature_hash,
        "source_train_count": len(train_rows), "source_dev_count": len(dev_rows),
        "trainable_rows": n, "dev_scored_rows": dev_n,
        "selection": "minimum source dev subject cross-entropy only"})


def predict(protocol: str, backbone: str, seed: int) -> None:
    ids, meta, roles = rows_and_metadata()
    views = protocol_rows(meta, roles, protocol)
    run_dir = _best_path(protocol, backbone, seed).parent
    trained = json.loads((run_dir / "TRAINED.json").read_text())
    if sha(_best_path(protocol, backbone, seed)) != trained["checkpoint_sha256"] or sha(OUT / "assignments.npz") != trained["assignment_sha256"]:
        raise RuntimeError("training artifact changed")
    with np.load(OUT / protocol / "expert_map.npz", allow_pickle=False) as mapped:
        table = mapped["table"].astype(np.int64)
        global_best = int(mapped["global_best"])
    contract = load_contract()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(_best_path(protocol, backbone, seed), map_location="cpu", weights_only=False)
    if backbone == "bert":
        path = Path(str(contract.values["models"]["bert"]["path"]))
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(path), num_labels=30,
            ignore_mismatched_sizes=True, local_files_only=True)
        texts = bert_texts(ids)
        available = np.ones(len(ids), dtype=bool)
    else:
        x, available, feature_hash = feature_matrix(backbone, ids)
        if feature_hash != trained["feature_sha256"]:
            raise RuntimeError("feature cache changed since training")
        dropout = float(contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"]["dropout"])
        model = AuditedBenchCoEHead(x.shape[1], 30, dropout)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    for view_name, rows in views.items():
        if view_name in {"train", "dev"}:
            continue
        classes = np.full(len(rows), -1, dtype=np.int16)
        selected = rows[available[rows]]
        at = np.flatnonzero(available[rows])
        if backbone == "bert":
            features = tokenizer([texts[i] for i in selected], truncation=True, padding="max_length",
                                 max_length=int(contract.values["models"]["bert"]["max_length"]), return_tensors="pt")
            dataset = TensorDataset(features["input_ids"], features["attention_mask"])
        else:
            dataset = TensorDataset(torch.from_numpy(x[selected]))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in loader:
                tensors = [t.to(device) for t in batch]
                logits = model(input_ids=tensors[0], attention_mask=tensors[1]).logits if backbone == "bert" else model(tensors[0])
                predictions.append(logits.argmax(-1).cpu().numpy())
        classes[at] = np.concatenate(predictions).astype(np.int16)
        actions = np.where(classes >= 0, table[np.maximum(classes, 0)], global_best)
        path = run_dir / f"predictions_{view_name}.npz"
        np.savez_compressed(path, uids=np.asarray(ids)[rows], rows=rows, predicted_subject=classes,
                            selected_expert=actions.astype(np.int8), fallback=(classes < 0))
        write_json(run_dir / f"PREDICTION_{view_name}.json", {"protocol": protocol, "backbone": backbone,
            "seed": seed, "view": view_name, "n": len(rows), "fallback_n": int((classes < 0).sum()),
            "predictions_sha256": sha(path), "checkpoint_sha256": trained["checkpoint_sha256"],
            "map_sha256": sha(OUT / protocol / "expert_map.npz"),
            "test_outcomes_read": False})
        print(json.dumps({"stage": "predict", "protocol": protocol, "backbone": backbone,
                          "seed": seed, "view": view_name, "n": len(rows)}), flush=True)


def evaluate(protocol: str, backbone: str, seed: int) -> None:
    ids, meta, roles = rows_and_metadata()
    views = protocol_rows(meta, roles, protocol)
    contract = load_contract()
    for key in ("y", "c"):
        item = contract.values["data"]["matrices"][key]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen {key} changed")
    with np.load(ROOT / "data/matrices/Y.npz", allow_pickle=False) as archive:
        y = archive["Y"]
    c = np.load(ROOT / "data/matrices/C.npy", mmap_mode="r", allow_pickle=False)
    bounds = json.loads((ROOT / "data/matrices/cost_bounds.json").read_text())
    run_dir = _best_path(protocol, backbone, seed).parent
    for view_name, expected_rows in views.items():
        if view_name in {"train", "dev"}:
            continue
        receipt = json.loads((run_dir / f"PREDICTION_{view_name}.json").read_text())
        path = run_dir / f"predictions_{view_name}.npz"
        if sha(path) != receipt["predictions_sha256"]:
            raise RuntimeError("prediction hash changed")
        if sha(OUT / protocol / "expert_map.npz") != receipt["map_sha256"]:
            raise RuntimeError("expert map changed after prediction")
        if sha(_best_path(protocol, backbone, seed)) != receipt["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed after prediction")
        with np.load(path, allow_pickle=False) as prediction:
            rows = prediction["rows"].astype(np.int64)
            actions = prediction["selected_expert"].astype(np.int64)
            if not np.array_equal(rows, expected_rows) or not np.array_equal(prediction["uids"], np.asarray(ids)[rows]):
                raise RuntimeError("prediction view membership changed")
        correct = y[rows, actions]
        cost = c[rows, actions]
        accuracy = float(correct.mean())
        mean_cost = float(cost.astype(np.float64).mean())
        result = {"protocol": protocol, "backbone": backbone, "seed": seed, "view": view_name,
                  "n": len(rows), "correct": int(correct.sum()), "accuracy_percent": 100 * accuracy,
                  "cost_usd_per_10k": 10000 * mean_cost,
                  "rank_score": 100 * rank_score(accuracy, mean_cost, bounds["cmin"], bounds["cmax"]),
                  "predictions_sha256": receipt["predictions_sha256"]}
        write_json(run_dir / f"metrics_{view_name}.json", result)
        print(json.dumps(result), flush=True)


def aggregate() -> None:
    rows = []
    for protocol in PROTOCOLS:
        for backbone in BACKBONES:
            for seed in SEEDS:
                for view in (("14-ID",) if protocol == "id14" else ("11-ID", "3-OOD")):
                    path = _best_path(protocol, backbone, seed).parent / f"metrics_{view}.json"
                    if path.is_file():
                        rows.append(json.loads(path.read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUT / "RESULTS.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary = []
    for protocol in PROTOCOLS:
        for backbone in BACKBONES:
            for view in (("14-ID",) if protocol == "id14" else ("11-ID", "3-OOD")):
                subset = [r for r in rows if (r["protocol"], r["backbone"], r["view"]) == (protocol, backbone, view)]
                if not subset:
                    continue
                summary.append({"protocol": protocol, "backbone": backbone, "view": view,
                    "seeds": len(subset), "n": subset[0]["n"],
                    **{key + "_mean": float(np.mean([r[key] for r in subset])) for key in ("accuracy_percent", "cost_usd_per_10k", "rank_score")},
                    **{key + "_sd": float(np.std([r[key] for r in subset], ddof=1)) if len(subset) > 1 else None for key in ("accuracy_percent", "cost_usd_per_10k", "rank_score")}})
    if summary:
        with (OUT / "SUMMARY.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps({"stage": "aggregate", "runs": len(rows), "groups": len(summary)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "map", "train", "predict", "evaluate", "aggregate"))
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage in {"map", "train", "predict", "evaluate"} and not args.protocol:
        parser.error("--protocol required")
    if args.stage in {"train", "predict", "evaluate"} and (not args.backbone or args.seed is None):
        parser.error("--backbone and --seed required")
    {"prepare": prepare, "map": lambda: fit_map(args.protocol),
     "train": lambda: train(args.protocol, args.backbone, args.seed),
     "predict": lambda: predict(args.protocol, args.backbone, args.seed),
     "evaluate": lambda: evaluate(args.protocol, args.backbone, args.seed),
     "aggregate": aggregate}[args.stage]()


if __name__ == "__main__":
    main()
