#!/usr/bin/env python3
"""Source-selected soft-label Query heads on frozen TinyLLaVA/Qwen3-VL features.

The fold, lambda, source-dev and test barrier protocol is shared with E20.
Only the frozen-feature head implementation differs from E20's full BERT fit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
from torch.nn import functional as F

import e20_query_softlabels as e20
from common import OUTPUT, ROOT, eval_matrices, read_json, scores, view_uids, write_csv, write_json
from routers.bench_coe_query.config import load_contract, sha256_file
from routers.bench_coe_query.feature_cache import load_full_feature_cache
from routers.bench_coe_query.heads import AuditedBenchCoEHead
from routers.bench_coe_query.reproducibility import seed_everything
from scripts import run_bench_coe_subject_mmmu as subject
from src.next_round_x.data import load_source, role_uids

BACKBONES = {"tinyllava": "TinyLLaVA", "qwen3vl": "Qwen3-VL"}
FEATURE_BANK: torch.Tensor | None = None
BACKBONE = ""
DEST = Path()
SEEDS = (0,)
FALLBACK_EXPERT = 12
PER_DATASET_COLUMNS = ("method", "selection", "seed", "dataset", "n", "A", "C", "R")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_inputs(protocol: str):
    global FEATURE_BANK
    contract = load_contract()
    pointer_path = ROOT / f"outputs/bench_coe_query/features/{BACKBONE}/LATEST_FULL.json"
    pointer = read_json(pointer_path)
    if pointer["contract_sha256"] != contract.sha256:
        raise RuntimeError("frozen feature contract hash changed")
    cache = ROOT / pointer["cache_root"]
    features, records, manifest, integrity = load_full_feature_cache(
        cache, dataset_root=ROOT, expected_contract_sha256=contract.sha256,
        expected_input_content=pointer["input_content"])
    if not integrity["passed"] or len(records) != 30493:
        raise RuntimeError("full feature-cache validation failed")
    data_paths = {key: ROOT / entry["path"]
                  for key, entry in contract.values["data"]["matrices"].items()
                  if key in ("y", "c", "cost_bounds")}
    data_paths.update({f"split_{key}": ROOT / entry["path"]
                       for key, entry in contract.values["data"]["splits"].items()})
    for key, path in data_paths.items():
        section, name = ("splits", key[6:]) if key.startswith("split_") else ("matrices", key)
        expected = contract.values["data"][section][name]["sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen {key} hash changed")
    registry_ids, _, _ = subject.rows_and_metadata()
    if len(registry_ids) != 30540:
        raise RuntimeError("sample registry size changed")
    registry = {uid: index for index, uid in enumerate(registry_ids)}
    if len(registry) != len(registry_ids):
        raise RuntimeError("duplicate registry UIDs")
    lookup = np.full(len(registry_ids), -1, dtype=np.int64)
    for index, record in enumerate(records):
        row = int(record["row_index"])
        if registry_ids[row] != record["sample_id"] or lookup[row] >= 0:
            raise RuntimeError("feature cache UID/registry alignment changed")
        lookup[row] = index
    if int((lookup >= 0).sum()) != 30493:
        raise RuntimeError("feature cache coverage changed")
    source = load_source(protocol)
    train_uids = source["uids"].tolist()
    if train_uids != role_uids(protocol, "train"):
        raise RuntimeError("source train UID order changed")
    train_rows = np.asarray([registry[uid] for uid in train_uids], dtype=np.int64)
    with np.load(ROOT / "data/matrices/Y.npz", allow_pickle=False) as archive:
        all_y = archive["Y"]
    all_c = np.load(ROOT / "data/matrices/C.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(all_y[train_rows], source["y"]) or not np.allclose(
            all_c[train_rows], source["c"]):
        raise RuntimeError("source Y/C rows changed")
    for fold in range(3):
        tr = np.flatnonzero((source["role"] == "fit") & (source["fold"] != fold))
        if int(source["y"][tr].mean(axis=0).argmax()) != FALLBACK_EXPERT:
            raise RuntimeError("fold fallback expert differs from source-only policy")
    if int(source["y"].mean(axis=0).argmax()) != FALLBACK_EXPERT:
        raise RuntimeError("full-source fallback expert changed")
    dev_uids = role_uids(protocol, "dev")
    dev_rows = np.asarray([registry[uid] for uid in dev_uids], dtype=np.int64)
    source_datasets = {uid.split("/", 1)[0] for uid in train_uids}
    if any(uid.split("/", 1)[0] not in source_datasets for uid in dev_uids):
        raise RuntimeError("source dev contains held-out datasets")
    target_views = ("14ID",) if protocol == "full14" else ("11ID", "3OOD")
    views = view_uids()
    device = torch.device("cuda")
    FEATURE_BANK = torch.tensor(features, dtype=torch.float16, device=device)

    def index_tensors(uids):
        rows = np.asarray([registry[uid] for uid in uids], dtype=np.int64)
        indices = lookup[rows]
        available = indices >= 0
        indices = np.maximum(indices, 0)
        return (torch.as_tensor(indices, dtype=torch.long, device=device),
                torch.as_tensor(available, dtype=torch.bool, device=device))

    token = {"train": index_tensors(train_uids), "dev": index_tensors(dev_uids)}
    for view in target_views:
        token[view] = index_tensors(views[view])
    DEST.mkdir(parents=True, exist_ok=True)
    write_json(DEST / f"PREFLIGHT_{protocol}.json", dict(
        status="passed", backbone=BACKBONE, protocol=protocol,
        contract_sha256=contract.sha256, feature_integrity=integrity,
        frozen_data_sha256={key: digest(path) for key, path in data_paths.items()},
        source_train_n=len(train_uids), source_dev_n=len(dev_uids),
        source_train_feature_n=int(token["train"][1].sum()),
        source_dev_feature_n=int(token["dev"][1].sum()),
        test_feature_n={view: int(token[view][1].sum()) for view in target_views},
        fallback_expert_index=FALLBACK_EXPERT,
        source_fold_fallback_verified=True))
    return contract, source, all_y[dev_rows], all_c[dev_rows], token, target_views


def supervised(inputs, target, active):
    positions, available = inputs
    selected = np.flatnonzero(active & available.cpu().numpy())
    if not len(selected):
        raise RuntimeError("no available soft-label training rows")
    index = torch.as_tensor(selected, dtype=torch.long, device=positions.device)
    return positions[index], torch.as_tensor(target[selected], dtype=torch.float32,
                                            device=positions.device)


@torch.no_grad()
def soft_loss(model, positions, target, batch_size):
    assert FEATURE_BANK is not None
    model.eval()
    total = 0.0
    for start in range(0, len(positions), batch_size):
        end = min(start + batch_size, len(positions))
        logits = model(FEATURE_BANK[positions[start:end]].float())
        loss = -(target[start:end] * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        total += float(loss.sum().item())
    return total / len(positions)


@torch.no_grad()
def predict(model, inputs, batch_size):
    assert FEATURE_BANK is not None
    positions, available = inputs
    chosen = torch.nonzero(available, as_tuple=False).flatten()
    actions = np.full(len(positions), FALLBACK_EXPERT, dtype=np.int8)
    model.eval()
    for start in range(0, len(chosen), batch_size):
        local = chosen[start:start + batch_size]
        logits = model(FEATURE_BANK[positions[local]].float())
        actions[local.cpu().numpy()] = logits.argmax(dim=-1).cpu().numpy().astype(np.int8)
    return actions


def fit_and_predict(contract, train_inputs, train_target, train_active,
                    dev_inputs, dev_target, dev_active, eval_inputs,
                    *, seed: int, lam: str, tag: str):
    assert FEATURE_BANK is not None
    seed_everything(seed)
    profile = contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"]
    batch_size = int(profile["batch_size_per_gpu"])
    model = AuditedBenchCoEHead(FEATURE_BANK.shape[1], 17,
                                float(profile["dropout"])).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(profile["learning_rate"]), weight_decay=float(profile["weight_decay"]))
    train_positions, train_values = supervised(train_inputs, train_target, train_active)
    dev_positions, dev_values = supervised(dev_inputs, dev_target, dev_active)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, int(profile["epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_positions), generator=generator).to("cuda")
        loss_total = 0.0
        for start in range(0, len(order), batch_size):
            local = order[start:start + batch_size]
            logits = model(FEATURE_BANK[train_positions[local]].float())
            loss = -(train_values[local] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite soft-label loss: {tag}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(local)
        dev_loss = soft_loss(model, dev_positions, dev_values, batch_size)
        history.append(dict(epoch=epoch, train_loss=loss_total / len(train_positions),
                            dev_loss=dev_loss))
        if dev_loss < best - 1e-9:
            best, best_epoch, bad = dev_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            bad += 1
        if epoch >= int(profile["min_epochs"]) and bad >= int(profile["patience"]):
            break
    if best_state is None:
        raise RuntimeError("no frozen-head checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    checkpoint = DEST / "models" / f"{tag}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(backbone=BACKBONE, tag=tag, seed=seed, lambda_cost=lam,
                    state_dict=best_state, best_epoch=best_epoch,
                    best_dev_soft_loss=best), checkpoint)
    actions = {name: predict(model, inputs, batch_size) for name, inputs in eval_inputs.items()}
    info = dict(backbone=BACKBONE, best_epoch=best_epoch, best_dev_soft_loss=best,
                epochs_run=len(history), history=history,
                trainable_n=len(train_positions), dev_n=len(dev_positions),
                missing_feature_fallback_expert_index=FALLBACK_EXPERT,
                checkpoint=str(checkpoint.relative_to(ROOT)),
                checkpoint_sha256=digest(checkpoint))
    print(json.dumps(dict(stage="head_complete", backbone=BACKBONE, tag=tag,
                          best_epoch=best_epoch, dev_loss=round(best, 6))), flush=True)
    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return actions, info


def per_dataset_results() -> None:
    views = view_uids()
    matrices, _, _ = eval_matrices(views)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    dataset = np.asarray([uid.split("/", 1)[0] for uid in views["3OOD"]])
    mapping = {"MMMU_DEV_VAL": "MMMU", "MathVision_MINI": "MathVision",
               "ChartQA_TEST": "ChartQA"}
    if dict(zip(*np.unique(dataset, return_counts=True))) != {
            "MMMU_DEV_VAL": 206, "MathVision_MINI": 51, "ChartQA_TEST": 510}:
        raise RuntimeError("held-out test dataset composition changed")
    rows = []
    for seed in SEEDS:
        root = DEST / "heldout11" / f"seed-{seed}"
        selected = read_json(root / "SOURCE_CV.json")["selected_lambda"]
        for selection, lam in (("source_cv", selected),):
            with np.load(root / f"final-lambda-{lam}.npz", allow_pickle=False) as frozen:
                if frozen["uid_3OOD"].tolist() != views["3OOD"]:
                    raise RuntimeError("held-out prediction UID order changed")
                actions = frozen["action_3OOD"].astype(np.int8)
            for name, label in mapping.items():
                mask = dataset == name
                A, C, R = scores(matrices["3OOD"][0][mask],
                                 matrices["3OOD"][1][mask], actions[mask], bounds)
                rows.append(dict(method=e20.METHOD, selection=selection, seed=seed,
                                 dataset=label, n=int(mask.sum()), A=A, C=C, R=R))
    write_csv(DEST / "per_dataset.csv", PER_DATASET_COLUMNS, rows)


def configure(backbone: str) -> None:
    global BACKBONE, DEST, SEEDS
    BACKBONE = backbone
    SEEDS = tuple(int(value) for value in os.environ.get("V19_QUERY_SEEDS", "0").split(","))
    if not SEEDS or len(set(SEEDS)) != len(SEEDS) or any(seed not in range(5) for seed in SEEDS):
        raise RuntimeError("V19_QUERY_SEEDS must be unique values from 0–4")
    DEST = OUTPUT / f"V19_QUERY_COST_{backbone.upper()}"
    e20.OUT = DEST
    e20.EXPERIMENT = DEST.name
    e20.METHOD = f"Query-{BACKBONES[backbone]}"
    e20.COMMAND = "v19_query_softlabels_frozen.py"
    e20.RESULT_NOTES = (f"Frozen {BACKBONES[backbone]} features and trained soft-label head; "
                        "test-selected lambda is diagnostic only")
    e20.TRAINING_DESCRIPTION = (
        f"Frozen {BACKBONES[backbone]} features validated through the full cache gate; "
        "the LayerNorm-Dropout-Linear head is trained independently for each "
        "released benchmark lambda, with cost_scale=100 and all-wrong rows "
        "excluded from soft cross-entropy. Source-dev soft loss selects the "
        "epoch; group-isolated three-fold source-fit OOF Rank selects lambda. "
        "Missing-image features use the train-global strongest expert, index 12, "
        "verified independently on every source fold. Test-selected lambda "
        "is a labelled supplement diagnostic only.")
    e20.SEEDS = SEEDS
    e20.prepare_inputs = prepare_inputs
    e20.fit_and_predict = fit_and_predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--protocol", choices=("full14", "heldout11"))
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    configure(args.backbone)
    if args.aggregate:
        e20.aggregate()
        per_dataset_results()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("frozen-head soft-label training requires CUDA")
        e20.run_protocol(args.protocol)


if __name__ == "__main__":
    main()
