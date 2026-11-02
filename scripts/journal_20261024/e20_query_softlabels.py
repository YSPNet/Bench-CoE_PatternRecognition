#!/usr/bin/env python3
"""V5 Query-Bench-CoE BERT soft-label cost protocol, source-only 3-fold CV.

Each lambda/fold independently fine-tunes the paper's BERT-base-uncased
classifier.  The released benchmark's soft-target constructor supplies the
targets.  Source dev loss chooses the epoch, and group-isolated source-fit
OOF Rank chooses lambda.  The final six lambda models are fit on the full
source train.  Their test actions are frozen before test outcomes are opened.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    eval_matrices, make_audit, read_json, scores, summarize,
                    utc_now, view_uids, write_csv, write_json)
from routers.bench_coe_query.backbones.utils import require_model_identity
from routers.bench_coe_query.bert_router import _linear_warmup_decay_scheduler, _optimizer
from routers.bench_coe_query.config import effective_model_path, load_contract, sha256_file
from routers.bench_coe_query.reproducibility import seed_everything
from routers.utils.soft_targets import build_soft_targets
from scripts import run_bench_coe_subject_mmmu as subject
from src.next_round_x.data import load_source, role_uids

OUT = OUTPUT / "E20"
EXPERIMENT = "E20"
METHOD = "Query-BERT"
COMMAND = "e20_query_softlabels.py"
RESULT_NOTES = "Fully trained soft-label BERT; test-selected lambda is diagnostic only"
TRAINING_DESCRIPTION = (
    "BERT fully fine-tuned independently at each released benchmark lambda; "
    "soft labels use routers.utils.soft_targets.build_soft_targets with cost_scale=100 "
    "and all-wrong rows excluded from cross-entropy. Source-dev soft loss chooses "
    "epoch. Three source-fit group-isolated OOF folds select lambda by pooled Rank. "
    "Test-selected lambda is a separately labelled supplement diagnostic.")
LAMBDA_GRID = ("0", "10", "100", "1000", "10000", "inf")
COST_SCALE = 100.0  # released benchmark soft-target default
CURVE_COLUMNS = ("method", "protocol", "seed", "lambda", "view", "A", "C", "R")
SELECT_COLUMNS = ("protocol", "seed", "view", "selection", "lambda", "source_oof_R", "test_R")


def log(**fields):
    print(json.dumps(fields, sort_keys=True, default=str), flush=True)


def targets(y, c, lam):
    values = build_soft_targets(y, c, lam=float(lam), scheme="exp",
                                fallback="cheapest", cost_scale=COST_SCALE)
    active = y.sum(axis=1) > 0
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("soft-target rows do not sum to one")
    return values, active


def text_tensors(ids, texts, tokenizer, max_length):
    enc = tokenizer([texts[uid] for uid in ids], truncation=True,
                    padding="max_length", max_length=max_length,
                    return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def loader(ids_tensor, mask_tensor, target=None, active=None, *, shuffle=False, seed=0, batch=64):
    if target is None:
        dataset = TensorDataset(ids_tensor, mask_tensor)
    else:
        select = np.flatnonzero(active)
        dataset = TensorDataset(ids_tensor[select], mask_tensor[select],
                                torch.from_numpy(target[select]))
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=0, pin_memory=True)


@torch.no_grad()
def loss_on(model, valid_loader, device):
    model.eval()
    total = 0.0
    seen = 0
    for ids, mask, target in valid_loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = -(target.to(device) * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
        total += float(loss) * len(ids)
        seen += len(ids)
    if seen == 0:
        raise RuntimeError("empty source-dev loss set")
    return total / seen


@torch.no_grad()
def predict(model, inputs, device, batch_size):
    model.eval()
    result = []
    for ids, mask in loader(*inputs, batch=batch_size):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        result.append(logits.float().argmax(dim=-1).cpu().numpy())
    return np.concatenate(result).astype(np.int8)


def fit_and_predict(contract, train_inputs, train_target, train_active,
                    dev_inputs, dev_target, dev_active, eval_inputs,
                    *, seed, lam, tag):
    seed_everything(seed)
    model_path = effective_model_path(contract, "bert")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), num_labels=17, local_files_only=True,
        ignore_mismatched_sizes=True).to("cuda")
    model.requires_grad_(True)
    cfg = contract.values["profiles"]["vlrb_adapted_v1"]["bert"]
    batch_size = int(cfg["batch_size_per_gpu"])
    epochs = int(cfg["epochs"])
    train_loader = loader(*train_inputs, train_target, train_active,
                          shuffle=True, seed=seed, batch=batch_size)
    dev_loader = loader(*dev_inputs, dev_target, dev_active, batch=batch_size)
    if len(train_loader) == 0 or len(dev_loader) == 0:
        raise RuntimeError("empty source train/dev soft-target loader")
    optimizer = _optimizer(model, float(cfg["learning_rate"]), float(cfg["weight_decay"]))
    steps = epochs * len(train_loader)
    scheduler = _linear_warmup_decay_scheduler(optimizer,
        warmup_steps=int(steps * float(cfg["warmup_ratio"])), total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0
        for ids, mask, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids.to("cuda"), attention_mask=mask.to("cuda")).logits
                loss = -(target.to("cuda") * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite soft-label loss: {tag}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach()) * len(ids)
            count += len(ids)
        dev_loss = loss_on(model, dev_loader, "cuda")
        history.append(dict(epoch=epoch, train_loss=running/count, dev_loss=dev_loss))
        log(stage="epoch", tag=tag, **history[-1])
        if dev_loss < best - 1e-9:
            best, bad, best_epoch = dev_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
        if epoch >= int(cfg["min_epochs"]) and bad >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("no BERT checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    actions = {name: predict(model, tensors, "cuda", batch_size)
               for name, tensors in eval_inputs.items()}
    del model, best_state, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return actions, dict(best_epoch=best_epoch, best_dev_loss=best,
                         epochs_run=len(history), history=history,
                         trainable_n=int(train_active.sum()), dev_n=int(dev_active.sum()))


def prepare_inputs(protocol):
    contract = load_contract()
    model_path = effective_model_path(contract, "bert")
    require_model_identity(model_path, expected_model_type="bert",
                           expected_architecture="BertForMaskedLM")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True,
                                               local_files_only=True)
    registry_ids, _, _ = subject.rows_and_metadata()
    text_list = subject.bert_texts(registry_ids)
    texts = dict(zip(registry_ids, text_list))
    if len(texts) != 30540:
        raise RuntimeError("frozen input text registry changed")
    source = load_source(protocol)
    train_uids = source["uids"].tolist()
    if train_uids != role_uids(protocol, "train"):
        raise RuntimeError("source train UID order differs from official split")
    registry = {uid: i for i, uid in enumerate(registry_ids)}
    positions = np.asarray([registry[uid] for uid in train_uids], dtype=np.int64)
    for matrix_name in ("y", "c"):
        entry = contract.values["data"]["matrices"][matrix_name]
        matrix_path = ROOT / str(entry["path"])
        if sha256_file(matrix_path) != str(entry["sha256"]):
            raise RuntimeError(f"frozen {matrix_name} matrix hash changed")
    with np.load(ROOT / "data/matrices/Y.npz", allow_pickle=False) as archive:
        all_y = archive["Y"]
    all_c = np.load(ROOT / "data/matrices/C.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(all_y[positions], source["y"]) or not np.allclose(all_c[positions], source["c"]):
        raise RuntimeError("strict matrix/source scope alignment changed")
    dev_uids = role_uids(protocol, "dev")
    dev_positions = np.asarray([registry[uid] for uid in dev_uids], dtype=np.int64)
    source_datasets = {uid.split("/", 1)[0] for uid in train_uids}
    if any(uid.split("/", 1)[0] not in source_datasets for uid in dev_uids):
        raise RuntimeError("source dev includes held-out dataset")
    all_views = view_uids()
    views = ("14ID",) if protocol == "full14" else ("11ID", "3OOD")
    max_length = int(contract.values["models"]["bert"]["max_length"])
    token = {"train": text_tensors(train_uids, texts, tokenizer, max_length),
             "dev": text_tensors(dev_uids, texts, tokenizer, max_length)}
    for view in views:
        token[view] = text_tensors(all_views[view], texts, tokenizer, max_length)
    return contract, source, all_y[dev_positions], all_c[dev_positions], token, views


def run_protocol(protocol):
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    contract, data, y_dev, c_dev, token, views = prepare_inputs(protocol)
    fit = np.flatnonzero(data["role"] == "fit")
    if len(fit) != (17871 if protocol == "full14" else 16269):
        raise RuntimeError("source-fit group split changed")
    audit = make_audit(EXPERIMENT, f"{COMMAND} --protocol {protocol}",
        TRAINING_DESCRIPTION,
        {view: len(ids) for view, ids in view_uids().items()})
    audit.update(status="running", c_min=bounds["cmin"], c_max=bounds["cmax"],
                 gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
                 lambda_grid=list(LAMBDA_GRID), protocol=protocol,
                 cost_scale=COST_SCALE, seed_policy=list(SEEDS))
    root = OUT / protocol
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "AUDIT.json", audit)
    try:
        for seed in SEEDS:
            seed_root = root / f"seed-{seed}"
            seed_root.mkdir(exist_ok=True)
            oof = {lam: np.full(len(fit), -1, dtype=np.int16) for lam in LAMBDA_GRID}
            fit_pos = {int(row): i for i, row in enumerate(fit)}
            for fold in range(3):
                tr = np.flatnonzero((data["role"] == "fit") & (data["fold"] != fold))
                va = np.flatnonzero((data["role"] == "fit") & (data["fold"] == fold))
                if set(data["groups"][tr]) & set(data["groups"][va]):
                    raise RuntimeError("source OOF dependency groups overlap")
                for lam in LAMBDA_GRID:
                    tag = f"{protocol}/seed-{seed}/fold-{fold}/lambda-{lam}"
                    part = seed_root / f"fold-{fold}-lambda-{lam}.npz"
                    info_path = seed_root / f"fold-{fold}-lambda-{lam}.json"
                    if part.is_file() and info_path.is_file():
                        with np.load(part, allow_pickle=False) as saved:
                            if saved["uids"].tolist() != data["uids"][va].tolist():
                                raise RuntimeError(f"OOF UID checkpoint mismatch: {tag}")
                            action = saved["actions"].astype(np.int8)
                    else:
                        y_train, c_train = data["y"][tr], data["c"][tr]
                        train_target, train_active = targets(y_train, c_train, lam)
                        dev_target, dev_active = targets(y_dev, c_dev, lam)
                        actions, info = fit_and_predict(contract,
                            (token["train"][0][tr], token["train"][1][tr]),
                            train_target, train_active, token["dev"], dev_target,
                            dev_active, {"oof": (token["train"][0][va],
                                                 token["train"][1][va])},
                            seed=seed, lam=lam, tag=tag)
                        action = actions["oof"]
                        np.savez_compressed(part, uids=data["uids"][va], actions=action)
                        write_json(info_path, dict(protocol=protocol, seed=seed,
                            fold=fold, **{"lambda": lam}, **info))
                        log(stage="fold_lambda_complete", tag=tag, n=len(va))
                    oof[lam][np.asarray([fit_pos[int(row)] for row in va])] = action
            cv = []
            for lam in LAMBDA_GRID:
                if np.any(oof[lam] < 0):
                    raise RuntimeError(f"incomplete OOF for lambda {lam}")
                A, C, R = scores(data["y"][fit], data["c"][fit], oof[lam], bounds)
                cv.append(dict(**{"lambda": lam}, A=A, C=C, R=R))
            winner = max(cv, key=lambda row: (row["R"], row["A"], -row["C"],
                                               -LAMBDA_GRID.index(row["lambda"])))
            write_json(seed_root / "SOURCE_CV.json", dict(protocol=protocol, seed=seed,
                source_fit_n=len(fit), source_cv=cv, selected_lambda=winner["lambda"],
                group_isolated=True, fold_count=3))
            log(stage="source_cv_selected", protocol=protocol, seed=seed,
                chosen_lambda=winner["lambda"], source_R=winner["R"])
            for lam in LAMBDA_GRID:
                tag = f"{protocol}/seed-{seed}/final/lambda-{lam}"
                part = seed_root / f"final-lambda-{lam}.npz"
                info_path = seed_root / f"final-lambda-{lam}.json"
                if part.is_file() and info_path.is_file():
                    with np.load(part, allow_pickle=False) as saved:
                        for view in views:
                            if saved[f"uid_{view}"].tolist() != view_uids()[view]:
                                raise RuntimeError(f"final UID checkpoint mismatch: {tag}/{view}")
                    continue
                train_target, train_active = targets(data["y"], data["c"], lam)
                dev_target, dev_active = targets(y_dev, c_dev, lam)
                actions, info = fit_and_predict(contract, token["train"],
                    train_target, train_active, token["dev"], dev_target, dev_active,
                    {view: token[view] for view in views}, seed=seed, lam=lam, tag=tag)
                np.savez_compressed(part,
                    **{f"uid_{view}": np.asarray(view_uids()[view]) for view in views},
                    **{f"action_{view}": actions[view] for view in views})
                write_json(info_path, dict(protocol=protocol, seed=seed,
                    **{"lambda": lam}, **info))
                log(stage="final_lambda_complete", tag=tag)
            write_json(seed_root / "COMPLETE.json", dict(protocol=protocol, seed=seed,
                selected_lambda=winner["lambda"], folds=3, lambdas=list(LAMBDA_GRID),
                views=list(views), status="complete"))
        audit["status"] = "preview_complete" if SEEDS == (0,) else "complete"
    except Exception as exc:
        path = root / "traceback.txt"
        path.write_text(traceback.format_exc())
        audit["failures"].append(dict(id=EXPERIMENT, seed=None, error=str(exc),
                                      traceback_path=str(path.relative_to(ROOT))))
        audit["status"] = "failed"
        raise
    finally:
        audit["end_time"] = utc_now()
        write_json(root / "AUDIT.json", audit)


def aggregate():
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    views = view_uids()
    matrices, _, _ = eval_matrices(views)
    result, curves, selected = [], [], []
    source_audits = {}
    for protocol in ("full14", "heldout11"):
        audit = read_json(OUT / protocol / "AUDIT.json")
        if audit.get("status") not in ("complete", "preview_complete"):
            raise RuntimeError(f"{protocol} is not complete")
        source_audits[protocol] = str((OUT / protocol / "AUDIT.json").relative_to(ROOT))
        target_views = ("14ID",) if protocol == "full14" else ("11ID", "3OOD")
        for seed in SEEDS:
            seed_root = OUT / protocol / f"seed-{seed}"
            chosen = read_json(seed_root / "SOURCE_CV.json")
            for view in target_views:
                candidates = []
                for lam in LAMBDA_GRID:
                    with np.load(seed_root / f"final-lambda-{lam}.npz", allow_pickle=False) as frozen:
                        if frozen[f"uid_{view}"].tolist() != views[view]:
                            raise RuntimeError("final test UID order changed")
                        action = frozen[f"action_{view}"].astype(np.int8)
                    A, C, R = scores(*matrices[view], action, bounds)
                    candidates.append(dict(**{"lambda": lam}, A=A, C=C, R=R))
                    curves.append(dict(method=METHOD, protocol=protocol,
                        seed=seed, **{"lambda": lam}, view=view, A=A, C=C, R=R))
                best = max(candidates, key=lambda item: item["R"])
                for selection, lam in (("source_cv", chosen["selected_lambda"]),
                                       ("best_test", best["lambda"])):
                    item = next(row for row in candidates if row["lambda"] == lam)
                    result.append(dict(experiment=EXPERIMENT, method=METHOD,
                        variant=f"lambda={lam}/select={selection}", view=view,
                        seed=seed, n_test=len(views[view]), A=item["A"], C=item["C"],
                        R=item["R"], notes=RESULT_NOTES))
                    source_R = next(row["R"] for row in chosen["source_cv"]
                                    if row["lambda"] == lam)
                    selected.append(dict(protocol=protocol, seed=seed, view=view,
                        selection=selection, **{"lambda": lam},
                        source_oof_R=source_R, test_R=item["R"]))
    write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, result)
    write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summarize(result))
    write_csv(OUT / "curves.csv", CURVE_COLUMNS, curves)
    write_csv(OUT / "selections.csv", SELECT_COLUMNS, selected)
    audit = make_audit(EXPERIMENT, f"{COMMAND} --aggregate",
        "Source-fit three-fold group-isolated OOF selects lambda for main rows. "
        "The best_test rows are labelled supplement diagnostics only.",
        {view: len(ids) for view, ids in views.items()})
    audit.update(status="preview_complete" if SEEDS == (0,) else "complete",
                 c_min=bounds["cmin"], c_max=bounds["cmax"],
                 source_audits=source_audits, lambda_grid=list(LAMBDA_GRID),
                 gpu="see per-protocol audits", end_time=utc_now())
    write_json(OUT / "AUDIT.json", audit)
    with (OUTPUT / "PROGRESS.md").open("a") as handle:
        handle.write(f"\n## {EXPERIMENT} — {audit['status']}\n\nCost-aware {METHOD}, "
                     f"seeds {list(SEEDS)}, source group-OOF and test-selected diagnostic.\n")
    log(stage="aggregate_complete", rows=len(result), curves=len(curves))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--protocol", choices=("full14", "heldout11"))
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("cost-aware Query-BERT requires CUDA")
        run_protocol(args.protocol)


if __name__ == "__main__":
    main()
