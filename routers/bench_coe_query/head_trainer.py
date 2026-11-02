"""LayerNorm+Dropout+Linear trainer for frozen TinyLLaVA/Qwen3-VL caches."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from routers.utils.rank_score import rank_score

from .checkpoints import DevLossCheckpointManager, load_checkpoint
from .config import ExperimentContract
from .data import load_vlrb_strict
from .feature_cache import load_full_feature_cache, load_merged_features
from .full_policy import (
    EXPECTED_MULTI_IMAGE_SPLIT_COUNTS,
)
from .heads import AuditedBenchCoEHead, trainable_parameter_names
from .labels import build_diagnostic_labels_for_split, compute_train_expert_accuracy
from .manifest import build_manifest, write_manifest
from .input_content import feature_input_content_fields
from .reproducibility import seed_everything
from .trainer import DevLossEarlyStopping, masked_cross_entropy


def build_head_training_labels(data, rows_all: np.ndarray, split_positions: dict[str, np.ndarray]) -> np.ndarray:
    """Build only train/dev targets; test Y is deliberately never indexed here."""

    train_accuracy = compute_train_expert_accuracy(
        data.y, data.split_to_rows["train"], data.model_order, expected_model_order=data.model_order
    )
    labels_all = np.full(len(rows_all), -100, dtype=np.int64)
    for split in ("train", "dev"):
        positions = split_positions[split]
        rows = rows_all[positions]
        result = build_diagnostic_labels_for_split(
            data.y, rows, train_accuracy, data.model_order, expected_model_order=data.model_order
        )
        labels_all[positions] = result.labels
    return labels_all


@torch.no_grad()
def _head_eval(model, loader, device, data, cmin: float, cmax: float) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    valid_count = 0
    hard_correct = 0
    routed_correct = 0
    costs = 0.0
    samples = 0
    for features, labels, rows in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss, count = masked_cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)
        if count:
            valid = labels != -100
            loss_sum += float(loss.cpu()) * count
            hard_correct += int((predictions[valid] == labels[valid]).sum().item())
            valid_count += count
        row_values = rows.numpy()
        prediction_values = predictions.cpu().numpy()
        routed_correct += int(data.y[row_values, prediction_values].sum())
        costs += float(data.c[row_values, prediction_values].astype(np.float64).sum())
        samples += len(row_values)
    if valid_count == 0 or samples == 0:
        raise RuntimeError("head dev evaluation has no usable rows")
    accuracy = routed_correct / samples
    average_cost = costs / samples
    return {
        "dev_hard_label_loss": loss_sum / valid_count,
        "dev_hard_label_accuracy": hard_correct / valid_count,
        "dev_routed_accuracy": accuracy,
        "dev_average_cost": average_cost,
        "dev_rank_score": rank_score(accuracy, average_cost, cmin, cmax, beta=0.1),
        "dev_trainable_samples": valid_count,
        "dev_samples": samples,
    }


@torch.no_grad()
def _predict(model, features: torch.Tensor, batch_size: int, device) -> np.ndarray:
    # There is intentionally no Y/C parameter in this prediction helper.
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    model.eval()
    logits = [model(batch[0].to(device)).float().cpu().numpy() for batch in loader]
    return np.concatenate(logits, axis=0)


def run_head_training(args, contract: ExperimentContract) -> int:
    seed_info = seed_everything(args.seed)
    data = load_vlrb_strict(args.dataset_dir, contract=contract)
    is_full = os.environ.get("RUN_FULL", "0") == "1" and args.max_samples is None
    cache_root = Path(args.feature_cache)
    manifest_path = cache_root / "merged/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature manifest missing: {manifest_path}")
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_manifest_sha256 = str(frozen_manifest["manifest_sha256"])
    cache_integrity = None
    if is_full:
        features_np, records, feature_manifest, cache_integrity = load_full_feature_cache(
            cache_root,
            dataset_root=contract.dataset_root,
            expected_contract_sha256=contract.sha256,
            expected_input_content=feature_input_content_fields(contract, args.backbone),
        )
    else:
        features_np, records, feature_manifest = load_merged_features(
            cache_root,
            expected_signature=str(frozen_manifest["signature"]),
            expected_manifest_sha256=feature_manifest_sha256,
        )
    if len({int(record["row_index"]) for record in records}) != len(records):
        raise RuntimeError("feature cache contains duplicate rows")
    rows_all = np.asarray([int(record["row_index"]) for record in records], dtype=np.int64)
    if any(data.samples[row].sample_id != str(record["sample_id"]) for row, record in zip(rows_all, records)):
        raise RuntimeError("feature cache sample IDs do not match matrix rows")

    split_positions: dict[str, np.ndarray] = {}
    for split in ("train", "dev", "test"):
        split_positions[split] = np.flatnonzero(
            np.asarray([data.samples[row].split == split for row in rows_all], dtype=bool)
        )
        if len(split_positions[split]) == 0:
            raise RuntimeError(f"feature cache has no {split} rows")
    if is_full:
        expected_feature_rows = int(contract.values["data"]["expected_samples"]) - int(
            contract.values["data"]["asset_policy"]["known_multi_image_samples"]
        )
        if len(rows_all) != expected_feature_rows:
            raise RuntimeError(
                f"full feature cache row count mismatch: {len(rows_all)} != {expected_feature_rows}"
            )
        for split, excluded in EXPECTED_MULTI_IMAGE_SPLIT_COUNTS.items():
            expected = len(data.split_to_rows[split]) - excluded
            if len(split_positions[split]) != expected:
                raise RuntimeError(
                    f"full {split} feature count mismatch: {len(split_positions[split])} != {expected}"
                )
    else:
        limits = contract.values["safety"]["smoke_limits"]
        for split, positions in split_positions.items():
            if len(positions) > int(limits[split]):
                raise RuntimeError(
                    f"RUN_FULL=0 cache exceeds {split} smoke limit: {len(positions)} > {limits[split]}"
                )

    labels_all = build_head_training_labels(data, rows_all, split_positions)

    features = torch.from_numpy(np.asarray(features_np, dtype=np.float32))
    labels = torch.from_numpy(labels_all)
    rows_tensor = torch.from_numpy(rows_all)
    profile = contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"]
    epochs = int(args.epochs if args.epochs is not None else (profile["epochs"] if is_full else 1))
    batch_size = int(profile["batch_size_per_gpu"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuditedBenchCoEHead(
        hidden_dim=int(features.shape[1]),
        num_experts=len(data.model_order),
        dropout=float(profile["dropout"]),
    ).to(device)
    expected_names = ("layer_norm.weight", "layer_norm.bias", "classifier.weight", "classifier.bias")
    if trainable_parameter_names(model) != expected_names:
        raise RuntimeError(f"unexpected trainable head parameters: {trainable_parameter_names(model)}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if optimized != trainable:
        raise RuntimeError("head optimizer contains anything other than LN/Linear parameters")

    generator = torch.Generator().manual_seed(args.seed)
    train_positions = split_positions["train"]
    dev_positions = split_positions["dev"]
    train_loader = DataLoader(
        TensorDataset(features[train_positions], labels[train_positions], rows_tensor[train_positions]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    dev_loader = DataLoader(
        TensorDataset(features[dev_positions], labels[dev_positions], rows_tensor[dev_positions]),
        batch_size=batch_size,
        shuffle=False,
    )
    with (data.dataset_root / "data/matrices/cost_bounds.json").open("r", encoding="utf-8") as handle:
        bounds = json.load(handle)
    cmin, cmax = float(bounds["cmin"]), float(bounds["cmax"])

    mode_root = f"runs/{args.backbone}/vlrb_adapted_v1" if is_full else f"smoke/{args.backbone}/head"
    output = contract.dataset_root / str(contract.values["safety"]["output_root"]) / mode_root / args.run_id / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(contract.plain(), sort_keys=False), encoding="utf-8")
    write_manifest(
        output / "manifest.json",
        build_manifest(
            contract,
            seed=args.seed,
            run_id=args.run_id,
            input_content_verification=(
                {
                    "status": "verified_through_full_feature_cache_pointer_gate_chain",
                    "input_content": cache_integrity.get("input_content"),
                    "cache_root": cache_integrity.get("cache_root"),
                    "pointer_sha256": cache_integrity.get("pointer_sha256"),
                    "passed": cache_integrity.get("passed") is True,
                }
                if is_full and cache_integrity is not None
                else None
            ),
        ),
    )
    manager = DevLossCheckpointManager(output / "checkpoints")
    stopper = DevLossEarlyStopping(patience=int(profile["patience"]), min_epochs=int(profile["min_epochs"]))
    history_path = output / "training_history.jsonl"
    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    resumed_from_checkpoint: str | None = None
    if args.resume:
        last_path = output / "checkpoints/last/checkpoint.pt"
        if not last_path.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing: {last_path}")
        checkpoint = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            expected_config_sha256=contract.sha256,
            expected_feature_manifest_sha256=feature_manifest_sha256,
            restore_rng=True,
            map_location=device,
        )
        resume_state = checkpoint.get("extra", {}).get("resume_state", {})
        if "train_generator_state" not in resume_state:
            raise RuntimeError("checkpoint predates exact-resume train generator state")
        generator.set_state(resume_state["train_generator_state"])
        stopper.best_loss = float(checkpoint["best_dev_hard_label_loss"])
        stopper.bad_epochs = int(resume_state.get("early_stopping_bad_epochs", 0))
        manager.best_loss = float(checkpoint["best_dev_hard_label_loss"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1
        resumed_from_checkpoint = str(last_path)
        if start_epoch > epochs:
            raise RuntimeError(
                f"resume checkpoint already reached epoch {checkpoint['epoch']}; requested total epochs={epochs}"
            )
        if history_path.is_file():
            history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        valid_count = 0
        hard_correct = 0
        for batch_features, batch_labels, _batch_rows in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss, count = masked_cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            if count:
                valid = batch_labels != -100
                loss_sum += float(loss.detach().cpu()) * count
                hard_correct += int((logits.detach().argmax(dim=-1)[valid] == batch_labels[valid]).sum().item())
                valid_count += count
        dev_metrics = _head_eval(model, dev_loader, device, data, cmin, cmax)
        record = {
            "epoch": epoch,
            "train_hard_label_loss": loss_sum / max(1, valid_count),
            "train_hard_label_accuracy": hard_correct / max(1, valid_count),
            "trainable_samples": valid_count,
            **dev_metrics,
        }
        history.append(record)
        _improved, stop = stopper.update(
            epoch=epoch, dev_hard_label_loss=float(dev_metrics["dev_hard_label_loss"])
        )
        manager.save(
            dev_hard_label_loss=float(dev_metrics["dev_hard_label_loss"]),
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=epoch,
            global_step=global_step,
            seed=args.seed,
            config_sha256=contract.sha256,
            feature_manifest_sha256=feature_manifest_sha256,
            extra={
                "backbone": args.backbone,
                "feature_signature": feature_manifest["signature"],
                "input_content": feature_input_content_fields(contract, args.backbone),
                "resume_state": {
                    "train_generator_state": generator.get_state(),
                    "early_stopping_bad_epochs": stopper.bad_epochs,
                },
            },
        )
        history_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
            encoding="utf-8",
        )
        print(json.dumps(record, sort_keys=True))
        if stop:
            break
    changed = {name: not torch.equal(initial[name], value.detach().cpu()) for name, value in model.state_dict().items()}
    if not all(changed.values()):
        raise RuntimeError(f"not every LN/Linear state tensor changed: {changed}")
    best_path = output / "checkpoints/best/checkpoint.pt"
    load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=contract.sha256,
        expected_feature_manifest_sha256=feature_manifest_sha256,
        map_location=device,
    )
    model.to(device)
    probe_features = features[: min(8, len(features))]
    round_trip_before = _predict(model, probe_features, batch_size, device)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=contract.sha256,
        expected_feature_manifest_sha256=feature_manifest_sha256,
        map_location=device,
    )
    round_trip_after = _predict(model, probe_features, batch_size, device)
    checkpoint_round_trip_prediction_equal = bool(np.array_equal(round_trip_before, round_trip_after))
    if not checkpoint_round_trip_prediction_equal:
        raise RuntimeError("head checkpoint save/load changed fixed-probe predictions")
    elapsed = time.perf_counter() - started
    history_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in history), encoding="utf-8"
    )
    resource = {
        "backbone": args.backbone,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "feature_manifest_sha256": feature_manifest_sha256,
        "feature_signature": feature_manifest["signature"],
        "trainable_parameter_names": list(trainable_parameter_names(model)),
        "optimizer_parameter_set_exact": True,
        "checkpoint_monitor": "dev_hard_label_loss",
        "checkpoint_round_trip_prediction_equal": checkpoint_round_trip_prediction_equal,
        "head_state_changed": changed,
        "backbone_parameters_present_in_optimizer": 0,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "seed_settings": seed_info,
        "prediction_generation": {
            "status": "not_performed_in_training_process",
            "required_entry_point": "routers.bench_coe_query.cli freeze-run",
            "required_process_boundary": "target_free_producer_then_separate_evaluator",
        },
        "feature_cache_integrity": cache_integrity,
    }
    (output / "resource_usage.json").write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "resource": resource}, indent=2, sort_keys=True))
    return 0
