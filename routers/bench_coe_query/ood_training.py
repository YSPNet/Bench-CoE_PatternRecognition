"""Train Prompt-15 routers on train_seen and select checkpoints on dev_seen only."""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .backbones.utils import force_offline_environment, parameter_summary_checksum, require_model_identity
from .bert_router import (
    BertTextDataset,
    _evaluate_dev,
    _gradient_norm,
    _linear_warmup_decay_scheduler,
    _optimizer,
    predict_bert_logits,
)
from .checkpoints import DevLossCheckpointManager, checkpoint_sha256, load_checkpoint
from .config import ExperimentContract, effective_model_path, load_contract
from .data import load_vlrb_strict
from .feature_cache import load_full_feature_cache
from .head_trainer import _head_eval, _predict
from .heads import AuditedBenchCoEHead, trainable_parameter_names
from .input_content import feature_input_content_fields, input_content_summary, verify_pinned_input_content
from .ood_common import (
    OOD_CONFIG_PATH,
    OOD_GATE_PATH,
    OOD_PROTOCOL_PATH,
    atomic_json,
    atomic_text,
    build_ood_training_labels,
    full_feature_cache_root,
    load_strict_json,
    ood_run_dir,
    ood_training_binding,
    row_indices_sha256,
)
from .ood_gate import assert_ood_execution_gate
from .ood_protocol import load_and_validate_protocol
from .reproducibility import seed_everything
from .trainer import DevLossEarlyStopping, masked_cross_entropy


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for record in history
        ),
    )


def _base_manifest(
    contract: ExperimentContract,
    *,
    backbone: str,
    seed: int,
    protocol: dict[str, Any],
    view: Any,
    binding_sha256: str,
    binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "prompt15_ood_training_run",
        "claim": "reconstructed held-out protocol; not exact Table A2 membership",
        "backbone": backbone,
        "seed": seed,
        "run_id": "reconstructed-heldout-v1",
        "contract_sha256": contract.sha256,
        "ood_config_sha256": protocol["ood_config_sha256"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "training_binding_sha256": binding_sha256,
        "training_binding": binding,
        "profile": "vlrb_adapted_v1",
        "architecture_and_hyperparameters": "identical_to_main_contract",
        "checkpoint_selection": {
            "view": "dev_seen",
            "monitor": "dev_hard_label_loss",
            "test_or_ood_allowed": False,
        },
        "training_target_scope": {
            "train_statistics": "train_seen_only",
            "train_labels": "train_seen_only",
            "dev_labels": "dev_seen_using_train_seen_priority",
            "held_out_train_or_dev_y_used": False,
            "test_y_or_c_used": False,
        },
        "roles": {
            name: {
                "count": len(view.rows[name]),
                "row_indices_sha256": row_indices_sha256(view.rows[name]),
                "manifest_sha256": protocol["views"][name]["sha256"],
            }
            for name in ("train_seen", "dev_seen")
        },
        "held_out_datasets": protocol["held_out_datasets_by_domain"],
    }


def _existing_completion(
    output: Path,
    *,
    backbone: str,
    seed: int,
    binding_sha256: str,
) -> dict[str, Any] | None:
    path = output / "OOD_TRAINING_COMPLETE.json"
    if not path.is_file():
        return None
    value = load_strict_json(path)
    checkpoint = output / "checkpoints/best/checkpoint.pt"
    if (
        value.get("passed") is not True
        or value.get("backbone") != backbone
        or value.get("seed") != seed
        or value.get("training_binding_sha256") != binding_sha256
        or not checkpoint.is_file()
        or value.get("best_checkpoint_sha256") != checkpoint_sha256(checkpoint)
    ):
        raise RuntimeError(f"stale or invalid OOD training completion: {path}")
    return value


def _train_bert(
    contract: ExperimentContract,
    data: Any,
    view: Any,
    train_labels: Any,
    dev_labels: Any,
    *,
    seed: int,
    output: Path,
    binding_sha256: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    profile = contract.values["profiles"]["vlrb_adapted_v1"]["bert"]
    epochs = int(profile["epochs"])
    batch_size = int(profile["batch_size_per_gpu"])
    max_length = int(contract.values["models"]["bert"]["max_length"])
    input_verification = verify_pinned_input_content(
        contract, model_names=("bert",), include_visual_assets=False
    )
    pinned_input = input_content_summary(contract)
    model_path = effective_model_path(contract, "bert")
    require_model_identity(
        model_path,
        expected_model_type="bert",
        expected_architecture="BertForMaskedLM",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=True, local_files_only=True
    )
    id2label = {index: model_id for index, model_id in enumerate(data.model_order)}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=len(data.model_order),
        id2label=id2label,
        label2id={model_id: index for index, model_id in id2label.items()},
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    model.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("OOD BERT is not fully trainable")
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)

    train_rows = tuple(view.rows["train_seen"])
    dev_rows = tuple(view.rows["dev_seen"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BertTextDataset(
            [data.samples[row] for row in train_rows],
            train_labels.labels,
            tokenizer,
            max_length,
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        BertTextDataset(
            [data.samples[row] for row in dev_rows],
            dev_labels.labels,
            tokenizer,
            max_length,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = _optimizer(
        model, float(profile["learning_rate"]), float(profile["weight_decay"])
    )
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * float(profile["warmup_ratio"]))
    scheduler = _linear_warmup_decay_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    manager = DevLossCheckpointManager(output / "checkpoints")
    stopper = DevLossEarlyStopping(
        patience=int(profile["patience"]), min_epochs=int(profile["min_epochs"])
    )
    history: list[dict[str, Any]] = []
    history_path = output / "training_history.jsonl"
    global_step = 0
    before_checksum = parameter_summary_checksum(model.bert)
    initial_probe = next(model.bert.parameters()).detach().cpu().clone()
    warnings_seen: list[str] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        trainable_count = 0
        hard_correct = 0
        routed_correct = 0
        rows_seen = 0
        gradient_norms: list[float] = []
        skipped_steps = 0
        for batch in train_loader:
            rows = batch.pop("row_index").numpy()
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits
                loss, count = masked_cross_entropy(logits, labels)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scaler.scale(loss).backward()
            for item in caught:
                message = str(item.message)
                if message not in warnings_seen:
                    warnings_seen.append(message)
            scaler.unscale_(optimizer)
            gradient_norm = _gradient_norm(model)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            skipped = float(scaler.get_scale()) < scale_before
            if skipped:
                skipped_steps += 1
            elif not math.isfinite(gradient_norm):
                raise RuntimeError("non-finite OOD BERT gradient was applied")
            if math.isfinite(gradient_norm):
                gradient_norms.append(gradient_norm)
            scheduler.step()
            global_step += 1
            predictions = logits.detach().argmax(dim=-1)
            if count:
                valid = labels != -100
                loss_sum += float(loss.detach().cpu()) * count
                hard_correct += int((predictions[valid] == labels[valid]).sum().item())
                trainable_count += count
            routed_correct += int(data.y[rows, predictions.cpu().numpy()].sum())
            rows_seen += len(rows)
        dev_metrics = _evaluate_dev(model, dev_loader, device, data)
        record = {
            "epoch": epoch,
            "train_hard_label_loss": loss_sum / max(1, trainable_count),
            "train_hard_label_accuracy": hard_correct / max(1, trainable_count),
            "train_routed_accuracy": routed_correct / max(1, rows_seen),
            "trainable_samples": trainable_count,
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
            "amp_optimizer_steps_skipped": skipped_steps,
            "learning_rate": float(scheduler.get_last_lr()[0]),
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
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            seed=seed,
            config_sha256=binding_sha256,
            extra={
                "backbone": "bert",
                "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
                "train_view": "train_seen",
                "dev_view": "dev_seen",
                "held_out_y_used": False,
                "input_content": pinned_input["models"]["bert"],
            },
        )
        _write_history(history_path, history)
        print(json.dumps({"backbone": "bert", "seed": seed, **record}, sort_keys=True))
        if stop:
            break

    after_checksum = parameter_summary_checksum(model.bert)
    if torch.equal(initial_probe, next(model.bert.parameters()).detach().cpu()):
        raise RuntimeError("OOD BERT backbone parameters did not update")
    if before_checksum == after_checksum:
        raise RuntimeError("OOD BERT parameter checksum did not change")
    best_path = output / "checkpoints/best/checkpoint.pt"
    checkpoint = load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=binding_sha256,
        map_location=device,
    )
    probe = [data.samples[row] for row in dev_rows[:2]]
    before = predict_bert_logits(
        model, tokenizer, probe, max_length=max_length, batch_size=batch_size, device=device
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=binding_sha256,
        map_location=device,
    )
    after = predict_bert_logits(
        model, tokenizer, probe, max_length=max_length, batch_size=batch_size, device=device
    )
    if not np.array_equal(before, after):
        raise RuntimeError("OOD BERT checkpoint round-trip changed probe predictions")
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "epochs_completed": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "best_dev_hard_label_loss": float(checkpoint["best_dev_hard_label_loss"]),
        "global_steps": global_step,
        "fully_trainable": True,
        "checkpoint_round_trip_prediction_equal": True,
        "input_content_verification": input_verification,
        "runtime_warnings": warnings_seen,
    }


def _train_head(
    contract: ExperimentContract,
    data: Any,
    view: Any,
    train_labels: Any,
    dev_labels: Any,
    *,
    backbone: str,
    seed: int,
    output: Path,
    binding_sha256: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    profile = contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"]
    cache_root = full_feature_cache_root(contract, backbone)
    features_np, records, feature_manifest, integrity = load_full_feature_cache(
        cache_root,
        dataset_root=contract.dataset_root,
        expected_contract_sha256=contract.sha256,
        expected_input_content=feature_input_content_fields(contract, backbone),
    )
    position_by_row = {
        int(record["row_index"]): position for position, record in enumerate(records)
    }
    if len(position_by_row) != len(records):
        raise RuntimeError("OOD feature cache contains duplicate rows")
    if any(
        data.samples[int(record["row_index"])].sample_id != str(record["sample_id"])
        for record in records
    ):
        raise RuntimeError("OOD feature cache sample IDs differ from strict registry")
    train_rows = tuple(view.rows["train_seen"])
    dev_rows = tuple(view.rows["dev_seen"])
    missing = set(train_rows + dev_rows).difference(position_by_row)
    if missing:
        raise RuntimeError(f"seen train/dev feature cache is missing {len(missing)} rows")
    train_positions = np.asarray([position_by_row[row] for row in train_rows], dtype=np.int64)
    dev_positions = np.asarray([position_by_row[row] for row in dev_rows], dtype=np.int64)
    train_features = torch.from_numpy(
        np.asarray(features_np[train_positions], dtype=np.float32)
    )
    dev_features = torch.from_numpy(np.asarray(features_np[dev_positions], dtype=np.float32))
    train_label_tensor = torch.from_numpy(
        np.asarray(train_labels.labels, dtype=np.int64).copy()
    )
    dev_label_tensor = torch.from_numpy(
        np.asarray(dev_labels.labels, dtype=np.int64).copy()
    )
    train_row_tensor = torch.tensor(train_rows, dtype=torch.long)
    dev_row_tensor = torch.tensor(dev_rows, dtype=torch.long)
    device = torch.device("cuda")
    model = AuditedBenchCoEHead(
        hidden_dim=int(features_np.shape[1]),
        num_experts=len(data.model_order),
        dropout=float(profile["dropout"]),
    ).to(device)
    expected_names = (
        "layer_norm.weight",
        "layer_norm.bias",
        "classifier.weight",
        "classifier.bias",
    )
    if trainable_parameter_names(model) != expected_names:
        raise RuntimeError("OOD head trainable parameter scope is not LN+Linear")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimized != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise RuntimeError("OOD head optimizer parameter scope mismatch")
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(profile["batch_size_per_gpu"])
    train_loader = DataLoader(
        TensorDataset(train_features, train_label_tensor, train_row_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    dev_loader = DataLoader(
        TensorDataset(dev_features, dev_label_tensor, dev_row_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    bounds = load_strict_json(contract.dataset_root / "data/matrices/cost_bounds.json")
    manager = DevLossCheckpointManager(output / "checkpoints")
    stopper = DevLossEarlyStopping(
        patience=int(profile["patience"]), min_epochs=int(profile["min_epochs"])
    )
    history: list[dict[str, Any]] = []
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    global_step = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(profile["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        valid_count = 0
        hard_correct = 0
        routed_correct = 0
        rows_seen = 0
        for batch_features, batch_labels, batch_rows in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss, count = masked_cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            predictions = logits.detach().argmax(dim=-1)
            if count:
                valid = batch_labels != -100
                loss_sum += float(loss.detach().cpu()) * count
                hard_correct += int((predictions[valid] == batch_labels[valid]).sum().item())
                valid_count += count
            rows_np = batch_rows.numpy()
            routed_correct += int(data.y[rows_np, predictions.cpu().numpy()].sum())
            rows_seen += len(rows_np)
        dev_metrics = _head_eval(
            model,
            dev_loader,
            device,
            data,
            float(bounds["cmin"]),
            float(bounds["cmax"]),
        )
        record = {
            "epoch": epoch,
            "train_hard_label_loss": loss_sum / max(1, valid_count),
            "train_hard_label_accuracy": hard_correct / max(1, valid_count),
            "train_routed_accuracy": routed_correct / max(1, rows_seen),
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
            seed=seed,
            config_sha256=binding_sha256,
            feature_manifest_sha256=str(feature_manifest["manifest_sha256"]),
            extra={
                "backbone": backbone,
                "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
                "train_view": "train_seen",
                "dev_view": "dev_seen",
                "held_out_y_used": False,
                "feature_signature": feature_manifest["signature"],
                "input_content": feature_input_content_fields(contract, backbone),
            },
        )
        _write_history(output / "training_history.jsonl", history)
        print(json.dumps({"backbone": backbone, "seed": seed, **record}, sort_keys=True))
        if stop:
            break

    changed = {
        name: not torch.equal(initial[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    }
    if not all(changed.values()):
        raise RuntimeError(f"not every OOD LN/Linear tensor changed: {changed}")
    best_path = output / "checkpoints/best/checkpoint.pt"
    checkpoint = load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=binding_sha256,
        expected_feature_manifest_sha256=str(feature_manifest["manifest_sha256"]),
        map_location=device,
    )
    probe = dev_features[:8]
    before = _predict(model, probe, batch_size, device)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=binding_sha256,
        expected_feature_manifest_sha256=str(feature_manifest["manifest_sha256"]),
        map_location=device,
    )
    after = _predict(model, probe, batch_size, device)
    if not np.array_equal(before, after):
        raise RuntimeError("OOD head checkpoint round-trip changed probe predictions")
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "epochs_completed": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "best_dev_hard_label_loss": float(checkpoint["best_dev_hard_label_loss"]),
        "global_steps": global_step,
        "trainable_parameter_names": list(expected_names),
        "backbone_parameters_present_in_optimizer": 0,
        "checkpoint_round_trip_prediction_equal": True,
        "feature_cache_integrity": integrity,
        "feature_cache_reuse_caveat": (
            "The pretrained backbone may have seen these benchmark datasets."
        ),
    }


def run_training(args: argparse.Namespace, contract: ExperimentContract) -> int:
    if args.backbone not in {"bert", "tinyllava", "qwen3vl"}:
        raise ValueError(f"unknown OOD backbone: {args.backbone}")
    if args.seed not in tuple(contract.values["reproducibility"]["seeds"]):
        raise ValueError(f"seed is outside the approved set: {args.seed}")
    force_offline_environment()
    gate = assert_ood_execution_gate(contract, phase="train", gate_path=args.gate)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("OOD training requires exactly one CUDA_VISIBLE_DEVICES GPU")
    seed_info = seed_everything(args.seed)
    data = load_vlrb_strict(args.dataset_dir, contract=contract)
    protocol, view = load_and_validate_protocol(
        contract,
        data,
        config_path=args.ood_config,
        protocol_path=args.protocol,
        validate_training_statistics=True,
    )
    binding_sha256, binding = ood_training_binding(contract, protocol)
    output = ood_run_dir(contract, args.backbone, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    existing = _existing_completion(
        output,
        backbone=args.backbone,
        seed=args.seed,
        binding_sha256=binding_sha256,
    )
    if existing is not None:
        print(json.dumps({"status": "already_complete", **existing}, sort_keys=True))
        return 0
    train_accuracy, train_labels, dev_labels = build_ood_training_labels(data, view)
    stored_accuracy = np.asarray(
        protocol["training_statistics"]["expert_accuracy"], dtype=np.float64
    )
    if not np.array_equal(train_accuracy, stored_accuracy):
        raise RuntimeError("live train_seen expert statistics differ from OOD protocol")
    manifest = _base_manifest(
        contract,
        backbone=args.backbone,
        seed=args.seed,
        protocol=protocol,
        view=view,
        binding_sha256=binding_sha256,
        binding=binding,
    )
    manifest["seed_settings"] = seed_info
    manifest["execution_gate_sha256"] = gate["gate_file_sha256"]
    manifest["training_statistics"] = {
        "source_view": "train_seen",
        "strongest_expert_index": int(np.argmax(train_accuracy)),
        "strongest_expert_id": data.model_order[int(np.argmax(train_accuracy))],
        "trainable_rows": int(train_labels.is_trainable.sum()),
        "all_wrong_rows": int((~train_labels.is_trainable).sum()),
        "held_out_y_used": False,
    }
    atomic_json(output / "OOD_RUN_MANIFEST.json", manifest)
    result = (
        _train_bert(
            contract,
            data,
            view,
            train_labels,
            dev_labels,
            seed=args.seed,
            output=output,
            binding_sha256=binding_sha256,
            protocol=protocol,
        )
        if args.backbone == "bert"
        else _train_head(
            contract,
            data,
            view,
            train_labels,
            dev_labels,
            backbone=args.backbone,
            seed=args.seed,
            output=output,
            binding_sha256=binding_sha256,
            protocol=protocol,
        )
    )
    checkpoint = output / "checkpoints/best/checkpoint.pt"
    completion = {
        "schema_version": 1,
        "kind": "prompt15_ood_training_completion",
        "passed": True,
        "backbone": args.backbone,
        "seed": args.seed,
        "training_binding_sha256": binding_sha256,
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "best_checkpoint": checkpoint.relative_to(contract.dataset_root).as_posix(),
        "best_checkpoint_sha256": checkpoint_sha256(checkpoint),
        "selection_view": "dev_seen",
        "selection_monitor": "dev_hard_label_loss",
        "test_or_ood_used_for_selection": False,
        "prediction_generation": "not_performed_in_training_process",
        "resource": result,
    }
    atomic_json(output / "OOD_TRAINING_COMPLETE.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("bert", "tinyllava", "qwen3vl"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--contract", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--ood-config", default=str(OOD_CONFIG_PATH))
    parser.add_argument("--protocol", default=str(OOD_PROTOCOL_PATH))
    parser.add_argument("--gate", default=str(OOD_GATE_PATH))
    args = parser.parse_args()
    return run_training(args, load_contract(args.contract))


if __name__ == "__main__":
    raise SystemExit(main())
