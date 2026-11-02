"""Offline Query-BERT full-parameter fine-tuning on frozen VL-RouterBench rows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .backbones.utils import force_offline_environment, parameter_summary_checksum, require_model_identity
from .checkpoints import DevLossCheckpointManager, load_checkpoint
from .config import ExperimentContract, effective_model_path
from .data import load_vlrb_strict
from .labels import build_diagnostic_labels_for_split, compute_train_expert_accuracy
from .input_content import input_content_summary, verify_pinned_input_content
from .manifest import build_manifest, write_manifest
from .reproducibility import seed_everything
from .selection import select_rows
from .trainer import DevLossEarlyStopping, masked_cross_entropy


class BertTextDataset(Dataset):
    def __init__(self, samples, labels: Sequence[int], tokenizer, max_length: int):
        if len(samples) != len(labels):
            raise ValueError("BERT samples/labels length mismatch")
        self.samples = tuple(samples)
        self.labels = tuple(int(label) for label in labels)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "row_index": torch.tensor(sample.row_index, dtype=torch.long),
        }


def _optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(marker in name for marker in no_decay)
            ],
            "weight_decay": float(weight_decay),
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(marker in name for marker in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=float(learning_rate))
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimized != trainable:
        raise RuntimeError("BERT optimizer parameter set does not equal all trainable model parameters")
    return optimizer


def _linear_warmup_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Transformers-compatible linear warmup/decay without Trainer/PEFT imports."""

    if total_steps <= 0 or warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(f"invalid warmup/total steps: {warmup_steps}/{total_steps}")

    def factor(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def _evaluate_dev(model, loader, device, data) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_trainable = 0
    hard_correct = 0
    routed_correct = 0
    rows_seen = 0
    for batch in loader:
        rows = batch.pop("row_index").numpy()
        labels = batch.pop("labels").to(device)
        inputs = {name: value.to(device) for name, value in batch.items()}
        logits = model(**inputs).logits
        loss, count = masked_cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)
        if count:
            valid = labels != -100
            total_loss += float(loss.item()) * count
            hard_correct += int((predictions[valid] == labels[valid]).sum().item())
            total_trainable += count
        predictions_cpu = predictions.cpu().numpy()
        routed_correct += int(data.y[rows, predictions_cpu].sum())
        rows_seen += len(rows)
    if total_trainable == 0:
        raise RuntimeError("dev subset contains no trainable hard labels")
    return {
        "dev_hard_label_loss": total_loss / total_trainable,
        "dev_hard_label_accuracy": hard_correct / total_trainable,
        "dev_routed_accuracy": routed_correct / rows_seen,
        "dev_trainable_samples": total_trainable,
        "dev_samples": rows_seen,
    }


@torch.no_grad()
def predict_bert_logits(model, tokenizer, samples, *, max_length: int, batch_size: int, device) -> np.ndarray:
    # This interface intentionally has no Y/C argument.
    dummy_labels = np.zeros(len(samples), dtype=np.int64)
    loader = DataLoader(
        BertTextDataset(samples, dummy_labels, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    outputs = []
    for batch in loader:
        batch.pop("row_index")
        batch.pop("labels")
        inputs = {name: value.to(device) for name, value in batch.items()}
        outputs.append(model(**inputs).logits.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _gradient_norm(model: torch.nn.Module) -> float:
    # Accumulate diagnostics in float64 so valid fp16/fp32 gradients do not
    # overflow merely while being summarized for JSON provenance.
    squares = [
        parameter.grad.detach().double().norm(2).square()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0


def run_bert_training(args, contract: ExperimentContract) -> int:
    force_offline_environment()
    seed_info = seed_everything(args.seed)
    data = load_vlrb_strict(args.dataset_dir, contract=contract)
    train_rows = select_rows(data, split="train", max_samples=args.max_samples)
    dev_rows = select_rows(data, split="dev", max_samples=args.max_samples)
    train_accuracy = compute_train_expert_accuracy(
        data.y, data.split_to_rows["train"], data.model_order, expected_model_order=data.model_order
    )
    train_labels = build_diagnostic_labels_for_split(
        data.y, train_rows, train_accuracy, data.model_order, expected_model_order=data.model_order
    )
    dev_labels = build_diagnostic_labels_for_split(
        data.y, dev_rows, train_accuracy, data.model_order, expected_model_order=data.model_order
    )

    profile = contract.values["profiles"]["vlrb_adapted_v1"]["bert"]
    is_full = os.environ.get("RUN_FULL", "0") == "1" and args.max_samples is None
    epochs = int(args.epochs if args.epochs is not None else (profile["epochs"] if is_full else 1))
    batch_size = int(profile["batch_size_per_gpu"])
    max_length = int(contract.values["models"]["bert"]["max_length"])
    input_content_verification = (
        verify_pinned_input_content(
            contract,
            model_names=("bert",),
            include_visual_assets=False,
        )
        if is_full
        else None
    )
    pinned_input_content = input_content_summary(contract)
    model_path = effective_model_path(contract, "bert")
    require_model_identity(model_path, expected_model_type="bert", expected_architecture="BertForMaskedLM")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
    id2label = {index: model for index, model in enumerate(data.model_order)}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=len(data.model_order),
        id2label=id2label,
        label2id={model: index for index, model in id2label.items()},
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    model.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("BERT is not fully trainable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_samples = [data.samples[row] for row in train_rows]
    dev_samples = [data.samples[row] for row in dev_rows]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        BertTextDataset(train_samples, train_labels.labels, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        BertTextDataset(dev_samples, dev_labels.labels, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = _optimizer(model, float(profile["learning_rate"]), float(profile["weight_decay"]))
    total_steps = max(1, epochs * len(train_loader))
    # The adapted implementation uses integer truncation. The pinned candidate
    # uses ceil; both equal 201 for the archived 3,350-step full schedule, but
    # truncation is important for the one-step smoke run because ceil would create a
    # zero-learning-rate-only run and falsely report that full fine-tuning did
    # not update the BERT backbone.
    warmup_steps = int(total_steps * float(profile["warmup_ratio"]))
    scheduler = _linear_warmup_decay_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mode_root = "runs/bert/vlrb_adapted_v1" if is_full else "smoke/bert"
    output = contract.dataset_root / str(contract.values["safety"]["output_root"]) / mode_root / args.run_id / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(contract.plain(), sort_keys=False), encoding="utf-8"
    )
    write_manifest(
        output / "manifest.json",
        build_manifest(
            contract,
            seed=args.seed,
            run_id=args.run_id,
            input_content_verification=input_content_verification,
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
            scheduler=scheduler,
            expected_config_sha256=contract.sha256,
            restore_rng=True,
            map_location=device,
        )
        resume_state = checkpoint.get("extra", {}).get("resume_state", {})
        if "train_generator_state" not in resume_state or "grad_scaler_state" not in resume_state:
            raise RuntimeError("checkpoint predates exact-resume generator/scaler state")
        generator.set_state(resume_state["train_generator_state"])
        scaler.load_state_dict(resume_state["grad_scaler_state"])
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
    before_checksum = parameter_summary_checksum(model.bert)
    started = time.perf_counter()
    runtime_warnings: list[str] = []
    initial_state_probe = next(model.bert.parameters()).detach().cpu().clone()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        trainable_count = 0
        hard_correct = 0
        routed_correct = 0
        rows_seen = 0
        gradient_norms: list[float] = []
        gradient_norm_nonfinite_steps = 0
        amp_optimizer_steps_attempted = 0
        amp_optimizer_steps_skipped = 0
        amp_skipped_step_events: list[dict[str, Any]] = []
        amp_scale_start = float(scaler.get_scale())
        amp_scale_min = amp_scale_start
        amp_scale_max = amp_scale_start
        for batch in train_loader:
            rows = batch.pop("row_index").numpy()
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**inputs).logits
                loss, count = masked_cross_entropy(logits, labels)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scaler.scale(loss).backward()
            runtime_warnings.extend(
                str(item.message) for item in caught if str(item.message) not in runtime_warnings
            )
            scaler.unscale_(optimizer)
            gradient_norm = _gradient_norm(model)
            gradient_norm_is_finite = math.isfinite(gradient_norm)
            if gradient_norm_is_finite:
                gradient_norms.append(gradient_norm)
            else:
                gradient_norm_nonfinite_steps += 1
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer_step_skipped = bool(use_amp and scale_after < scale_before)
            amp_optimizer_steps_attempted += 1
            amp_scale_min = min(amp_scale_min, scale_before, scale_after)
            amp_scale_max = max(amp_scale_max, scale_before, scale_after)
            if optimizer_step_skipped:
                amp_optimizer_steps_skipped += 1
                amp_skipped_step_events.append(
                    {
                        "global_step_attempt": global_step + 1,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                        "gradient_norm_finite": gradient_norm_is_finite,
                    }
                )
            if not gradient_norm_is_finite and not optimizer_step_skipped:
                raise RuntimeError(
                    "non-finite BERT gradient was not rejected by GradScaler"
                )
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
        gradient_norm_status = (
            "all_finite"
            if gradient_norm_nonfinite_steps == 0
            else "nonfinite_observed_only_on_amp_skipped_steps"
        )
        record = {
            "epoch": epoch,
            "train_hard_label_loss": loss_sum / max(1, trainable_count),
            "train_hard_label_accuracy": hard_correct / max(1, trainable_count),
            "train_routed_accuracy": routed_correct / max(1, rows_seen),
            "trainable_samples": trainable_count,
            "gradient_norm_mean": (
                float(np.mean(gradient_norms)) if gradient_norms else None
            ),
            "gradient_norm_max": (
                float(np.max(gradient_norms)) if gradient_norms else None
            ),
            "gradient_norm_status": gradient_norm_status,
            "gradient_norm_finite_steps": len(gradient_norms),
            "gradient_norm_nonfinite_steps": gradient_norm_nonfinite_steps,
            "amp_diagnostics": {
                "enabled": use_amp,
                "optimizer_steps_attempted": amp_optimizer_steps_attempted,
                "optimizer_steps_applied": (
                    amp_optimizer_steps_attempted - amp_optimizer_steps_skipped
                ),
                "optimizer_steps_skipped": amp_optimizer_steps_skipped,
                "scale_start": amp_scale_start,
                "scale_end": float(scaler.get_scale()),
                "scale_min": amp_scale_min,
                "scale_max": amp_scale_max,
                "skipped_step_events": amp_skipped_step_events,
                "scheduler_advanced_on_every_attempt": True,
            },
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **dev_metrics,
        }
        history.append(record)
        _improved, should_stop = stopper.update(
            epoch=epoch, dev_hard_label_loss=float(dev_metrics["dev_hard_label_loss"])
        )
        manager.save(
            dev_hard_label_loss=float(dev_metrics["dev_hard_label_loss"]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            seed=args.seed,
            config_sha256=contract.sha256,
            extra={
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "input_content": {
                    "manifest_file_sha256": pinned_input_content["manifest_file_sha256"],
                    "manifest_sha256": pinned_input_content["manifest_sha256"],
                    "model": pinned_input_content["models"]["bert"],
                    "verified_for_full_run": input_content_verification is not None,
                },
                "resume_state": {
                    "train_generator_state": generator.get_state(),
                    "grad_scaler_state": scaler.state_dict(),
                    "early_stopping_bad_epochs": stopper.bad_epochs,
                },
            },
        )
        history_path.write_text(
            "".join(
                json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                for item in history
            ),
            encoding="utf-8",
        )
        print(json.dumps(record, sort_keys=True, allow_nan=False))
        if should_stop:
            break

    after_checksum = parameter_summary_checksum(model.bert)
    parameter_updated = not torch.equal(initial_state_probe, next(model.bert.parameters()).detach().cpu())
    if not parameter_updated or before_checksum == after_checksum:
        raise RuntimeError("BERT smoke invariant failed: backbone parameters did not update")
    best_path = output / "checkpoints/best/checkpoint.pt"
    load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=contract.sha256,
        map_location=device,
    )
    model.to(device)
    probe_samples = dev_samples[: min(2, len(dev_samples))]
    round_trip_before = predict_bert_logits(
        model,
        tokenizer,
        probe_samples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    load_checkpoint(
        best_path,
        model=model,
        expected_config_sha256=contract.sha256,
        map_location=device,
    )
    round_trip_after = predict_bert_logits(
        model,
        tokenizer,
        probe_samples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    checkpoint_round_trip_prediction_equal = bool(np.array_equal(round_trip_before, round_trip_after))
    if not checkpoint_round_trip_prediction_equal:
        raise RuntimeError("BERT checkpoint save/load changed fixed-probe predictions")
    elapsed = time.perf_counter() - started
    history_payload = "".join(
        json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in history
    )
    history_path.write_text(history_payload, encoding="utf-8")
    resource = {
        "elapsed_seconds": elapsed,
        "device": str(device),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "samples_per_second_train": len(train_rows) * len(history) / elapsed if elapsed else None,
        "backbone_checksum_before": before_checksum,
        "backbone_checksum_after": after_checksum,
        "backbone_updated": parameter_updated,
        "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        "optimizer_covers_all_trainable_parameters": True,
        "checkpoint_monitor": "dev_hard_label_loss",
        "checkpoint_round_trip_prediction_equal": checkpoint_round_trip_prediction_equal,
        "runtime_warnings": runtime_warnings,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "seed_settings": seed_info,
        "prediction_generation": {
            "status": "not_performed_in_training_process",
            "required_entry_point": "routers.bench_coe_query.cli freeze-run",
            "required_process_boundary": "target_free_producer_then_separate_evaluator",
        },
    }
    (output / "resource_usage.json").write_text(
        json.dumps(resource, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "resource": resource}, indent=2, sort_keys=True))
    return 0
