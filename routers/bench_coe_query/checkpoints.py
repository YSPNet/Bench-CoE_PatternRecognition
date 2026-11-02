"""Atomic last/best checkpoints selected exclusively by dev hard-label loss."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .errors import ContractError
from .reproducibility import capture_rng_state, restore_rng_state


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    best_dev_hard_label_loss: float,
    seed: int,
    config_sha256: str,
    feature_manifest_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "schema_version": 1,
        "monitor": "dev_hard_label_loss",
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_dev_hard_label_loss": float(best_dev_hard_label_loss),
        "seed": int(seed),
        "config_sha256": str(config_sha256),
        "feature_manifest_sha256": feature_manifest_sha256,
        "extra": dict(extra or {}),
    }
    output = Path(path)
    _atomic_torch_save(checkpoint, output)
    return {"path": str(output), "sha256": checkpoint_sha256(output)}


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_config_sha256: str | None = None,
    expected_feature_manifest_sha256: str | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("monitor") != "dev_hard_label_loss":
        raise ContractError("checkpoint monitor is not dev_hard_label_loss")
    if expected_config_sha256 is not None and checkpoint.get("config_sha256") != expected_config_sha256:
        raise ContractError("checkpoint/config hash mismatch")
    if (
        expected_feature_manifest_sha256 is not None
        and checkpoint.get("feature_manifest_sha256") != expected_feature_manifest_sha256
    ):
        raise ContractError("checkpoint/feature manifest hash mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None:
        if checkpoint.get("optimizer_state") is None:
            raise ContractError("checkpoint has no optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        if checkpoint.get("scheduler_state") is None:
            raise ContractError("checkpoint has no scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint


@dataclass
class DevLossCheckpointManager:
    output_dir: Path
    best_loss: float = float("inf")

    def save(
        self,
        *,
        dev_hard_label_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        global_step: int,
        seed: int,
        config_sha256: str,
        feature_manifest_sha256: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not torch.isfinite(torch.tensor(dev_hard_label_loss)):
            raise ContractError("dev_hard_label_loss must be finite")
        common = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_dev_hard_label_loss=min(self.best_loss, dev_hard_label_loss),
            seed=seed,
            config_sha256=config_sha256,
            feature_manifest_sha256=feature_manifest_sha256,
            extra=extra,
        )
        last = save_checkpoint(self.output_dir / "last/checkpoint.pt", **common)
        improved = dev_hard_label_loss < self.best_loss - 1e-6
        best = None
        if improved:
            self.best_loss = float(dev_hard_label_loss)
            common["best_dev_hard_label_loss"] = self.best_loss
            best = save_checkpoint(self.output_dir / "best/checkpoint.pt", **common)
        return {"last": last, "best": best, "improved": improved, "monitor": "dev_hard_label_loss"}
