"""Common loss and dev-only early-stopping primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits/labels shape mismatch")
    valid = labels != ignore_index
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits, labels, ignore_index=ignore_index), count


@dataclass
class DevLossEarlyStopping:
    patience: int
    min_epochs: int
    tolerance: float = 1e-6
    best_loss: float = float("inf")
    bad_epochs: int = 0

    def update(self, *, epoch: int, dev_hard_label_loss: float) -> tuple[bool, bool]:
        improved = dev_hard_label_loss < self.best_loss - self.tolerance
        if improved:
            self.best_loss = float(dev_hard_label_loss)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = epoch >= self.min_epochs and self.bad_epochs >= self.patience
        return improved, should_stop
