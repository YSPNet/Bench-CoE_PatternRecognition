"""Cost-aware query-routing targets and masked losses.

This module deliberately has no filesystem access.  Callers must supply rows
from the frozen train/dev protocol and explicit observation masks.  In
particular, a missing observation is never converted to an incorrect answer or
zero cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .errors import DataIntegrityError


TARGET_MODES = (
    "strongest_correct_hard",
    "cheapest_correct_hard",
    "uniform_correct_soft",
    "cost_correct_soft",
    "cheapest_uniform_limit",
)


@dataclass(frozen=True)
class CostAwareTargets:
    """Immutable target matrix plus explicit selection-loss validity."""

    probabilities: np.ndarray
    hard_labels: np.ndarray
    selection_loss_mask: np.ndarray
    correct_expert_count: np.ndarray
    mode: str
    lambda_cost: float
    kappa: float

    @classmethod
    def create(
        cls,
        *,
        probabilities: np.ndarray,
        hard_labels: np.ndarray,
        selection_loss_mask: np.ndarray,
        correct_expert_count: np.ndarray,
        mode: str,
        lambda_cost: float,
        kappa: float,
    ) -> "CostAwareTargets":
        for value in (
            probabilities,
            hard_labels,
            selection_loss_mask,
            correct_expert_count,
        ):
            value.flags.writeable = False
        return cls(
            probabilities=probabilities,
            hard_labels=hard_labels,
            selection_loss_mask=selection_loss_mask,
            correct_expert_count=correct_expert_count,
            mode=mode,
            lambda_cost=float(lambda_cost),
            kappa=float(kappa),
        )


def _validated_inputs(
    y: np.ndarray,
    costs: np.ndarray,
    correctness_observed: np.ndarray | None,
    cost_observed: np.ndarray | None,
    fit_global_accuracy: np.ndarray,
    model_order: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outcomes = np.asarray(y)
    values = np.asarray(costs, dtype=np.float64)
    if outcomes.ndim != 2 or values.shape != outcomes.shape:
        raise DataIntegrityError(
            f"Y/C must be aligned 2-D matrices, got {outcomes.shape}/{values.shape}"
        )
    if not np.isin(outcomes, (0, 1)).all():
        raise DataIntegrityError("Y must contain only binary correctness values")
    observed_y = (
        np.ones_like(outcomes, dtype=bool)
        if correctness_observed is None
        else np.asarray(correctness_observed, dtype=bool)
    )
    observed_c = (
        np.ones_like(outcomes, dtype=bool)
        if cost_observed is None
        else np.asarray(cost_observed, dtype=bool)
    )
    if observed_y.shape != outcomes.shape or observed_c.shape != outcomes.shape:
        raise DataIntegrityError("observation masks must have the same shape as Y/C")
    observed_cost_values = values[observed_c]
    if not np.isfinite(observed_cost_values).all() or (observed_cost_values < 0).any():
        raise DataIntegrityError("observed costs must be finite and non-negative")
    accuracy = np.asarray(fit_global_accuracy, dtype=np.float64)
    if accuracy.shape != (outcomes.shape[1],) or not np.isfinite(accuracy).all():
        raise DataIntegrityError("fit_global_accuracy must be finite with one value per expert")
    if model_order is not None:
        order = tuple(str(item) for item in model_order)
        if len(order) != outcomes.shape[1] or len(set(order)) != len(order):
            raise DataIntegrityError("model_order must uniquely match the expert columns")
    return outcomes, values, observed_y, observed_c, accuracy


def fit_cost_scale(
    costs: np.ndarray,
    fit_rows: Sequence[int] | np.ndarray,
    *,
    cost_observed: np.ndarray | None = None,
) -> tuple[float, str]:
    """Return the fit-only median positive cost and an auditable status."""

    matrix = np.asarray(costs, dtype=np.float64)
    if matrix.ndim != 2:
        raise DataIntegrityError("C must be a 2-D matrix")
    rows = np.asarray(fit_rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise DataIntegrityError("fit rows must be a non-empty unique sequence")
    if (rows < 0).any() or (rows >= len(matrix)).any():
        raise DataIntegrityError("fit rows contain an out-of-range index")
    observed = (
        np.ones_like(matrix, dtype=bool)
        if cost_observed is None
        else np.asarray(cost_observed, dtype=bool)
    )
    if observed.shape != matrix.shape:
        raise DataIntegrityError("cost observation mask shape mismatch")
    selected = matrix[rows][observed[rows]]
    if selected.size and (not np.isfinite(selected).all() or (selected < 0).any()):
        raise DataIntegrityError("fit observed costs must be finite and non-negative")
    positive = selected[selected > 0]
    if not len(positive):
        return 1.0, "ALL_ZERO_COST"
    return float(np.median(positive)), "FIT_MEDIAN_POSITIVE"


def build_targets(
    y: np.ndarray,
    costs: np.ndarray,
    correctness_observed: np.ndarray | None,
    fit_global_accuracy: np.ndarray,
    mode: str,
    lambda_cost: float,
    kappa: float,
    *,
    cost_observed: np.ndarray | None = None,
    model_order: Sequence[str] | None = None,
    ignore_index: int = -100,
) -> CostAwareTargets:
    """Build one of the four fair Route-1 targets (plus its limit branch).

    Canonical column order is the final deterministic tie-break.  Cost-aware
    modes fail closed when a correct expert's cost is unobserved.
    """

    if mode not in TARGET_MODES:
        raise ValueError(f"unsupported target mode {mode!r}; expected one of {TARGET_MODES}")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if np.isnan(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be non-negative")
    if np.isinf(lambda_cost) and mode != "cheapest_uniform_limit":
        raise ValueError("infinite lambda requires mode='cheapest_uniform_limit'")

    outcomes, values, observed_y, observed_c, accuracy = _validated_inputs(
        y,
        costs,
        correctness_observed,
        cost_observed,
        fit_global_accuracy,
        model_order,
    )
    correct = (outcomes == 1) & observed_y
    counts = correct.sum(axis=1, dtype=np.int64).astype(np.int16)
    valid = counts > 0
    probabilities = np.zeros(outcomes.shape, dtype=np.float64)
    hard = np.full(outcomes.shape[0], int(ignore_index), dtype=np.int64)
    canonical = np.arange(outcomes.shape[1], dtype=np.int64)

    for row in np.flatnonzero(valid):
        columns = np.flatnonzero(correct[row])
        if mode in {
            "cheapest_correct_hard",
            "cost_correct_soft",
            "cheapest_uniform_limit",
        } and not observed_c[row, columns].all():
            raise DataIntegrityError(
                f"row {row} has an unobserved correct-expert cost required by {mode}"
            )

        if mode == "strongest_correct_hard":
            chosen = int(columns[np.lexsort((canonical[columns], -accuracy[columns]))[0]])
            hard[row] = chosen
            probabilities[row, chosen] = 1.0
        elif mode == "cheapest_correct_hard":
            # lowest cost, then highest fit accuracy, then canonical index
            chosen = int(
                columns[
                    np.lexsort(
                        (canonical[columns], -accuracy[columns], values[row, columns])
                    )[0]
                ]
            )
            hard[row] = chosen
            probabilities[row, chosen] = 1.0
        elif mode == "uniform_correct_soft":
            probabilities[row, columns] = 1.0 / len(columns)
            hard[row] = int(columns[np.lexsort((canonical[columns], -accuracy[columns]))[0]])
        elif mode == "cheapest_uniform_limit":
            minimum = values[row, columns].min()
            cheapest = columns[values[row, columns] == minimum]
            probabilities[row, cheapest] = 1.0 / len(cheapest)
            hard[row] = int(
                cheapest[np.lexsort((canonical[cheapest], -accuracy[cheapest]))[0]]
            )
        else:
            # Subtracting the maximum log weight is a stable logsumexp
            # normalization without evaluating exp(-inf) for a special limit.
            log_weights = -float(lambda_cost) * values[row, columns] / float(kappa)
            shifted = log_weights - np.max(log_weights)
            weights = np.exp(shifted)
            weights /= weights.sum(dtype=np.float64)
            probabilities[row, columns] = weights
            maximum = weights.max()
            tied = columns[weights == maximum]
            hard[row] = int(tied[np.lexsort((canonical[tied], -accuracy[tied]))[0]])

    if valid.any() and not np.allclose(probabilities[valid].sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("valid target rows must sum to one")
    if (probabilities[~valid] != 0).any():
        raise AssertionError("all-wrong target rows must remain zero and masked")
    return CostAwareTargets.create(
        probabilities=probabilities,
        hard_labels=hard,
        selection_loss_mask=valid.astype(bool),
        correct_expert_count=counts,
        mode=mode,
        lambda_cost=float(lambda_cost),
        kappa=float(kappa),
    )


def masked_soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_rows: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Mean soft cross entropy over valid rows, safe for an all-wrong batch."""

    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits/targets shape mismatch")
    if valid_rows.shape != (logits.shape[0],):
        raise ValueError("valid_rows shape mismatch")
    mask = valid_rows.to(dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    selected_targets = targets[mask]
    if not torch.allclose(
        selected_targets.sum(dim=-1),
        torch.ones(count, dtype=selected_targets.dtype, device=selected_targets.device),
        atol=1e-5,
    ):
        raise ValueError("valid soft targets must sum to one")
    loss = -(selected_targets * F.log_softmax(logits[mask], dim=-1)).sum(dim=-1).mean()
    return loss, count


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    binary_targets: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Element-normalized independent BCE for Route 2 quality prediction."""

    if logits.ndim != 2 or binary_targets.shape != logits.shape or observed.shape != logits.shape:
        raise ValueError("logits/targets/observed shape mismatch")
    mask = observed.to(dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    target = binary_targets.to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask]), count
