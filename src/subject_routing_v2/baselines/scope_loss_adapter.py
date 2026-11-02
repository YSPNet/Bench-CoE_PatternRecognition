"""Formula-checked SCOPE CRM/RCCR loss adaptation for a lightweight head.

This module adapts equations (9)--(12) from SCOPE-Router to the frozen
Bench-CoE source protocol. It deliberately does not implement SCOPE's model
profiles or dual-tower architecture, so callers must describe it as a loss
adaptation rather than a full SCOPE reproduction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ImportError as error:  # pragma: no cover - guarded at construction
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = error
else:
    _TORCH_IMPORT_ERROR = None


class ScopeAdapterError(ValueError):
    """Raised when SCOPE targets or adapter inputs violate their contract."""


def _validated_outcomes(
    quality: Any,
    cost: Any,
    observed_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(quality, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)
    mask = (
        np.ones_like(y, dtype=bool)
        if observed_mask is None
        else np.asarray(observed_mask, dtype=bool)
    )
    if y.ndim != 2 or c.shape != y.shape or mask.shape != y.shape:
        raise ScopeAdapterError("quality, cost, and mask must be equal 2-D matrices")
    if not np.isfinite(y[mask]).all() or not np.isfinite(c[mask]).all():
        raise ScopeAdapterError("observed outcomes must be finite")
    if not np.isin(y[mask], (0.0, 1.0)).all():
        raise ScopeAdapterError("CRM adaptation requires binary correctness")
    if (c[mask] < 0).any():
        raise ScopeAdapterError("observed cost must be non-negative")
    return y, c, mask


def source_cost_alpha(
    quality: Any,
    cost: Any,
    observed_mask: Any | None = None,
) -> tuple[float, dict[str, Any]]:
    """Freeze an inverse-USD cost scale using only the supplied source rows.

    The robust scale is the median positive excess cost among correct experts
    relative to the cheapest correct expert on the same query. It makes the
    exponent dimensionless without copying another benchmark's price scale.
    """

    y, c, mask = _validated_outcomes(quality, cost, observed_mask)
    correct = mask & (y > 0.5)
    cheapest = np.min(np.where(correct, c, np.inf), axis=1)
    valid_rows = np.isfinite(cheapest)
    gaps = c - cheapest[:, None]
    positive = gaps[correct & valid_rows[:, None] & (gaps > 0.0)]
    fallback_used = False
    if positive.size:
        scale = float(np.median(positive))
    else:
        observed_costs = c[mask & (c > 0.0)]
        if not observed_costs.size:
            raise ScopeAdapterError("cannot derive a positive source cost scale")
        scale = float(np.median(observed_costs))
        fallback_used = True
    if not math.isfinite(scale) or scale <= 0.0:
        raise ScopeAdapterError("derived source cost scale is not positive finite")
    return 1.0 / scale, {
        "estimator": "reciprocal_median_positive_correct_excess_cost",
        "cost_unit": "usd_proxy_per_request",
        "alpha_unit": "inverse_usd_proxy_per_request",
        "positive_gap_cells": int(positive.size),
        "median_positive_gap_usd_proxy_per_request": scale,
        "fallback_to_observed_cost_median": fallback_used,
    }


def crm_relevance_targets(
    quality: Any,
    cost: Any,
    *,
    lambda_value: float,
    cost_alpha: float,
    observed_mask: Any | None = None,
) -> np.ndarray:
    """Equation (9): unnormalized cost-aware relevance in ``[0, 1]``.

    All-wrong rows remain all-zero. Unobserved cells are zero in the returned
    value and must still be excluded by the mask supplied to the loss.
    """

    y, c, mask = _validated_outcomes(quality, cost, observed_mask)
    if lambda_value < 0 or cost_alpha <= 0:
        raise ScopeAdapterError("lambda must be non-negative and cost_alpha positive")
    correct = mask & (y > 0.5)
    cheapest = np.min(np.where(correct, c, np.inf), axis=1)
    has_correct = np.isfinite(cheapest)
    targets = np.zeros_like(y, dtype=np.float64)
    if has_correct.any():
        row_cost = c[has_correct]
        minimum = cheapest[has_correct, None]
        if math.isinf(float(lambda_value)):
            relevance = np.equal(row_cost, minimum).astype(np.float64)
        else:
            exponent = -float(lambda_value) * float(cost_alpha) * (
                row_cost - minimum
            )
            relevance = np.exp(np.minimum(exponent, 0.0))
        targets[has_correct] = np.where(correct[has_correct], relevance, 0.0)
    return np.clip(targets, 0.0, 1.0).astype(np.float32)


def correct_set_soft_targets(
    quality: Any,
    cost: Any,
    *,
    lambda_value: float,
    cost_alpha: float,
    observed_mask: Any | None = None,
) -> np.ndarray:
    """VL-RouterBench row-normalized correct-set soft labels.

    Correct rows normalize equation-(9)-style weights to unit mass. Following
    the repository implementation, all-wrong rows fall back to the cheapest
    observed expert rather than becoming all-negative.
    """

    y, c, mask = _validated_outcomes(quality, cost, observed_mask)
    relevance = crm_relevance_targets(
        y,
        c,
        lambda_value=lambda_value,
        cost_alpha=cost_alpha,
        observed_mask=mask,
    ).astype(np.float64)
    mass = relevance.sum(axis=1)
    targets = np.zeros_like(relevance)
    has_correct = mass > 0.0
    targets[has_correct] = relevance[has_correct] / mass[has_correct, None]
    for row in np.flatnonzero(~has_correct):
        observed = np.flatnonzero(mask[row])
        if not observed.size:
            raise ScopeAdapterError("all-wrong row has no observed cost fallback")
        cheapest = observed[np.argmin(c[row, observed])]
        targets[row, cheapest] = 1.0
    return targets.astype(np.float32)


def masked_sigmoid_bce_with_logits(
    logits: Any,
    targets: Any,
    observed_mask: Any,
) -> Any:
    """Equation (10), retaining independent sigmoid scores per expert."""

    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required") from _TORCH_IMPORT_ERROR
    target_tensor = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    mask = torch.as_tensor(observed_mask, dtype=torch.bool, device=logits.device)
    if logits.shape != target_tensor.shape or mask.shape != logits.shape:
        raise ScopeAdapterError("CRM logits, targets, and mask must match")
    if not bool(mask.any()):
        raise ScopeAdapterError("CRM needs at least one observed pair")
    losses = functional.binary_cross_entropy_with_logits(
        logits, target_tensor, reduction="none"
    )
    return losses[mask].mean()


def routing_consistency_loss(
    query_features: Any,
    relevance_targets: Any,
    *,
    temperature: float,
) -> Any:
    """Equation (11), with exact empty-valid-anchor handling."""

    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required") from _TORCH_IMPORT_ERROR
    if temperature <= 0:
        raise ScopeAdapterError("RCCR temperature must be positive")
    query = torch.as_tensor(query_features)
    target = torch.as_tensor(
        relevance_targets, dtype=query.dtype, device=query.device
    ).clamp_min(0.0)
    if query.ndim != 2 or target.ndim != 2 or query.shape[0] != target.shape[0]:
        raise ScopeAdapterError("RCCR query/target batch shapes are inconsistent")
    mass = target.sum(dim=1, keepdim=True)
    valid = mass.squeeze(1) > 0.0
    if int(valid.sum().item()) <= 1:
        return query.sum() * 0.0
    query = functional.normalize(query[valid], dim=1)
    normalized_target = target[valid] / mass[valid]
    weights = normalized_target @ normalized_target.T
    weights.fill_diagonal_(0.0)
    positive_mass = weights.sum(dim=1, keepdim=True)
    has_positive = positive_mass.squeeze(1) > 0.0
    if not bool(has_positive.any()):
        return query.sum() * 0.0
    normalized_weights = weights / positive_mass.clamp_min(1e-12)
    similarity = (query @ query.T) / float(temperature)
    diagonal = torch.eye(similarity.shape[0], dtype=torch.bool, device=query.device)
    similarity = similarity.masked_fill(
        diagonal, -torch.finfo(similarity.dtype).max
    )
    log_probabilities = functional.log_softmax(similarity, dim=1)
    return -(
        normalized_weights[has_positive] * log_probabilities[has_positive]
    ).sum(dim=1).mean()


@dataclass(slots=True)
class _AffineRoutingHead:
    weight: Any
    bias: Any


class ScopeLossAdapter:
    """CPU-only affine query head used for controlled loss comparisons."""

    VALID_OBJECTIVES = {"bce_only", "crm_only", "crm_rccr", "vlrb_soft_ce"}

    def __init__(
        self,
        *,
        objective: str,
        seed: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        rccr_weight: float,
        rccr_temperature: float,
        cpu_threads: int,
    ) -> None:
        if objective not in self.VALID_OBJECTIVES:
            raise ScopeAdapterError(f"unknown objective: {objective}")
        if epochs <= 0 or batch_size <= 1 or learning_rate <= 0 or weight_decay < 0:
            raise ScopeAdapterError("invalid optimizer configuration")
        if rccr_weight < 0 or rccr_temperature <= 0 or cpu_threads <= 0:
            raise ScopeAdapterError("invalid RCCR/CPU configuration")
        self.objective = objective
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.rccr_weight = float(rccr_weight)
        self.rccr_temperature = float(rccr_temperature)
        self.cpu_threads = int(cpu_threads)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.head: _AffineRoutingHead | None = None
        self.fit_summary: dict[str, Any] = {}

    @property
    def score_semantics(self) -> str:
        if self.objective == "bce_only":
            return "independent_sigmoid_correctness_score"
        if self.objective in {"crm_only", "crm_rccr"}:
            return "independent_sigmoid_cost_relevance_not_correctness_probability"
        return "softmax_assignment_mass_not_correctness_probability"

    def fit(
        self,
        features: Any,
        *,
        binary_targets: Any,
        relevance_targets: Any,
        soft_targets: Any,
        observed_mask: Any,
    ) -> "ScopeLossAdapter":
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required") from _TORCH_IMPORT_ERROR
        started = time.perf_counter()
        x = np.asarray(features, dtype=np.float32)
        binary = np.asarray(binary_targets, dtype=np.float32)
        relevance = np.asarray(relevance_targets, dtype=np.float32)
        soft = np.asarray(soft_targets, dtype=np.float32)
        mask = np.asarray(observed_mask, dtype=bool)
        if x.ndim != 2 or binary.ndim != 2 or x.shape[0] != binary.shape[0]:
            raise ScopeAdapterError("feature/target shapes are inconsistent")
        if (
            relevance.shape != binary.shape
            or soft.shape != binary.shape
            or mask.shape != binary.shape
        ):
            raise ScopeAdapterError("all target/mask shapes must match")
        if not np.isfinite(x).all() or not np.isfinite(relevance).all():
            raise ScopeAdapterError("fit arrays must be finite")
        self.feature_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        self.feature_scale = scale
        scaled = (x - self.feature_mean) / self.feature_scale

        torch.set_num_threads(self.cpu_threads)
        torch.manual_seed(self.seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        model = torch.nn.Linear(x.shape[1], binary.shape[1], bias=True, device="cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        x_tensor = torch.from_numpy(np.ascontiguousarray(scaled))
        binary_tensor = torch.from_numpy(binary)
        relevance_tensor = torch.from_numpy(relevance)
        soft_tensor = torch.from_numpy(soft)
        mask_tensor = torch.from_numpy(mask.copy())
        epoch_losses: list[float] = []
        primary_losses: list[float] = []
        rccr_losses: list[float] = []
        for _epoch in range(self.epochs):
            order = torch.randperm(x.shape[0], generator=generator)
            total = 0.0
            total_primary = 0.0
            total_rccr = 0.0
            batches = 0
            for start in range(0, x.shape[0], self.batch_size):
                indices = order[start : start + self.batch_size]
                logits = model(x_tensor[indices])
                if self.objective == "bce_only":
                    primary = masked_sigmoid_bce_with_logits(
                        logits, binary_tensor[indices], mask_tensor[indices]
                    )
                    rccr = logits.sum() * 0.0
                elif self.objective in {"crm_only", "crm_rccr"}:
                    primary = masked_sigmoid_bce_with_logits(
                        logits, relevance_tensor[indices], mask_tensor[indices]
                    )
                    rccr = routing_consistency_loss(
                        logits,
                        relevance_tensor[indices],
                        temperature=self.rccr_temperature,
                    )
                else:
                    log_probabilities = functional.log_softmax(logits, dim=1)
                    primary = -(
                        soft_tensor[indices] * log_probabilities
                    ).sum(dim=1).mean()
                    rccr = logits.sum() * 0.0
                coefficient = (
                    self.rccr_weight if self.objective == "crm_rccr" else 0.0
                )
                loss = primary + coefficient * rccr
                if not bool(torch.isfinite(loss)):
                    raise ScopeAdapterError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total += float(loss.detach())
                total_primary += float(primary.detach())
                total_rccr += float(rccr.detach())
                batches += 1
            epoch_losses.append(total / batches)
            primary_losses.append(total_primary / batches)
            rccr_losses.append(total_rccr / batches)
        self.head = _AffineRoutingHead(
            weight=model.weight.detach().cpu().numpy().copy(),
            bias=model.bias.detach().cpu().numpy().copy(),
        )
        self.fit_summary = {
            "implementation": "cpu_affine_query_head",
            "objective": self.objective,
            "score_semantics": self.score_semantics,
            "routing_space_for_RCCR": "expert_logit_vector",
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "final_total_loss": epoch_losses[-1],
            "final_primary_loss": primary_losses[-1],
            "final_RCCR_loss": rccr_losses[-1],
            "all_wrong_rows_retained": int(
                np.all((~mask) | (binary <= 0.5), axis=1).sum()
            ),
            "observed_supervision_cells": int(mask.sum()),
            "wall_seconds": time.perf_counter() - started,
            "device": "cpu",
        }
        return self

    def predict_scores(self, features: Any) -> np.ndarray:
        if self.head is None or self.feature_mean is None or self.feature_scale is None:
            raise ScopeAdapterError("adapter is not fitted")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.feature_mean.size:
            raise ScopeAdapterError("prediction feature shape mismatch")
        logits = ((x - self.feature_mean) / self.feature_scale) @ self.head.weight.T
        logits += self.head.bias
        if self.objective == "vlrb_soft_ce":
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(shifted)
            scores = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        else:
            positive = logits >= 0.0
            scores = np.empty_like(logits, dtype=np.float32)
            scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
            exponent = np.exp(logits[~positive])
            scores[~positive] = exponent / (1.0 + exponent)
        if not np.isfinite(scores).all():
            raise ScopeAdapterError("prediction produced non-finite scores")
        return scores.astype(np.float64)
