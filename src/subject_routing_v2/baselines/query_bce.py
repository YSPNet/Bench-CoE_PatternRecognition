"""Independent multi-label BCE or continuous-quality query heads."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class QueryHeadError(ValueError):
    """Raised for malformed query-head inputs."""


def independent_bce_loss_and_gradient(
    logits: Any, targets: Any, observed_mask: Any
) -> tuple[float, np.ndarray]:
    """Return masked independent-sigmoid BCE and its logit gradient.

    This function exists both as a transparent contract and as a regression
    test against accidentally replacing multi-label BCE with softmax CE.
    """

    raw_logits = np.asarray(logits, dtype=np.float64)
    raw_targets = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    if raw_logits.shape != raw_targets.shape or mask.shape != raw_logits.shape:
        raise QueryHeadError("BCE logits, targets, and mask must have equal shapes")
    if not np.isin(raw_targets[mask], (0.0, 1.0)).all():
        raise QueryHeadError("BCE targets must be binary on observed cells")
    count = int(mask.sum())
    if count == 0:
        raise QueryHeadError("BCE requires at least one observed cell")
    losses = np.maximum(raw_logits, 0.0) - raw_logits * raw_targets
    losses += np.log1p(np.exp(-np.abs(raw_logits)))
    probabilities = np.empty_like(raw_logits)
    positive = raw_logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-raw_logits[positive]))
    exponent = np.exp(raw_logits[~positive])
    probabilities[~positive] = exponent / (1.0 + exponent)
    gradient = np.zeros_like(raw_logits)
    gradient[mask] = (probabilities[mask] - raw_targets[mask]) / count
    return float(losses[mask].mean()), gradient


@dataclass(slots=True)
class _ConstantHead:
    probability: float

    def predict(self, rows: int) -> np.ndarray:
        return np.full(rows, self.probability, dtype=np.float64)


class IndependentQualityHeads:
    """Fold-local independent quality heads sharing only matrix operations."""

    def __init__(
        self,
        *,
        quality_kind: str,
        alpha: float,
        seed: int,
        max_iter: int,
        learning_rate_init: float,
        batch_size: int,
    ) -> None:
        if quality_kind not in {"binary_correctness", "bounded_quality"}:
            raise QueryHeadError(f"unsupported quality kind: {quality_kind}")
        if alpha < 0 or max_iter <= 0 or learning_rate_init <= 0 or batch_size <= 0:
            raise QueryHeadError("invalid quality-head hyperparameter")
        self.quality_kind = quality_kind
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.learning_rate_init = float(learning_rate_init)
        self.batch_size = int(batch_size)
        self.scaler: StandardScaler | None = None
        self.model: Any | None = None
        self.heads: list[Any] | None = None
        self.expert_count: int | None = None
        self.fit_summary: dict[str, Any] = {}

    def fit(
        self, features: Any, quality: Any, observed_mask: Any
    ) -> "IndependentQualityHeads":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(quality, dtype=np.float64)
        mask = np.asarray(observed_mask, dtype=bool)
        if x.ndim != 2 or y.ndim != 2 or y.shape != mask.shape or x.shape[0] != y.shape[0]:
            raise QueryHeadError("quality fit shapes are inconsistent")
        if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
            raise QueryHeadError("quality fit contains non-finite observed values")
        if self.quality_kind == "binary_correctness" and not np.isin(y[mask], (0, 1)).all():
            raise QueryHeadError("binary quality must contain only 0/1")
        self.scaler = StandardScaler().fit(x)
        scaled = self.scaler.transform(x)
        self.expert_count = y.shape[1]
        all_observed = bool(mask.all())
        if self.quality_kind == "binary_correctness" and all_observed:
            model = MLPClassifier(
                hidden_layer_sizes=(),
                activation="identity",
                solver="adam",
                alpha=self.alpha,
                batch_size=min(self.batch_size, x.shape[0]),
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                shuffle=True,
                random_state=self.seed,
                tol=1e-4,
                n_iter_no_change=8,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(scaled, y.astype(np.int8))
            self.model = model
            self.heads = None
            self.fit_summary = {
                "implementation": "vectorized_independent_sigmoid_linear_heads",
                "iterations": int(model.n_iter_),
                "all_wrong_rows_retained": int(np.all(y == 0, axis=1).sum()),
                "observed_cells": int(mask.sum()),
            }
            return self

        heads: list[Any] = []
        iterations = []
        for expert in range(y.shape[1]):
            observed = mask[:, expert]
            if not observed.any():
                raise QueryHeadError(f"expert {expert} has no observed quality")
            target = y[observed, expert]
            if self.quality_kind == "binary_correctness":
                if np.unique(target).size == 1:
                    probability = float((target.sum() + 1.0) / (target.size + 2.0))
                    head: Any = _ConstantHead(probability)
                    iterations.append(0)
                else:
                    head = SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=self.alpha,
                        max_iter=self.max_iter,
                        tol=1e-4,
                        random_state=self.seed + expert,
                        average=True,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        head.fit(scaled[observed], target.astype(np.int8))
                    iterations.append(int(head.n_iter_))
            else:
                head = Ridge(alpha=self.alpha, fit_intercept=True)
                head.fit(scaled[observed], target)
                iterations.append(1)
            heads.append(head)
        self.model = None
        self.heads = heads
        self.fit_summary = {
            "implementation": (
                "masked_independent_sigmoid_heads"
                if self.quality_kind == "binary_correctness"
                else "masked_independent_continuous_ridge_heads"
            ),
            "iterations_by_expert": iterations,
            "all_wrong_rows_retained": int(
                np.all((~mask) | (y == 0), axis=1).sum()
            ),
            "observed_cells": int(mask.sum()),
        }
        return self

    def predict_quality(self, features: Any) -> np.ndarray:
        if self.scaler is None or self.expert_count is None:
            raise QueryHeadError("quality heads are not fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.scaler.n_features_in_:
            raise QueryHeadError("quality prediction feature shape mismatch")
        scaled = self.scaler.transform(x)
        if self.model is not None:
            output = np.asarray(self.model.predict_proba(scaled), dtype=np.float64)
            if output.ndim == 3:
                output = np.column_stack([values[:, 1] for values in output])
        else:
            assert self.heads is not None
            columns = []
            for head in self.heads:
                if isinstance(head, _ConstantHead):
                    values = head.predict(x.shape[0])
                elif self.quality_kind == "binary_correctness":
                    values = head.predict_proba(scaled)[:, 1]
                else:
                    values = head.predict(scaled)
                columns.append(np.asarray(values, dtype=np.float64))
            output = np.column_stack(columns)
        if output.shape != (x.shape[0], self.expert_count):
            raise QueryHeadError("quality prediction output shape mismatch")
        return np.clip(output, 0.0, 1.0)

