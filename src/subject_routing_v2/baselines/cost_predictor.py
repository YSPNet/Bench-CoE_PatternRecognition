"""Pre-generation cost predictors kept separate from quality heads."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class CostPredictorError(ValueError):
    """Raised for invalid cost-prediction data."""


def _validate(
    features: Any, costs: Any, observed_mask: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    c = np.asarray(costs, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    if x.ndim != 2 or c.ndim != 2 or c.shape != mask.shape or x.shape[0] != c.shape[0]:
        raise CostPredictorError("cost fit shapes are inconsistent")
    if not np.isfinite(x).all() or not np.isfinite(c[mask]).all() or (c[mask] <= 0).any():
        raise CostPredictorError("observed costs must be positive and finite")
    if (mask.sum(axis=0) == 0).any():
        raise CostPredictorError("an expert has no observed cost")
    return x, c, mask


class TrainMeanCostPredictor:
    """Mandatory train-only per-expert mean baseline."""

    predictor_id = "train_only_expert_mean_cost"

    def __init__(self) -> None:
        self.means: np.ndarray | None = None
        self.feature_dimension: int | None = None

    def fit(
        self, features: Any, costs: Any, observed_mask: Any
    ) -> "TrainMeanCostPredictor":
        x, c, mask = _validate(features, costs, observed_mask)
        self.means = np.divide(
            np.where(mask, c, 0.0).sum(axis=0), mask.sum(axis=0), dtype=np.float64
        )
        self.feature_dimension = x.shape[1]
        return self

    def predict_cost(self, features: Any) -> np.ndarray:
        if self.means is None or self.feature_dimension is None:
            raise CostPredictorError("mean cost predictor is not fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_dimension:
            raise CostPredictorError("cost prediction feature shape mismatch")
        return np.broadcast_to(self.means, (x.shape[0], self.means.size)).copy()


class LogCostRidgePredictor:
    """Optional CPU input→log-cost model using only the audited InputView array."""

    predictor_id = "fold_local_input_log_cost_ridge"

    def __init__(self, *, alpha: float) -> None:
        if float(alpha) <= 0:
            raise CostPredictorError("ridge alpha must be positive")
        self.alpha = float(alpha)
        self.scaler: StandardScaler | None = None
        self.model: Ridge | None = None
        self.heads: list[Ridge] = []

    def fit(
        self, features: Any, costs: Any, observed_mask: Any
    ) -> "LogCostRidgePredictor":
        x, c, mask = _validate(features, costs, observed_mask)
        self.scaler = StandardScaler().fit(x)
        scaled = self.scaler.transform(x)
        self.heads = []
        if mask.all():
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)
            self.model.fit(scaled, np.log(c))
            return self
        self.model = None
        for expert in range(c.shape[1]):
            observed = mask[:, expert]
            head = Ridge(alpha=self.alpha, fit_intercept=True)
            head.fit(scaled[observed], np.log(c[observed, expert]))
            self.heads.append(head)
        return self

    def predict_cost(self, features: Any) -> np.ndarray:
        if self.scaler is None or (self.model is None and not self.heads):
            raise CostPredictorError("log-cost predictor is not fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.scaler.n_features_in_:
            raise CostPredictorError("cost prediction feature shape mismatch")
        scaled = self.scaler.transform(x)
        if self.model is not None:
            output = np.exp(self.model.predict(scaled))
        else:
            output = np.column_stack([np.exp(head.predict(scaled)) for head in self.heads])
        if not np.isfinite(output).all() or (output <= 0).any():
            raise CostPredictorError("cost prediction is not positive finite")
        return output
