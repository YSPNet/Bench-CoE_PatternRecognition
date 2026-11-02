"""Scalable FineRouter-shaped strict and hybrid adapters.

The original paper combines graph task discovery and query-level task-aware
quality prediction.  This cache-only adaptation uses joint semantic/preference
clustering followed by a deployable linear task student.  The strict class
exposes only task-level fixed tables; the hybrid class adds query-level quality
and is intentionally typed separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.preprocessing import StandardScaler


class FineRouterAdapterError(ValueError):
    """Raised when an adapter would violate its frozen routing identity."""


def _validate(
    features: Any,
    quality: Any,
    cost: Any,
    observed_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(quality, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    if x.ndim != 2 or y.ndim != 2 or c.shape != y.shape or mask.shape != y.shape:
        raise FineRouterAdapterError("feature/outcome matrices have invalid shapes")
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise FineRouterAdapterError("feature and outcome rows differ")
    if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
        raise FineRouterAdapterError("observed feature/quality values must be finite")
    if not np.isin(y[mask], (0.0, 1.0)).all():
        raise FineRouterAdapterError("FineRouter V2 adaptation requires binary quality")
    if not np.isfinite(c[mask]).all() or (c[mask] <= 0.0).any():
        raise FineRouterAdapterError("observed costs must be positive finite")
    if (mask.sum(axis=0) == 0).any():
        raise FineRouterAdapterError("every expert needs observed source outcomes")
    return x, y, c, mask


def _row_l2(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def _row_minmax(values: np.ndarray) -> np.ndarray:
    lower = values.min(axis=1, keepdims=True)
    width = values.max(axis=1, keepdims=True) - lower
    output = np.full_like(values, 0.5, dtype=np.float64)
    return np.divide(values - lower, width, out=output, where=width > 1e-12)


def _profile_tables(
    states: np.ndarray,
    state_ids: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    quality_mask: np.ndarray,
    *,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_quality = np.divide(
        np.where(quality_mask, quality, 0.0).sum(axis=0),
        quality_mask.sum(axis=0),
    )
    global_cost = cost.mean(axis=0)
    state_quality = np.empty((len(state_ids), quality.shape[1]), dtype=np.float64)
    state_cost = np.empty_like(state_quality)
    support = np.empty(len(state_ids), dtype=np.int64)
    for position, state in enumerate(state_ids):
        rows = states == state
        count = int(rows.sum())
        if not count:
            raise FineRouterAdapterError("task student produced an empty deploy state")
        support[position] = count
        weight = count / (count + float(shrinkage))
        state_quality[position] = weight * quality[rows].mean(axis=0) + (
            1.0 - weight
        ) * global_quality
        state_cost[position] = weight * cost[rows].mean(axis=0) + (
            1.0 - weight
        ) * global_cost
    return state_quality, state_cost, support


def assert_state_budget_invariant(
    states: Sequence[int], choices: Sequence[int]
) -> None:
    """Require every task code to choose exactly one expert for a setting."""

    state_array = np.asarray(states, dtype=np.int64)
    choice_array = np.asarray(choices, dtype=np.int64)
    if state_array.shape != choice_array.shape or state_array.ndim != 1:
        raise FineRouterAdapterError("state/choice vectors must align")
    for state in np.unique(state_array):
        if np.unique(choice_array[state_array == state]).size != 1:
            raise FineRouterAdapterError(
                f"strict task {int(state)} routes to multiple experts"
            )


@dataclass(slots=True)
class FineRouterStrictAdapter:
    """Joint-discovery task student with task→fixed profile tables."""

    n_tasks: int
    seed: int
    semantic_weight: float = 1.0
    preference_weight: float = 1.0
    student_alpha: float = 1e-4
    student_max_iter: int = 200
    minimum_support: int = 64
    profile_shrinkage: float = 20.0
    scaler: StandardScaler | None = None
    clusterer: MiniBatchKMeans | None = None
    student: SGDClassifier | None = None
    state_ids: np.ndarray | None = None
    state_quality: np.ndarray | None = None
    state_cost: np.ndarray | None = None
    fit_summary: dict[str, Any] | None = None

    routing_identity = "STRICT_DISCRETE_TASK_LOOKUP"
    fidelity = "FINEROUTER_STYLE_SCALABLE_ADAPTATION_NOT_GRAPH_NOT_LEIDEN"

    def fit(
        self,
        features: Any,
        quality: Any,
        cost: Any,
        observed_mask: Any,
    ) -> "FineRouterStrictAdapter":
        x, y, c, mask = _validate(features, quality, cost, observed_mask)
        if self.n_tasks <= 1 or self.n_tasks > x.shape[0]:
            raise FineRouterAdapterError("n_tasks must be in [2, n_rows]")
        if self.semantic_weight <= 0 or self.preference_weight <= 0:
            raise FineRouterAdapterError("joint discovery weights must be positive")
        if self.minimum_support <= 0 or self.profile_shrinkage < 0:
            raise FineRouterAdapterError("invalid support/shrinkage")
        semantic = _row_l2(x)
        preference = _row_l2(np.where(mask, y, 0.0))
        joint = _row_l2(
            np.concatenate(
                [
                    self.semantic_weight * semantic,
                    self.preference_weight * preference,
                ],
                axis=1,
            )
        )
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.n_tasks,
            random_state=self.seed,
            n_init=3,
            max_iter=100,
            batch_size=min(2048, len(x)),
            reassignment_ratio=0.0,
        )
        discovered = self.clusterer.fit_predict(joint).astype(np.int64)
        discovered_support = np.bincount(discovered, minlength=self.n_tasks)
        if (discovered_support < self.minimum_support).any():
            raise FineRouterAdapterError(
                "discovered task violates preregistered minimum support"
            )
        self.scaler = StandardScaler().fit(x)
        scaled = self.scaler.transform(x)
        self.student = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.student_alpha,
            max_iter=self.student_max_iter,
            tol=1e-4,
            random_state=self.seed,
            average=True,
        )
        self.student.fit(scaled, discovered)
        deployed_states = self.student.predict(scaled).astype(np.int64)
        self.state_ids = np.unique(deployed_states)
        if self.state_ids.size != self.n_tasks:
            raise FineRouterAdapterError("task student collapsed a discovered task")
        self.state_quality, self.state_cost, deployed_support = _profile_tables(
            deployed_states,
            self.state_ids,
            y,
            c,
            mask,
            shrinkage=self.profile_shrinkage,
        )
        self.fit_summary = {
            "discovery_algorithm": "joint_semantic_preference_minibatch_kmeans",
            "paper_graph_or_Leiden_executed": False,
            "task_classifier": "fixed_MiniLM_SGD_multinomial_log_loss",
            "n_tasks": self.n_tasks,
            "discovered_support": discovered_support.astype(int).tolist(),
            "deployed_state_ids": self.state_ids.astype(int).tolist(),
            "deployed_support": deployed_support.astype(int).tolist(),
            "student_training_assignment_accuracy_diagnostic": float(
                np.mean(deployed_states == discovered)
            ),
            "all_wrong_rows_retained": int(
                np.all((~mask) | (y <= 0.5), axis=1).sum()
            ),
            "quality_cells_observed": int(mask.sum()),
            "source_or_dataset_id_used": False,
        }
        return self

    def predict_states(self, features: Any) -> np.ndarray:
        if self.scaler is None or self.student is None or self.state_ids is None:
            raise FineRouterAdapterError("strict adapter is not fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.scaler.n_features_in_:
            raise FineRouterAdapterError("strict prediction feature shape mismatch")
        states = self.student.predict(self.scaler.transform(x)).astype(np.int64)
        if not np.isin(states, self.state_ids).all():
            raise FineRouterAdapterError("task student emitted an unknown state")
        return states

    def predict_tables(
        self, features: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.state_ids is None or self.state_quality is None or self.state_cost is None:
            raise FineRouterAdapterError("strict task profiles are not fitted")
        states = self.predict_states(features)
        position = {int(state): row for row, state in enumerate(self.state_ids)}
        rows = np.asarray([position[int(state)] for state in states], dtype=np.int64)
        return self.state_quality[rows], self.state_cost[rows], states


@dataclass(slots=True)
class FineRouterHybridAdapter:
    """Query-varying quality head layered on a fitted strict task adapter."""

    strict: FineRouterStrictAdapter
    ridge: float = 10.0
    alpha_query: float = 0.5
    state_shrinkage: float = 20.0
    quality_model: Ridge | None = None
    residual_by_state: np.ndarray | None = None
    prior_by_state: np.ndarray | None = None
    fit_summary: dict[str, Any] | None = None

    routing_identity = "HYBRID_QUERY_ROUTING"
    fidelity = "FINEROUTER_FULL_SHAPED_ADAPTATION_NOT_ORIGINAL_MOE"

    def fit(
        self,
        features: Any,
        quality: Any,
        cost: Any,
        observed_mask: Any,
    ) -> "FineRouterHybridAdapter":
        x, y, _c, mask = _validate(features, quality, cost, observed_mask)
        if not mask.all():
            raise FineRouterAdapterError(
                "current vectorized hybrid adapter requires complete FULL_Y cells"
            )
        if self.ridge <= 0 or not 0.0 <= self.alpha_query <= 1.0:
            raise FineRouterAdapterError("invalid hybrid ridge/alpha")
        if self.strict.scaler is None or self.strict.state_ids is None:
            raise FineRouterAdapterError("hybrid needs a fitted strict task adapter")
        states = self.strict.predict_states(x)
        scaled = self.strict.scaler.transform(x)
        self.quality_model = Ridge(alpha=self.ridge, fit_intercept=True).fit(scaled, y)
        base = np.clip(self.quality_model.predict(scaled), 0.0, 1.0)
        global_prior = y.mean(axis=0)
        residual = []
        prior = []
        support = []
        for state in self.strict.state_ids:
            rows = states == state
            count = int(rows.sum())
            support.append(count)
            weight = count / (count + self.state_shrinkage)
            residual.append(weight * (y[rows] - base[rows]).mean(axis=0))
            prior.append(
                (y[rows].sum(axis=0) + self.state_shrinkage * global_prior)
                / (count + self.state_shrinkage)
            )
        self.residual_by_state = np.asarray(residual, dtype=np.float64)
        self.prior_by_state = np.asarray(prior, dtype=np.float64)
        self.fit_summary = {
            "quality_head": "global_multioutput_ridge_plus_task_residual",
            "paper_task_specific_MLP_executed": False,
            "quality_target": "binary_correctness_MSE",
            "state_support": support,
            "same_state_may_route_to_different_experts": True,
            "routing_identity": self.routing_identity,
        }
        return self

    def predict_quality(self, features: Any) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.quality_model is None
            or self.residual_by_state is None
            or self.prior_by_state is None
            or self.strict.scaler is None
            or self.strict.state_ids is None
        ):
            raise FineRouterAdapterError("hybrid adapter is not fitted")
        x = np.asarray(features, dtype=np.float64)
        states = self.strict.predict_states(x)
        state_position = {
            int(state): row for row, state in enumerate(self.strict.state_ids)
        }
        rows = np.asarray(
            [state_position[int(state)] for state in states], dtype=np.int64
        )
        base = np.clip(
            self.quality_model.predict(self.strict.scaler.transform(x)), 0.0, 1.0
        )
        task_query = base + self.residual_by_state[rows]
        quality = self.alpha_query * _row_minmax(task_query) + (
            1.0 - self.alpha_query
        ) * _row_minmax(self.prior_by_state[rows])
        return np.clip(quality, 0.0, 1.0), states
