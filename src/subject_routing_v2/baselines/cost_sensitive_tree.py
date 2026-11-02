"""Transparent strict cost-sensitive decision-tree task baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .finerouter_strict_adapter import (
    FineRouterAdapterError,
    _profile_tables,
    _validate,
)


@dataclass(slots=True)
class CostSensitiveTree:
    """Distil cheapest-best expert IDs into a bounded-leaf strict taxonomy."""

    max_leaf_nodes: int
    minimum_leaf_support: int
    seed: int
    profile_shrinkage: float = 20.0
    model: DecisionTreeClassifier | None = None
    state_ids: np.ndarray | None = None
    state_quality: np.ndarray | None = None
    state_cost: np.ndarray | None = None
    fit_summary: dict[str, Any] | None = None

    routing_identity = "STRICT_DISCRETE_TASK_LOOKUP"
    fidelity = "ORDINARY_COST_SENSITIVE_WINNER_ID_TREE_CONTROL"

    def fit(
        self,
        features: Any,
        quality: Any,
        cost: Any,
        observed_mask: Any,
    ) -> "CostSensitiveTree":
        x, y, c, mask = _validate(features, quality, cost, observed_mask)
        if self.max_leaf_nodes <= 1 or self.minimum_leaf_support <= 0:
            raise FineRouterAdapterError("invalid decision-tree capacity")
        winner = np.empty(len(x), dtype=np.int64)
        for row in range(len(x)):
            observed = np.flatnonzero(mask[row])
            best_quality = np.max(y[row, observed])
            tied = observed[y[row, observed] == best_quality]
            winner[row] = tied[np.argmin(c[row, tied])]
        self.model = DecisionTreeClassifier(
            criterion="log_loss",
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.minimum_leaf_support,
            random_state=self.seed,
            class_weight="balanced",
        ).fit(x, winner)
        states = self.model.apply(x).astype(np.int64)
        self.state_ids = np.unique(states)
        self.state_quality, self.state_cost, support = _profile_tables(
            states,
            self.state_ids,
            y,
            c,
            mask,
            shrinkage=self.profile_shrinkage,
        )
        self.fit_summary = {
            "teacher_target": "highest_binary_quality_then_lowest_item_cost",
            "all_wrong_rows_retained": int(np.all(y <= 0.5, axis=1).sum()),
            "requested_max_leaf_nodes": self.max_leaf_nodes,
            "effective_leaf_codes": int(self.state_ids.size),
            "leaf_support": support.astype(int).tolist(),
            "tree_depth": int(self.model.get_depth()),
            "winner_ID_training_accuracy_diagnostic": float(
                np.mean(self.model.predict(x) == winner)
            ),
            "final_evaluation_target": "selected_expert_quality_and_cost_not_winner_classification_accuracy",
        }
        return self

    def predict_tables(
        self, features: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.model is None
            or self.state_ids is None
            or self.state_quality is None
            or self.state_cost is None
        ):
            raise FineRouterAdapterError("cost-sensitive tree is not fitted")
        x = np.asarray(features, dtype=np.float64)
        states = self.model.apply(x).astype(np.int64)
        position = {int(state): row for row, state in enumerate(self.state_ids)}
        if any(int(state) not in position for state in states):
            raise FineRouterAdapterError("tree emitted an unfitted leaf state")
        rows = np.asarray([position[int(state)] for state in states], dtype=np.int64)
        return self.state_quality[rows], self.state_cost[rows], states
