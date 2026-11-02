"""Same-information query-routing baselines for subject-routing V2."""

from .cost_predictor import LogCostRidgePredictor, TrainMeanCostPredictor
from .cost_sensitive_tree import CostSensitiveTree
from .independent_profiles import (
    PROFILE_VARIANTS,
    IndependentProfile,
    IndependentProfileError,
    estimate_independent_profiles,
    select_expert_index,
    stratified_group_holdout,
)
from .finerouter_strict_adapter import (
    FineRouterAdapterError,
    FineRouterHybridAdapter,
    FineRouterStrictAdapter,
    assert_state_budget_invariant,
)
from .query_bce import IndependentQualityHeads, independent_bce_loss_and_gradient
from .query_knn import KNNQualityEstimator
from .scope_loss_adapter import (
    ScopeAdapterError,
    ScopeLossAdapter,
    correct_set_soft_targets,
    crm_relevance_targets,
    masked_sigmoid_bce_with_logits,
    routing_consistency_loss,
    source_cost_alpha,
)

__all__ = [
    "CostSensitiveTree",
    "PROFILE_VARIANTS",
    "IndependentProfile",
    "IndependentProfileError",
    "estimate_independent_profiles",
    "select_expert_index",
    "stratified_group_holdout",
    "FineRouterAdapterError",
    "FineRouterHybridAdapter",
    "FineRouterStrictAdapter",
    "IndependentQualityHeads",
    "KNNQualityEstimator",
    "LogCostRidgePredictor",
    "ScopeAdapterError",
    "ScopeLossAdapter",
    "TrainMeanCostPredictor",
    "correct_set_soft_targets",
    "crm_relevance_targets",
    "independent_bce_loss_and_gradient",
    "masked_sigmoid_bce_with_logits",
    "routing_consistency_loss",
    "source_cost_alpha",
    "assert_state_budget_invariant",
]
