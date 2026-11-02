"""Deterministic strict-subject policy tables built only from train profiles."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactIntegrityError, canonical_sha256
from .data_contracts import UnknownExpertError
from .profiling import CapabilityProfileArtifact


POLICY_ALGORITHM_ID = "strict_subject_train_profile_policy_tables_v1"
_PROFILE_ESTIMATE_VARIANTS = frozenset(
    {
        "raw_mean",
        "parent_global_shrunk_mean",
        "leave_child_out_hierarchically_shrunk_mean",
    }
)


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


def _atomic_envelope(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen strict policy table: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"payload": dict(payload), "payload_sha256": canonical_sha256(payload)}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True, slots=True)
class BudgetDefinition:
    budget_id: str
    cap: float
    allowed_experts: tuple[str, ...]
    cost_unit: str = "usd_proxy_per_request"

    def __post_init__(self) -> None:
        if not self.budget_id or not np.isfinite(self.cap) or self.cap < 0:
            raise ArtifactIntegrityError("budget ID/cap is invalid")
        if not self.allowed_experts or len(self.allowed_experts) != len(
            set(self.allowed_experts)
        ):
            raise ArtifactIntegrityError(
                "budget expert set must be non-empty and unique"
            )
        if self.cost_unit != "usd_proxy_per_request":
            raise ArtifactIntegrityError("budget cost unit mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "cap": self.cap,
            "cost_unit": self.cost_unit,
            "allowed_experts": list(self.allowed_experts),
        }


def derive_training_cost_budgets(
    profile: CapabilityProfileArtifact, quantiles: Sequence[float]
) -> tuple[BudgetDefinition, ...]:
    """Create expert eligibility caps from profile-train global mean cost only."""

    requested = tuple(float(value) for value in quantiles)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(value < 0 or value > 1 for value in requested)
    ):
        raise ValueError("budget quantiles must be unique values in [0, 1]")
    global_row = profile.state_position[profile.global_state_id]
    costs = np.asarray(profile.raw_cost_mean[global_row], dtype=np.float64)
    if not np.isfinite(costs).all():
        raise ArtifactIntegrityError("global train mean cost is missing for an expert")
    result = []
    for quantile in requested:
        cap = float(np.quantile(costs, quantile))
        allowed = tuple(
            expert
            for expert, cost in zip(profile.expert_ids, costs)
            if float(cost) <= cap + 1e-15
        )
        label = f"q{int(round(quantile * 100)):03d}"
        result.append(BudgetDefinition(label, cap, allowed, profile.cost_unit))
    return tuple(result)


def _lambda_id(value: float) -> str:
    text = format(float(value), ".12g").replace("-", "m").replace(".", "p")
    return f"quality_minus_lambda_cost__lambda_{text}"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    expert_id: str
    quality_estimate: float
    train_mean_cost: float
    normalized_train_cost: float
    objective_value: float
    evidence_semantics: str
    tie_break: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "quality_estimate": self.quality_estimate,
            "train_mean_cost": self.train_mean_cost,
            "normalized_train_cost": self.normalized_train_cost,
            "objective_value": self.objective_value,
            "evidence_semantics": self.evidence_semantics,
            "tie_break": self.tie_break,
        }


@dataclass(frozen=True, slots=True)
class StrictPolicyTableArtifact:
    profile_sha256: str
    method_id: str
    environment_id: str
    expert_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    global_state_id: str
    budgets: tuple[BudgetDefinition, ...]
    strategy_lambdas: Mapping[str, float]
    quality_variant: str
    cost_variant: str
    decisions: Mapping[str, Mapping[str, Mapping[str, PolicyDecision]]]
    cost_normalizer: float
    cost_unit: str
    tie_tolerance: float
    selected_on: str = "train_profile_only"
    algorithm_id: str = POLICY_ALGORITHM_ID

    def __post_init__(self) -> None:
        _require_sha256(self.profile_sha256, name="capability profile hash")
        if (
            self.algorithm_id != POLICY_ALGORITHM_ID
            or self.selected_on != "train_profile_only"
        ):
            raise ArtifactIntegrityError(
                "unexpected strict-policy identity/selection split"
            )
        if self.quality_variant not in _PROFILE_ESTIMATE_VARIANTS:
            raise ArtifactIntegrityError("unsupported policy quality estimate")
        if self.cost_variant not in _PROFILE_ESTIMATE_VARIANTS:
            raise ArtifactIntegrityError("unsupported policy cost estimate")
        if self.global_state_id not in self.state_ids:
            raise ArtifactIntegrityError("strict policy lacks a global fallback state")
        if len(self.expert_ids) != len(set(self.expert_ids)) or len(
            self.state_ids
        ) != len(set(self.state_ids)):
            raise ArtifactIntegrityError(
                "strict policy expert/state IDs must be unique"
            )
        if self.cost_normalizer <= 0 or self.cost_unit != "usd_proxy_per_request":
            raise ArtifactIntegrityError("strict policy cost scaling is invalid")
        if self.tie_tolerance < 0:
            raise ArtifactIntegrityError(
                "strict policy tie tolerance cannot be negative"
            )
        budget_by_id = {budget.budget_id: budget for budget in self.budgets}
        if len(budget_by_id) != len(self.budgets):
            raise ArtifactIntegrityError("strict policy budget IDs must be unique")
        if set(self.decisions) != set(self.strategy_lambdas):
            raise ArtifactIntegrityError("strict policy strategies/decisions differ")
        for strategy, budget_tables in self.decisions.items():
            if set(budget_tables) != set(budget_by_id):
                raise ArtifactIntegrityError(
                    f"strategy {strategy} does not cover every budget"
                )
            for budget_id, state_table in budget_tables.items():
                if set(state_table) != set(self.state_ids):
                    raise ArtifactIntegrityError(
                        "strict policy table does not cover every state"
                    )
                allowed = set(budget_by_id[budget_id].allowed_experts)
                if any(
                    decision.expert_id not in allowed
                    for decision in state_table.values()
                ):
                    raise ArtifactIntegrityError(
                        "strict policy selected a budget-disallowed expert"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_strict_train_profile_policy_tables",
            "algorithm_id": self.algorithm_id,
            "profile_sha256": self.profile_sha256,
            "method_id": self.method_id,
            "environment_id": self.environment_id,
            "expert_ids": list(self.expert_ids),
            "state_ids": list(self.state_ids),
            "global_state_id": self.global_state_id,
            "budgets": [budget.to_dict() for budget in self.budgets],
            "strategies": [
                {
                    "strategy_id": strategy,
                    "lambda_scaled": value,
                    "cost_scale": "train_profile_positive_cost_median",
                }
                for strategy, value in self.strategy_lambdas.items()
            ],
            "quality_variant": self.quality_variant,
            "quality_semantics": "empirical_estimate_not_calibrated_probability",
            "cost_variant": self.cost_variant,
            "cost_normalizer": self.cost_normalizer,
            "cost_unit": self.cost_unit,
            "tie_tolerance": self.tie_tolerance,
            "tie_order": ["training_mean_cost", "canonical_expert_order"],
            "selected_on": self.selected_on,
            "noninferiority_claim": False,
            "decisions": {
                strategy: {
                    budget: {
                        state: decision.to_dict()
                        for state, decision in state_table.items()
                    }
                    for budget, state_table in budget_tables.items()
                }
                for strategy, budget_tables in self.decisions.items()
            },
            "strict_subject_invariant": "one_expert_per_frozen_state_and_budget",
            "test_actual_C_used_for_selection": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrictPolicyTableArtifact":
        budgets = tuple(
            BudgetDefinition(
                budget_id=str(budget["budget_id"]),
                cap=float(budget["cap"]),
                allowed_experts=tuple(
                    str(expert) for expert in budget["allowed_experts"]
                ),
                cost_unit=str(budget["cost_unit"]),
            )
            for budget in value["budgets"]
        )
        lambdas = {
            str(strategy["strategy_id"]): float(strategy["lambda_scaled"])
            for strategy in value["strategies"]
        }
        decisions = {
            str(strategy): {
                str(budget): {
                    str(state): PolicyDecision(
                        expert_id=str(decision["expert_id"]),
                        quality_estimate=float(decision["quality_estimate"]),
                        train_mean_cost=float(decision["train_mean_cost"]),
                        normalized_train_cost=float(decision["normalized_train_cost"]),
                        objective_value=float(decision["objective_value"]),
                        evidence_semantics=str(decision["evidence_semantics"]),
                        tie_break=str(decision["tie_break"]),
                    )
                    for state, decision in state_table.items()
                }
                for budget, state_table in budget_tables.items()
            }
            for strategy, budget_tables in value["decisions"].items()
        }
        return cls(
            profile_sha256=str(value["profile_sha256"]),
            method_id=str(value["method_id"]),
            environment_id=str(value["environment_id"]),
            expert_ids=tuple(str(expert) for expert in value["expert_ids"]),
            state_ids=tuple(str(state) for state in value["state_ids"]),
            global_state_id=str(value["global_state_id"]),
            budgets=budgets,
            strategy_lambdas=lambdas,
            quality_variant=str(value["quality_variant"]),
            cost_variant=str(value["cost_variant"]),
            decisions=decisions,
            cost_normalizer=float(value["cost_normalizer"]),
            cost_unit=str(value["cost_unit"]),
            tie_tolerance=float(value["tie_tolerance"]),
            selected_on=str(value["selected_on"]),
            algorithm_id=str(value["algorithm_id"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StrictPolicyTableArtifact":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = value.get("payload")
        if (
            not isinstance(payload, dict)
            or payload.get("kind")
            != "subject_routing_strict_train_profile_policy_tables"
            or value.get("payload_sha256") != canonical_sha256(payload)
        ):
            raise ArtifactIntegrityError(
                "strict policy table artifact digest/type mismatch"
            )
        return cls.from_dict(payload)

    def route(
        self, states: Sequence[str], *, budget_id: str, strategy_id: str
    ) -> tuple[str, ...]:
        if strategy_id not in self.decisions:
            raise KeyError(f"unknown frozen policy strategy: {strategy_id}")
        if budget_id not in self.decisions[strategy_id]:
            raise KeyError(f"unknown frozen policy budget: {budget_id}")
        table = self.decisions[strategy_id][budget_id]
        return tuple(
            table[state if state in table else self.global_state_id].expert_id
            for state in map(str, states)
        )


def build_strict_policy_tables(
    profile: CapabilityProfileArtifact,
    budgets: Sequence[BudgetDefinition],
    *,
    lambdas: Sequence[float],
    quality_variant: str = "parent_global_shrunk_mean",
    cost_variant: str = "parent_global_shrunk_mean",
    tie_tolerance: float = 1e-12,
) -> StrictPolicyTableArtifact:
    """Select exactly one expert for each frozen state/budget from train estimates."""

    budget_values = tuple(budgets)
    if not budget_values:
        raise ValueError("at least one budget is required")
    canonical_position = {
        expert: column for column, expert in enumerate(profile.expert_ids)
    }
    for budget in budget_values:
        unknown = set(budget.allowed_experts).difference(canonical_position)
        if unknown:
            raise UnknownExpertError(
                f"budget contains unknown expert: {sorted(unknown)[0]}"
            )
        expected_order = tuple(
            expert
            for expert in profile.expert_ids
            if expert in set(budget.allowed_experts)
        )
        if budget.allowed_experts != expected_order:
            raise ArtifactIntegrityError(
                "budget expert set must preserve canonical order"
            )
    lambda_values = tuple(float(value) for value in lambdas)
    if any(not np.isfinite(value) or value < 0 for value in lambda_values):
        raise ValueError("policy lambda values must be finite and non-negative")
    strategies: dict[str, float] = {"accuracy_only": 0.0}
    for value in lambda_values:
        strategies[_lambda_id(value)] = value
    quality = (
        profile.raw_quality_mean
        if quality_variant == "raw_mean"
        else profile.shrunk_quality_mean
    )
    cost = (
        profile.raw_cost_mean
        if cost_variant == "raw_mean"
        else profile.shrunk_cost_mean
    )
    global_row = profile.state_position[profile.global_state_id]
    decisions: dict[str, dict[str, dict[str, PolicyDecision]]] = {}
    for strategy, lambda_cost in strategies.items():
        by_budget: dict[str, dict[str, PolicyDecision]] = {}
        for budget in budget_values:
            by_state: dict[str, PolicyDecision] = {}
            for state in profile.state_ids:
                row = profile.state_position[state]
                candidates = []
                for expert in budget.allowed_experts:
                    column = canonical_position[expert]
                    estimate = float(quality[row, column])
                    train_cost = float(cost[row, column])
                    if not np.isfinite(estimate):
                        estimate = float(quality[global_row, column])
                    if not np.isfinite(train_cost):
                        train_cost = float(cost[global_row, column])
                    if not np.isfinite(estimate) or not np.isfinite(train_cost):
                        continue
                    normalized_cost = train_cost / profile.cost_normalizer
                    objective = (
                        estimate
                        if strategy == "accuracy_only"
                        else estimate - lambda_cost * normalized_cost
                    )
                    candidates.append(
                        (expert, estimate, train_cost, normalized_cost, objective)
                    )
                if not candidates:
                    raise ArtifactIntegrityError(
                        f"state {state}/budget {budget.budget_id} has no observed train candidate"
                    )
                best_objective = max(candidate[4] for candidate in candidates)
                tied = [
                    candidate
                    for candidate in candidates
                    if abs(candidate[4] - best_objective) <= tie_tolerance
                ]
                tied.sort(key=lambda item: (item[2], canonical_position[item[0]]))
                expert, estimate, train_cost, normalized_cost, objective = tied[0]
                by_state[state] = PolicyDecision(
                    expert_id=expert,
                    quality_estimate=estimate,
                    train_mean_cost=train_cost,
                    normalized_train_cost=normalized_cost,
                    objective_value=objective,
                    evidence_semantics="point_estimate_selection_no_noninferiority_claim",
                    tie_break=(
                        "training_mean_cost_then_canonical_expert_order"
                        if len(tied) > 1
                        else "not_needed"
                    ),
                )
            by_budget[budget.budget_id] = by_state
        decisions[strategy] = by_budget
    return StrictPolicyTableArtifact(
        profile_sha256=profile.sha256,
        method_id=profile.method_id,
        environment_id=profile.environment_id,
        expert_ids=profile.expert_ids,
        state_ids=profile.state_ids,
        global_state_id=profile.global_state_id,
        budgets=budget_values,
        strategy_lambdas=strategies,
        quality_variant=str(quality_variant),
        cost_variant=str(cost_variant),
        decisions=decisions,
        cost_normalizer=profile.cost_normalizer,
        cost_unit=profile.cost_unit,
        tie_tolerance=float(tie_tolerance),
    )
