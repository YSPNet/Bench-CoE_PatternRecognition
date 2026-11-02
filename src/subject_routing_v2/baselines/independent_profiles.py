"""Independent capability-profile estimators for strict routing tables.

The partition/student is deliberately outside this module.  Callers provide
the *deployed* state assignments on a disjoint profile split.  We then expose
three estimators over exactly the same rows: pooled empirical means,
source-balanced empirical means, and parent/global empirical shrinkage.

Raw empirical means are retained for audit.  They are never labelled as
confidence values, and rare or empty states use a deterministic parent/global
fallback for deployment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PROFILE_VARIANTS = ("pooled", "dataset_balanced", "shrinkage")


class IndependentProfileError(ValueError):
    """Raised when a profile or split violates the independent-fold contract."""


def _as_unique_strings(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(map(str, values))
    if not result or len(result) != len(set(result)):
        raise IndependentProfileError(f"{name} must be non-empty and unique")
    return result


def _depths(
    parent_by_state: Mapping[str, str | None], *, global_state_id: str
) -> dict[str, int]:
    parents = {
        str(state): None if parent is None else str(parent)
        for state, parent in parent_by_state.items()
    }
    if parents.get(global_state_id, "missing") is not None:
        raise IndependentProfileError("global state must exist and have no parent")
    depths: dict[str, int] = {}
    for state in parents:
        current: str | None = state
        seen: set[str] = set()
        depth = 0
        while current != global_state_id:
            if current is None or current in seen or current not in parents:
                raise IndependentProfileError("profile state hierarchy is invalid")
            seen.add(current)
            current = parents[current]
            depth += 1
        depths[state] = depth
    return depths


def _is_descendant(
    state: str,
    ancestor: str,
    parent_by_state: Mapping[str, str | None],
) -> bool:
    current: str | None = state
    seen: set[str] = set()
    while current is not None and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        current = parent_by_state[current]
    return False


def stratified_group_holdout(
    uids: Sequence[str],
    source_ids: Sequence[str],
    group_ids: Sequence[str],
    *,
    holdout_fraction: float,
    seed: int,
    salt: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split groups within each source while preserving the original UID order."""

    ids = _as_unique_strings(uids, name="split UIDs")
    sources = tuple(map(str, source_ids))
    groups = tuple(map(str, group_ids))
    if len(sources) != len(ids) or len(groups) != len(ids):
        raise IndependentProfileError("split UID/source/group rows differ")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise IndependentProfileError("holdout fraction must lie in (0,1)")

    rows_by_key: dict[tuple[str, str], list[int]] = {}
    for row, (source, group) in enumerate(zip(sources, groups)):
        rows_by_key.setdefault((source, group), []).append(row)
    groups_by_source: dict[str, list[tuple[str, str]]] = {}
    for key in rows_by_key:
        groups_by_source.setdefault(key[0], []).append(key)

    heldout_keys: set[tuple[str, str]] = set()
    for source, keys in sorted(groups_by_source.items()):
        if len(keys) < 2:
            raise IndependentProfileError(
                f"source {source} has fewer than two dependency groups"
            )

        def digest(key: tuple[str, str]) -> str:
            raw = f"{int(seed)}\0{salt}\0{key[0]}\0{key[1]}".encode("utf-8")
            return hashlib.sha256(raw).hexdigest()

        ordered = sorted(keys, key=lambda key: (digest(key), key[1]))
        target = max(
            1,
            int(
                round(
                    sum(len(rows_by_key[key]) for key in ordered)
                    * float(holdout_fraction)
                )
            ),
        )
        selected_rows = 0
        for key in ordered[:-1]:
            if selected_rows >= target and heldout_keys.intersection(ordered):
                break
            heldout_keys.add(key)
            selected_rows += len(rows_by_key[key])

    heldout_rows = {
        row for key in heldout_keys for row in rows_by_key[key]
    }
    fit = tuple(uid for row, uid in enumerate(ids) if row not in heldout_rows)
    heldout = tuple(uid for row, uid in enumerate(ids) if row in heldout_rows)
    if not fit or not heldout or set(fit).intersection(heldout):
        raise IndependentProfileError("group holdout produced an invalid split")
    fit_set = set(fit)
    heldout_set = set(heldout)
    fit_groups = {
        (sources[row], groups[row])
        for row, uid in enumerate(ids)
        if uid in fit_set
    }
    heldout_groups = {
        (sources[row], groups[row])
        for row, uid in enumerate(ids)
        if uid in heldout_set
    }
    if fit_groups.intersection(heldout_groups):
        raise IndependentProfileError("dependency group crossed the holdout split")
    return fit, heldout


@dataclass(frozen=True, slots=True)
class IndependentProfile:
    """Three deployable profile tables plus long-form audit records."""

    expert_ids: tuple[str, ...]
    routing_state_ids: tuple[str, ...]
    global_state_id: str
    quality_by_variant: Mapping[str, np.ndarray]
    cost_by_variant: Mapping[str, np.ndarray]
    state_support: np.ndarray
    source_coverage: np.ndarray
    records: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        shape = (len(self.routing_state_ids), len(self.expert_ids))
        if set(self.quality_by_variant) != set(PROFILE_VARIANTS) or set(
            self.cost_by_variant
        ) != set(PROFILE_VARIANTS):
            raise IndependentProfileError("profile variants are incomplete")
        for values in (*self.quality_by_variant.values(), *self.cost_by_variant.values()):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise IndependentProfileError("deployable profile table is invalid")
        if np.asarray(self.state_support).shape != (shape[0],) or np.asarray(
            self.source_coverage
        ).shape != (shape[0],):
            raise IndependentProfileError("profile support vectors are invalid")

    @property
    def state_position(self) -> dict[str, int]:
        return {state: row for row, state in enumerate(self.routing_state_ids)}


def estimate_independent_profiles(
    *,
    deployed_states: Sequence[str],
    routing_state_ids: Sequence[str],
    parent_by_state: Mapping[str, str | None],
    global_state_id: str,
    source_ids: Sequence[str],
    expert_ids: Sequence[str],
    quality: Any,
    cost: Any,
    quality_observed_mask: Any,
    cost_observed_mask: Any,
    minimum_state_support: int,
    minimum_source_coverage: int,
    kappa: float,
) -> IndependentProfile:
    """Estimate pooled, source-balanced and parent-shrunk fixed tables."""

    states = tuple(map(str, deployed_states))
    routing_states = _as_unique_strings(
        routing_state_ids, name="routing state IDs"
    )
    experts = _as_unique_strings(expert_ids, name="expert IDs")
    sources = tuple(map(str, source_ids))
    y = np.asarray(quality, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)
    y_mask = np.asarray(quality_observed_mask, dtype=bool)
    c_mask = np.asarray(cost_observed_mask, dtype=bool)
    if (
        y.ndim != 2
        or y.shape != c.shape
        or y.shape != y_mask.shape
        or y.shape != c_mask.shape
        or y.shape != (len(states), len(experts))
        or len(sources) != len(states)
    ):
        raise IndependentProfileError("profile arrays are misaligned")
    if set(states).difference(routing_states):
        raise IndependentProfileError("profile assignment names an unknown state")
    if not np.isfinite(y[y_mask]).all() or not np.isfinite(c[c_mask]).all():
        raise IndependentProfileError("observed profile cells must be finite")
    if not np.isin(y[y_mask], (0.0, 1.0)).all() or (c[c_mask] <= 0).any():
        raise IndependentProfileError("profile quality/cost semantics are invalid")
    if minimum_state_support <= 0 or minimum_source_coverage <= 0 or kappa < 0:
        raise IndependentProfileError("profile fallback parameters are invalid")

    global_id = str(global_state_id)
    parents = {
        str(state): None if parent is None else str(parent)
        for state, parent in parent_by_state.items()
    }
    if global_id not in parents:
        parents[global_id] = None
    if set(routing_states).difference(parents):
        raise IndependentProfileError("routing state is absent from hierarchy")
    depths = _depths(parents, global_state_id=global_id)

    node_order = tuple(sorted(parents, key=lambda state: (depths[state], state)))
    node_rows: dict[str, np.ndarray] = {}
    for node in node_order:
        if node == global_id:
            rows = np.arange(len(states), dtype=np.int64)
        else:
            rows = np.asarray(
                [
                    row
                    for row, state in enumerate(states)
                    if _is_descendant(state, node, parents)
                ],
                dtype=np.int64,
            )
        node_rows[node] = rows

    width = len(experts)
    pooled_raw: dict[str, np.ndarray] = {}
    balanced_raw: dict[str, np.ndarray] = {}
    quality_n: dict[str, np.ndarray] = {}
    cost_raw: dict[str, np.ndarray] = {}
    balanced_cost_raw: dict[str, np.ndarray] = {}
    cost_n: dict[str, np.ndarray] = {}
    coverage_by_node: dict[str, np.ndarray] = {}
    covered_sources_by_node: dict[str, tuple[tuple[str, ...], ...]] = {}

    for node in node_order:
        rows = node_rows[node]
        q_raw = np.full(width, np.nan, dtype=np.float64)
        q_balanced = np.full(width, np.nan, dtype=np.float64)
        q_count = np.zeros(width, dtype=np.int64)
        c_raw = np.full(width, np.nan, dtype=np.float64)
        c_balanced = np.full(width, np.nan, dtype=np.float64)
        c_count = np.zeros(width, dtype=np.int64)
        coverage = np.zeros(width, dtype=np.int64)
        covered_sources: list[tuple[str, ...]] = []
        for expert in range(width):
            observed_q = rows[y_mask[rows, expert]] if len(rows) else rows
            observed_c = rows[c_mask[rows, expert]] if len(rows) else rows
            q_count[expert] = len(observed_q)
            c_count[expert] = len(observed_c)
            if len(observed_q):
                q_raw[expert] = float(y[observed_q, expert].mean())
            if len(observed_c):
                c_raw[expert] = float(c[observed_c, expert].mean())
            source_means_q = []
            source_means_c = []
            supported_sources = []
            for source in sorted(set(sources[row] for row in rows)):
                source_rows = rows[
                    np.asarray([sources[int(row)] == source for row in rows])
                ]
                q_rows = source_rows[y_mask[source_rows, expert]]
                c_rows = source_rows[c_mask[source_rows, expert]]
                if len(q_rows):
                    source_means_q.append(float(y[q_rows, expert].mean()))
                    supported_sources.append(source)
                if len(c_rows):
                    source_means_c.append(float(c[c_rows, expert].mean()))
            if source_means_q:
                q_balanced[expert] = float(np.mean(source_means_q))
            if source_means_c:
                c_balanced[expert] = float(np.mean(source_means_c))
            coverage[expert] = len(supported_sources)
            covered_sources.append(tuple(supported_sources))
        pooled_raw[node] = q_raw
        balanced_raw[node] = q_balanced
        quality_n[node] = q_count
        cost_raw[node] = c_raw
        balanced_cost_raw[node] = c_balanced
        cost_n[node] = c_count
        coverage_by_node[node] = coverage
        covered_sources_by_node[node] = tuple(covered_sources)

    if (
        not np.isfinite(pooled_raw[global_id]).all()
        or not np.isfinite(cost_raw[global_id]).all()
        or not np.isfinite(balanced_raw[global_id]).all()
        or not np.isfinite(balanced_cost_raw[global_id]).all()
    ):
        raise IndependentProfileError("global fallback lacks an observed expert cell")

    effective_quality: dict[str, dict[str, np.ndarray]] = {
        variant: {} for variant in PROFILE_VARIANTS
    }
    effective_cost: dict[str, dict[str, np.ndarray]] = {
        variant: {} for variant in PROFILE_VARIANTS
    }
    quality_reason: dict[str, dict[str, list[str]]] = {
        variant: {} for variant in PROFILE_VARIANTS
    }
    cost_reason: dict[str, dict[str, list[str]]] = {
        variant: {} for variant in PROFILE_VARIANTS
    }

    for node in node_order:
        parent = parents[node]
        support = len(node_rows[node])
        for variant in PROFILE_VARIANTS:
            if node == global_id:
                q_base = (
                    balanced_raw[node]
                    if variant == "dataset_balanced"
                    else pooled_raw[node]
                )
                c_base = (
                    balanced_cost_raw[node]
                    if variant == "dataset_balanced"
                    else cost_raw[node]
                )
                effective_quality[variant][node] = q_base.copy()
                effective_cost[variant][node] = c_base.copy()
                quality_reason[variant][node] = ["GLOBAL_EMPIRICAL"] * width
                cost_reason[variant][node] = ["GLOBAL_EMPIRICAL"] * width
                continue
            assert parent is not None
            parent_q = effective_quality[variant][parent]
            parent_c = effective_cost[variant][parent]
            raw_q = (
                balanced_raw[node]
                if variant == "dataset_balanced"
                else pooled_raw[node]
            )
            raw_c = (
                balanced_cost_raw[node]
                if variant == "dataset_balanced"
                else cost_raw[node]
            )
            q_values = np.empty(width, dtype=np.float64)
            c_values = np.empty(width, dtype=np.float64)
            q_reasons: list[str] = []
            c_reasons: list[str] = []
            for expert in range(width):
                q_eligible = (
                    support >= minimum_state_support
                    and coverage_by_node[node][expert] >= minimum_source_coverage
                    and np.isfinite(raw_q[expert])
                )
                c_eligible = (
                    support >= minimum_state_support
                    and coverage_by_node[node][expert] >= minimum_source_coverage
                    and np.isfinite(raw_c[expert])
                )
                if not q_eligible:
                    q_values[expert] = parent_q[expert]
                    q_reasons.append(
                        "EMPTY_PARENT_BACKOFF"
                        if quality_n[node][expert] == 0
                        else "RARE_OR_LOW_COVERAGE_PARENT_BACKOFF"
                    )
                elif variant == "shrinkage":
                    weight = quality_n[node][expert] / (
                        quality_n[node][expert] + float(kappa)
                    )
                    q_values[expert] = (
                        weight * raw_q[expert] + (1.0 - weight) * parent_q[expert]
                    )
                    q_reasons.append("PARENT_EMPIRICAL_SHRINKAGE")
                else:
                    q_values[expert] = raw_q[expert]
                    q_reasons.append("DIRECT_EMPIRICAL")
                if not c_eligible:
                    c_values[expert] = parent_c[expert]
                    c_reasons.append(
                        "EMPTY_PARENT_BACKOFF"
                        if cost_n[node][expert] == 0
                        else "RARE_OR_LOW_COVERAGE_PARENT_BACKOFF"
                    )
                elif variant == "shrinkage":
                    weight = cost_n[node][expert] / (
                        cost_n[node][expert] + float(kappa)
                    )
                    c_values[expert] = (
                        weight * raw_c[expert] + (1.0 - weight) * parent_c[expert]
                    )
                    c_reasons.append("PARENT_EMPIRICAL_SHRINKAGE")
                else:
                    c_values[expert] = raw_c[expert]
                    c_reasons.append("DIRECT_EMPIRICAL")
            effective_quality[variant][node] = q_values
            effective_cost[variant][node] = c_values
            quality_reason[variant][node] = q_reasons
            cost_reason[variant][node] = c_reasons

    records: list[dict[str, Any]] = []
    for state in routing_states:
        rows = node_rows[state]
        all_wrong = int(
            np.all((~y_mask[rows]) | (y[rows] <= 0.5), axis=1).sum()
        ) if len(rows) else 0
        partial_missing = int(
            ((~y_mask[rows]).any(axis=1) | (~c_mask[rows]).any(axis=1)).sum()
        ) if len(rows) else 0
        parent = parents[state]
        for variant in PROFILE_VARIANTS:
            for expert, expert_id in enumerate(experts):
                records.append(
                    {
                        "state_id": state,
                        "parent_state_id": parent,
                        "global_state_id": global_id,
                        "profile_variant": variant,
                        "expert_id": expert_id,
                        "state_support": int(len(rows)),
                        "all_wrong_rows_retained": all_wrong,
                        "partially_missing_rows_retained": partial_missing,
                        "source_coverage_count": int(
                            coverage_by_node[state][expert]
                        ),
                        "source_coverage": list(
                            covered_sources_by_node[state][expert]
                        ),
                        "quality_observed_n": int(quality_n[state][expert]),
                        "cost_observed_n": int(cost_n[state][expert]),
                        "raw_pooled_quality_mean": (
                            None
                            if not np.isfinite(pooled_raw[state][expert])
                            else float(pooled_raw[state][expert])
                        ),
                        "raw_dataset_balanced_quality_mean": (
                            None
                            if not np.isfinite(balanced_raw[state][expert])
                            else float(balanced_raw[state][expert])
                        ),
                        "raw_pooled_cost_mean_usd_proxy": (
                            None
                            if not np.isfinite(cost_raw[state][expert])
                            else float(cost_raw[state][expert])
                        ),
                        "raw_dataset_balanced_cost_mean_usd_proxy": (
                            None
                            if not np.isfinite(balanced_cost_raw[state][expert])
                            else float(balanced_cost_raw[state][expert])
                        ),
                        "deployed_quality_estimate": float(
                            effective_quality[variant][state][expert]
                        ),
                        "deployed_cost_estimate_usd_proxy": float(
                            effective_cost[variant][state][expert]
                        ),
                        "quality_fallback_or_estimator": quality_reason[variant][
                            state
                        ][expert],
                        "cost_fallback_or_estimator": cost_reason[variant][state][
                            expert
                        ],
                        "minimum_state_support": int(minimum_state_support),
                        "minimum_source_coverage": int(minimum_source_coverage),
                        "kappa": float(kappa if variant == "shrinkage" else 0.0),
                        "raw_mean_is_confidence": False,
                        "estimate_semantics": "EMPIRICAL_POINT_ESTIMATE_NOT_CONFIDENCE",
                    }
                )

    state_support = np.asarray(
        [len(node_rows[state]) for state in routing_states], dtype=np.int64
    )
    source_coverage = np.asarray(
        [
            len(set(sources[int(row)] for row in node_rows[state]))
            for state in routing_states
        ],
        dtype=np.int64,
    )
    return IndependentProfile(
        expert_ids=experts,
        routing_state_ids=routing_states,
        global_state_id=global_id,
        quality_by_variant={
            variant: np.vstack(
                [effective_quality[variant][state] for state in routing_states]
            )
            for variant in PROFILE_VARIANTS
        },
        cost_by_variant={
            variant: np.vstack(
                [effective_cost[variant][state] for state in routing_states]
            )
            for variant in PROFILE_VARIANTS
        },
        state_support=state_support,
        source_coverage=source_coverage,
        records=tuple(records),
    )


def select_expert_index(
    quality: Sequence[float],
    cost: Sequence[float],
    allowed_indices: Sequence[int],
    *,
    tie_tolerance: float = 1e-12,
) -> int:
    """Quality first, then lower estimated cost, then canonical column order."""

    q = np.asarray(quality, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    if (
        q.ndim != 1
        or c.shape != q.shape
        or not len(allowed)
        or not np.isfinite(q).all()
        or not np.isfinite(c).all()
        or (allowed < 0).any()
        or (allowed >= len(q)).any()
    ):
        raise IndependentProfileError("expert selection inputs are invalid")
    best = float(q[allowed].max())
    tied = allowed[np.isclose(q[allowed], best, atol=tie_tolerance, rtol=0.0)]
    cheapest = float(c[tied].min())
    tied = tied[np.isclose(c[tied], cheapest, atol=1e-15, rtol=0.0)]
    return int(tied.min())
