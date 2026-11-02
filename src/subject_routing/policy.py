"""Frozen strict-subject policy fitting and runtime routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .artifacts import ArtifactIntegrityError, ClusterArtifact, PolicyArtifact
from .data_contracts import TrainingLabels, UnknownExpertError
from .leakage_guard import Operation, UIDAccessAudit, UIDScopeError


class UnknownBudgetError(KeyError):
    """Raised when runtime requests a budget absent from the frozen table."""


def _descends_from(
    state: str, ancestor: str, parent_by_state: Mapping[str, str | None]
) -> bool:
    current: str | None = state
    seen: set[str] = set()
    while current is not None and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        current = parent_by_state.get(current)
    return False


def fit_profile_policy(
    cluster: ClusterArtifact,
    profile_assignments: Mapping[str, str],
    labels: TrainingLabels,
    budgets: Mapping[str, Sequence[str]],
    *,
    audit: UIDAccessAudit,
) -> PolicyArtifact:
    """Fit one expert per frozen state/budget using profile-train labels only."""

    if audit.operation not in {Operation.FIT, Operation.TRAIN}:
        raise UIDScopeError("fit_profile requires fit/train UID scope")
    uids = tuple(profile_assignments)
    audit.require(uids, component="fit_profile")
    matrix = labels.matrix.subset(uids)
    if tuple(matrix.expert_ids) == ():
        raise ArtifactIntegrityError("profile expert pool is empty")
    known_states = {node.state_id for node in cluster.nodes}
    if any(state not in known_states for state in profile_assignments.values()):
        raise ArtifactIntegrityError("profile assignment references unknown cluster state")
    parent = {node.state_id: node.parent_state_id for node in cluster.nodes}
    global_rows = np.arange(len(uids), dtype=np.int64)
    table: dict[str, dict[str, str]] = {}
    frozen_budgets: dict[str, tuple[str, ...]] = {}
    expert_position = {expert: column for column, expert in enumerate(matrix.expert_ids)}

    for budget_id, allowed_values in budgets.items():
        allowed = tuple(str(expert) for expert in allowed_values)
        if not allowed or len(allowed) != len(set(allowed)):
            raise ArtifactIntegrityError(f"budget {budget_id} expert IDs must be non-empty/unique")
        unknown = [expert for expert in allowed if expert not in expert_position]
        if unknown:
            raise UnknownExpertError(f"budget {budget_id} contains unknown expert: {unknown[0]}")
        frozen_budgets[str(budget_id)] = allowed
        mapping: dict[str, str] = {}
        for state in [node.state_id for node in cluster.nodes] + [cluster.unknown_state_id]:
            if state == cluster.unknown_state_id:
                rows = global_rows
            else:
                rows = np.asarray(
                    [
                        row
                        for row, uid in enumerate(uids)
                        if _descends_from(profile_assignments[uid], state, parent)
                    ],
                    dtype=np.int64,
                )
                if not len(rows):
                    rows = global_rows
            scores = []
            for expert in allowed:
                column = expert_position[expert]
                observed = matrix.observed_mask[rows, column]
                score = (
                    float(matrix.y[rows[observed], column].mean())
                    if observed.any()
                    else float("-inf")
                )
                scores.append(score)
            if not np.isfinite(scores).any():
                raise ArtifactIntegrityError("profile has no observed allowed-expert outcome")
            # np.argmax gives the canonical allowed-order tie break.
            mapping[state] = allowed[int(np.argmax(scores))]
        table[str(budget_id)] = mapping
    return PolicyArtifact(
        cluster_artifact_sha256=cluster.sha256,
        expert_ids=matrix.expert_ids,
        budgets=frozen_budgets,
        table=table,
        unknown_state_id=cluster.unknown_state_id,
    )


@dataclass(frozen=True, slots=True)
class StrictSubjectRouter:
    """Runtime invariant: input -> frozen state -> one expert per budget."""

    policy: PolicyArtifact

    def route(self, states: Sequence[str], *, budget_id: str) -> tuple[str, ...]:
        budget = str(budget_id)
        if budget not in self.policy.table:
            raise UnknownBudgetError(f"budget is not frozen in policy table: {budget}")
        mapping = self.policy.table[budget]
        experts = []
        for raw_state in states:
            state = str(raw_state)
            resolved = state if state in mapping else self.policy.unknown_state_id
            experts.append(mapping[resolved])
        return tuple(experts)
