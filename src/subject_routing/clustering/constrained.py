"""Bounded soft-constraint refinement for frozen spherical clusters.

The implementation deliberately treats teacher relations as noisy evidence:
constraints remain weighted terms, contradictory terms are retained and
audited, and no must-link transitive closure is constructed.  The resulting
centres are a frozen inductive assigner; deployment never calls the teacher.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts import ArtifactIntegrityError, canonical_sha256
from ..features.fusion import l2_normalize
from ..interfaces import AssignmentBatch
from ..leakage_guard import Operation, UIDAccessAudit, UIDScopeError


ALGORITHM_ID = "bounded_soft_constrained_spherical_refinement_v1"
_DIMENSIONS = frozenset({"subject", "visual_structure", "operation", "task_category"})
_FIT_OPERATIONS = frozenset({Operation.FIT, Operation.TRAIN})


def _uid_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


@dataclass(frozen=True, slots=True)
class SoftPairConstraint:
    """A signed, non-transitive pair relation.

    ``target=1`` favours a shared cluster and ``target=-1`` favours distinct
    clusters.  Neither value is a hard must/cannot link.
    """

    a_uid: str
    b_uid: str
    dimension: str
    target: int
    weight: float
    evidence_id: str
    outcome_informed: bool = False

    def __post_init__(self) -> None:
        if not self.a_uid or not self.b_uid or self.a_uid == self.b_uid:
            raise ValueError("pair constraints require two distinct non-empty UIDs")
        if self.dimension not in _DIMENSIONS:
            raise ValueError("unsupported pair-constraint dimension")
        if self.target not in {-1, 1}:
            raise ValueError("pair constraint target must be -1 or 1")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("pair constraint weight must be finite and positive")
        if not self.evidence_id:
            raise ValueError("pair constraint requires an evidence ID")

    @property
    def canonical_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.a_uid, self.b_uid)))  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_uid": self.a_uid,
            "b_uid": self.b_uid,
            "dimension": self.dimension,
            "target": self.target,
            "weight": float(self.weight),
            "evidence_id": self.evidence_id,
            "outcome_informed": self.outcome_informed,
        }


@dataclass(frozen=True, slots=True)
class SoftTripletConstraint:
    """Soft evidence that an anchor is closer to positive than negative."""

    anchor_uid: str
    positive_uid: str
    negative_uid: str
    dimension: str
    weight: float
    evidence_id: str
    outcome_informed: bool = False

    def __post_init__(self) -> None:
        uids = (self.anchor_uid, self.positive_uid, self.negative_uid)
        if any(not uid for uid in uids) or len(set(uids)) != 3:
            raise ValueError("triplet constraints require three distinct non-empty UIDs")
        if self.dimension not in _DIMENSIONS | {"combined"}:
            raise ValueError("unsupported triplet-constraint dimension")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("triplet constraint weight must be finite and positive")
        if not self.evidence_id:
            raise ValueError("triplet constraint requires an evidence ID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_uid": self.anchor_uid,
            "positive_uid": self.positive_uid,
            "negative_uid": self.negative_uid,
            "dimension": self.dimension,
            "weight": float(self.weight),
            "evidence_id": self.evidence_id,
            "outcome_informed": self.outcome_informed,
        }


@dataclass(frozen=True, slots=True)
class ConstraintConflictAudit:
    pair_evidence_count: int
    triplet_evidence_count: int
    ignored_unknown_or_tie_count: int
    pair_conflicts: tuple[dict[str, Any], ...]
    triplet_conflicts: tuple[dict[str, Any], ...]
    transitive_closure_edges_added: int = 0

    @property
    def conflict_count(self) -> int:
        return len(self.pair_conflicts) + len(self.triplet_conflicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_evidence_count": self.pair_evidence_count,
            "triplet_evidence_count": self.triplet_evidence_count,
            "ignored_unknown_or_tie_count": self.ignored_unknown_or_tie_count,
            "pair_conflicts": list(self.pair_conflicts),
            "triplet_conflicts": list(self.triplet_conflicts),
            "conflict_count": self.conflict_count,
            "resolution": "preserve_and_audit_soft_evidence",
            "transitive_closure_edges_added": self.transitive_closure_edges_added,
        }


def audit_constraint_conflicts(
    pair_constraints: Sequence[SoftPairConstraint],
    triplet_constraints: Sequence[SoftTripletConstraint],
    *,
    ignored_unknown_or_tie_count: int = 0,
) -> ConstraintConflictAudit:
    """Report direct contradictions without deleting either observation."""

    if ignored_unknown_or_tie_count < 0:
        raise ValueError("ignored unknown/tie count cannot be negative")
    pair_groups: dict[tuple[tuple[str, str], str], list[SoftPairConstraint]] = {}
    for constraint in pair_constraints:
        key = (constraint.canonical_pair, constraint.dimension)
        pair_groups.setdefault(key, []).append(constraint)
    pair_conflicts: list[dict[str, Any]] = []
    for (pair, dimension), group in sorted(pair_groups.items()):
        targets = {item.target for item in group}
        if targets == {-1, 1}:
            pair_conflicts.append(
                {
                    "uids": list(pair),
                    "dimension": dimension,
                    "positive_weight": float(sum(item.weight for item in group if item.target == 1)),
                    "negative_weight": float(sum(item.weight for item in group if item.target == -1)),
                    "evidence_ids": sorted(item.evidence_id for item in group),
                }
            )

    triplet_groups: dict[tuple[str, frozenset[str], str], list[SoftTripletConstraint]] = {}
    for constraint in triplet_constraints:
        key = (
            constraint.anchor_uid,
            frozenset((constraint.positive_uid, constraint.negative_uid)),
            constraint.dimension,
        )
        triplet_groups.setdefault(key, []).append(constraint)
    triplet_conflicts: list[dict[str, Any]] = []
    for (anchor, alternatives, dimension), group in sorted(
        triplet_groups.items(), key=lambda item: (item[0][0], sorted(item[0][1]), item[0][2])
    ):
        directions = {(item.positive_uid, item.negative_uid) for item in group}
        if len(directions) > 1:
            triplet_conflicts.append(
                {
                    "anchor_uid": anchor,
                    "alternative_uids": sorted(alternatives),
                    "dimension": dimension,
                    "evidence_ids": sorted(item.evidence_id for item in group),
                }
            )
    return ConstraintConflictAudit(
        pair_evidence_count=len(pair_constraints),
        triplet_evidence_count=len(triplet_constraints),
        ignored_unknown_or_tie_count=int(ignored_unknown_or_tie_count),
        pair_conflicts=tuple(pair_conflicts),
        triplet_conflicts=tuple(triplet_conflicts),
    )


@dataclass(frozen=True, slots=True)
class ConstrainedSphericalArtifact:
    environment_id: str
    fit_role: str
    information_access: str
    feature_artifact_sha256: str
    fit_uids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    centers: np.ndarray
    assignments: tuple[str, ...]
    support: tuple[int, ...]
    constraint_sha256: str
    initial_assignment_sha256: str
    geometry_weight: float
    iterations: int
    accepted_moves: int
    maximum_iterations: int
    maximum_accepted_moves: int
    stopped_reason: str
    initial_objective: Mapping[str, float]
    final_objective: Mapping[str, float]
    conflict_audit: Mapping[str, Any]
    algorithm_id: str = ALGORITHM_ID

    def __post_init__(self) -> None:
        _require_sha256(self.feature_artifact_sha256, name="feature artifact hash")
        _require_sha256(self.constraint_sha256, name="constraint hash")
        _require_sha256(self.initial_assignment_sha256, name="initial assignment hash")
        if self.fit_role not in {"train", "train_seen", "discovery"}:
            raise ArtifactIntegrityError("constrained fit role must be train/discovery")
        if self.information_access not in {
            "train_input_teacher_constraints",
            "train_outcome_informed_teacher_constraints",
        }:
            raise ArtifactIntegrityError("invalid constrained information-access declaration")
        if not self.fit_uids or len(self.fit_uids) != len(set(self.fit_uids)):
            raise ArtifactIntegrityError("constrained fit UIDs must be unique and non-empty")
        if not self.cluster_ids or len(self.cluster_ids) != len(set(self.cluster_ids)):
            raise ArtifactIntegrityError("constrained cluster IDs must be unique and non-empty")
        centers = np.asarray(self.centers, dtype=np.float64).copy()
        if centers.shape[0] != len(self.cluster_ids) or centers.ndim != 2:
            raise ArtifactIntegrityError("constrained centre matrix shape mismatch")
        if not np.isfinite(centers).all() or not np.allclose(
            np.linalg.norm(centers, axis=1), 1.0, atol=1e-6
        ):
            raise ArtifactIntegrityError("constrained centres must be finite unit vectors")
        centers.flags.writeable = False
        object.__setattr__(self, "centers", centers)
        if len(self.assignments) != len(self.fit_uids):
            raise ArtifactIntegrityError("constrained assignment row count mismatch")
        if set(self.assignments) != set(self.cluster_ids):
            raise ArtifactIntegrityError("constrained artifact contains an empty/unknown state")
        expected = tuple(self.assignments.count(state) for state in self.cluster_ids)
        if tuple(self.support) != expected or any(value <= 0 for value in expected):
            raise ArtifactIntegrityError("constrained support does not match assignments")
        if self.iterations < 0 or self.iterations > self.maximum_iterations:
            raise ArtifactIntegrityError("constrained iteration bound violated")
        if self.accepted_moves < 0 or self.accepted_moves > self.maximum_accepted_moves:
            raise ArtifactIntegrityError("constrained move bound violated")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_soft_constrained_spherical_cluster",
            "algorithm_id": self.algorithm_id,
            "environment_id": self.environment_id,
            "fit_role": self.fit_role,
            "information_access": self.information_access,
            "label_access_split": "train" if "outcome_informed" in self.information_access else "none",
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "fit_uid_sha256": _uid_sha256(self.fit_uids),
            "fit_uids": list(self.fit_uids),
            "cluster_ids": list(self.cluster_ids),
            "centers": self.centers.tolist(),
            "assignments": list(self.assignments),
            "support": list(self.support),
            "constraint_sha256": self.constraint_sha256,
            "initial_assignment_sha256": self.initial_assignment_sha256,
            "geometry_weight": float(self.geometry_weight),
            "iterations": self.iterations,
            "accepted_moves": self.accepted_moves,
            "maximum_iterations": self.maximum_iterations,
            "maximum_accepted_moves": self.maximum_accepted_moves,
            "stopped_reason": self.stopped_reason,
            "initial_objective": dict(self.initial_objective),
            "final_objective": dict(self.final_objective),
            "conflict_audit": dict(self.conflict_audit),
            "row_drop_count": 0,
            "transitive_closure_edges_added": 0,
            "dev_or_test_refit": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _unit_centres(
    values: np.ndarray, labels: np.ndarray, previous: np.ndarray
) -> np.ndarray:
    result = []
    for cluster in range(previous.shape[0]):
        rows = values[labels == cluster]
        if not len(rows):
            raise RuntimeError("soft refinement attempted to create an empty cluster")
        centre = rows.mean(axis=0)
        norm = float(np.linalg.norm(centre))
        if norm <= 1e-12:
            centre = previous[cluster]
            norm = float(np.linalg.norm(centre))
        result.append(centre / max(norm, 1e-12))
    return np.asarray(result, dtype=np.float64)


def _objective(
    values: np.ndarray,
    labels: np.ndarray,
    centres: np.ndarray,
    pair_rows: Sequence[tuple[int, int, int, float]],
    triplet_rows: Sequence[tuple[int, int, int, float]],
    geometry_weight: float,
) -> dict[str, float]:
    geometry = float(
        geometry_weight
        * np.maximum(0.0, 1.0 - np.sum(values * centres[labels], axis=1)).sum()
    )
    pair_penalty = 0.0
    for a_row, b_row, target, weight in pair_rows:
        violated = labels[a_row] != labels[b_row] if target == 1 else labels[a_row] == labels[b_row]
        pair_penalty += float(weight) * float(violated)
    triplet_penalty = 0.0
    for anchor, positive, negative, weight in triplet_rows:
        # Two bounded, interpretable soft terms; neither is a hard relation.
        triplet_penalty += 0.5 * float(weight) * float(labels[anchor] != labels[positive])
        triplet_penalty += 0.5 * float(weight) * float(labels[anchor] == labels[negative])
    return {
        "geometry": geometry,
        "pair_penalty": pair_penalty,
        "triplet_penalty": triplet_penalty,
        "total": geometry + pair_penalty + triplet_penalty,
    }


def refine_soft_constrained_spherical(
    features: np.ndarray,
    uids: Sequence[str],
    *,
    cluster_ids: Sequence[str],
    initial_centers: np.ndarray,
    initial_assignments: Sequence[str],
    pair_constraints: Sequence[SoftPairConstraint] = (),
    triplet_constraints: Sequence[SoftTripletConstraint] = (),
    feature_artifact_sha256: str,
    environment_id: str,
    fit_role: str,
    audit: UIDAccessAudit,
    geometry_weight: float = 1.0,
    maximum_iterations: int = 10,
    maximum_accepted_moves: int = 256,
    minimum_improvement: float = 1e-12,
    ignored_unknown_or_tie_count: int = 0,
) -> ConstrainedSphericalArtifact:
    """Refine assignments by deterministic bounded coordinate descent."""

    if audit.operation not in _FIT_OPERATIONS:
        raise UIDScopeError("constraint refinement requires fit/train UID scope")
    ids = tuple(str(uid) for uid in uids)
    states = tuple(str(state) for state in cluster_ids)
    assignments = tuple(str(state) for state in initial_assignments)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("constraint refinement UIDs must be unique and non-empty")
    audit.require(ids, component="soft_constrained_spherical.fit")
    if len(assignments) != len(ids) or set(assignments) != set(states):
        raise ValueError("initial assignments must cover every non-empty frozen cluster")
    if geometry_weight <= 0 or not np.isfinite(geometry_weight):
        raise ValueError("geometry weight must be finite and positive")
    if maximum_iterations <= 0 or maximum_accepted_moves <= 0 or minimum_improvement < 0:
        raise ValueError("invalid finite refinement budget")
    _require_sha256(feature_artifact_sha256, name="feature artifact hash")

    values = np.asarray(l2_normalize(features), dtype=np.float64)
    centres = np.asarray(initial_centers, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != len(ids):
        raise ValueError("feature and UID row counts differ")
    if centres.shape != (len(states), values.shape[1]):
        raise ValueError("initial centre shape differs from clusters/features")
    centre_norms = np.linalg.norm(centres, axis=1)
    if not np.isfinite(centres).all() or (centre_norms <= 1e-12).any():
        raise ValueError("initial centres must be finite and non-zero")
    centres /= centre_norms[:, None]
    state_to_index = {state: index for index, state in enumerate(states)}
    labels = np.asarray([state_to_index[state] for state in assignments], dtype=np.int64)
    uid_to_row = {uid: row for row, uid in enumerate(ids)}

    constraint_uids = {
        uid
        for item in pair_constraints
        for uid in (item.a_uid, item.b_uid)
    } | {
        uid
        for item in triplet_constraints
        for uid in (item.anchor_uid, item.positive_uid, item.negative_uid)
    }
    outside = sorted(constraint_uids.difference(uid_to_row))
    if outside:
        raise UIDScopeError(f"constraint references UID outside fit scope: {outside[0]}")
    if any(item.outcome_informed for item in (*pair_constraints, *triplet_constraints)):
        if audit.operation is not Operation.TRAIN:
            raise UIDScopeError("outcome-informed constraints require a train audit")
        information_access = "train_outcome_informed_teacher_constraints"
    else:
        information_access = "train_input_teacher_constraints"

    pair_rows = tuple(
        (uid_to_row[item.a_uid], uid_to_row[item.b_uid], item.target, float(item.weight))
        for item in pair_constraints
    )
    triplet_rows = tuple(
        (
            uid_to_row[item.anchor_uid],
            uid_to_row[item.positive_uid],
            uid_to_row[item.negative_uid],
            float(item.weight),
        )
        for item in triplet_constraints
    )
    conflict_audit = audit_constraint_conflicts(
        pair_constraints,
        triplet_constraints,
        ignored_unknown_or_tie_count=ignored_unknown_or_tie_count,
    )
    constraint_payload = {
        "pairs": [item.to_dict() for item in pair_constraints],
        "triplets": [item.to_dict() for item in triplet_constraints],
        "ignored_unknown_or_tie_count": ignored_unknown_or_tie_count,
    }
    initial_assignment_sha256 = canonical_sha256(list(assignments))
    initial_objective = _objective(
        values, labels, centres, pair_rows, triplet_rows, geometry_weight
    )
    accepted_moves = 0
    stopped_reason = "maximum_iterations_exhausted"
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        changed = False
        for row in range(len(ids)):
            if accepted_moves >= maximum_accepted_moves:
                stopped_reason = "maximum_accepted_moves_exhausted"
                break
            current = int(labels[row])
            if int(np.count_nonzero(labels == current)) <= 1:
                continue
            current_total = _objective(
                values, labels, centres, pair_rows, triplet_rows, geometry_weight
            )["total"]
            best_cluster = current
            best_total = current_total
            for candidate in range(len(states)):
                if candidate == current:
                    continue
                trial = labels.copy()
                trial[row] = candidate
                candidate_total = _objective(
                    values, trial, centres, pair_rows, triplet_rows, geometry_weight
                )["total"]
                if candidate_total < best_total - minimum_improvement:
                    best_cluster = candidate
                    best_total = candidate_total
            if best_cluster != current:
                labels[row] = best_cluster
                accepted_moves += 1
                changed = True
        centres = _unit_centres(values, labels, centres)
        if accepted_moves >= maximum_accepted_moves:
            stopped_reason = "maximum_accepted_moves_exhausted"
            break
        if not changed:
            stopped_reason = "coordinate_fixed_point"
            break

    final_objective = _objective(
        values, labels, centres, pair_rows, triplet_rows, geometry_weight
    )
    final_assignments = tuple(states[int(index)] for index in labels)
    support = tuple(int(np.count_nonzero(labels == index)) for index in range(len(states)))
    return ConstrainedSphericalArtifact(
        environment_id=str(environment_id),
        fit_role=str(fit_role),
        information_access=information_access,
        feature_artifact_sha256=str(feature_artifact_sha256),
        fit_uids=ids,
        cluster_ids=states,
        centers=centres,
        assignments=final_assignments,
        support=support,
        constraint_sha256=canonical_sha256(constraint_payload),
        initial_assignment_sha256=initial_assignment_sha256,
        geometry_weight=float(geometry_weight),
        iterations=iterations,
        accepted_moves=accepted_moves,
        maximum_iterations=int(maximum_iterations),
        maximum_accepted_moves=int(maximum_accepted_moves),
        stopped_reason=stopped_reason,
        initial_objective=initial_objective,
        final_objective=final_objective,
        conflict_audit=conflict_audit.to_dict(),
    )


def assign_frozen_constrained(
    artifact: ConstrainedSphericalArtifact,
    features: np.ndarray,
    *,
    feature_artifact_sha256: str,
) -> AssignmentBatch:
    """Assign new inputs to frozen refined centres without teacher access."""

    if str(feature_artifact_sha256) != artifact.feature_artifact_sha256:
        raise ArtifactIntegrityError("runtime feature hash differs from constrained artifact")
    values = np.asarray(l2_normalize(features), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != artifact.centers.shape[1]:
        raise ArtifactIntegrityError("runtime feature dimension differs from refined centres")
    scores = values @ artifact.centers.T
    selected = scores.argmax(axis=1)
    frozen_scores = np.asarray(scores, dtype=np.float32)
    frozen_scores.flags.writeable = False
    return AssignmentBatch(
        states=tuple(artifact.cluster_ids[int(index)] for index in selected),
        assignment_scores=frozen_scores,
        score_semantics="cosine_assignment_score_not_correctness_probability",
    )
