"""Interoperable fit/assign/profile/route interface for subject methods."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .artifacts import (
    ArtifactIntegrityError,
    AssignerArtifact,
    ClusterArtifact,
    ClusterNode,
    DeploymentBundle,
    PolicyArtifact,
)
from .data_contracts import TrainingLabels
from .features.fusion import l2_normalize
from .leakage_guard import Operation, UIDAccessAudit, UIDScopeError
from .policy import StrictSubjectRouter, fit_profile_policy


def _uid_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AssignmentBatch:
    states: tuple[str, ...]
    assignment_scores: np.ndarray
    score_semantics: str = "assignment_score_not_correctness_probability"

    def __post_init__(self) -> None:
        if self.assignment_scores.ndim != 2 or self.assignment_scores.shape[0] != len(
            self.states
        ):
            raise ArtifactIntegrityError("assignment score row count mismatch")
        if not np.isfinite(self.assignment_scores).all():
            raise ArtifactIntegrityError("assignment scores must be finite")


class SubjectMethod(Protocol):
    def fit_discovery(self, *args: Any, **kwargs: Any) -> ClusterArtifact: ...

    def fit_assigner(self, *args: Any, **kwargs: Any) -> AssignerArtifact: ...

    def assign(self, *args: Any, **kwargs: Any) -> AssignmentBatch: ...

    def fit_profile(self, *args: Any, **kwargs: Any) -> PolicyArtifact: ...

    def route(self, *args: Any, **kwargs: Any) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class FrozenSubjectPipeline:
    """Reference implementation shared by baseline/frontier subject methods."""

    @staticmethod
    def fit_discovery(
        *,
        method_id: str,
        environment_id: str,
        uids: Sequence[str],
        discovered_states: Sequence[str],
        nodes: Sequence[ClusterNode],
        feature_artifact_sha256: str,
        information_access: str,
        audit: UIDAccessAudit,
    ) -> ClusterArtifact:
        if audit.operation is not Operation.FIT:
            raise UIDScopeError("fit_discovery requires a fit UID scope")
        ids = tuple(str(uid) for uid in uids)
        states = tuple(str(state) for state in discovered_states)
        if len(ids) != len(states):
            raise ArtifactIntegrityError("discovery UID/state count mismatch")
        audit.require(ids, component="fit_discovery")
        return ClusterArtifact(
            method_id=str(method_id),
            environment_id=str(environment_id),
            information_access=str(information_access),
            label_access_split=(
                "none" if information_access == "input_only" else "train"
            ),
            feature_artifact_sha256=str(feature_artifact_sha256),
            fit_uid_sha256=_uid_sha256(ids),
            nodes=tuple(nodes),
            discovery_assignments=dict(zip(ids, states)),
        )

    @staticmethod
    def fit_assigner(
        cluster: ClusterArtifact,
        features: np.ndarray,
        uids: Sequence[str],
        *,
        audit: UIDAccessAudit,
        unknown_score_threshold: float = -1.0,
    ) -> AssignerArtifact:
        if audit.operation is not Operation.FIT:
            raise UIDScopeError("fit_assigner requires a fit UID scope")
        ids = tuple(str(uid) for uid in uids)
        audit.require(ids, component="fit_assigner")
        values = l2_normalize(features)
        if values.shape[0] != len(ids):
            raise ArtifactIntegrityError("assigner feature/UID count mismatch")
        if any(uid not in cluster.discovery_assignments for uid in ids):
            raise ArtifactIntegrityError("assigner UID absent from discovery artifact")
        leaf_states = tuple(
            node.state_id for node in cluster.nodes if node.state_kind == "leaf"
        )
        prototypes = []
        for state in leaf_states:
            rows = [
                row
                for row, uid in enumerate(ids)
                if cluster.discovery_assignments[uid] == state
            ]
            if not rows:
                raise ArtifactIntegrityError(f"leaf state has no discovery support: {state}")
            prototypes.append(l2_normalize(values[rows].mean(axis=0, keepdims=True))[0])
        return AssignerArtifact(
            mechanism="fixed_prototypes",
            feature_artifact_sha256=cluster.feature_artifact_sha256,
            cluster_artifact_sha256=cluster.sha256,
            state_ids=leaf_states,
            feature_dimension=values.shape[1],
            parameters={
                "prototypes": np.asarray(prototypes).tolist(),
                "unknown_score_threshold": float(unknown_score_threshold),
            },
            unknown_state_id=cluster.unknown_state_id,
        )

    @staticmethod
    def assign(
        assigner: AssignerArtifact,
        features: np.ndarray,
        *,
        feature_artifact_sha256: str,
    ) -> AssignmentBatch:
        if str(feature_artifact_sha256) != assigner.feature_artifact_sha256:
            raise ArtifactIntegrityError("runtime feature hash differs from frozen assigner")
        values = l2_normalize(features)
        if values.shape[1] != assigner.feature_dimension:
            raise ArtifactIntegrityError("runtime feature dimension mismatch")
        if assigner.mechanism == "fixed_prototypes":
            prototypes = l2_normalize(np.asarray(assigner.parameters["prototypes"]))
            scores = values @ prototypes.T
            selected = np.argmax(scores, axis=1)
            threshold = float(assigner.parameters.get("unknown_score_threshold", -1.0))
            states = tuple(
                assigner.unknown_state_id
                if scores[row, column] < threshold
                else assigner.state_ids[column]
                for row, column in enumerate(selected)
            )
        elif assigner.mechanism == "student_linear":
            weights = np.asarray(assigner.parameters["weights"], dtype=np.float32)
            bias = np.asarray(assigner.parameters["bias"], dtype=np.float32)
            scores = values @ weights.T + bias
            selected = np.argmax(scores, axis=1)
            threshold = float(assigner.parameters.get("unknown_score_threshold", float("-inf")))
            states = tuple(
                assigner.unknown_state_id
                if scores[row, column] < threshold
                else assigner.state_ids[column]
                for row, column in enumerate(selected)
            )
        else:  # pragma: no cover - artifact constructor already blocks this
            raise ArtifactIntegrityError("unsupported deployable assigner")
        frozen_scores = np.asarray(scores, dtype=np.float32)
        frozen_scores.flags.writeable = False
        return AssignmentBatch(states=states, assignment_scores=frozen_scores)

    @staticmethod
    def fit_profile(
        cluster: ClusterArtifact,
        profile_assignments: Mapping[str, str],
        labels: TrainingLabels,
        budgets: Mapping[str, Sequence[str]],
        *,
        audit: UIDAccessAudit,
    ) -> PolicyArtifact:
        return fit_profile_policy(
            cluster, profile_assignments, labels, budgets, audit=audit
        )

    @staticmethod
    def route(
        policy: PolicyArtifact, states: Sequence[str], *, budget_id: str
    ) -> tuple[str, ...]:
        return StrictSubjectRouter(policy).route(states, budget_id=budget_id)

    @staticmethod
    def bundle(
        cluster: ClusterArtifact,
        assigner: AssignerArtifact,
        policy: PolicyArtifact,
    ) -> DeploymentBundle:
        return DeploymentBundle(cluster=cluster, assigner=assigner, policy=policy)
