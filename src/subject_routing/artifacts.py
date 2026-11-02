"""Hash-linked, label-free deployment artifacts for subject routing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class ArtifactIntegrityError(ValueError):
    """Raised for malformed, stale, or mismatched subject artifacts."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


@dataclass(frozen=True, slots=True)
class ClusterNode:
    state_id: str
    name: str
    parent_state_id: str | None
    depth: int
    state_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "name": self.name,
            "parent_state_id": self.parent_state_id,
            "depth": self.depth,
            "state_kind": self.state_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClusterNode":
        return cls(
            state_id=str(value["state_id"]),
            name=str(value["name"]),
            parent_state_id=(
                None
                if value.get("parent_state_id") is None
                else str(value["parent_state_id"])
            ),
            depth=int(value["depth"]),
            state_kind=str(value["state_kind"]),
        )


@dataclass(frozen=True, slots=True)
class ClusterArtifact:
    method_id: str
    environment_id: str
    information_access: str
    label_access_split: str
    feature_artifact_sha256: str
    fit_uid_sha256: str
    nodes: tuple[ClusterNode, ...]
    discovery_assignments: Mapping[str, str]
    unknown_state_id: str = "__unknown__"

    def __post_init__(self) -> None:
        _require_digest(self.feature_artifact_sha256, name="feature_artifact_sha256")
        _require_digest(self.fit_uid_sha256, name="fit_uid_sha256")
        if self.information_access not in {"input_only", "train_outcome_guided"}:
            raise ArtifactIntegrityError("unsupported information access type")
        expected_split = "none" if self.information_access == "input_only" else "train"
        if self.label_access_split != expected_split:
            raise ArtifactIntegrityError("cluster label-access metadata is inconsistent")
        node_ids = [node.state_id for node in self.nodes]
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise ArtifactIntegrityError("cluster node IDs must be non-empty and unique")
        if self.unknown_state_id in node_ids:
            raise ArtifactIntegrityError("unknown state must be separate from semantic nodes")
        by_id = {node.state_id: node for node in self.nodes}
        for node in self.nodes:
            if node.parent_state_id is None:
                if node.depth != 0:
                    raise ArtifactIntegrityError("root node depth must be zero")
            elif node.parent_state_id not in by_id:
                raise ArtifactIntegrityError("cluster node references an unknown parent")
            elif by_id[node.parent_state_id].depth + 1 != node.depth:
                raise ArtifactIntegrityError("cluster node depth/parent relationship is invalid")
        if any(state not in by_id for state in self.discovery_assignments.values()):
            raise ArtifactIntegrityError("discovery assignment references an unknown state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "environment_id": self.environment_id,
            "information_access": self.information_access,
            "label_access_split": self.label_access_split,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "fit_uid_sha256": self.fit_uid_sha256,
            "nodes": [node.to_dict() for node in self.nodes],
            "discovery_assignments": dict(sorted(self.discovery_assignments.items())),
            "unknown_state_id": self.unknown_state_id,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClusterArtifact":
        return cls(
            method_id=str(value["method_id"]),
            environment_id=str(value["environment_id"]),
            information_access=str(value["information_access"]),
            label_access_split=str(value["label_access_split"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            fit_uid_sha256=str(value["fit_uid_sha256"]),
            nodes=tuple(ClusterNode.from_dict(node) for node in value["nodes"]),
            discovery_assignments=dict(value["discovery_assignments"]),
            unknown_state_id=str(value["unknown_state_id"]),
        )


@dataclass(frozen=True, slots=True)
class AssignerArtifact:
    mechanism: str
    feature_artifact_sha256: str
    cluster_artifact_sha256: str
    state_ids: tuple[str, ...]
    feature_dimension: int
    parameters: Mapping[str, Any]
    unknown_state_id: str

    def __post_init__(self) -> None:
        _require_digest(self.feature_artifact_sha256, name="assigner feature hash")
        _require_digest(self.cluster_artifact_sha256, name="assigner cluster hash")
        if self.mechanism not in {"fixed_prototypes", "student_linear"}:
            raise ArtifactIntegrityError(
                "only inductive fixed-prototype or student classifiers are deployable"
            )
        if not self.state_ids or len(self.state_ids) != len(set(self.state_ids)):
            raise ArtifactIntegrityError("assigner state IDs must be unique")
        if self.feature_dimension <= 0:
            raise ArtifactIntegrityError("assigner feature dimension must be positive")
        expected = "prototypes" if self.mechanism == "fixed_prototypes" else "weights"
        matrix = np.asarray(self.parameters.get(expected), dtype=np.float64)
        if matrix.shape != (len(self.state_ids), self.feature_dimension):
            raise ArtifactIntegrityError(f"assigner {expected} shape mismatch")
        if not np.isfinite(matrix).all():
            raise ArtifactIntegrityError("assigner parameters must be finite")
        if self.mechanism == "student_linear":
            bias = np.asarray(self.parameters.get("bias"), dtype=np.float64)
            if bias.shape != (len(self.state_ids),) or not np.isfinite(bias).all():
                raise ArtifactIntegrityError("student bias shape/value mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "cluster_artifact_sha256": self.cluster_artifact_sha256,
            "state_ids": list(self.state_ids),
            "feature_dimension": self.feature_dimension,
            "parameters": dict(self.parameters),
            "unknown_state_id": self.unknown_state_id,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssignerArtifact":
        return cls(
            mechanism=str(value["mechanism"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            cluster_artifact_sha256=str(value["cluster_artifact_sha256"]),
            state_ids=tuple(str(item) for item in value["state_ids"]),
            feature_dimension=int(value["feature_dimension"]),
            parameters=dict(value["parameters"]),
            unknown_state_id=str(value["unknown_state_id"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyArtifact:
    cluster_artifact_sha256: str
    expert_ids: tuple[str, ...]
    budgets: Mapping[str, tuple[str, ...]]
    table: Mapping[str, Mapping[str, str]]
    unknown_state_id: str

    def __post_init__(self) -> None:
        _require_digest(self.cluster_artifact_sha256, name="policy cluster hash")
        if not self.expert_ids or len(self.expert_ids) != len(set(self.expert_ids)):
            raise ArtifactIntegrityError("policy expert pool must be unique")
        if not self.budgets or set(self.budgets) != set(self.table):
            raise ArtifactIntegrityError("policy budget definitions/table differ")
        pool = set(self.expert_ids)
        for budget, allowed in self.budgets.items():
            if not allowed or not set(allowed).issubset(pool):
                raise ArtifactIntegrityError(f"budget {budget} contains an unknown expert")
            canonical_subset = tuple(
                expert for expert in self.expert_ids if expert in set(allowed)
            )
            if tuple(allowed) != canonical_subset:
                raise ArtifactIntegrityError(
                    f"budget {budget} experts must preserve canonical model order"
                )
            mapping = self.table[budget]
            if self.unknown_state_id not in mapping:
                raise ArtifactIntegrityError(f"budget {budget} has no frozen unknown fallback")
            if any(expert not in allowed for expert in mapping.values()):
                raise ArtifactIntegrityError(f"budget {budget} policy selects a disallowed expert")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_artifact_sha256": self.cluster_artifact_sha256,
            "expert_ids": list(self.expert_ids),
            "budgets": {key: list(value) for key, value in sorted(self.budgets.items())},
            "table": {
                budget: dict(sorted(mapping.items()))
                for budget, mapping in sorted(self.table.items())
            },
            "unknown_state_id": self.unknown_state_id,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyArtifact":
        return cls(
            cluster_artifact_sha256=str(value["cluster_artifact_sha256"]),
            expert_ids=tuple(str(item) for item in value["expert_ids"]),
            budgets={key: tuple(items) for key, items in value["budgets"].items()},
            table={key: dict(mapping) for key, mapping in value["table"].items()},
            unknown_state_id=str(value["unknown_state_id"]),
        )


@dataclass(frozen=True, slots=True)
class DeploymentBundle:
    cluster: ClusterArtifact
    assigner: AssignerArtifact
    policy: PolicyArtifact

    def __post_init__(self) -> None:
        if self.assigner.cluster_artifact_sha256 != self.cluster.sha256:
            raise ArtifactIntegrityError("assigner is not bound to cluster artifact")
        if self.policy.cluster_artifact_sha256 != self.cluster.sha256:
            raise ArtifactIntegrityError("policy is not bound to cluster artifact")
        if self.assigner.feature_artifact_sha256 != self.cluster.feature_artifact_sha256:
            raise ArtifactIntegrityError("cluster and assigner feature hashes differ")
        cluster_states = {node.state_id for node in self.cluster.nodes}
        if not set(self.assigner.state_ids).issubset(cluster_states):
            raise ArtifactIntegrityError("assigner states are absent from cluster hierarchy")
        required_states = cluster_states | {self.cluster.unknown_state_id}
        for mapping in self.policy.table.values():
            if not required_states.issubset(mapping):
                raise ArtifactIntegrityError("policy table does not cover every frozen state")

    def to_dict(self) -> dict[str, Any]:
        payloads = {
            "cluster": self.cluster.to_dict(),
            "assigner": self.assigner.to_dict(),
            "policy": self.policy.to_dict(),
        }
        return {
            "schema_version": 1,
            "kind": "subject_routing_label_free_deployment_bundle",
            "artifacts": {
                name: {"sha256": canonical_sha256(payload), "payload": payload}
                for name, payload in payloads.items()
            },
            "represented_runtime_objects": ["cluster", "assigner", "fixed_policy_table"],
            "explicitly_absent_runtime_objects": [
                "training_label_store",
                "evaluation_label_store",
                "Y",
                "C",
                "expert_outputs",
            ],
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        target = Path(path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite deployment bundle: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["bundle_sha256"] = canonical_sha256(payload)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> "DeploymentBundle":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        recorded_bundle = value.pop("bundle_sha256", None)
        if recorded_bundle != canonical_sha256(value):
            raise ArtifactIntegrityError("deployment bundle digest mismatch")
        artifacts = value.get("artifacts", {})
        for name in ("cluster", "assigner", "policy"):
            entry = artifacts.get(name, {})
            if entry.get("sha256") != canonical_sha256(entry.get("payload")):
                raise ArtifactIntegrityError(f"deployment {name} digest mismatch")
        return cls(
            cluster=ClusterArtifact.from_dict(artifacts["cluster"]["payload"]),
            assigner=AssignerArtifact.from_dict(artifacts["assigner"]["payload"]),
            policy=PolicyArtifact.from_dict(artifacts["policy"]["payload"]),
        )
