"""Scalable input-only bisecting spherical semantic tree."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..artifacts import ArtifactIntegrityError, canonical_sha256
from ..features.fusion import l2_normalize
from ..interfaces import AssignmentBatch
from ..leakage_guard import Operation, UIDAccessAudit, UIDScopeError
from .spherical_kmeans import SphericalKMeans


ALGORITHM_ID = "bisecting_spherical_semantic_tree_v1"
_FIT_ROLES = frozenset({"train", "train_seen", "discovery"})
_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")
_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "can",
        "does",
        "following",
        "from",
        "has",
        "image",
        "into",
        "most",
        "not",
        "question",
        "that",
        "the",
        "this",
        "what",
        "when",
        "which",
        "with",
        "would",
    }
)


def _uid_sha256(uids: Sequence[str]) -> str:
    raw = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


def _atomic_envelope(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen tree artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"payload": payload, "payload_sha256": canonical_sha256(payload)}
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


def _load_envelope(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value.get("payload")
    if not isinstance(payload, dict) or value.get("payload_sha256") != canonical_sha256(payload):
        raise ArtifactIntegrityError("semantic-tree artifact digest mismatch")
    return payload


def _unit_mean(features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    mean = features[rows].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        mean = features[int(rows[0])]
        norm = float(np.linalg.norm(mean))
    return mean / max(norm, 1e-12)


def _tokenize(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple(token for token in _TOKEN_PATTERN.findall(normalized) if token not in _STOPWORDS)


@dataclass(frozen=True, slots=True)
class SemanticTreeNode:
    state_id: str
    parent_state_id: str | None
    child_state_ids: tuple[str, ...]
    depth: int
    support: int
    center: np.ndarray
    within_distance_sum: float
    within_distance_mean: float
    within_distance_max: float
    representative_uids: tuple[str, ...]
    keywords: tuple[dict[str, Any], ...]
    split_gain: float | None

    def __post_init__(self) -> None:
        center = np.array(self.center, dtype=np.float64, copy=True)
        if center.ndim != 1 or not np.isfinite(center).all() or not np.isclose(
            np.linalg.norm(center), 1.0, atol=1e-6
        ):
            raise ArtifactIntegrityError("semantic-tree node center must be a finite unit vector")
        center.flags.writeable = False
        object.__setattr__(self, "center", center)
        if self.depth < 0 or self.support <= 0:
            raise ArtifactIntegrityError("tree node depth/support is invalid")
        if len(self.child_state_ids) not in {0, 2}:
            raise ArtifactIntegrityError("bisecting tree nodes must have zero or two children")
        distances = (
            self.within_distance_sum,
            self.within_distance_mean,
            self.within_distance_max,
        )
        if not all(np.isfinite(value) and value >= 0 for value in distances):
            raise ArtifactIntegrityError("node within-distance metrics must be finite/non-negative")
        if not self.representative_uids or len(self.representative_uids) != len(
            set(self.representative_uids)
        ):
            raise ArtifactIntegrityError("node must have unique representative train UIDs")
        if self.child_state_ids and (self.split_gain is None or self.split_gain < -1e-8):
            raise ArtifactIntegrityError("internal node must record non-negative geometric split gain")
        if not self.child_state_ids and self.split_gain is not None:
            raise ArtifactIntegrityError("leaf node cannot record a split gain")
        for keyword in self.keywords:
            if set(keyword) != {"term", "score", "count"}:
                raise ArtifactIntegrityError("keyword record schema mismatch")
            if not keyword["term"] or not np.isfinite(float(keyword["score"])):
                raise ArtifactIntegrityError("keyword record is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "parent_state_id": self.parent_state_id,
            "child_state_ids": list(self.child_state_ids),
            "depth": self.depth,
            "support": self.support,
            "center": self.center.tolist(),
            "within_distance": {
                "sum": self.within_distance_sum,
                "mean": self.within_distance_mean,
                "max": self.within_distance_max,
            },
            "representative_uids": list(self.representative_uids),
            "keywords": list(self.keywords),
            "keyword_semantics": "input_only_tfidf_descriptor_not_ground_truth_subject",
            "split_gain": self.split_gain,
            "fallback_state_id": self.state_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticTreeNode":
        within = value["within_distance"]
        return cls(
            state_id=str(value["state_id"]),
            parent_state_id=(
                None if value.get("parent_state_id") is None else str(value["parent_state_id"])
            ),
            child_state_ids=tuple(str(item) for item in value["child_state_ids"]),
            depth=int(value["depth"]),
            support=int(value["support"]),
            center=np.asarray(value["center"], dtype=np.float64),
            within_distance_sum=float(within["sum"]),
            within_distance_mean=float(within["mean"]),
            within_distance_max=float(within["max"]),
            representative_uids=tuple(str(item) for item in value["representative_uids"]),
            keywords=tuple(dict(item) for item in value["keywords"]),
            split_gain=(None if value.get("split_gain") is None else float(value["split_gain"])),
        )


@dataclass(frozen=True, slots=True)
class SemanticTreeArtifact:
    environment_id: str
    fit_role: str
    feature_artifact_sha256: str
    fit_uids: tuple[str, ...]
    fit_uid_sha256: str
    nodes: tuple[SemanticTreeNode, ...]
    fit_paths: tuple[tuple[str, ...], ...]
    seed: int
    n_init_per_split: int
    max_leaves_requested: int
    max_depth_limit: int
    minimum_child_support: int
    minimum_split_gain: float
    split_attempts: int
    accepted_splits: int
    max_score_matrix_elements: int
    algorithm_id: str = ALGORITHM_ID
    information_access: str = "input_only"

    def __post_init__(self) -> None:
        _require_sha256(self.feature_artifact_sha256, name="feature artifact hash")
        _require_sha256(self.fit_uid_sha256, name="fit UID hash")
        if self.fit_role not in _FIT_ROLES or self.information_access != "input_only":
            raise ArtifactIntegrityError("semantic tree must be input-only and train/discovery-fit")
        if self.algorithm_id != ALGORITHM_ID:
            raise ArtifactIntegrityError("unexpected semantic-tree algorithm identity")
        if not self.fit_uids or len(self.fit_uids) != len(set(self.fit_uids)):
            raise ArtifactIntegrityError("tree fit UIDs must be non-empty and unique")
        if _uid_sha256(self.fit_uids) != self.fit_uid_sha256:
            raise ArtifactIntegrityError("tree fit UID hash mismatch")
        if len(self.fit_paths) != len(self.fit_uids):
            raise ArtifactIntegrityError("tree path count differs from fit UID count")
        node_by_id = {node.state_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes) or "root" not in node_by_id:
            raise ArtifactIntegrityError("tree node IDs must be unique and include root")
        root = node_by_id["root"]
        if root.parent_state_id is not None or root.depth != 0:
            raise ArtifactIntegrityError("tree root metadata is invalid")
        dimensions = {node.center.shape for node in self.nodes}
        if len(dimensions) != 1:
            raise ArtifactIntegrityError("tree node center dimensions differ")
        for node in self.nodes:
            for child_id in node.child_state_ids:
                child = node_by_id.get(child_id)
                if child is None or child.parent_state_id != node.state_id:
                    raise ArtifactIntegrityError("tree parent/child link is inconsistent")
                if child.depth != node.depth + 1:
                    raise ArtifactIntegrityError("tree child depth is inconsistent")
            if node.depth > self.max_depth_limit:
                raise ArtifactIntegrityError("tree exceeds frozen maximum depth")
        memberships: dict[str, set[int]] = {node_id: set() for node_id in node_by_id}
        for row, path in enumerate(self.fit_paths):
            if not path or path[0] != "root" or len(path) != len(set(path)):
                raise ArtifactIntegrityError("training path must be simple and start at root")
            for position, node_id in enumerate(path):
                if node_id not in node_by_id:
                    raise ArtifactIntegrityError("training path references an unknown node")
                memberships[node_id].add(row)
                if position:
                    parent = node_by_id[path[position - 1]]
                    if node_id not in parent.child_state_ids:
                        raise ArtifactIntegrityError("training path violates a parent/child edge")
            if node_by_id[path[-1]].child_state_ids:
                raise ArtifactIntegrityError("fit paths must end at construction leaves")
        uid_set = set(self.fit_uids)
        uid_position = {uid: row for row, uid in enumerate(self.fit_uids)}
        for node in self.nodes:
            if len(memberships[node.state_id]) != node.support:
                raise ArtifactIntegrityError("tree node support differs from path membership")
            if not set(node.representative_uids).issubset(uid_set):
                raise ArtifactIntegrityError("representative UID is outside fit scope")
            if any(
                uid_position[uid] not in memberships[node.state_id]
                for uid in node.representative_uids
            ):
                raise ArtifactIntegrityError("representative UID is outside its tree node")
            if node.child_state_ids:
                left, right = (memberships[child] for child in node.child_state_ids)
                if left & right or left | right != memberships[node.state_id]:
                    raise ArtifactIntegrityError("children are not a mutually exclusive parent cover")
        leaves = [node for node in self.nodes if not node.child_state_ids]
        if len(leaves) != self.accepted_splits + 1 or len(leaves) > self.max_leaves_requested:
            raise ArtifactIntegrityError("semantic-tree effective leaf count is inconsistent")
        if self.split_attempts < self.accepted_splits or self.max_score_matrix_elements < 0:
            raise ArtifactIntegrityError("tree construction counters are inconsistent")

    @property
    def node_by_id(self) -> dict[str, SemanticTreeNode]:
        return {node.state_id: node for node in self.nodes}

    @property
    def leaf_state_ids(self) -> tuple[str, ...]:
        return tuple(node.state_id for node in self.nodes if not node.child_state_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_input_only_bisecting_spherical_tree",
            "algorithm_id": self.algorithm_id,
            "information_access": self.information_access,
            "label_access_split": "none",
            "claim": "input_derived_pseudo_subject_tree_not_ground_truth_taxonomy",
            "environment_id": self.environment_id,
            "fit_role": self.fit_role,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "fit_uid_sha256": self.fit_uid_sha256,
            "fit_uids": list(self.fit_uids),
            "nodes": [node.to_dict() for node in self.nodes],
            "fit_paths": [list(path) for path in self.fit_paths],
            "seed": self.seed,
            "n_init_per_split": self.n_init_per_split,
            "max_leaves_requested": self.max_leaves_requested,
            "effective_leaves": len(self.leaf_state_ids),
            "max_depth_limit": self.max_depth_limit,
            "minimum_child_support": self.minimum_child_support,
            "minimum_split_gain": self.minimum_split_gain,
            "construction_cost": {
                "split_attempts": self.split_attempts,
                "accepted_splits": self.accepted_splits,
                "dense_N_by_N_materialized": False,
                "max_score_matrix_elements": self.max_score_matrix_elements,
                "storage_class": "O(ND_plus_tree_nodes_times_D_plus_paths)",
            },
            "row_drop_count": 0,
            "dev_or_test_refit": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticTreeArtifact":
        cost = value["construction_cost"]
        return cls(
            environment_id=str(value["environment_id"]),
            fit_role=str(value["fit_role"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            fit_uids=tuple(str(uid) for uid in value["fit_uids"]),
            fit_uid_sha256=str(value["fit_uid_sha256"]),
            nodes=tuple(SemanticTreeNode.from_dict(node) for node in value["nodes"]),
            fit_paths=tuple(tuple(str(state) for state in path) for path in value["fit_paths"]),
            seed=int(value["seed"]),
            n_init_per_split=int(value["n_init_per_split"]),
            max_leaves_requested=int(value["max_leaves_requested"]),
            max_depth_limit=int(value["max_depth_limit"]),
            minimum_child_support=int(value["minimum_child_support"]),
            minimum_split_gain=float(value["minimum_split_gain"]),
            split_attempts=int(cost["split_attempts"]),
            accepted_splits=int(cost["accepted_splits"]),
            max_score_matrix_elements=int(cost["max_score_matrix_elements"]),
            algorithm_id=str(value["algorithm_id"]),
            information_access=str(value["information_access"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticTreeArtifact":
        return cls.from_dict(_load_envelope(Path(path)))


@dataclass(frozen=True, slots=True)
class PrunedTreeArtifact:
    tree_sha256: str
    feature_artifact_sha256: str
    fit_uid_sha256: str
    target_leaf_count: int
    effective_leaf_count: int
    max_depth: int
    minimum_support: int
    active_state_ids: tuple[str, ...]
    fallback_state_ids: tuple[str, ...]
    training_routing_states: tuple[str, ...]
    selection_trace: tuple[dict[str, Any], ...]
    information_access: str = "input_only_geometry_only"

    def __post_init__(self) -> None:
        _require_sha256(self.tree_sha256, name="tree hash")
        _require_sha256(self.feature_artifact_sha256, name="feature hash")
        _require_sha256(self.fit_uid_sha256, name="fit UID hash")
        if min(self.target_leaf_count, self.effective_leaf_count, self.minimum_support) <= 0:
            raise ArtifactIntegrityError("pruning leaf/support values must be positive")
        if self.max_depth < 0 or self.effective_leaf_count != len(self.active_state_ids):
            raise ArtifactIntegrityError("pruning depth/effective leaf metadata is inconsistent")
        if len(self.active_state_ids) != len(set(self.active_state_ids)):
            raise ArtifactIntegrityError("active pruning states must be unique")
        if self.information_access != "input_only_geometry_only":
            raise ArtifactIntegrityError("tree pruning cannot use expert outcomes")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_frozen_semantic_tree_pruning",
            "tree_sha256": self.tree_sha256,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "fit_uid_sha256": self.fit_uid_sha256,
            "target_leaf_count": self.target_leaf_count,
            "effective_leaf_count": self.effective_leaf_count,
            "max_depth": self.max_depth,
            "minimum_support": self.minimum_support,
            "active_state_ids": list(self.active_state_ids),
            "fallback_state_ids": list(self.fallback_state_ids),
            "training_routing_states": list(self.training_routing_states),
            "selection_trace": list(self.selection_trace),
            "selection_signal": "input_geometry_split_gain_only",
            "information_access": self.information_access,
            "expert_performance_used": False,
            "dev_or_test_used": False,
        }

    def validate_against(self, tree: SemanticTreeArtifact) -> None:
        if self.tree_sha256 != tree.sha256:
            raise ArtifactIntegrityError("pruning artifact is not linked to this semantic tree")
        if self.feature_artifact_sha256 != tree.feature_artifact_sha256:
            raise ArtifactIntegrityError("pruning/tree feature hashes differ")
        if self.fit_uid_sha256 != tree.fit_uid_sha256:
            raise ArtifactIntegrityError("pruning/tree fit UID hashes differ")
        nodes = tree.node_by_id
        active = set(self.active_state_ids)
        if not active.issubset(nodes):
            raise ArtifactIntegrityError("pruning references an unknown active state")
        if any(nodes[state].depth > self.max_depth for state in active):
            raise ArtifactIntegrityError("active state exceeds pruning max depth")
        for state in active:
            if state != "root" and nodes[state].support < self.minimum_support:
                raise ArtifactIntegrityError("active state violates pruning minimum support")
        expected_states = []
        for path in tree.fit_paths:
            matches = [state for state in path if state in active]
            if len(matches) != 1:
                raise ArtifactIntegrityError("active states do not form a complete tree frontier")
            expected_states.append(matches[0])
        if tuple(expected_states) != self.training_routing_states:
            raise ArtifactIntegrityError("frozen training routing states differ from tree paths")
        expected_fallback = tuple(node.state_id for node in tree.nodes if node.child_state_ids)
        if self.fallback_state_ids != expected_fallback:
            raise ArtifactIntegrityError("pruning does not preserve every internal fallback state")

    def save(self, path: str | Path, tree: SemanticTreeArtifact) -> None:
        self.validate_against(tree)
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PrunedTreeArtifact":
        return cls(
            tree_sha256=str(value["tree_sha256"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            fit_uid_sha256=str(value["fit_uid_sha256"]),
            target_leaf_count=int(value["target_leaf_count"]),
            effective_leaf_count=int(value["effective_leaf_count"]),
            max_depth=int(value["max_depth"]),
            minimum_support=int(value["minimum_support"]),
            active_state_ids=tuple(str(item) for item in value["active_state_ids"]),
            fallback_state_ids=tuple(str(item) for item in value["fallback_state_ids"]),
            training_routing_states=tuple(
                str(item) for item in value["training_routing_states"]
            ),
            selection_trace=tuple(dict(item) for item in value["selection_trace"]),
            information_access=str(value["information_access"]),
        )

    @classmethod
    def load(
        cls, path: str | Path, tree: SemanticTreeArtifact
    ) -> "PrunedTreeArtifact":
        result = cls.from_dict(_load_envelope(Path(path)))
        result.validate_against(tree)
        return result


@dataclass(frozen=True, slots=True)
class TreeAssignmentTrace:
    batch: AssignmentBatch
    paths: tuple[tuple[str, ...], ...]
    binary_decisions: tuple[int, ...]


class BisectingSphericalTree:
    """Top-down split of the highest-distortion eligible leaf, never NxN."""

    def __init__(
        self,
        *,
        max_leaves: int,
        max_depth: int,
        minimum_child_support: int,
        minimum_split_gain: float,
        seed: int,
        n_init_per_split: int = 5,
        keyword_count: int = 8,
        representative_count: int = 3,
    ) -> None:
        if min(max_leaves, minimum_child_support, n_init_per_split, keyword_count) <= 0:
            raise ValueError("tree leaf/support/init/keyword settings must be positive")
        if max_depth < 0 or minimum_split_gain < 0 or representative_count <= 0:
            raise ValueError("tree depth/gain/representative settings are invalid")
        self.max_leaves = int(max_leaves)
        self.max_depth = int(max_depth)
        self.minimum_child_support = int(minimum_child_support)
        self.minimum_split_gain = float(minimum_split_gain)
        self.seed = int(seed)
        self.n_init_per_split = int(n_init_per_split)
        self.keyword_count = int(keyword_count)
        self.representative_count = int(representative_count)
        self.artifact_: SemanticTreeArtifact | None = None

    def fit(
        self,
        features: np.ndarray,
        uids: Sequence[str],
        documents: Sequence[str],
        *,
        environment_id: str,
        fit_role: str,
        feature_artifact_sha256: str,
        audit: UIDAccessAudit,
    ) -> SemanticTreeArtifact:
        if audit.operation is not Operation.FIT:
            raise UIDScopeError("semantic-tree fit requires a fit UID scope")
        if fit_role not in _FIT_ROLES:
            raise UIDScopeError("semantic tree may fit only train/train_seen/discovery")
        ids = tuple(str(uid) for uid in uids)
        texts = tuple(str(document) for document in documents)
        if not ids or len(ids) != len(set(ids)) or len(texts) != len(ids):
            raise ValueError("tree UIDs/documents must be unique, non-empty, and aligned")
        audit.require(ids, component="bisecting_spherical_semantic_tree.fit")
        _require_sha256(feature_artifact_sha256, name="feature artifact hash")
        values = np.asarray(l2_normalize(features), dtype=np.float64)
        if values.shape[0] != len(ids):
            raise ValueError("tree feature and UID row counts differ")

        tokenized = tuple(_tokenize(text) for text in texts)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        rows_by_node: dict[str, np.ndarray] = {
            "root": np.arange(len(ids), dtype=np.int64)
        }
        centers: dict[str, np.ndarray] = {"root": _unit_mean(values, rows_by_node["root"])}
        parents: dict[str, str | None] = {"root": None}
        children: dict[str, tuple[str, ...]] = {"root": ()}
        depths: dict[str, int] = {"root": 0}
        gains: dict[str, float | None] = {"root": None}
        leaves: set[str] = {"root"}
        unsplittable: set[str] = set()
        fit_paths: list[list[str]] = [["root"] for _ in ids]
        split_attempts = 0
        accepted_splits = 0
        max_score_elements = 0

        def distortion(node_id: str) -> float:
            rows = rows_by_node[node_id]
            return float(
                np.maximum(0.0, 1.0 - values[rows] @ centers[node_id]).sum()
            )

        while len(leaves) < self.max_leaves:
            eligible = [
                node_id
                for node_id in leaves
                if node_id not in unsplittable
                and depths[node_id] < self.max_depth
                and len(rows_by_node[node_id]) >= 2 * self.minimum_child_support
            ]
            if not eligible:
                break
            node_id = sorted(eligible, key=lambda item: (-distortion(item), item))[0]
            node_rows = rows_by_node[node_id]
            node_uids = tuple(ids[int(row)] for row in node_rows)
            node_seed = self.seed ^ int(hashlib.sha256(node_id.encode()).hexdigest()[:8], 16)
            split_attempts += 1
            split = SphericalKMeans(
                2,
                seed=node_seed,
                n_init=self.n_init_per_split,
                max_iter=100,
                tolerance=1e-6,
            ).fit(
                values[node_rows],
                node_uids,
                environment_id=environment_id,
                fit_role=fit_role,
                feature_artifact_sha256=feature_artifact_sha256,
                audit=audit,
            )
            max_score_elements = max(max_score_elements, len(node_rows) * 2)
            local = np.asarray(
                [split.cluster_ids.index(state) for state in split.assignments], dtype=np.int64
            )
            child_rows = tuple(node_rows[local == side] for side in range(2))
            if min(len(rows) for rows in child_rows) < self.minimum_child_support:
                unsplittable.add(node_id)
                continue
            child_centers = tuple(_unit_mean(values, rows) for rows in child_rows)
            child_distortion = sum(
                float(np.maximum(0.0, 1.0 - values[rows] @ center).sum())
                for rows, center in zip(child_rows, child_centers)
            )
            gain = distortion(node_id) - child_distortion
            if gain + 1e-10 < self.minimum_split_gain:
                unsplittable.add(node_id)
                continue
            child_ids = (f"{node_id}/L", f"{node_id}/R")
            children[node_id] = child_ids
            gains[node_id] = max(0.0, float(gain))
            leaves.remove(node_id)
            leaves.update(child_ids)
            for child_id, rows, center in zip(child_ids, child_rows, child_centers):
                rows_by_node[child_id] = rows
                centers[child_id] = center
                parents[child_id] = node_id
                children[child_id] = ()
                depths[child_id] = depths[node_id] + 1
                gains[child_id] = None
                for row in rows:
                    fit_paths[int(row)].append(child_id)
            accepted_splits += 1

        nodes = []
        for state_id in sorted(rows_by_node, key=lambda item: (depths[item], item)):
            rows = rows_by_node[state_id]
            center = centers[state_id]
            distances = np.maximum(0.0, 1.0 - values[rows] @ center)
            representative_order = np.lexsort((rows, -values[rows] @ center))
            representatives = tuple(
                ids[int(rows[index])]
                for index in representative_order[: self.representative_count]
            )
            counts: Counter[str] = Counter()
            for row in rows:
                counts.update(tokenized[int(row)])
            scored_keywords = []
            total_terms = max(1, sum(counts.values()))
            for term, count in counts.items():
                inverse_frequency = math.log(
                    (len(ids) + 1) / (document_frequency[term] + 1)
                ) + 1.0
                score = count / total_terms * inverse_frequency
                scored_keywords.append((float(score), term, int(count)))
            scored_keywords.sort(key=lambda item: (-item[0], item[1]))
            keywords = tuple(
                {"term": term, "score": score, "count": count}
                for score, term, count in scored_keywords[: self.keyword_count]
            )
            nodes.append(
                SemanticTreeNode(
                    state_id=state_id,
                    parent_state_id=parents[state_id],
                    child_state_ids=children[state_id],
                    depth=depths[state_id],
                    support=int(len(rows)),
                    center=center,
                    within_distance_sum=float(distances.sum()),
                    within_distance_mean=float(distances.mean()),
                    within_distance_max=float(distances.max()),
                    representative_uids=representatives,
                    keywords=keywords,
                    split_gain=gains[state_id],
                )
            )
        artifact = SemanticTreeArtifact(
            environment_id=str(environment_id),
            fit_role=fit_role,
            feature_artifact_sha256=str(feature_artifact_sha256),
            fit_uids=ids,
            fit_uid_sha256=_uid_sha256(ids),
            nodes=tuple(nodes),
            fit_paths=tuple(tuple(path) for path in fit_paths),
            seed=self.seed,
            n_init_per_split=self.n_init_per_split,
            max_leaves_requested=self.max_leaves,
            max_depth_limit=self.max_depth,
            minimum_child_support=self.minimum_child_support,
            minimum_split_gain=self.minimum_split_gain,
            split_attempts=split_attempts,
            accepted_splits=accepted_splits,
            max_score_matrix_elements=max_score_elements,
        )
        self.artifact_ = artifact
        return artifact


def prune_tree(
    tree: SemanticTreeArtifact,
    *,
    target_leaf_count: int,
    max_depth: int,
    minimum_support: int,
) -> PrunedTreeArtifact:
    """Freeze an input-geometry-only frontier before any expert profiling."""

    if target_leaf_count <= 0 or max_depth < 0 or minimum_support <= 0:
        raise ValueError("invalid fixed tree-pruning criterion")
    nodes = tree.node_by_id
    active: set[str] = {"root"}
    trace: list[dict[str, Any]] = []
    while len(active) < target_leaf_count:
        eligible = []
        for state in active:
            node = nodes[state]
            if not node.child_state_ids or node.depth >= max_depth:
                continue
            if any(nodes[child].support < minimum_support for child in node.child_state_ids):
                continue
            eligible.append(state)
        if not eligible:
            break
        chosen = sorted(
            eligible,
            key=lambda state: (-(nodes[state].split_gain or 0.0), state),
        )[0]
        active.remove(chosen)
        active.update(nodes[chosen].child_state_ids)
        trace.append(
            {
                "split_state_id": chosen,
                "geometric_split_gain": nodes[chosen].split_gain,
                "active_count_after": len(active),
            }
        )
    active_order = tuple(node.state_id for node in tree.nodes if node.state_id in active)
    training_states = tuple(
        next(state for state in path if state in active) for path in tree.fit_paths
    )
    fallback = tuple(node.state_id for node in tree.nodes if node.child_state_ids)
    artifact = PrunedTreeArtifact(
        tree_sha256=tree.sha256,
        feature_artifact_sha256=tree.feature_artifact_sha256,
        fit_uid_sha256=tree.fit_uid_sha256,
        target_leaf_count=int(target_leaf_count),
        effective_leaf_count=len(active_order),
        max_depth=int(max_depth),
        minimum_support=int(minimum_support),
        active_state_ids=active_order,
        fallback_state_ids=fallback,
        training_routing_states=training_states,
        selection_trace=tuple(trace),
    )
    artifact.validate_against(tree)
    return artifact


def assign_frozen_tree_with_trace(
    tree: SemanticTreeArtifact,
    pruning: PrunedTreeArtifact,
    features: np.ndarray,
    *,
    feature_artifact_sha256: str,
    minimum_child_similarity: float = -1.0,
) -> TreeAssignmentTrace:
    """Route rows independently down frozen child centers; never updates the tree."""

    pruning.validate_against(tree)
    if str(feature_artifact_sha256) != tree.feature_artifact_sha256:
        raise ArtifactIntegrityError("runtime feature hash differs from frozen semantic tree")
    if not -1.0 <= minimum_child_similarity <= 1.0:
        raise ValueError("minimum child similarity must lie in [-1, 1]")
    values = np.asarray(l2_normalize(features), dtype=np.float64)
    nodes = tree.node_by_id
    if values.shape[1] != nodes["root"].center.shape[0]:
        raise ArtifactIntegrityError("runtime feature dimension differs from semantic tree")
    active = set(pruning.active_state_ids)
    states = []
    paths = []
    decisions = []
    for feature in values:
        state = "root"
        path = [state]
        decision_count = 0
        while state not in active and nodes[state].child_state_ids:
            children = nodes[state].child_state_ids
            similarities = np.asarray([feature @ nodes[child].center for child in children])
            decision_count += len(children)
            selected = int(np.argmax(similarities))
            if float(similarities[selected]) < minimum_child_similarity:
                break
            state = children[selected]
            path.append(state)
        states.append(state)
        paths.append(tuple(path))
        decisions.append(decision_count)
    routing_states = tuple(
        node.state_id
        for node in tree.nodes
        if node.state_id in set(pruning.active_state_ids) | set(pruning.fallback_state_ids)
    )
    score_matrix = values @ np.vstack([nodes[state].center for state in routing_states]).T
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    score_matrix.flags.writeable = False
    batch = AssignmentBatch(
        states=tuple(states),
        assignment_scores=score_matrix,
        score_semantics="cosine_to_frozen_routing_state_centers_not_probability",
    )
    return TreeAssignmentTrace(
        batch=batch,
        paths=tuple(paths),
        binary_decisions=tuple(decisions),
    )


def assign_frozen_tree(
    tree: SemanticTreeArtifact,
    pruning: PrunedTreeArtifact,
    features: np.ndarray,
    *,
    feature_artifact_sha256: str,
    minimum_child_similarity: float = -1.0,
) -> AssignmentBatch:
    return assign_frozen_tree_with_trace(
        tree,
        pruning,
        features,
        feature_artifact_sha256=feature_artifact_sha256,
        minimum_child_similarity=minimum_child_similarity,
    ).batch
