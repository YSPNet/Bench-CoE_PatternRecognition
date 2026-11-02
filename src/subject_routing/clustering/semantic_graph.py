"""Train-scoped sparse semantic kNN graphs and frozen Leiden deployment.

Graph construction is input-only and materializes at most one query block by
all training rows, never a dense N-by-N matrix.  Community detection is loaded
lazily from the real ``igraph`` + ``leidenalg`` packages; no substitute
community algorithm is accepted when those optional dependencies are absent.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ..artifacts import ArtifactIntegrityError, canonical_sha256
from ..features.fusion import l2_normalize
from ..interfaces import AssignmentBatch
from ..leakage_guard import Operation, UIDAccessAudit, UIDScopeError


GRAPH_ALGORITHM_ID = "input_only_sparse_blockwise_cosine_knn_v1"
LEIDEN_ALGORITHM_ID = "igraph_leidenalg_weighted_rb_configuration_v1"
PROTOTYPE_ALGORITHM_ID = "frozen_leiden_community_cosine_prototypes_v1"
REAL_LEIDEN_BACKEND = "leidenalg.find_partition.RBConfigurationVertexPartition"
FIXTURE_BACKEND = "fixture_membership_not_leiden_execution"
NO_EDGE_BACKEND = "constant_fallback_no_graph_edges"
_FIT_ROLES = frozenset({"train", "train_seen", "discovery"})
_SYMMETRIZATIONS = frozenset({"union_max", "mutual_mean"})


class LeidenDependencyError(RuntimeError):
    """Raised when the exact real Leiden stack is unavailable."""


def _uid_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


def _edge_hash(uids: Sequence[str], edges: Sequence["GraphEdge"], *, directed: bool) -> str:
    payload = [
        {
            "source_uid": uids[edge.source],
            "target_uid": uids[edge.target],
            "similarity": edge.similarity,
        }
        for edge in edges
    ]
    return canonical_sha256({"directed": directed, "edges": payload})


def _atomic_envelope(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen semantic-graph artifact: {path}")
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


def _load_envelope(path: Path, *, kind: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != kind
        or value.get("payload_sha256") != canonical_sha256(payload)
    ):
        raise ArtifactIntegrityError(f"{kind} artifact digest/type mismatch")
    return payload


@dataclass(frozen=True, slots=True, order=True)
class GraphEdge:
    source: int
    target: int
    similarity: float

    def __post_init__(self) -> None:
        if self.source < 0 or self.target < 0 or self.source == self.target:
            raise ArtifactIntegrityError("graph edges require distinct non-negative nodes")
        if not np.isfinite(self.similarity) or not -1.0 <= self.similarity <= 1.0:
            raise ArtifactIntegrityError("cosine edge similarity must be finite in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "similarity": self.similarity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphEdge":
        return cls(
            source=int(value["source"]),
            target=int(value["target"]),
            similarity=float(value["similarity"]),
        )


@dataclass(frozen=True, slots=True)
class SemanticKNNGraphArtifact:
    environment_id: str
    fit_role: str
    feature_artifact_sha256: str
    fit_uids: tuple[str, ...]
    fit_uid_sha256: str
    feature_dimension: int
    requested_top_k: int
    effective_top_k: int
    minimum_similarity: float
    symmetrization: str
    query_block_size: int
    directed_edges: tuple[GraphEdge, ...]
    undirected_edges: tuple[GraphEdge, ...]
    directed_edge_uid_sha256: str
    undirected_edge_uid_sha256: str
    isolated_node_indices: tuple[int, ...]
    peak_score_matrix_elements: int
    estimated_peak_score_bytes: int
    estimated_sparse_adjacency_bytes: int
    information_access: str = "input_only"
    algorithm_id: str = GRAPH_ALGORITHM_ID

    def __post_init__(self) -> None:
        _require_sha256(self.feature_artifact_sha256, name="graph feature hash")
        if self.fit_role not in _FIT_ROLES:
            raise ArtifactIntegrityError("semantic graph fit role must be train/discovery")
        if self.information_access != "input_only" or self.algorithm_id != GRAPH_ALGORITHM_ID:
            raise ArtifactIntegrityError("semantic graph must remain the input-only algorithm")
        if not self.fit_uids or len(self.fit_uids) != len(set(self.fit_uids)):
            raise ArtifactIntegrityError("semantic graph UIDs must be non-empty and unique")
        if _uid_sha256(self.fit_uids) != self.fit_uid_sha256:
            raise ArtifactIntegrityError("semantic graph UID hash mismatch")
        if self.feature_dimension <= 0 or self.requested_top_k <= 0:
            raise ArtifactIntegrityError("semantic graph dimension/top-k must be positive")
        expected_k = min(self.requested_top_k, len(self.fit_uids) - 1)
        if self.effective_top_k != expected_k:
            raise ArtifactIntegrityError("semantic graph effective top-k is inconsistent")
        if self.symmetrization not in _SYMMETRIZATIONS:
            raise ArtifactIntegrityError("unsupported directed-to-undirected symmetrization")
        if not 0.0 <= self.minimum_similarity <= 1.0 or self.query_block_size <= 0:
            raise ArtifactIntegrityError("invalid graph similarity/block configuration")

        node_count = len(self.fit_uids)
        for edge in (*self.directed_edges, *self.undirected_edges):
            if edge.source >= node_count or edge.target >= node_count:
                raise ArtifactIntegrityError("semantic graph edge references an unknown node")
            if edge.similarity + 1e-7 < self.minimum_similarity:
                raise ArtifactIntegrityError("semantic graph retained an edge below threshold")
        if tuple(sorted(self.directed_edges)) != self.directed_edges:
            raise ArtifactIntegrityError("directed semantic edges must be canonically ordered")
        if any(edge.source >= edge.target for edge in self.undirected_edges):
            raise ArtifactIntegrityError("undirected semantic edges require source < target")
        if tuple(sorted(self.undirected_edges)) != self.undirected_edges:
            raise ArtifactIntegrityError("undirected semantic edges must be canonically ordered")
        if len({(edge.source, edge.target) for edge in self.directed_edges}) != len(
            self.directed_edges
        ):
            raise ArtifactIntegrityError("duplicate directed semantic edges")
        if len({(edge.source, edge.target) for edge in self.undirected_edges}) != len(
            self.undirected_edges
        ):
            raise ArtifactIntegrityError("duplicate undirected semantic edges")
        directed_degree = Counter(edge.source for edge in self.directed_edges)
        if any(count > self.effective_top_k for count in directed_degree.values()):
            raise ArtifactIntegrityError("a graph node exceeds the frozen directed top-k")
        expected_undirected = _symmetrize(self.directed_edges, self.symmetrization)
        if expected_undirected != self.undirected_edges:
            raise ArtifactIntegrityError("undirected edges do not match frozen symmetrization")
        if _edge_hash(self.fit_uids, self.directed_edges, directed=True) != (
            self.directed_edge_uid_sha256
        ):
            raise ArtifactIntegrityError("directed edge UID hash mismatch")
        if _edge_hash(self.fit_uids, self.undirected_edges, directed=False) != (
            self.undirected_edge_uid_sha256
        ):
            raise ArtifactIntegrityError("undirected edge UID hash mismatch")
        degree = np.zeros(node_count, dtype=np.int64)
        for edge in self.undirected_edges:
            degree[edge.source] += 1
            degree[edge.target] += 1
        expected_isolated = tuple(int(row) for row in np.flatnonzero(degree == 0))
        if self.isolated_node_indices != expected_isolated:
            raise ArtifactIntegrityError("recorded isolated graph nodes are inconsistent")
        maximum_peak = min(self.query_block_size, node_count) * node_count
        if not 0 <= self.peak_score_matrix_elements <= maximum_peak:
            raise ArtifactIntegrityError("graph peak score-matrix counter is invalid")
        if min(self.estimated_peak_score_bytes, self.estimated_sparse_adjacency_bytes) < 0:
            raise ArtifactIntegrityError("graph memory estimates cannot be negative")

    @property
    def node_count(self) -> int:
        return len(self.fit_uids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_input_only_sparse_semantic_knn_graph",
            "algorithm_id": self.algorithm_id,
            "information_access": self.information_access,
            "label_access_split": "none",
            "graph_kind": "semantic_input_only_not_outcome_informed",
            "environment_id": self.environment_id,
            "fit_role": self.fit_role,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "fit_uid_sha256": self.fit_uid_sha256,
            "fit_uids": list(self.fit_uids),
            "feature_dimension": self.feature_dimension,
            "knn": {
                "metric": "cosine_on_L2_normalized_vectors",
                "requested_top_k": self.requested_top_k,
                "effective_top_k": self.effective_top_k,
                "minimum_similarity": self.minimum_similarity,
                "similarity_range": [-1.0, 1.0],
                "tie_break": "lower_frozen_UID_position",
                "query_block_size": self.query_block_size,
            },
            "symmetrization": self.symmetrization,
            "directed_edges": [edge.to_dict() for edge in self.directed_edges],
            "undirected_edges": [edge.to_dict() for edge in self.undirected_edges],
            "directed_edge_uid_sha256": self.directed_edge_uid_sha256,
            "undirected_edge_uid_sha256": self.undirected_edge_uid_sha256,
            "counts": {
                "nodes": self.node_count,
                "directed_edges": len(self.directed_edges),
                "undirected_edges": len(self.undirected_edges),
                "isolated_nodes": len(self.isolated_node_indices),
            },
            "isolated_node_indices": list(self.isolated_node_indices),
            "memory": {
                "dense_N_by_N_materialized": False,
                "peak_score_matrix_elements": self.peak_score_matrix_elements,
                "estimated_peak_score_bytes": self.estimated_peak_score_bytes,
                "estimated_sparse_adjacency_bytes": self.estimated_sparse_adjacency_bytes,
            },
            "dev_or_test_nodes_added": False,
            "row_drop_count": 0,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticKNNGraphArtifact":
        knn = value["knn"]
        memory = value["memory"]
        return cls(
            environment_id=str(value["environment_id"]),
            fit_role=str(value["fit_role"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            fit_uids=tuple(str(uid) for uid in value["fit_uids"]),
            fit_uid_sha256=str(value["fit_uid_sha256"]),
            feature_dimension=int(value["feature_dimension"]),
            requested_top_k=int(knn["requested_top_k"]),
            effective_top_k=int(knn["effective_top_k"]),
            minimum_similarity=float(knn["minimum_similarity"]),
            symmetrization=str(value["symmetrization"]),
            query_block_size=int(knn["query_block_size"]),
            directed_edges=tuple(GraphEdge.from_dict(edge) for edge in value["directed_edges"]),
            undirected_edges=tuple(
                GraphEdge.from_dict(edge) for edge in value["undirected_edges"]
            ),
            directed_edge_uid_sha256=str(value["directed_edge_uid_sha256"]),
            undirected_edge_uid_sha256=str(value["undirected_edge_uid_sha256"]),
            isolated_node_indices=tuple(int(row) for row in value["isolated_node_indices"]),
            peak_score_matrix_elements=int(memory["peak_score_matrix_elements"]),
            estimated_peak_score_bytes=int(memory["estimated_peak_score_bytes"]),
            estimated_sparse_adjacency_bytes=int(memory["estimated_sparse_adjacency_bytes"]),
            information_access=str(value["information_access"]),
            algorithm_id=str(value["algorithm_id"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticKNNGraphArtifact":
        return cls.from_dict(
            _load_envelope(
                Path(path), kind="subject_routing_input_only_sparse_semantic_knn_graph"
            )
        )


def _symmetrize(
    directed_edges: Sequence[GraphEdge], strategy: str
) -> tuple[GraphEdge, ...]:
    if strategy not in _SYMMETRIZATIONS:
        raise ValueError(f"unsupported graph symmetrization: {strategy}")
    directions = {(edge.source, edge.target): edge.similarity for edge in directed_edges}
    pairs = sorted({tuple(sorted((source, target))) for source, target in directions})
    result: list[GraphEdge] = []
    for source, target in pairs:
        forward = directions.get((source, target))
        reverse = directions.get((target, source))
        if strategy == "mutual_mean":
            if forward is None or reverse is None:
                continue
            weight = (forward + reverse) / 2.0
        else:
            weight = max(value for value in (forward, reverse) if value is not None)
        result.append(GraphEdge(source, target, float(weight)))
    return tuple(result)


def build_sparse_cosine_knn_graph(
    features: np.ndarray,
    uids: Sequence[str],
    *,
    environment_id: str,
    fit_role: str,
    feature_artifact_sha256: str,
    top_k: int,
    minimum_similarity: float,
    symmetrization: str,
    query_block_size: int,
    audit: UIDAccessAudit,
) -> SemanticKNNGraphArtifact:
    """Build an exact blockwise top-k graph without storing dense N-by-N scores."""

    if audit.operation is not Operation.FIT:
        raise UIDScopeError("semantic kNN graph construction requires a fit UID scope")
    ids = tuple(str(uid) for uid in uids)
    audit.require(ids, component="semantic_knn_graph_fit")
    if not ids or len(ids) != len(set(ids)):
        raise ArtifactIntegrityError("semantic graph UIDs must be non-empty and unique")
    if top_k <= 0 or query_block_size <= 0 or not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("invalid semantic kNN graph parameters")
    if symmetrization not in _SYMMETRIZATIONS:
        raise ValueError("unsupported graph symmetrization")
    values = l2_normalize(features)
    if values.shape[0] != len(ids):
        raise ArtifactIntegrityError("semantic graph feature/UID count mismatch")
    node_count = len(ids)
    effective_k = min(int(top_k), node_count - 1)
    directed: list[GraphEdge] = []
    peak_elements = 0
    for start in range(0, node_count, query_block_size):
        stop = min(start + query_block_size, node_count)
        scores = np.asarray(values[start:stop] @ values.T, dtype=np.float32)
        peak_elements = max(peak_elements, int(scores.size))
        for offset in range(stop - start):
            source = start + offset
            scores[offset, source] = -np.inf
            if effective_k == 0:
                continue
            candidates = np.argpartition(-scores[offset], effective_k - 1)[:effective_k]
            ordered = sorted(
                (int(target) for target in candidates),
                key=lambda target: (-float(scores[offset, target]), target),
            )
            for target in ordered:
                similarity = float(np.clip(scores[offset, target], -1.0, 1.0))
                if similarity >= minimum_similarity:
                    directed.append(GraphEdge(source, target, similarity))
    directed_edges = tuple(sorted(directed))
    undirected_edges = _symmetrize(directed_edges, symmetrization)
    degree = np.zeros(node_count, dtype=np.int64)
    for edge in undirected_edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    isolated = tuple(int(row) for row in np.flatnonzero(degree == 0))
    adjacency_entries = 2 * len(undirected_edges)
    sparse_bytes = adjacency_entries * (4 + 8) + (node_count + 1) * 8
    return SemanticKNNGraphArtifact(
        environment_id=str(environment_id),
        fit_role=str(fit_role),
        feature_artifact_sha256=str(feature_artifact_sha256),
        fit_uids=ids,
        fit_uid_sha256=_uid_sha256(ids),
        feature_dimension=int(values.shape[1]),
        requested_top_k=int(top_k),
        effective_top_k=effective_k,
        minimum_similarity=float(minimum_similarity),
        symmetrization=str(symmetrization),
        query_block_size=int(query_block_size),
        directed_edges=directed_edges,
        undirected_edges=undirected_edges,
        directed_edge_uid_sha256=_edge_hash(ids, directed_edges, directed=True),
        undirected_edge_uid_sha256=_edge_hash(ids, undirected_edges, directed=False),
        isolated_node_indices=isolated,
        peak_score_matrix_elements=peak_elements,
        estimated_peak_score_bytes=peak_elements * np.dtype(np.float32).itemsize,
        estimated_sparse_adjacency_bytes=int(sparse_bytes),
    )


@dataclass(frozen=True, slots=True)
class LeidenDependencyAudit:
    igraph_found: bool
    leidenalg_found: bool
    igraph_version: str | None
    leidenalg_version: str | None
    graph_class_available: bool
    find_partition_available: bool
    rb_partition_available: bool
    expected_igraph_version: str | None
    expected_leidenalg_version: str | None
    usable: bool
    blocker: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_modules": ["igraph", "leidenalg"],
            "igraph_found": self.igraph_found,
            "leidenalg_found": self.leidenalg_found,
            "igraph_version": self.igraph_version,
            "leidenalg_version": self.leidenalg_version,
            "graph_class_available": self.graph_class_available,
            "find_partition_available": self.find_partition_available,
            "rb_partition_available": self.rb_partition_available,
            "expected_igraph_version": self.expected_igraph_version,
            "expected_leidenalg_version": self.expected_leidenalg_version,
            "usable": self.usable,
            "blocker": self.blocker,
        }


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def probe_leiden_dependencies(
    *,
    expected_igraph_version: str | None = None,
    expected_leidenalg_version: str | None = None,
) -> LeidenDependencyAudit:
    igraph_found = importlib.util.find_spec("igraph") is not None
    leidenalg_found = importlib.util.find_spec("leidenalg") is not None
    if not igraph_found or not leidenalg_found:
        missing = [
            name
            for name, found in (("igraph", igraph_found), ("leidenalg", leidenalg_found))
            if not found
        ]
        return LeidenDependencyAudit(
            igraph_found=igraph_found,
            leidenalg_found=leidenalg_found,
            igraph_version=_distribution_version("igraph", "python-igraph"),
            leidenalg_version=_distribution_version("leidenalg"),
            graph_class_available=False,
            find_partition_available=False,
            rb_partition_available=False,
            expected_igraph_version=expected_igraph_version,
            expected_leidenalg_version=expected_leidenalg_version,
            usable=False,
            blocker=f"missing_required_modules:{','.join(missing)}",
        )
    try:
        igraph = importlib.import_module("igraph")
        leidenalg = importlib.import_module("leidenalg")
        igraph_version = _distribution_version("igraph", "python-igraph")
        leidenalg_version = _distribution_version("leidenalg")
        graph_available = callable(getattr(igraph, "Graph", None))
        find_available = callable(getattr(leidenalg, "find_partition", None))
        partition_available = hasattr(leidenalg, "RBConfigurationVertexPartition")
    except (ImportError, AttributeError, TypeError, ValueError):
        return LeidenDependencyAudit(
            igraph_found=True,
            leidenalg_found=True,
            igraph_version=None,
            leidenalg_version=None,
            graph_class_available=False,
            find_partition_available=False,
            rb_partition_available=False,
            expected_igraph_version=expected_igraph_version,
            expected_leidenalg_version=expected_leidenalg_version,
            usable=False,
            blocker="leiden_dependency_API_probe_failed",
        )
    blocker: str | None = None
    if not graph_available or not find_available or not partition_available:
        blocker = "required_leiden_API_missing"
    elif expected_igraph_version is None or expected_leidenalg_version is None:
        blocker = "exact_dependency_versions_not_frozen"
    elif igraph_version != expected_igraph_version:
        blocker = "installed_igraph_version_differs_from_frozen_version"
    elif leidenalg_version != expected_leidenalg_version:
        blocker = "installed_leidenalg_version_differs_from_frozen_version"
    return LeidenDependencyAudit(
        igraph_found=True,
        leidenalg_found=True,
        igraph_version=igraph_version,
        leidenalg_version=leidenalg_version,
        graph_class_available=graph_available,
        find_partition_available=find_available,
        rb_partition_available=partition_available,
        expected_igraph_version=expected_igraph_version,
        expected_leidenalg_version=expected_leidenalg_version,
        usable=blocker is None,
        blocker=blocker,
    )


def require_leiden_dependencies(
    *, expected_igraph_version: str, expected_leidenalg_version: str
) -> tuple[Any, Any, LeidenDependencyAudit]:
    audit = probe_leiden_dependencies(
        expected_igraph_version=expected_igraph_version,
        expected_leidenalg_version=expected_leidenalg_version,
    )
    if not audit.usable:
        raise LeidenDependencyError(
            f"real igraph/leidenalg stack is unavailable: {audit.blocker}; "
            "no alternative community algorithm may be labeled Leiden"
        )
    return importlib.import_module("igraph"), importlib.import_module("leidenalg"), audit


@dataclass(frozen=True, slots=True)
class LeidenCommunityArtifact:
    graph_sha256: str
    graph_fit_uid_sha256: str
    resolution: float
    seed: int
    minimum_community_support: int
    raw_membership: tuple[int, ...]
    raw_community_sizes: tuple[dict[str, int], ...]
    raw_label_to_state: tuple[tuple[int, str], ...]
    retained_state_ids: tuple[str, ...]
    training_states: tuple[str, ...]
    fallback_state_id: str
    isolated_node_count: int
    low_support_node_count: int
    backend: str
    igraph_version: str
    leidenalg_version: str
    information_access: str = "input_only_graph"
    algorithm_id: str = LEIDEN_ALGORITHM_ID

    def __post_init__(self) -> None:
        _require_sha256(self.graph_sha256, name="semantic graph hash")
        _require_sha256(self.graph_fit_uid_sha256, name="semantic graph UID hash")
        if self.resolution <= 0 or self.minimum_community_support <= 0:
            raise ArtifactIntegrityError("Leiden resolution/support must be positive")
        if any(label < 0 for label in self.raw_membership):
            raise ArtifactIntegrityError("Leiden membership labels must be non-negative")
        if not self.raw_membership or len(self.training_states) != len(self.raw_membership):
            raise ArtifactIntegrityError("Leiden membership/state arrays are invalid")
        mapping = dict(self.raw_label_to_state)
        if len(mapping) != len(self.raw_label_to_state) or tuple(mapping.values()) != (
            self.retained_state_ids
        ):
            raise ArtifactIntegrityError("Leiden raw/state mapping is inconsistent")
        if len(set(self.retained_state_ids)) != len(self.retained_state_ids):
            raise ArtifactIntegrityError("Leiden state IDs must be unique")
        if self.fallback_state_id in self.retained_state_ids:
            raise ArtifactIntegrityError("Leiden fallback must be separate from communities")
        raw_counts = Counter(self.raw_membership)
        recorded = {
            int(record["raw_label"]): int(record["support"])
            for record in self.raw_community_sizes
        }
        if recorded != dict(sorted(raw_counts.items())):
            raise ArtifactIntegrityError("raw Leiden community sizes are inconsistent")
        if min(self.isolated_node_count, self.low_support_node_count) < 0:
            raise ArtifactIntegrityError("Leiden fallback counters cannot be negative")
        if self.training_states.count(self.fallback_state_id) < self.isolated_node_count:
            raise ArtifactIntegrityError("isolated graph nodes must use fallback")
        allowed_states = set(self.retained_state_ids) | {self.fallback_state_id}
        if not set(self.training_states).issubset(allowed_states):
            raise ArtifactIntegrityError("Leiden assignments reference non-frozen states")
        if self.backend == REAL_LEIDEN_BACKEND:
            if not self.igraph_version or not self.leidenalg_version:
                raise ArtifactIntegrityError("real Leiden artifact must freeze both versions")
        elif self.backend not in {FIXTURE_BACKEND, NO_EDGE_BACKEND}:
            raise ArtifactIntegrityError("unrecognized Leiden community backend")

    @property
    def actual_community_count(self) -> int:
        return len(self.raw_community_sizes)

    @property
    def retained_community_count(self) -> int:
        return len(self.retained_state_ids)

    @property
    def coverage(self) -> float:
        return sum(state != self.fallback_state_id for state in self.training_states) / len(
            self.training_states
        )

    def to_dict(self) -> dict[str, Any]:
        hierarchy = [
            {
                "state_id": self.fallback_state_id,
                "parent_state_id": None,
                "depth": 0,
                "state_kind": "global_fallback_root",
            }
        ] + [
            {
                "state_id": state,
                "parent_state_id": self.fallback_state_id,
                "depth": 1,
                "state_kind": "leiden_community",
            }
            for state in self.retained_state_ids
        ]
        return {
            "schema_version": 1,
            "kind": "subject_routing_frozen_leiden_communities",
            "algorithm_id": self.algorithm_id,
            "backend": self.backend,
            "information_access": self.information_access,
            "label_access_split": "none",
            "graph_kind": "semantic_input_only_not_outcome_informed",
            "graph_sha256": self.graph_sha256,
            "graph_fit_uid_sha256": self.graph_fit_uid_sha256,
            "resolution": self.resolution,
            "seed": self.seed,
            "minimum_community_support": self.minimum_community_support,
            "dependency": {
                "igraph_version": self.igraph_version,
                "leidenalg_version": self.leidenalg_version,
            },
            "raw_membership": list(self.raw_membership),
            "raw_community_sizes": list(self.raw_community_sizes),
            "raw_label_to_state": [
                {"raw_label": label, "state_id": state}
                for label, state in self.raw_label_to_state
            ],
            "retained_state_ids": list(self.retained_state_ids),
            "training_states": list(self.training_states),
            "hierarchy": hierarchy,
            "fallback": {
                "state_id": self.fallback_state_id,
                "isolated_node_count": self.isolated_node_count,
                "low_support_node_count": self.low_support_node_count,
                "fallback_count": self.training_states.count(self.fallback_state_id),
                "row_drop_count": 0,
            },
            "actual_community_count": self.actual_community_count,
            "retained_community_count": self.retained_community_count,
            "coverage": self.coverage,
            "deployment_paths": {
                "prototype_assigner": "SUPPORTED_FROZEN_INDUCTIVE",
                "B07_student_pseudo_labels": "training_states",
                "evaluation_time_community_detection": False,
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeidenCommunityArtifact":
        dependency = value["dependency"]
        fallback = value["fallback"]
        return cls(
            graph_sha256=str(value["graph_sha256"]),
            graph_fit_uid_sha256=str(value["graph_fit_uid_sha256"]),
            resolution=float(value["resolution"]),
            seed=int(value["seed"]),
            minimum_community_support=int(value["minimum_community_support"]),
            raw_membership=tuple(int(label) for label in value["raw_membership"]),
            raw_community_sizes=tuple(dict(record) for record in value["raw_community_sizes"]),
            raw_label_to_state=tuple(
                (int(record["raw_label"]), str(record["state_id"]))
                for record in value["raw_label_to_state"]
            ),
            retained_state_ids=tuple(str(state) for state in value["retained_state_ids"]),
            training_states=tuple(str(state) for state in value["training_states"]),
            fallback_state_id=str(fallback["state_id"]),
            isolated_node_count=int(fallback["isolated_node_count"]),
            low_support_node_count=int(fallback["low_support_node_count"]),
            backend=str(value["backend"]),
            igraph_version=str(dependency["igraph_version"]),
            leidenalg_version=str(dependency["leidenalg_version"]),
            information_access=str(value["information_access"]),
            algorithm_id=str(value["algorithm_id"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LeidenCommunityArtifact":
        return cls.from_dict(
            _load_envelope(Path(path), kind="subject_routing_frozen_leiden_communities")
        )


def community_artifact_from_membership(
    graph: SemanticKNNGraphArtifact,
    membership: Sequence[int],
    *,
    resolution: float,
    seed: int,
    minimum_community_support: int,
    backend: str,
    igraph_version: str,
    leidenalg_version: str,
    fallback_state_id: str = "__global__",
) -> LeidenCommunityArtifact:
    """Freeze memberships, canonicalize labels, and retain every node in denominator."""

    labels = tuple(int(label) for label in membership)
    if len(labels) != graph.node_count:
        raise ArtifactIntegrityError("community membership differs from graph node count")
    isolated = set(graph.isolated_node_indices)
    eligible_counts = Counter(label for row, label in enumerate(labels) if row not in isolated)
    retained = [
        label
        for label, count in eligible_counts.items()
        if count >= int(minimum_community_support)
    ]
    retained.sort(
        key=lambda label: min(
            graph.fit_uids[row]
            for row, candidate in enumerate(labels)
            if candidate == label and row not in isolated
        )
    )
    mapping = tuple(
        (label, f"community-{position:04d}") for position, label in enumerate(retained)
    )
    lookup = dict(mapping)
    states = tuple(
        fallback_state_id if row in isolated else lookup.get(label, fallback_state_id)
        for row, label in enumerate(labels)
    )
    low_support_count = sum(
        1 for row, label in enumerate(labels) if row not in isolated and label not in lookup
    )
    counts = Counter(labels)
    return LeidenCommunityArtifact(
        graph_sha256=graph.sha256,
        graph_fit_uid_sha256=graph.fit_uid_sha256,
        resolution=float(resolution),
        seed=int(seed),
        minimum_community_support=int(minimum_community_support),
        raw_membership=labels,
        raw_community_sizes=tuple(
            {"raw_label": label, "support": support} for label, support in sorted(counts.items())
        ),
        raw_label_to_state=mapping,
        retained_state_ids=tuple(state for _label, state in mapping),
        training_states=states,
        fallback_state_id=str(fallback_state_id),
        isolated_node_count=len(isolated),
        low_support_node_count=low_support_count,
        backend=str(backend),
        igraph_version=str(igraph_version),
        leidenalg_version=str(leidenalg_version),
    )


def run_frozen_leiden(
    graph: SemanticKNNGraphArtifact,
    *,
    resolution: float,
    seed: int,
    minimum_community_support: int,
    expected_igraph_version: str,
    expected_leidenalg_version: str,
    fallback_state_id: str = "__global__",
) -> LeidenCommunityArtifact:
    """Run only the real weighted Leiden implementation on a frozen train graph."""

    if not graph.undirected_edges:
        return community_artifact_from_membership(
            graph,
            [0] * graph.node_count,
            resolution=resolution,
            seed=seed,
            minimum_community_support=minimum_community_support,
            backend=NO_EDGE_BACKEND,
            igraph_version="NOT_USED_NO_EDGES",
            leidenalg_version="NOT_USED_NO_EDGES",
            fallback_state_id=fallback_state_id,
        )
    igraph, leidenalg, audit = require_leiden_dependencies(
        expected_igraph_version=expected_igraph_version,
        expected_leidenalg_version=expected_leidenalg_version,
    )
    network = igraph.Graph(
        n=graph.node_count,
        edges=[(edge.source, edge.target) for edge in graph.undirected_edges],
        directed=False,
    )
    partition = leidenalg.find_partition(
        network,
        leidenalg.RBConfigurationVertexPartition,
        weights=[edge.similarity for edge in graph.undirected_edges],
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    return community_artifact_from_membership(
        graph,
        partition.membership,
        resolution=resolution,
        seed=seed,
        minimum_community_support=minimum_community_support,
        backend=REAL_LEIDEN_BACKEND,
        igraph_version=str(audit.igraph_version),
        leidenalg_version=str(audit.leidenalg_version),
        fallback_state_id=fallback_state_id,
    )


@dataclass(frozen=True, slots=True)
class GraphPrototypeAssignerArtifact:
    graph_sha256: str
    community_sha256: str
    feature_artifact_sha256: str
    feature_dimension: int
    state_ids: tuple[str, ...]
    prototypes: np.ndarray
    minimum_similarity: float
    fallback_state_id: str
    information_access: str = "input_only_graph_pseudo_labels"
    algorithm_id: str = PROTOTYPE_ALGORITHM_ID

    def __post_init__(self) -> None:
        _require_sha256(self.graph_sha256, name="prototype graph hash")
        _require_sha256(self.community_sha256, name="prototype community hash")
        _require_sha256(self.feature_artifact_sha256, name="prototype feature hash")
        if self.feature_dimension <= 0 or not -1.0 <= self.minimum_similarity <= 1.0:
            raise ArtifactIntegrityError("invalid graph prototype dimension/threshold")
        if len(self.state_ids) != len(set(self.state_ids)):
            raise ArtifactIntegrityError("graph prototype state IDs must be unique")
        if self.fallback_state_id in self.state_ids:
            raise ArtifactIntegrityError("graph prototype fallback must be a separate state")
        values = np.asarray(self.prototypes, dtype=np.float64)
        if values.shape != (len(self.state_ids), self.feature_dimension):
            raise ArtifactIntegrityError("graph prototype matrix shape mismatch")
        if values.size and (
            not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)
        ):
            raise ArtifactIntegrityError("graph community prototypes must be finite unit vectors")
        values.flags.writeable = False
        object.__setattr__(self, "prototypes", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_frozen_leiden_prototype_assigner",
            "algorithm_id": self.algorithm_id,
            "information_access": self.information_access,
            "graph_sha256": self.graph_sha256,
            "community_sha256": self.community_sha256,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "feature_dimension": self.feature_dimension,
            "state_ids": list(self.state_ids),
            "prototypes": self.prototypes.tolist(),
            "minimum_similarity": self.minimum_similarity,
            "fallback_state_id": self.fallback_state_id,
            "evaluation_time_community_detection": False,
            "test_time_state_creation": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphPrototypeAssignerArtifact":
        return cls(
            graph_sha256=str(value["graph_sha256"]),
            community_sha256=str(value["community_sha256"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            feature_dimension=int(value["feature_dimension"]),
            state_ids=tuple(str(state) for state in value["state_ids"]),
            prototypes=np.asarray(value["prototypes"], dtype=np.float64),
            minimum_similarity=float(value["minimum_similarity"]),
            fallback_state_id=str(value["fallback_state_id"]),
            information_access=str(value["information_access"]),
            algorithm_id=str(value["algorithm_id"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphPrototypeAssignerArtifact":
        return cls.from_dict(
            _load_envelope(
                Path(path), kind="subject_routing_frozen_leiden_prototype_assigner"
            )
        )


def fit_graph_prototype_assigner(
    graph: SemanticKNNGraphArtifact,
    communities: LeidenCommunityArtifact,
    features: np.ndarray,
    uids: Sequence[str],
    *,
    minimum_similarity: float,
    audit: UIDAccessAudit,
) -> GraphPrototypeAssignerArtifact:
    if audit.operation is not Operation.FIT:
        raise UIDScopeError("community prototype fit requires a fit UID scope")
    ids = tuple(str(uid) for uid in uids)
    audit.require(ids, component="leiden_community_prototype_fit")
    if ids != graph.fit_uids or communities.graph_sha256 != graph.sha256:
        raise ArtifactIntegrityError("community prototype inputs differ from frozen graph")
    values = l2_normalize(features)
    if values.shape != (graph.node_count, graph.feature_dimension):
        raise ArtifactIntegrityError("community prototype features are invalid or misaligned")
    prototypes: list[np.ndarray] = []
    for state in communities.retained_state_ids:
        rows = [row for row, assigned in enumerate(communities.training_states) if assigned == state]
        if not rows:
            raise ArtifactIntegrityError("retained graph community has no prototype rows")
        prototypes.append(l2_normalize(values[rows].mean(axis=0, keepdims=True))[0])
    matrix = (
        np.asarray(prototypes, dtype=np.float64)
        if prototypes
        else np.empty((0, graph.feature_dimension), dtype=np.float64)
    )
    return GraphPrototypeAssignerArtifact(
        graph_sha256=graph.sha256,
        community_sha256=communities.sha256,
        feature_artifact_sha256=graph.feature_artifact_sha256,
        feature_dimension=graph.feature_dimension,
        state_ids=communities.retained_state_ids,
        prototypes=matrix,
        minimum_similarity=float(minimum_similarity),
        fallback_state_id=communities.fallback_state_id,
    )


def assign_graph_prototypes(
    assigner: GraphPrototypeAssignerArtifact,
    features: np.ndarray,
    uids: Sequence[str],
    *,
    feature_artifact_sha256: str,
    audit: UIDAccessAudit,
) -> AssignmentBatch:
    """Assign each new row independently; community detection is never rerun."""

    if audit.operation is not Operation.ASSIGN:
        raise UIDScopeError("community prototype assignment requires an assign UID scope")
    ids = tuple(str(uid) for uid in uids)
    audit.require(ids, component="leiden_community_prototype_assign")
    if feature_artifact_sha256 != assigner.feature_artifact_sha256:
        raise ArtifactIntegrityError("runtime feature hash differs from graph prototype artifact")
    values = l2_normalize(features)
    if values.shape != (len(ids), assigner.feature_dimension):
        raise ArtifactIntegrityError("community assignment features are invalid or misaligned")
    if not assigner.state_ids:
        scores = np.ones((len(ids), 1), dtype=np.float32)
        states = (assigner.fallback_state_id,) * len(ids)
    else:
        similarities = np.asarray(values @ assigner.prototypes.T, dtype=np.float32)
        selected = np.argmax(similarities, axis=1)
        best = similarities[np.arange(len(ids)), selected]
        states = tuple(
            assigner.state_ids[int(column)]
            if float(best[row]) >= assigner.minimum_similarity
            else assigner.fallback_state_id
            for row, column in enumerate(selected)
        )
        scores = np.concatenate(
            (similarities, (1.0 - best[:, None]).astype(np.float32)), axis=1
        )
    scores.flags.writeable = False
    return AssignmentBatch(
        states=states,
        assignment_scores=scores,
        score_semantics="cosine_boundary_score_not_correctness_probability",
    )


def community_label_stability(
    first: LeidenCommunityArtifact, second: LeidenCommunityArtifact
) -> float:
    """Permutation-invariant auxiliary ARI across seeds; never a routing objective."""

    if (
        first.graph_sha256 != second.graph_sha256
        or first.graph_fit_uid_sha256 != second.graph_fit_uid_sha256
    ):
        raise ArtifactIntegrityError("community stability requires the exact same frozen graph")
    return float(adjusted_rand_score(first.training_states, second.training_states))
