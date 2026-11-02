"""Deterministic, input-only spherical K-means with frozen induction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..artifacts import ArtifactIntegrityError, canonical_sha256
from ..features.fusion import l2_normalize
from ..interfaces import AssignmentBatch
from ..leakage_guard import Operation, UIDAccessAudit, UIDScopeError


ALGORITHM_ID = "semantic_spherical_kmeans_subject_lloyd_v1"
EMPTY_CLUSTER_POLICY = "steal_farthest_row_from_cluster_with_support_gt_1"
_FIT_ROLES = frozenset({"train", "train_seen", "discovery"})


def _uid_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


@dataclass(frozen=True, slots=True)
class KGridPlan:
    n_samples: int
    minimum_leaf_support: int
    max_candidates: int | None
    accepted: tuple[int, ...]
    rejected: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "minimum_leaf_support": self.minimum_leaf_support,
            "max_candidates": self.max_candidates,
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
        }


def plan_k_grid(
    candidates: Sequence[int],
    *,
    n_samples: int,
    minimum_leaf_support: int,
    max_candidates: int | None = None,
) -> KGridPlan:
    """Prune a preregistered grid without looking at labels or dev/test."""

    proposed = tuple(int(value) for value in candidates)
    if not proposed or len(proposed) != len(set(proposed)) or any(value <= 0 for value in proposed):
        raise ValueError("K candidates must be non-empty, positive, and unique")
    if 1 not in proposed:
        raise ValueError("the K=1 reference must remain in every grid")
    if n_samples <= 0 or minimum_leaf_support <= 0:
        raise ValueError("n_samples and minimum_leaf_support must be positive")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided")
    support_limit = n_samples // minimum_leaf_support
    accepted: list[int] = []
    rejected: list[dict[str, Any]] = []
    for value in proposed:
        if value > n_samples:
            reason = "exceeds_sample_count"
        elif value != 1 and value > support_limit:
            reason = "exceeds_floor_n_over_minimum_leaf_support"
        elif max_candidates is not None and len(accepted) >= max_candidates:
            reason = "exceeds_candidate_budget"
        else:
            accepted.append(value)
            continue
        rejected.append({"k": value, "reason": reason})
    return KGridPlan(
        n_samples=int(n_samples),
        minimum_leaf_support=int(minimum_leaf_support),
        max_candidates=max_candidates,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )


@dataclass(frozen=True, slots=True)
class SphericalKMeansArtifact:
    environment_id: str
    fit_role: str
    feature_artifact_sha256: str
    fit_uids: tuple[str, ...]
    fit_uid_sha256: str
    cluster_ids: tuple[str, ...]
    centers: np.ndarray
    assignments: tuple[str, ...]
    support: tuple[int, ...]
    within_distance: tuple[dict[str, float], ...]
    requested_k: int
    effective_k: int
    seed: int
    n_init: int
    max_iter: int
    iterations: int
    tolerance: float
    converged: bool
    objective_cosine_distance_sum: float
    empty_cluster_repairs: int
    approximation: str = "exact_full_batch_lloyd"
    information_access: str = "input_only"
    algorithm_id: str = ALGORITHM_ID
    empty_cluster_policy: str = EMPTY_CLUSTER_POLICY

    def __post_init__(self) -> None:
        _require_sha256(self.feature_artifact_sha256, name="feature artifact hash")
        _require_sha256(self.fit_uid_sha256, name="fit UID hash")
        if self.fit_role not in _FIT_ROLES:
            raise ArtifactIntegrityError("spherical K-means fit role must be train/discovery")
        if self.information_access != "input_only":
            raise ArtifactIntegrityError("semantic spherical K-means must be input-only")
        if self.algorithm_id != ALGORITHM_ID or self.approximation != "exact_full_batch_lloyd":
            raise ArtifactIntegrityError("unexpected spherical K-means algorithm identity")
        if self.empty_cluster_policy != EMPTY_CLUSTER_POLICY:
            raise ArtifactIntegrityError("unexpected empty-cluster policy")
        if not self.fit_uids or len(self.fit_uids) != len(set(self.fit_uids)):
            raise ArtifactIntegrityError("fit UIDs must be non-empty and unique")
        if _uid_sha256(self.fit_uids) != self.fit_uid_sha256:
            raise ArtifactIntegrityError("fit UID hash does not match frozen UID order")
        if self.requested_k <= 0 or self.effective_k != self.requested_k:
            raise ArtifactIntegrityError("effective K must equal requested K after repair")
        if len(self.cluster_ids) != self.requested_k or len(set(self.cluster_ids)) != len(
            self.cluster_ids
        ):
            raise ArtifactIntegrityError("cluster IDs must cover requested K exactly")
        centers = np.array(self.centers, dtype=np.float64, copy=True)
        if centers.ndim != 2 or centers.shape[0] != self.requested_k:
            raise ArtifactIntegrityError("center matrix shape does not match requested K")
        if not np.isfinite(centers).all() or not np.allclose(
            np.linalg.norm(centers, axis=1), 1.0, atol=1e-6
        ):
            raise ArtifactIntegrityError("all spherical centers must have unit norm")
        centers.flags.writeable = False
        object.__setattr__(self, "centers", centers)
        if len(self.assignments) != len(self.fit_uids):
            raise ArtifactIntegrityError("training assignment count differs from fit UID count")
        if set(self.assignments) != set(self.cluster_ids):
            raise ArtifactIntegrityError("artifact contains an empty or unknown cluster")
        expected_support = tuple(self.assignments.count(cluster) for cluster in self.cluster_ids)
        if tuple(self.support) != expected_support or any(count <= 0 for count in self.support):
            raise ArtifactIntegrityError("recorded cluster support is inconsistent")
        if len(self.within_distance) != self.requested_k:
            raise ArtifactIntegrityError("within-distance records must cover every cluster")
        for cluster, count, record in zip(self.cluster_ids, self.support, self.within_distance):
            if record.get("cluster_id") != cluster or int(record.get("support", -1)) != count:
                raise ArtifactIntegrityError("within-distance cluster metadata mismatch")
            metrics = [record.get("sum"), record.get("mean"), record.get("max")]
            if not all(value is not None and np.isfinite(float(value)) for value in metrics):
                raise ArtifactIntegrityError("within-distance metrics must be finite")
        if not np.isfinite(self.objective_cosine_distance_sum):
            raise ArtifactIntegrityError("objective must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_input_only_spherical_kmeans",
            "algorithm_id": self.algorithm_id,
            "approximation": self.approximation,
            "information_access": self.information_access,
            "label_access_split": "none",
            "environment_id": self.environment_id,
            "fit_role": self.fit_role,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "fit_uid_sha256": self.fit_uid_sha256,
            "fit_uids": list(self.fit_uids),
            "cluster_ids": list(self.cluster_ids),
            "centers": self.centers.tolist(),
            "assignments": list(self.assignments),
            "support": list(self.support),
            "within_distance": list(self.within_distance),
            "requested_k": self.requested_k,
            "effective_k": self.effective_k,
            "seed": self.seed,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "iterations": self.iterations,
            "tolerance": self.tolerance,
            "converged": self.converged,
            "objective_cosine_distance_sum": self.objective_cosine_distance_sum,
            "empty_cluster_policy": self.empty_cluster_policy,
            "empty_cluster_repairs": self.empty_cluster_repairs,
            "row_drop_count": 0,
            "dev_or_test_refit": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite frozen cluster artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        envelope = {"payload": payload, "payload_sha256": canonical_sha256(payload)}
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(envelope, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SphericalKMeansArtifact":
        return cls(
            environment_id=str(value["environment_id"]),
            fit_role=str(value["fit_role"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            fit_uids=tuple(str(uid) for uid in value["fit_uids"]),
            fit_uid_sha256=str(value["fit_uid_sha256"]),
            cluster_ids=tuple(str(cluster) for cluster in value["cluster_ids"]),
            centers=np.asarray(value["centers"], dtype=np.float64),
            assignments=tuple(str(cluster) for cluster in value["assignments"]),
            support=tuple(int(count) for count in value["support"]),
            within_distance=tuple(dict(record) for record in value["within_distance"]),
            requested_k=int(value["requested_k"]),
            effective_k=int(value["effective_k"]),
            seed=int(value["seed"]),
            n_init=int(value["n_init"]),
            max_iter=int(value["max_iter"]),
            iterations=int(value["iterations"]),
            tolerance=float(value["tolerance"]),
            converged=bool(value["converged"]),
            objective_cosine_distance_sum=float(value["objective_cosine_distance_sum"]),
            empty_cluster_repairs=int(value["empty_cluster_repairs"]),
            approximation=str(value["approximation"]),
            information_access=str(value["information_access"]),
            algorithm_id=str(value["algorithm_id"]),
            empty_cluster_policy=str(value["empty_cluster_policy"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SphericalKMeansArtifact":
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or envelope.get("payload_sha256") != canonical_sha256(
            payload
        ):
            raise ArtifactIntegrityError("spherical K-means artifact digest mismatch")
        return cls.from_dict(payload)


def _unit_mean(rows: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    center = rows.mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm <= 1e-12:
        center = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(center))
    return center / max(norm, 1e-12)


def _initialize(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k == 1:
        return _unit_mean(features, fallback=features[0])[None, :]
    n_rows = features.shape[0]
    chosen: list[int] = [int(rng.integers(n_rows))]
    centers = [features[chosen[0]]]
    while len(centers) < k:
        similarities = features @ np.asarray(centers).T
        distance = np.maximum(0.0, 1.0 - similarities.max(axis=1))
        distance[np.asarray(chosen, dtype=np.int64)] = 0.0
        weights = distance * distance
        if float(weights.sum()) <= 1e-15:
            index = next(row for row in range(n_rows) if row not in set(chosen))
        else:
            index = int(rng.choice(n_rows, p=weights / weights.sum()))
            if index in chosen:  # numerical guard
                index = next(row for row in range(n_rows) if row not in set(chosen))
        chosen.append(index)
        centers.append(features[index])
    return np.asarray(centers, dtype=np.float64)


def _repair_empty(
    assignments: np.ndarray, scores: np.ndarray, k: int
) -> tuple[np.ndarray, int]:
    repaired = assignments.copy()
    counts = np.bincount(repaired, minlength=k)
    repairs = 0
    for empty in np.flatnonzero(counts == 0):
        candidates = np.flatnonzero(counts[repaired] > 1)
        if not len(candidates):  # impossible when n >= K
            raise RuntimeError("cannot repair an empty cluster without deleting a sample")
        losses = 1.0 - scores[candidates, repaired[candidates]]
        row = int(candidates[int(np.argmax(losses))])
        donor = int(repaired[row])
        repaired[row] = int(empty)
        counts[donor] -= 1
        counts[empty] += 1
        repairs += 1
    return repaired, repairs


class SphericalKMeans:
    """Full-batch cosine Lloyd iterations with unit centers every update."""

    def __init__(
        self,
        n_clusters: int,
        *,
        seed: int,
        n_init: int = 10,
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        if n_clusters <= 0 or n_init <= 0 or max_iter <= 0 or tolerance < 0:
            raise ValueError("invalid spherical K-means hyperparameter")
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.artifact_: SphericalKMeansArtifact | None = None

    def fit(
        self,
        features: np.ndarray,
        uids: Sequence[str],
        *,
        environment_id: str,
        fit_role: str,
        feature_artifact_sha256: str,
        audit: UIDAccessAudit,
    ) -> SphericalKMeansArtifact:
        if audit.operation is not Operation.FIT:
            raise UIDScopeError("spherical K-means fit requires a fit UID scope")
        if fit_role not in _FIT_ROLES:
            raise UIDScopeError("spherical K-means may fit only train/train_seen/discovery")
        ids = tuple(str(uid) for uid in uids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("fit UIDs must be non-empty and unique")
        audit.require(ids, component="semantic_spherical_kmeans_subject.fit")
        values = np.asarray(l2_normalize(features), dtype=np.float64)
        if values.shape[0] != len(ids):
            raise ValueError("feature and fit UID row counts differ")
        if self.n_clusters > len(ids):
            raise ValueError("K cannot exceed the number of allowed fit samples")
        _require_sha256(feature_artifact_sha256, name="feature artifact hash")

        best: dict[str, Any] | None = None
        for init_index in range(self.n_init):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, init_index]))
            centers = _initialize(values, self.n_clusters, rng)
            previous: np.ndarray | None = None
            repairs_total = 0
            converged = False
            assignments = np.zeros(len(ids), dtype=np.int64)
            iteration = 0
            for iteration in range(1, self.max_iter + 1):
                scores = values @ centers.T
                assignments, repairs = _repair_empty(
                    scores.argmax(axis=1), scores, self.n_clusters
                )
                repairs_total += repairs
                updated = np.vstack(
                    [
                        _unit_mean(
                            values[assignments == cluster],
                            fallback=centers[cluster],
                        )
                        for cluster in range(self.n_clusters)
                    ]
                )
                shift = float(np.max(1.0 - np.sum(centers * updated, axis=1)))
                stable = previous is not None and np.array_equal(assignments, previous)
                centers = updated
                if stable or shift <= self.tolerance:
                    converged = True
                    break
                previous = assignments.copy()
            distances = np.maximum(
                0.0, 1.0 - np.sum(values * centers[assignments], axis=1)
            )
            objective = float(distances.sum())
            candidate = {
                "centers": centers,
                "assignments": assignments,
                "distances": distances,
                "objective": objective,
                "iterations": iteration,
                "converged": converged,
                "repairs": repairs_total,
                "init_index": init_index,
            }
            if best is None or (objective, init_index) < (
                best["objective"],
                best["init_index"],
            ):
                best = candidate
        assert best is not None
        cluster_ids = tuple(f"subject-{index:04d}" for index in range(self.n_clusters))
        integer_assignments = np.asarray(best["assignments"], dtype=np.int64)
        assignments = tuple(cluster_ids[index] for index in integer_assignments)
        support = tuple(
            int(np.count_nonzero(integer_assignments == cluster))
            for cluster in range(self.n_clusters)
        )
        distances = np.asarray(best["distances"], dtype=np.float64)
        within = []
        for cluster, cluster_id in enumerate(cluster_ids):
            selected = distances[integer_assignments == cluster]
            within.append(
                {
                    "cluster_id": cluster_id,
                    "support": int(len(selected)),
                    "sum": float(selected.sum()),
                    "mean": float(selected.mean()),
                    "max": float(selected.max()),
                }
            )
        artifact = SphericalKMeansArtifact(
            environment_id=str(environment_id),
            fit_role=fit_role,
            feature_artifact_sha256=str(feature_artifact_sha256),
            fit_uids=ids,
            fit_uid_sha256=_uid_sha256(ids),
            cluster_ids=cluster_ids,
            centers=np.asarray(best["centers"], dtype=np.float64),
            assignments=assignments,
            support=support,
            within_distance=tuple(within),
            requested_k=self.n_clusters,
            effective_k=len(set(assignments)),
            seed=self.seed,
            n_init=self.n_init,
            max_iter=self.max_iter,
            iterations=int(best["iterations"]),
            tolerance=self.tolerance,
            converged=bool(best["converged"]),
            objective_cosine_distance_sum=float(best["objective"]),
            empty_cluster_repairs=int(best["repairs"]),
        )
        self.artifact_ = artifact
        return artifact


def assign_frozen(
    artifact: SphericalKMeansArtifact,
    features: np.ndarray,
    *,
    feature_artifact_sha256: str,
) -> AssignmentBatch:
    """Nearest frozen-center assignment; never updates or reorders centers."""

    if str(feature_artifact_sha256) != artifact.feature_artifact_sha256:
        raise ArtifactIntegrityError("runtime feature hash differs from frozen K-means artifact")
    values = np.asarray(l2_normalize(features), dtype=np.float64)
    if values.shape[1] != artifact.centers.shape[1]:
        raise ArtifactIntegrityError("runtime feature dimension differs from frozen centers")
    scores = values @ artifact.centers.T
    selected = scores.argmax(axis=1)
    frozen_scores = np.asarray(scores, dtype=np.float32)
    frozen_scores.flags.writeable = False
    return AssignmentBatch(
        states=tuple(artifact.cluster_ids[index] for index in selected),
        assignment_scores=frozen_scores,
        score_semantics="cosine_assignment_score_not_correctness_probability",
    )
