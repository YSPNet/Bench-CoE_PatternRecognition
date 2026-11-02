"""Fail-closed adapter for contrib ``hdbscan`` inductive assignment.

The similarly named ``sklearn.cluster.HDBSCAN`` is deliberately unsupported:
it does not provide the contrib package's frozen ``approximate_predict`` API.
This module has no import-time dependency on either implementation, so the
mandatory fallback mechanics remain testable when the optional package is not
installed.  Such mechanics tests are not evidence that HDBSCAN itself ran.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import pickle
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts import ArtifactIntegrityError, canonical_sha256
from ..interfaces import AssignmentBatch
from ..leakage_guard import Operation, UIDAccessAudit, UIDScopeError


ALGORITHM_ID = "contrib_hdbscan_frozen_approximate_predict_v1"
CONTRIB_DISTRIBUTION = "hdbscan"
CONTRIB_MODULE = "hdbscan"
ASSIGNMENT_BACKEND = "hdbscan.prediction.approximate_predict"
FALLBACK_ONLY_BACKEND = "constant_frozen_fallback"
FORBIDDEN_SKLEARN_CLASS = "sklearn.cluster.HDBSCAN"
_FIT_ROLES = frozenset({"train", "train_seen", "discovery"})
_FALLBACK_POLICIES = frozenset({"global", "fixed_parent"})


class HDBSCANDependencyError(RuntimeError):
    """Raised when the real contrib dependency cannot satisfy the frozen contract."""


def _uid_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_envelope(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen HDBSCAN artifact: {path}")
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
class HDBSCANDependencyAudit:
    executable: str
    module_found: bool
    distribution_version: str | None
    module_origin: str | None
    class_module: str | None
    prediction_data_parameter: bool
    approximate_predict_available: bool
    sklearn_same_name_available: bool
    expected_version: str | None
    usable: bool
    blocker: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_distribution": CONTRIB_DISTRIBUTION,
            "required_module": CONTRIB_MODULE,
            "forbidden_substitute": FORBIDDEN_SKLEARN_CLASS,
            "executable": self.executable,
            "module_found": self.module_found,
            "distribution_version": self.distribution_version,
            "module_origin": self.module_origin,
            "class_module": self.class_module,
            "prediction_data_parameter": self.prediction_data_parameter,
            "approximate_predict_available": self.approximate_predict_available,
            "sklearn_same_name_available": self.sklearn_same_name_available,
            "expected_version": self.expected_version,
            "usable": self.usable,
            "blocker": self.blocker,
        }


def probe_contrib_hdbscan(*, expected_version: str | None = None) -> HDBSCANDependencyAudit:
    """Inspect the actual contrib package without accepting sklearn's class."""

    import sys

    sklearn_available = False
    try:
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        sklearn_available = hasattr(sklearn_cluster, "HDBSCAN")
    except (ImportError, AttributeError):
        pass

    spec = importlib.util.find_spec(CONTRIB_MODULE)
    if spec is None:
        return HDBSCANDependencyAudit(
            executable=sys.executable,
            module_found=False,
            distribution_version=None,
            module_origin=None,
            class_module=None,
            prediction_data_parameter=False,
            approximate_predict_available=False,
            sklearn_same_name_available=sklearn_available,
            expected_version=expected_version,
            usable=False,
            blocker="contrib_hdbscan_not_installed",
        )

    try:
        module = importlib.import_module(CONTRIB_MODULE)
        prediction = importlib.import_module("hdbscan.prediction")
        hdbscan_class = getattr(module, "HDBSCAN")
        version = importlib.metadata.version(CONTRIB_DISTRIBUTION)
        class_module = str(getattr(hdbscan_class, "__module__", ""))
        has_prediction_data = "prediction_data" in inspect.signature(hdbscan_class).parameters
        has_approximate_predict = callable(getattr(prediction, "approximate_predict", None))
    except (ImportError, AttributeError, TypeError, ValueError, importlib.metadata.PackageNotFoundError):
        return HDBSCANDependencyAudit(
            executable=sys.executable,
            module_found=True,
            distribution_version=None,
            module_origin=str(spec.origin) if spec.origin else None,
            class_module=None,
            prediction_data_parameter=False,
            approximate_predict_available=False,
            sklearn_same_name_available=sklearn_available,
            expected_version=expected_version,
            usable=False,
            blocker="contrib_hdbscan_api_probe_failed",
        )

    blocker: str | None = None
    if not class_module.startswith("hdbscan."):
        blocker = "resolved_HDBSCAN_is_not_from_contrib_hdbscan"
    elif not has_prediction_data:
        blocker = "prediction_data_constructor_parameter_missing"
    elif not has_approximate_predict:
        blocker = "hdbscan_prediction_approximate_predict_missing"
    elif expected_version is None:
        blocker = "exact_dependency_version_not_frozen"
    elif version != expected_version:
        blocker = "installed_dependency_version_differs_from_frozen_version"
    return HDBSCANDependencyAudit(
        executable=sys.executable,
        module_found=True,
        distribution_version=version,
        module_origin=str(spec.origin) if spec.origin else None,
        class_module=class_module,
        prediction_data_parameter=has_prediction_data,
        approximate_predict_available=has_approximate_predict,
        sklearn_same_name_available=sklearn_available,
        expected_version=expected_version,
        usable=blocker is None,
        blocker=blocker,
    )


def require_contrib_hdbscan(*, expected_version: str) -> tuple[type[Any], Any, str]:
    """Return the verified contrib class/function or fail before fitting."""

    audit = probe_contrib_hdbscan(expected_version=expected_version)
    if not audit.usable:
        raise HDBSCANDependencyError(
            f"contrib hdbscan dependency is unavailable: {audit.blocker}; "
            f"sklearn.cluster.HDBSCAN is not a substitute"
        )
    module = importlib.import_module(CONTRIB_MODULE)
    prediction = importlib.import_module("hdbscan.prediction")
    return module.HDBSCAN, prediction.approximate_predict, str(audit.distribution_version)


@dataclass(frozen=True, slots=True)
class FrozenHDBSCANArtifact:
    environment_id: str
    fit_role: str
    feature_artifact_sha256: str
    projection_artifact_sha256: str
    fit_uids: tuple[str, ...]
    fit_uid_sha256: str
    feature_dimension: int
    parameters: Mapping[str, Any]
    dependency_version: str
    raw_labels: tuple[int, ...]
    strengths: tuple[float, ...]
    raw_cluster_sizes: tuple[dict[str, int], ...]
    raw_label_to_state: tuple[tuple[int, str], ...]
    retained_state_ids: tuple[str, ...]
    training_states: tuple[str, ...]
    fallback_state_id: str
    fallback_policy: str
    minimum_routing_support: int
    noise_count: int
    low_support_count: int
    model_backend: str
    model_file_path: str | None
    model_file_sha256: str | None
    fallback_only_reason: str | None
    information_access: str = "input_only"
    algorithm_id: str = ALGORITHM_ID

    def __post_init__(self) -> None:
        _require_sha256(self.feature_artifact_sha256, name="feature artifact hash")
        _require_sha256(self.projection_artifact_sha256, name="projection artifact hash")
        if self.fit_role not in _FIT_ROLES:
            raise ArtifactIntegrityError("HDBSCAN fit role must be train/discovery")
        if self.information_access != "input_only" or self.algorithm_id != ALGORITHM_ID:
            raise ArtifactIntegrityError("unexpected HDBSCAN information or algorithm identity")
        if not self.fit_uids or len(self.fit_uids) != len(set(self.fit_uids)):
            raise ArtifactIntegrityError("fit UIDs must be non-empty and unique")
        if _uid_sha256(self.fit_uids) != self.fit_uid_sha256:
            raise ArtifactIntegrityError("fit UID hash does not match frozen UID order")
        if self.feature_dimension <= 0 or self.minimum_routing_support <= 0:
            raise ArtifactIntegrityError("feature dimension and support threshold must be positive")
        if self.fallback_policy not in _FALLBACK_POLICIES:
            raise ArtifactIntegrityError("unsupported HDBSCAN fallback policy")
        if not self.fallback_state_id or self.fallback_state_id in self.retained_state_ids:
            raise ArtifactIntegrityError("fallback state must be separate from density states")
        if not self.dependency_version:
            raise ArtifactIntegrityError("a real contrib dependency version must be recorded")

        row_count = len(self.fit_uids)
        if not (
            len(self.raw_labels)
            == len(self.strengths)
            == len(self.training_states)
            == row_count
        ):
            raise ArtifactIntegrityError("HDBSCAN training arrays differ from fit UID count")
        strength_values = np.asarray(self.strengths, dtype=np.float64)
        if not np.isfinite(strength_values).all() or np.any(strength_values < 0) or np.any(
            strength_values > 1
        ):
            raise ArtifactIntegrityError("HDBSCAN strengths must be finite in [0, 1]")
        if any(label < -1 for label in self.raw_labels):
            raise ArtifactIntegrityError("HDBSCAN raw labels must use -1 or non-negative integers")

        mapping = dict(self.raw_label_to_state)
        if len(mapping) != len(self.raw_label_to_state) or any(label < 0 for label in mapping):
            raise ArtifactIntegrityError("retained raw-label mapping is invalid")
        if tuple(mapping.values()) != self.retained_state_ids or len(set(mapping.values())) != len(
            mapping
        ):
            raise ArtifactIntegrityError("density state order/mapping is inconsistent")
        sizes = {int(row["raw_label"]): int(row["support"]) for row in self.raw_cluster_sizes}
        observed = Counter(label for label in self.raw_labels if label >= 0)
        if sizes != dict(sorted(observed.items())):
            raise ArtifactIntegrityError("raw cluster-size distribution is inconsistent")
        expected_retained = tuple(
            label for label, support in sorted(observed.items()) if support >= self.minimum_routing_support
        )
        if tuple(mapping) != expected_retained:
            raise ArtifactIntegrityError("retained labels violate frozen support threshold")
        expected_states = tuple(mapping.get(label, self.fallback_state_id) for label in self.raw_labels)
        if self.training_states != expected_states:
            raise ArtifactIntegrityError("training fallback assignments are inconsistent")
        if self.noise_count != self.raw_labels.count(-1):
            raise ArtifactIntegrityError("recorded HDBSCAN noise count is inconsistent")
        expected_low_support = sum(
            1 for label in self.raw_labels if label >= 0 and label not in mapping
        )
        if self.low_support_count != expected_low_support:
            raise ArtifactIntegrityError("recorded low-support count is inconsistent")

        if self.model_backend == ASSIGNMENT_BACKEND:
            if self.model_file_path is None or self.model_file_sha256 is None:
                raise ArtifactIntegrityError("approximate_predict requires a frozen model file")
            _require_sha256(self.model_file_sha256, name="HDBSCAN model file hash")
            if self.parameters.get("prediction_data") is not True:
                raise ArtifactIntegrityError("contrib HDBSCAN must be fit with prediction_data=True")
            if self.fallback_only_reason is not None:
                raise ArtifactIntegrityError("model-backed artifact cannot be fallback-only")
        elif self.model_backend == FALLBACK_ONLY_BACKEND:
            if self.model_file_path is not None or self.model_file_sha256 is not None:
                raise ArtifactIntegrityError("fallback-only artifact cannot reference a model")
            if self.retained_state_ids or self.raw_label_to_state:
                raise ArtifactIntegrityError("fallback-only artifact cannot retain density clusters")
            if not self.fallback_only_reason:
                raise ArtifactIntegrityError("fallback-only artifact must record its reason")
        else:
            raise ArtifactIntegrityError("unknown HDBSCAN assignment backend")

    @property
    def actual_cluster_count(self) -> int:
        return len(self.raw_cluster_sizes)

    @property
    def retained_cluster_count(self) -> int:
        return len(self.retained_state_ids)

    @property
    def noise_rate(self) -> float:
        return self.noise_count / len(self.fit_uids)

    @property
    def coverage(self) -> float:
        covered = sum(state != self.fallback_state_id for state in self.training_states)
        return covered / len(self.fit_uids)

    @property
    def fallback_rate(self) -> float:
        return 1.0 - self.coverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "subject_routing_input_only_contrib_hdbscan",
            "algorithm_id": self.algorithm_id,
            "information_access": self.information_access,
            "label_access_split": "none",
            "environment_id": self.environment_id,
            "fit_role": self.fit_role,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "projection_artifact_sha256": self.projection_artifact_sha256,
            "fit_uid_sha256": self.fit_uid_sha256,
            "fit_uids": list(self.fit_uids),
            "feature_dimension": self.feature_dimension,
            "parameters": dict(self.parameters),
            "dependency": {
                "distribution": CONTRIB_DISTRIBUTION,
                "version": self.dependency_version,
                "forbidden_substitute": FORBIDDEN_SKLEARN_CLASS,
            },
            "raw_labels": list(self.raw_labels),
            "strengths": list(self.strengths),
            "raw_cluster_sizes": list(self.raw_cluster_sizes),
            "raw_label_to_state": [
                {"raw_label": label, "state_id": state}
                for label, state in self.raw_label_to_state
            ],
            "retained_state_ids": list(self.retained_state_ids),
            "training_states": list(self.training_states),
            "fallback": {
                "state_id": self.fallback_state_id,
                "policy": self.fallback_policy,
                "minimum_routing_support": self.minimum_routing_support,
                "noise_count": self.noise_count,
                "low_support_count": self.low_support_count,
                "fallback_count": self.training_states.count(self.fallback_state_id),
                "fallback_rate": self.fallback_rate,
                "fallback_decisions_per_row": 1,
                "extra_online_expert_calls": 0,
                "row_drop_count": 0,
            },
            "coverage": self.coverage,
            "noise_rate": self.noise_rate,
            "actual_cluster_count": self.actual_cluster_count,
            "retained_cluster_count": self.retained_cluster_count,
            "model": {
                "backend": self.model_backend,
                "path": self.model_file_path,
                "sha256": self.model_file_sha256,
                "prediction_data": self.parameters.get("prediction_data"),
                "fallback_only_reason": self.fallback_only_reason,
            },
            "test_time_new_cluster_creation": False,
            "row_drop_count": 0,
            "dev_or_test_refit": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_json_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenHDBSCANArtifact":
        dependency = value["dependency"]
        fallback = value["fallback"]
        model = value["model"]
        return cls(
            environment_id=str(value["environment_id"]),
            fit_role=str(value["fit_role"]),
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            projection_artifact_sha256=str(value["projection_artifact_sha256"]),
            fit_uids=tuple(str(uid) for uid in value["fit_uids"]),
            fit_uid_sha256=str(value["fit_uid_sha256"]),
            feature_dimension=int(value["feature_dimension"]),
            parameters=dict(value["parameters"]),
            dependency_version=str(dependency["version"]),
            raw_labels=tuple(int(label) for label in value["raw_labels"]),
            strengths=tuple(float(strength) for strength in value["strengths"]),
            raw_cluster_sizes=tuple(dict(row) for row in value["raw_cluster_sizes"]),
            raw_label_to_state=tuple(
                (int(row["raw_label"]), str(row["state_id"]))
                for row in value["raw_label_to_state"]
            ),
            retained_state_ids=tuple(str(state) for state in value["retained_state_ids"]),
            training_states=tuple(str(state) for state in value["training_states"]),
            fallback_state_id=str(fallback["state_id"]),
            fallback_policy=str(fallback["policy"]),
            minimum_routing_support=int(fallback["minimum_routing_support"]),
            noise_count=int(fallback["noise_count"]),
            low_support_count=int(fallback["low_support_count"]),
            model_backend=str(model["backend"]),
            model_file_path=(None if model.get("path") is None else str(model["path"])),
            model_file_sha256=(
                None if model.get("sha256") is None else str(model["sha256"])
            ),
            fallback_only_reason=(
                None
                if model.get("fallback_only_reason") is None
                else str(model["fallback_only_reason"])
            ),
            information_access=str(value["information_access"]),
            algorithm_id=str(value["algorithm_id"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrozenHDBSCANArtifact":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = value.get("payload")
        if not isinstance(payload, dict) or value.get("payload_sha256") != canonical_sha256(
            payload
        ):
            raise ArtifactIntegrityError("HDBSCAN artifact digest mismatch")
        return cls.from_dict(payload)


def artifact_from_training_output(
    *,
    environment_id: str,
    fit_role: str,
    feature_artifact_sha256: str,
    projection_artifact_sha256: str,
    fit_uids: Sequence[str],
    feature_dimension: int,
    parameters: Mapping[str, Any],
    dependency_version: str,
    raw_labels: Sequence[int],
    strengths: Sequence[float],
    minimum_routing_support: int,
    fallback_state_id: str,
    fallback_policy: str,
    model_backend: str,
    model_file_path: str | None,
    model_file_sha256: str | None,
    fallback_only_reason: str | None = None,
) -> FrozenHDBSCANArtifact:
    """Freeze real fit outputs or explicitly marked mechanics fixtures."""

    ids = tuple(str(uid) for uid in fit_uids)
    labels = tuple(int(label) for label in raw_labels)
    probabilities = tuple(float(strength) for strength in strengths)
    counts = Counter(label for label in labels if label >= 0)
    retained_labels = tuple(
        label for label, support in sorted(counts.items()) if support >= minimum_routing_support
    )
    if model_backend == FALLBACK_ONLY_BACKEND:
        retained_labels = ()
    mapping = tuple(
        (label, f"density-{position:04d}") for position, label in enumerate(retained_labels)
    )
    lookup = dict(mapping)
    states = tuple(lookup.get(label, fallback_state_id) for label in labels)
    return FrozenHDBSCANArtifact(
        environment_id=str(environment_id),
        fit_role=str(fit_role),
        feature_artifact_sha256=str(feature_artifact_sha256),
        projection_artifact_sha256=str(projection_artifact_sha256),
        fit_uids=ids,
        fit_uid_sha256=_uid_sha256(ids),
        feature_dimension=int(feature_dimension),
        parameters=dict(parameters),
        dependency_version=str(dependency_version),
        raw_labels=labels,
        strengths=probabilities,
        raw_cluster_sizes=tuple(
            {"raw_label": label, "support": support} for label, support in sorted(counts.items())
        ),
        raw_label_to_state=mapping,
        retained_state_ids=tuple(state for _label, state in mapping),
        training_states=states,
        fallback_state_id=str(fallback_state_id),
        fallback_policy=str(fallback_policy),
        minimum_routing_support=int(minimum_routing_support),
        noise_count=labels.count(-1),
        low_support_count=sum(1 for label in labels if label >= 0 and label not in lookup),
        model_backend=str(model_backend),
        model_file_path=model_file_path,
        model_file_sha256=model_file_sha256,
        fallback_only_reason=fallback_only_reason,
    )


@dataclass(frozen=True, slots=True)
class HDBSCANAssignmentResult:
    batch: AssignmentBatch
    raw_labels: tuple[int, ...]
    strengths: tuple[float, ...]
    fallback_reasons: tuple[str | None, ...]
    frozen_state_ids: tuple[str, ...]
    fallback_state_id: str

    @property
    def coverage(self) -> float:
        if not self.batch.states:
            return 0.0
        return sum(state != self.fallback_state_id for state in self.batch.states) / len(
            self.batch.states
        )

    @property
    def noise_rate(self) -> float:
        if not self.raw_labels:
            return 0.0
        return self.raw_labels.count(-1) / len(self.raw_labels)

    def to_report(self) -> dict[str, Any]:
        counts = Counter(self.batch.states)
        reasons = Counter(reason for reason in self.fallback_reasons if reason is not None)
        return {
            "rows": len(self.batch.states),
            "coverage": self.coverage,
            "noise_rate": self.noise_rate,
            "fallback_rate": 1.0 - self.coverage,
            "state_distribution": dict(sorted(counts.items())),
            "fallback_reason_distribution": dict(sorted(reasons.items())),
            "frozen_density_state_count": len(self.frozen_state_ids),
            "test_time_new_cluster_count": 0,
            "fallback_decisions_per_row": 1,
            "extra_online_expert_calls": 0,
            "row_drop_count": 0,
            "score_semantics": self.batch.score_semantics,
        }


def apply_frozen_prediction_output(
    artifact: FrozenHDBSCANArtifact,
    raw_labels: Sequence[int],
    strengths: Sequence[float],
) -> HDBSCANAssignmentResult:
    """Map approximate-predict output to frozen states without creating clusters."""

    labels = tuple(int(label) for label in raw_labels)
    probabilities = tuple(float(strength) for strength in strengths)
    if len(labels) != len(probabilities):
        raise ArtifactIntegrityError("prediction labels and strengths differ in length")
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ArtifactIntegrityError("prediction strengths must be finite in [0, 1]")

    mapping = dict(artifact.raw_label_to_state)
    score_state_ids = artifact.retained_state_ids + (artifact.fallback_state_id,)
    state_position = {state: column for column, state in enumerate(score_state_ids)}
    scores = np.zeros((len(labels), len(score_state_ids)), dtype=np.float32)
    states: list[str] = []
    reasons: list[str | None] = []
    for row, (label, strength) in enumerate(zip(labels, probabilities)):
        state = mapping.get(label)
        if label == -1:
            reason = "noise"
        elif state is None:
            reason = "unknown_or_low_support_raw_label"
        else:
            reason = None
        if state is None:
            state = artifact.fallback_state_id
            scores[row, state_position[state]] = 1.0
        else:
            scores[row, state_position[state]] = strength
            scores[row, state_position[artifact.fallback_state_id]] = 1.0 - strength
        states.append(state)
        reasons.append(reason)
    scores.flags.writeable = False
    return HDBSCANAssignmentResult(
        batch=AssignmentBatch(
            states=tuple(states),
            assignment_scores=scores,
            score_semantics="hdbscan_membership_strength_not_correctness_probability",
        ),
        raw_labels=labels,
        strengths=probabilities,
        fallback_reasons=tuple(reasons),
        frozen_state_ids=artifact.retained_state_ids,
        fallback_state_id=artifact.fallback_state_id,
    )


class ContribHDBSCANAdapter:
    """Fit contrib HDBSCAN once and assign only with frozen approximate_predict."""

    def __init__(
        self,
        *,
        expected_dependency_version: str,
        min_cluster_size: int,
        min_samples: int | None,
        minimum_routing_support: int,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        cluster_selection_epsilon: float = 0.0,
        allow_single_cluster: bool = True,
        fallback_state_id: str = "__global__",
        fallback_policy: str = "global",
    ) -> None:
        if not expected_dependency_version:
            raise ValueError("an exact contrib hdbscan dependency version is required")
        if min_cluster_size < 2 or (min_samples is not None and min_samples <= 0):
            raise ValueError("invalid HDBSCAN density hyperparameters")
        if minimum_routing_support <= 0 or cluster_selection_epsilon < 0:
            raise ValueError("invalid HDBSCAN routing/fallback hyperparameters")
        if fallback_policy not in _FALLBACK_POLICIES:
            raise ValueError("fallback policy must be global or fixed_parent")
        self.expected_dependency_version = str(expected_dependency_version)
        self.parameters = {
            "min_cluster_size": int(min_cluster_size),
            "min_samples": None if min_samples is None else int(min_samples),
            "metric": str(metric),
            "cluster_selection_method": str(cluster_selection_method),
            "cluster_selection_epsilon": float(cluster_selection_epsilon),
            "allow_single_cluster": bool(allow_single_cluster),
            "prediction_data": True,
        }
        self.minimum_routing_support = int(minimum_routing_support)
        self.fallback_state_id = str(fallback_state_id)
        self.fallback_policy = str(fallback_policy)

    def fit(
        self,
        features: np.ndarray,
        uids: Sequence[str],
        *,
        environment_id: str,
        fit_role: str,
        feature_artifact_sha256: str,
        projection_artifact_sha256: str,
        model_output_path: str | Path,
        audit: UIDAccessAudit,
    ) -> FrozenHDBSCANArtifact:
        if audit.operation is not Operation.FIT:
            raise UIDScopeError("HDBSCAN fit requires a fit UID scope")
        ids = tuple(str(uid) for uid in uids)
        audit.require(ids, component="contrib_hdbscan_fit")
        values = np.asarray(features, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] != len(ids)
            or not np.isfinite(values).all()
            or len(ids) != len(set(ids))
        ):
            raise ArtifactIntegrityError("HDBSCAN fit features/UIDs are invalid or misaligned")
        _require_sha256(feature_artifact_sha256, name="feature artifact hash")
        _require_sha256(projection_artifact_sha256, name="projection artifact hash")

        hdbscan_class, _predict, version = require_contrib_hdbscan(
            expected_version=self.expected_dependency_version
        )
        model = hdbscan_class(**self.parameters)
        model.fit(values)
        labels = np.asarray(model.labels_, dtype=np.int64)
        strengths = np.asarray(model.probabilities_, dtype=np.float64)
        target = Path(model_output_path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite frozen HDBSCAN model: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return artifact_from_training_output(
            environment_id=environment_id,
            fit_role=fit_role,
            feature_artifact_sha256=feature_artifact_sha256,
            projection_artifact_sha256=projection_artifact_sha256,
            fit_uids=ids,
            feature_dimension=values.shape[1],
            parameters=self.parameters,
            dependency_version=version,
            raw_labels=labels,
            strengths=strengths,
            minimum_routing_support=self.minimum_routing_support,
            fallback_state_id=self.fallback_state_id,
            fallback_policy=self.fallback_policy,
            model_backend=ASSIGNMENT_BACKEND,
            model_file_path=str(target),
            model_file_sha256=_file_sha256(target),
        )

    @staticmethod
    def assign(
        artifact: FrozenHDBSCANArtifact,
        features: np.ndarray,
        uids: Sequence[str],
        *,
        feature_artifact_sha256: str,
        audit: UIDAccessAudit,
    ) -> HDBSCANAssignmentResult:
        if audit.operation is not Operation.ASSIGN:
            raise UIDScopeError("HDBSCAN assignment requires an assign UID scope")
        ids = tuple(str(uid) for uid in uids)
        audit.require(ids, component="contrib_hdbscan_approximate_predict")
        if feature_artifact_sha256 != artifact.feature_artifact_sha256:
            raise ArtifactIntegrityError("runtime feature hash differs from frozen HDBSCAN fit")
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (len(ids), artifact.feature_dimension) or not np.isfinite(
            values
        ).all():
            raise ArtifactIntegrityError("HDBSCAN assignment features are invalid or misaligned")
        if artifact.model_backend == FALLBACK_ONLY_BACKEND or not artifact.retained_state_ids:
            return apply_frozen_prediction_output(
                artifact, [-1] * len(ids), [0.0] * len(ids)
            )

        _hdbscan_class, approximate_predict, _version = require_contrib_hdbscan(
            expected_version=artifact.dependency_version
        )
        model_path = Path(str(artifact.model_file_path))
        if not model_path.is_file() or _file_sha256(model_path) != artifact.model_file_sha256:
            raise ArtifactIntegrityError("frozen HDBSCAN model file is missing or hash-mismatched")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)  # noqa: S301 - local hash-pinned artifact only
        labels, strengths = approximate_predict(model, values)
        return apply_frozen_prediction_output(artifact, labels, strengths)


def fallback_only_artifact(
    *,
    fit_uids: Sequence[str],
    feature_dimension: int,
    min_cluster_size: int,
    feature_artifact_sha256: str,
    projection_artifact_sha256: str,
    reason: str,
    environment_id: str = "synthetic_mechanics_fixture_not_real_hdbscan_result",
    fit_role: str = "discovery",
    fallback_state_id: str = "__global__",
) -> FrozenHDBSCANArtifact:
    """Build an explicit all-fallback artifact for too-small/all-noise handling tests."""

    ids = tuple(str(uid) for uid in fit_uids)
    return artifact_from_training_output(
        environment_id=environment_id,
        fit_role=fit_role,
        feature_artifact_sha256=feature_artifact_sha256,
        projection_artifact_sha256=projection_artifact_sha256,
        fit_uids=ids,
        feature_dimension=feature_dimension,
        parameters={"min_cluster_size": int(min_cluster_size), "prediction_data": True},
        dependency_version="MECHANICS_FIXTURE_NO_HDBSCAN_EXECUTION",
        raw_labels=[-1] * len(ids),
        strengths=[0.0] * len(ids),
        minimum_routing_support=max(1, int(min_cluster_size)),
        fallback_state_id=fallback_state_id,
        fallback_policy="global",
        model_backend=FALLBACK_ONLY_BACKEND,
        model_file_path=None,
        model_file_sha256=None,
        fallback_only_reason=str(reason),
    )
