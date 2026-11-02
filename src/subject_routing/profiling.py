"""Shared train-only capability profiling for every subject discovery method."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactIntegrityError, canonical_sha256
from .data_contracts import TrainingLabels, UnexplainedCostError
from .leakage_guard import Operation, UIDAccessAudit, UIDScopeError


PROFILE_ALGORITHM_ID = "shared_train_profile_parent_global_shrinkage_v1"
_TRAIN_OPERATIONS = frozenset({Operation.FIT, Operation.TRAIN})


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


def _atomic_envelope(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen capability artifact: {path}")
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


def _nullable(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _immutable(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ScopedTrainingLabelProvenance:
    requested_uid_sha256: str
    requested_rows: int
    expert_ids: tuple[str, ...]
    quality_files: tuple[dict[str, Any], ...]
    cost_matrix_path: str
    cost_matrix_sha256: str
    cost_access: str
    global_y_npz_opened: bool = False
    test_uid_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_uid_sha256": self.requested_uid_sha256,
            "requested_rows": self.requested_rows,
            "expert_ids": list(self.expert_ids),
            "quality_files": list(self.quality_files),
            "cost_matrix_path": self.cost_matrix_path,
            "cost_matrix_sha256": self.cost_matrix_sha256,
            "cost_access": self.cost_access,
            "global_y_npz_opened": self.global_y_npz_opened,
            "test_uid_requested": self.test_uid_requested,
        }


def materialize_scoped_training_labels(
    requested_uids: Sequence[str],
    *,
    allowed_train_uids: Sequence[str],
    expert_ids: Sequence[str],
    oracle_score_root: str | Path,
    cost_matrix_path: str | Path,
    sample_registry_path: str | Path,
    model_registry_path: str | Path,
    expected_cost_sha256: str,
    expected_sample_registry_sha256: str,
    expected_model_registry_sha256: str,
    expected_quality_sha256: Mapping[str, str],
    audit: UIDAccessAudit,
) -> tuple[TrainingLabels, ScopedTrainingLabelProvenance]:
    """Materialize only allowed train rows; never open the compressed global Y.

    Quality is predicate-filtered from per-source Parquet files.  Cost uses a
    read-only NPY memmap and copies only the explicitly allowed row indices.
    """

    if audit.operation not in _TRAIN_OPERATIONS:
        raise UIDScopeError("profile label materialization requires fit/train UID scope")
    ids = tuple(str(uid) for uid in requested_uids)
    allowed = frozenset(str(uid) for uid in allowed_train_uids)
    if not ids or len(ids) != len(set(ids)):
        raise ArtifactIntegrityError("profile UIDs must be non-empty and unique")
    outside = sorted(set(ids).difference(allowed))
    if outside:
        raise UIDScopeError(f"profile label request is outside allowed train UIDs: {outside[0]}")
    audit.require(ids, component="scoped_profile_label_materializer")

    sample_path = Path(sample_registry_path)
    model_path = Path(model_registry_path)
    cost_path = Path(cost_matrix_path)
    expected_files = (
        (sample_path, expected_sample_registry_sha256, "sample registry"),
        (model_path, expected_model_registry_sha256, "model registry"),
        (cost_path, expected_cost_sha256, "cost matrix"),
    )
    for path, expected, name in expected_files:
        if not path.is_file() or _file_sha256(path) != expected:
            raise ArtifactIntegrityError(f"{name} is missing or hash-mismatched")
    samples = tuple(str(uid) for uid in pickle.loads(sample_path.read_bytes()))
    models = tuple(str(model) for model in pickle.loads(model_path.read_bytes()))
    canonical_experts = tuple(str(expert) for expert in expert_ids)
    if models != canonical_experts:
        raise ArtifactIntegrityError("model registry differs from frozen canonical expert order")
    row_by_uid = {uid: row for row, uid in enumerate(samples)}
    missing_rows = [uid for uid in ids if uid not in row_by_uid]
    if missing_rows:
        raise ArtifactIntegrityError(f"profile UID absent from sample registry: {missing_rows[0]}")

    # Import only in this explicitly scoped offline-label function.
    import pyarrow.parquet as pq

    quality = np.full((len(ids), len(models)), np.nan, dtype=np.float64)
    output_row = {uid: row for row, uid in enumerate(ids)}
    output_column = {model: column for column, model in enumerate(models)}
    uids_by_source: dict[str, list[str]] = {}
    for uid in ids:
        source = uid.split("/", 1)[0]
        uids_by_source.setdefault(source, []).append(uid)
    quality_records: list[dict[str, Any]] = []
    root = Path(oracle_score_root)
    for source, source_uids in sorted(uids_by_source.items()):
        score_path = root / f"{source.casefold()}.parquet"
        expected_hash = expected_quality_sha256.get(source)
        if expected_hash is None:
            raise ArtifactIntegrityError(f"no frozen quality provenance for source: {source}")
        if not score_path.is_file() or _file_sha256(score_path) != expected_hash:
            raise ArtifactIntegrityError(f"quality file is missing or hash-mismatched: {source}")
        table = pq.read_table(
            score_path,
            columns=["sample_id", "model_id", "quality"],
            filters=[("sample_id", "in", source_uids)],
        )
        records = table.to_pylist()
        for record in records:
            uid = str(record["sample_id"])
            model = str(record["model_id"])
            if uid not in output_row or model not in output_column:
                raise ArtifactIntegrityError("predicate-filtered quality row is outside frozen scope")
            row = output_row[uid]
            column = output_column[model]
            if np.isfinite(quality[row, column]):
                raise ArtifactIntegrityError("duplicate quality cell in scoped profile materialization")
            quality[row, column] = float(record["quality"])
        quality_records.append(
            {
                "source": source,
                "path": str(score_path),
                "sha256": expected_hash,
                "requested_samples": len(source_uids),
                "returned_cells": len(records),
                "read_mode": "predicate_filtered_requested_sample_ids_only",
            }
        )
    if not np.isfinite(quality).all() or not np.isin(quality, (0.0, 1.0)).all():
        raise ArtifactIntegrityError("scoped profile quality matrix is incomplete or non-binary")

    cost_source = np.load(cost_path, mmap_mode="r", allow_pickle=False)
    if cost_source.shape != (len(samples), len(models)):
        raise ArtifactIntegrityError("cost memmap shape differs from frozen registries")
    selected_rows = np.asarray([row_by_uid[uid] for uid in ids], dtype=np.int64)
    cost = np.asarray(cost_source[selected_rows], dtype=np.float64)
    if not np.isfinite(cost).all() or (cost < 0).any():
        raise ArtifactIntegrityError("scoped profile cost rows are invalid")
    labels = TrainingLabels.create(
        uids=ids,
        expert_ids=models,
        y=quality,
        c=cost,
        observed_mask=None,
        cost_observed_mask=None,
        complete_matrix=True,
        quality_kind="binary_correctness",
        cost_unit="usd_proxy_per_request",
        zero_cost_semantics="unexplained_placeholder",
    )
    return labels, ScopedTrainingLabelProvenance(
        requested_uid_sha256=_uid_sha256(ids),
        requested_rows=len(ids),
        expert_ids=models,
        quality_files=tuple(quality_records),
        cost_matrix_path=str(cost_path),
        cost_matrix_sha256=expected_cost_sha256,
        cost_access="read_only_memmap_explicit_allowed_row_indices_only",
    )


@dataclass(frozen=True, slots=True)
class CapabilityProfileArtifact:
    method_id: str
    environment_id: str
    subject_artifact_sha256: str
    assigner_artifact_sha256: str
    label_artifact_sha256: str
    profile_uids: tuple[str, ...]
    profile_uid_sha256: str
    deployed_assignments: tuple[str, ...]
    assignment_sha256: str
    state_ids: tuple[str, ...]
    parent_by_state: Mapping[str, str | None]
    global_state_id: str
    expert_ids: tuple[str, ...]
    state_support: np.ndarray
    all_wrong_support: np.ndarray
    partially_missing_support: np.ndarray
    quality_observed: np.ndarray
    success_sum: np.ndarray
    raw_quality_mean: np.ndarray
    shrunk_quality_mean: np.ndarray
    cost_observed: np.ndarray
    raw_cost_mean: np.ndarray
    raw_cost_median: np.ndarray
    shrunk_cost_mean: np.ndarray
    kappa: float
    cost_normalizer: float
    cost_unit: str
    information_access: str = "train_profile_outcomes_only"
    algorithm_id: str = PROFILE_ALGORITHM_ID

    def __post_init__(self) -> None:
        for name, digest in (
            ("subject artifact", self.subject_artifact_sha256),
            ("assigner artifact", self.assigner_artifact_sha256),
            ("label artifact", self.label_artifact_sha256),
        ):
            _require_sha256(digest, name=name)
        if self.algorithm_id != PROFILE_ALGORITHM_ID or self.information_access != (
            "train_profile_outcomes_only"
        ):
            raise ArtifactIntegrityError("unexpected capability-profile identity")
        if not self.profile_uids or len(self.profile_uids) != len(set(self.profile_uids)):
            raise ArtifactIntegrityError("profile UIDs must be non-empty and unique")
        if _uid_sha256(self.profile_uids) != self.profile_uid_sha256:
            raise ArtifactIntegrityError("profile UID hash mismatch")
        if len(self.deployed_assignments) != len(self.profile_uids):
            raise ArtifactIntegrityError("deployed assignment count differs from profile UIDs")
        expected_assignment_hash = canonical_sha256(
            {
                "uids": list(self.profile_uids),
                "states": list(self.deployed_assignments),
            }
        )
        if self.assignment_sha256 != expected_assignment_hash:
            raise ArtifactIntegrityError("deployed profile assignment hash mismatch")
        if not self.state_ids or len(self.state_ids) != len(set(self.state_ids)):
            raise ArtifactIntegrityError("profile state IDs must be non-empty and unique")
        if self.global_state_id not in self.state_ids:
            raise ArtifactIntegrityError("profile requires a global fallback state")
        if set(self.parent_by_state) != set(self.state_ids):
            raise ArtifactIntegrityError("profile parent map must cover every state")
        allowed_states = set(self.state_ids)
        if any(state not in allowed_states for state in self.deployed_assignments):
            raise ArtifactIntegrityError("profile assignments reference unknown states")
        for state, parent in self.parent_by_state.items():
            if state == self.global_state_id and parent is not None:
                raise ArtifactIntegrityError("global profile state cannot have a parent")
            if parent is not None and (parent not in allowed_states or parent == state):
                raise ArtifactIntegrityError("profile parent map is invalid")
        for state in self.state_ids:
            seen: set[str] = set()
            current: str | None = state
            while current is not None:
                if current in seen:
                    raise ArtifactIntegrityError("profile parent map contains a cycle")
                seen.add(current)
                current = self.parent_by_state[current]
        if self.kappa < 0 or not np.isfinite(self.cost_normalizer) or self.cost_normalizer <= 0:
            raise ArtifactIntegrityError("profile shrinkage/cost normalizer is invalid")
        if self.cost_unit != "usd_proxy_per_request":
            raise ArtifactIntegrityError("profile cost unit is not the frozen USD proxy")

        rows, columns = len(self.state_ids), len(self.expert_ids)
        vectors = (self.state_support, self.all_wrong_support, self.partially_missing_support)
        matrices = (
            self.quality_observed,
            self.success_sum,
            self.raw_quality_mean,
            self.shrunk_quality_mean,
            self.cost_observed,
            self.raw_cost_mean,
            self.raw_cost_median,
            self.shrunk_cost_mean,
        )
        if any(np.asarray(value).shape != (rows,) for value in vectors):
            raise ArtifactIntegrityError("profile state-vector shape mismatch")
        if any(np.asarray(value).shape != (rows, columns) for value in matrices):
            raise ArtifactIntegrityError("profile state/expert matrix shape mismatch")
        if (np.asarray(self.state_support) < 0).any() or (
            np.asarray(self.quality_observed) < 0
        ).any() or (np.asarray(self.cost_observed) < 0).any():
            raise ArtifactIntegrityError("profile support counts cannot be negative")
        successes = np.asarray(self.success_sum, dtype=np.float64)
        quality_n = np.asarray(self.quality_observed, dtype=np.float64)
        if np.any(successes < 0) or np.any(successes - quality_n > 1e-9):
            raise ArtifactIntegrityError("profile successes exceed observed support")
        for quality in (self.raw_quality_mean, self.shrunk_quality_mean):
            finite = np.isfinite(quality)
            if np.any(np.asarray(quality)[finite] < 0) or np.any(np.asarray(quality)[finite] > 1):
                raise ArtifactIntegrityError("profile quality means must lie in [0, 1]")
        for cost in (self.raw_cost_mean, self.raw_cost_median, self.shrunk_cost_mean):
            finite = np.isfinite(cost)
            if np.any(np.asarray(cost)[finite] < 0):
                raise ArtifactIntegrityError("profile costs cannot be negative")
        for name in (
            "state_support",
            "all_wrong_support",
            "partially_missing_support",
            "quality_observed",
            "success_sum",
            "raw_quality_mean",
            "shrunk_quality_mean",
            "cost_observed",
            "raw_cost_mean",
            "raw_cost_median",
            "shrunk_cost_mean",
        ):
            array = np.asarray(getattr(self, name)).copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    @property
    def state_position(self) -> dict[str, int]:
        return {state: row for row, state in enumerate(self.state_ids)}

    @property
    def expert_position(self) -> dict[str, int]:
        return {expert: column for column, expert in enumerate(self.expert_ids)}

    def to_dict(self) -> dict[str, Any]:
        records = []
        for row, state in enumerate(self.state_ids):
            experts = []
            for column, expert in enumerate(self.expert_ids):
                experts.append(
                    {
                        "expert_id": expert,
                        "quality_observed_n": int(self.quality_observed[row, column]),
                        "success_sum": float(self.success_sum[row, column]),
                        "raw_quality_mean": _nullable(self.raw_quality_mean[row, column]),
                        "shrunk_quality_mean": _nullable(
                            self.shrunk_quality_mean[row, column]
                        ),
                        "quality_observed_mask": bool(self.quality_observed[row, column] > 0),
                        "cost_observed_n": int(self.cost_observed[row, column]),
                        "raw_cost_mean": _nullable(self.raw_cost_mean[row, column]),
                        "raw_cost_median": _nullable(self.raw_cost_median[row, column]),
                        "shrunk_cost_mean": _nullable(self.shrunk_cost_mean[row, column]),
                        "cost_observed_mask": bool(self.cost_observed[row, column] > 0),
                    }
                )
            records.append(
                {
                    "state_id": state,
                    "parent_state_id": self.parent_by_state[state],
                    "assigned_or_descendant_support": int(self.state_support[row]),
                    "all_wrong_rows": int(self.all_wrong_support[row]),
                    "partially_missing_rows": int(self.partially_missing_support[row]),
                    "experts": experts,
                }
            )
        return {
            "schema_version": 1,
            "kind": "subject_routing_shared_train_capability_profile",
            "algorithm_id": self.algorithm_id,
            "information_access": self.information_access,
            "label_access_split": "train_profile_only",
            "method_id": self.method_id,
            "environment_id": self.environment_id,
            "subject_artifact_sha256": self.subject_artifact_sha256,
            "assigner_artifact_sha256": self.assigner_artifact_sha256,
            "label_artifact_sha256": self.label_artifact_sha256,
            "profile_uids": list(self.profile_uids),
            "profile_uid_sha256": self.profile_uid_sha256,
            "deployed_assignments": list(self.deployed_assignments),
            "assignment_sha256": self.assignment_sha256,
            "state_ids": list(self.state_ids),
            "global_state_id": self.global_state_id,
            "expert_ids": list(self.expert_ids),
            "kappa": self.kappa,
            "shrinkage_semantics": "empirical_mean_not_calibrated_probability",
            "cost": {
                "unit": self.cost_unit,
                "normalizer": self.cost_normalizer,
                "normalizer_fit_split": "profile_train_only",
                "normalizer_statistic": "median_of_positive_observed_profile_cost_cells",
                "robust_statistic": "median",
            },
            "row_semantics": {
                "all_wrong_retained": True,
                "all_wrong_converted_to_positive_label": False,
                "missing_is_wrong": False,
                "row_drop_count": 0,
            },
            "states": records,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityProfileArtifact":
        expert_ids = tuple(str(expert) for expert in value["expert_ids"])
        states = value["states"]
        fields = (
            "quality_observed_n",
            "success_sum",
            "raw_quality_mean",
            "shrunk_quality_mean",
            "cost_observed_n",
            "raw_cost_mean",
            "raw_cost_median",
            "shrunk_cost_mean",
        )
        matrices: dict[str, list[list[float]]] = {field: [] for field in fields}
        for state in states:
            if tuple(record["expert_id"] for record in state["experts"]) != expert_ids:
                raise ArtifactIntegrityError("profile expert record order mismatch")
            for field in fields:
                matrices[field].append(
                    [
                        np.nan if record[field] is None else record[field]
                        for record in state["experts"]
                    ]
                )
        cost = value["cost"]
        return cls(
            method_id=str(value["method_id"]),
            environment_id=str(value["environment_id"]),
            subject_artifact_sha256=str(value["subject_artifact_sha256"]),
            assigner_artifact_sha256=str(value["assigner_artifact_sha256"]),
            label_artifact_sha256=str(value["label_artifact_sha256"]),
            profile_uids=tuple(str(uid) for uid in value["profile_uids"]),
            profile_uid_sha256=str(value["profile_uid_sha256"]),
            deployed_assignments=tuple(str(state) for state in value["deployed_assignments"]),
            assignment_sha256=str(value["assignment_sha256"]),
            state_ids=tuple(str(state) for state in value["state_ids"]),
            parent_by_state={
                str(state["state_id"]): (
                    None
                    if state["parent_state_id"] is None
                    else str(state["parent_state_id"])
                )
                for state in states
            },
            global_state_id=str(value["global_state_id"]),
            expert_ids=expert_ids,
            state_support=np.asarray(
                [state["assigned_or_descendant_support"] for state in states], dtype=np.int64
            ),
            all_wrong_support=np.asarray(
                [state["all_wrong_rows"] for state in states], dtype=np.int64
            ),
            partially_missing_support=np.asarray(
                [state["partially_missing_rows"] for state in states], dtype=np.int64
            ),
            quality_observed=np.asarray(matrices["quality_observed_n"], dtype=np.int64),
            success_sum=np.asarray(matrices["success_sum"], dtype=np.float64),
            raw_quality_mean=np.asarray(matrices["raw_quality_mean"], dtype=np.float64),
            shrunk_quality_mean=np.asarray(matrices["shrunk_quality_mean"], dtype=np.float64),
            cost_observed=np.asarray(matrices["cost_observed_n"], dtype=np.int64),
            raw_cost_mean=np.asarray(matrices["raw_cost_mean"], dtype=np.float64),
            raw_cost_median=np.asarray(matrices["raw_cost_median"], dtype=np.float64),
            shrunk_cost_mean=np.asarray(matrices["shrunk_cost_mean"], dtype=np.float64),
            kappa=float(value["kappa"]),
            cost_normalizer=float(cost["normalizer"]),
            cost_unit=str(cost["unit"]),
            information_access=str(value["information_access"]),
            algorithm_id=str(value["algorithm_id"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CapabilityProfileArtifact":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = value.get("payload")
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "subject_routing_shared_train_capability_profile"
            or value.get("payload_sha256") != canonical_sha256(payload)
        ):
            raise ArtifactIntegrityError("capability profile artifact digest/type mismatch")
        return cls.from_dict(payload)


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


def profile_subject_states(
    *,
    method_id: str,
    environment_id: str,
    subject_artifact_sha256: str,
    assigner_artifact_sha256: str,
    label_artifact_sha256: str,
    profile_assignments: Mapping[str, str],
    state_ids: Sequence[str],
    parent_by_state: Mapping[str, str | None],
    global_state_id: str,
    labels: TrainingLabels,
    canonical_expert_order: Sequence[str],
    kappa: float,
    audit: UIDAccessAudit,
) -> CapabilityProfileArtifact:
    """Compute raw and parent/global-shrunk train capability/cost tables."""

    if audit.operation not in _TRAIN_OPERATIONS:
        raise UIDScopeError("capability profiling requires fit/train UID scope")
    uids = tuple(str(uid) for uid in profile_assignments)
    audit.require(uids, component="shared_subject_capability_profiler")
    matrix = labels.matrix.subset(uids).reorder_experts(canonical_expert_order)
    if np.any(matrix.cost_observed_mask & (matrix.c == 0)) and (
        matrix.zero_cost_semantics != "legitimate_zero"
    ):
        raise UnexplainedCostError("profile contains unexplained observed zero cost")
    states = tuple(str(state) for state in state_ids)
    if global_state_id not in states:
        states = states + (str(global_state_id),)
    if len(states) != len(set(states)):
        raise ArtifactIntegrityError("profile state IDs must be unique")
    parents = {
        state: (
            None
            if state == global_state_id
            else (
                str(parent_by_state[state])
                if parent_by_state.get(state) is not None
                else str(global_state_id)
            )
        )
        for state in states
    }
    resolved_assignments = tuple(
        str(profile_assignments[uid])
        if str(profile_assignments[uid]) in states
        else str(global_state_id)
        for uid in uids
    )
    positive_costs = matrix.c[matrix.cost_observed_mask & (matrix.c > 0)]
    if not len(positive_costs):
        raise ArtifactIntegrityError("profile has no positive observed train cost for normalization")
    cost_normalizer = float(np.median(positive_costs))
    state_count, expert_count = len(states), len(matrix.expert_ids)
    state_support = np.zeros(state_count, dtype=np.int64)
    all_wrong_support = np.zeros(state_count, dtype=np.int64)
    partial_support = np.zeros(state_count, dtype=np.int64)
    quality_n = np.zeros((state_count, expert_count), dtype=np.int64)
    successes = np.zeros((state_count, expert_count), dtype=np.float64)
    raw_quality = np.full((state_count, expert_count), np.nan, dtype=np.float64)
    cost_n = np.zeros((state_count, expert_count), dtype=np.int64)
    raw_cost = np.full((state_count, expert_count), np.nan, dtype=np.float64)
    median_cost = np.full((state_count, expert_count), np.nan, dtype=np.float64)
    row_classes = matrix.row_classes()
    for state_row, state in enumerate(states):
        if state == global_state_id:
            rows = np.arange(len(uids), dtype=np.int64)
        else:
            rows = np.asarray(
                [
                    row
                    for row, assigned in enumerate(resolved_assignments)
                    if _descends_from(assigned, state, parents)
                ],
                dtype=np.int64,
            )
        state_support[state_row] = len(rows)
        all_wrong_support[state_row] = int(row_classes["all_wrong"][rows].sum())
        partial_support[state_row] = int(row_classes["partially_missing"][rows].sum())
        for column in range(expert_count):
            quality_mask = matrix.observed_mask[rows, column]
            quality_rows = rows[quality_mask]
            quality_n[state_row, column] = len(quality_rows)
            if len(quality_rows):
                successes[state_row, column] = float(matrix.y[quality_rows, column].sum())
                raw_quality[state_row, column] = (
                    successes[state_row, column] / len(quality_rows)
                )
            cost_mask = matrix.cost_observed_mask[rows, column]
            cost_rows = rows[cost_mask]
            cost_n[state_row, column] = len(cost_rows)
            if len(cost_rows):
                observed_cost = matrix.c[cost_rows, column]
                raw_cost[state_row, column] = float(observed_cost.mean())
                median_cost[state_row, column] = float(np.median(observed_cost))

    shrunk_quality = np.full_like(raw_quality, np.nan)
    shrunk_cost = np.full_like(raw_cost, np.nan)
    state_position = {state: row for row, state in enumerate(states)}
    global_row = state_position[global_state_id]
    shrunk_quality[global_row] = raw_quality[global_row]
    shrunk_cost[global_row] = raw_cost[global_row]

    def depth(state: str) -> int:
        result = 0
        current = parents[state]
        seen = {state}
        while current is not None and current not in seen:
            result += 1
            seen.add(current)
            current = parents[current]
        return result

    for state in sorted(
        (candidate for candidate in states if candidate != global_state_id),
        key=lambda candidate: (depth(candidate), states.index(candidate)),
    ):
        row = state_position[state]
        parent = parents[state]
        parent_row = global_row if parent is None else state_position[parent]
        for column in range(expert_count):
            q_prior = shrunk_quality[parent_row, column]
            q_n = quality_n[row, column]
            if q_n:
                if np.isfinite(q_prior) and kappa > 0:
                    shrunk_quality[row, column] = (
                        successes[row, column] + float(kappa) * q_prior
                    ) / (q_n + float(kappa))
                else:
                    shrunk_quality[row, column] = raw_quality[row, column]
            else:
                shrunk_quality[row, column] = q_prior
            c_prior = shrunk_cost[parent_row, column]
            c_n = cost_n[row, column]
            if c_n:
                if np.isfinite(c_prior) and kappa > 0:
                    shrunk_cost[row, column] = (
                        raw_cost[row, column] * c_n + float(kappa) * c_prior
                    ) / (c_n + float(kappa))
                else:
                    shrunk_cost[row, column] = raw_cost[row, column]
            else:
                shrunk_cost[row, column] = c_prior

    return CapabilityProfileArtifact(
        method_id=str(method_id),
        environment_id=str(environment_id),
        subject_artifact_sha256=str(subject_artifact_sha256),
        assigner_artifact_sha256=str(assigner_artifact_sha256),
        label_artifact_sha256=str(label_artifact_sha256),
        profile_uids=uids,
        profile_uid_sha256=_uid_sha256(uids),
        deployed_assignments=resolved_assignments,
        assignment_sha256=canonical_sha256(
            {"uids": list(uids), "states": list(resolved_assignments)}
        ),
        state_ids=states,
        parent_by_state=parents,
        global_state_id=str(global_state_id),
        expert_ids=matrix.expert_ids,
        state_support=state_support,
        all_wrong_support=all_wrong_support,
        partially_missing_support=partial_support,
        quality_observed=quality_n,
        success_sum=successes,
        raw_quality_mean=raw_quality,
        shrunk_quality_mean=shrunk_quality,
        cost_observed=cost_n,
        raw_cost_mean=raw_cost,
        raw_cost_median=median_cost,
        shrunk_cost_mean=shrunk_cost,
        kappa=float(kappa),
        cost_normalizer=cost_normalizer,
        cost_unit=matrix.cost_unit,
    )
