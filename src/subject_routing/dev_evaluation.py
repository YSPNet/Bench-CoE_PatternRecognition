"""Frozen-prediction development evaluation for subject-routing candidates."""

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

from routers.utils.rank_score import rank_score

from .artifacts import ArtifactIntegrityError, canonical_sha256
from .data_contracts import EvaluationLabels
from .features.fusion import l2_normalize
from .leakage_guard import (
    EvaluationLabelStore,
    Operation,
    UIDAccessAudit,
    UIDScopeError,
)


PREDICTION_KIND = "subject_routing_frozen_dev_prediction_bundle"
PROTOTYPE_KIND = "subject_routing_official_expert_prototype_deploy"
_FORBIDDEN_PREDICTION_KEYS = frozenset(
    {
        "actual_cost",
        "answer",
        "answers",
        "c",
        "gold",
        "labels",
        "outcomes",
        "quality",
        "targets",
        "test_predictions",
        "y",
    }
)


def _uid_sha256(uids: Sequence[str]) -> str:
    raw = json.dumps(
        [str(uid) for uid in uids], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA256")
    return digest


def _atomic_envelope(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
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


def _read_envelope(path: Path, *, expected_kind: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != expected_kind
        or value.get("payload_sha256") != canonical_sha256(payload)
    ):
        raise ArtifactIntegrityError(f"frozen artifact digest/type mismatch: {path}")
    return payload


@dataclass(frozen=True, slots=True)
class ScopedDevLabelProvenance:
    requested_uid_sha256: str
    requested_rows: int
    expert_ids: tuple[str, ...]
    quality_files: tuple[dict[str, Any], ...]
    cost_matrix_path: str
    cost_matrix_sha256: str
    prediction_file_sha256: str
    prediction_receipt_sha256: str
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
            "cost_access": "read_only_memmap_explicit_dev_rows_only",
            "prediction_file_sha256": self.prediction_file_sha256,
            "prediction_receipt_sha256": self.prediction_receipt_sha256,
            "predictions_loaded_and_hash_verified_before_label_access": True,
            "global_y_npz_opened": self.global_y_npz_opened,
            "test_uid_requested": self.test_uid_requested,
        }


def materialize_scoped_dev_labels(
    requested_uids: Sequence[str],
    *,
    allowed_dev_uids: Sequence[str],
    expert_ids: Sequence[str],
    oracle_score_root: str | Path,
    cost_matrix_path: str | Path,
    sample_registry_path: str | Path,
    model_registry_path: str | Path,
    expected_cost_sha256: str,
    expected_sample_registry_sha256: str,
    expected_model_registry_sha256: str,
    expected_quality_sha256: Mapping[str, str],
    verified_prediction_file_sha256: str,
    verified_prediction_receipt_sha256: str,
    audit: UIDAccessAudit,
) -> tuple[EvaluationLabels, ScopedDevLabelProvenance]:
    """Read only explicit dev rows after a frozen prediction receipt is verified."""

    if audit.operation is not Operation.DEV_EVALUATE:
        raise UIDScopeError("dev materialization requires dev_evaluate operation")
    ids = tuple(str(uid) for uid in requested_uids)
    allowed = frozenset(str(uid) for uid in allowed_dev_uids)
    if not ids or len(ids) != len(set(ids)):
        raise ArtifactIntegrityError("dev UIDs must be non-empty and unique")
    outside = sorted(set(ids).difference(allowed))
    if outside:
        raise UIDScopeError(
            f"dev label request is outside allowed dev UIDs: {outside[0]}"
        )
    audit.require(ids, component="scoped_dev_label_materializer")
    prediction_sha = _require_sha256(
        verified_prediction_file_sha256, name="prediction file hash"
    )
    receipt_sha = _require_sha256(
        verified_prediction_receipt_sha256, name="prediction receipt hash"
    )

    sample_path = Path(sample_registry_path)
    model_path = Path(model_registry_path)
    cost_path = Path(cost_matrix_path)
    for path, expected, name in (
        (sample_path, expected_sample_registry_sha256, "sample registry"),
        (model_path, expected_model_registry_sha256, "model registry"),
        (cost_path, expected_cost_sha256, "cost matrix"),
    ):
        if not path.is_file() or _file_sha256(path) != expected:
            raise ArtifactIntegrityError(f"{name} is missing or hash-mismatched")
    samples = tuple(str(uid) for uid in pickle.loads(sample_path.read_bytes()))
    models = tuple(str(model) for model in pickle.loads(model_path.read_bytes()))
    canonical_experts = tuple(str(expert) for expert in expert_ids)
    if models != canonical_experts:
        raise ArtifactIntegrityError(
            "model registry differs from canonical expert order"
        )
    row_by_uid = {uid: row for row, uid in enumerate(samples)}
    if any(uid not in row_by_uid for uid in ids):
        raise ArtifactIntegrityError(
            "dev UID is absent from the frozen sample registry"
        )

    import pyarrow.parquet as pq

    quality = np.full((len(ids), len(models)), np.nan, dtype=np.float64)
    output_row = {uid: row for row, uid in enumerate(ids)}
    output_column = {model: column for column, model in enumerate(models)}
    uids_by_source: dict[str, list[str]] = {}
    for uid in ids:
        uids_by_source.setdefault(uid.split("/", 1)[0], []).append(uid)
    quality_records: list[dict[str, Any]] = []
    root = Path(oracle_score_root)
    for source, source_uids in sorted(uids_by_source.items()):
        score_path = root / f"{source.casefold()}.parquet"
        expected_hash = expected_quality_sha256.get(source)
        if expected_hash is None:
            raise ArtifactIntegrityError(
                f"no frozen quality provenance for source: {source}"
            )
        if not score_path.is_file() or _file_sha256(score_path) != expected_hash:
            raise ArtifactIntegrityError(f"dev quality file changed: {source}")
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
                raise ArtifactIntegrityError(
                    "predicate-filtered dev quality escaped scope"
                )
            row = output_row[uid]
            column = output_column[model]
            if np.isfinite(quality[row, column]):
                raise ArtifactIntegrityError("duplicate dev quality cell")
            quality[row, column] = float(record["quality"])
        quality_records.append(
            {
                "source": source,
                "path": str(score_path),
                "sha256": expected_hash,
                "requested_samples": len(source_uids),
                "returned_cells": len(records),
                "read_mode": "predicate_filtered_requested_dev_sample_ids_only",
            }
        )
    if not np.isfinite(quality).all() or not np.isin(quality, (0.0, 1.0)).all():
        raise ArtifactIntegrityError(
            "scoped dev quality matrix is incomplete or non-binary"
        )

    cost_source = np.load(cost_path, mmap_mode="r", allow_pickle=False)
    selected_rows = np.asarray([row_by_uid[uid] for uid in ids], dtype=np.int64)
    cost = np.asarray(cost_source[selected_rows], dtype=np.float64)
    if cost.shape != quality.shape or not np.isfinite(cost).all() or (cost <= 0).any():
        raise ArtifactIntegrityError("scoped dev cost matrix is invalid")
    labels = EvaluationLabels.create(
        split="dev",
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
    store = EvaluationLabelStore(labels)
    scoped = store.read(ids, audit=audit)
    return scoped, ScopedDevLabelProvenance(
        requested_uid_sha256=_uid_sha256(ids),
        requested_rows=len(ids),
        expert_ids=models,
        quality_files=tuple(quality_records),
        cost_matrix_path=str(cost_path),
        cost_matrix_sha256=expected_cost_sha256,
        prediction_file_sha256=prediction_sha,
        prediction_receipt_sha256=receipt_sha,
    )


@dataclass(frozen=True, slots=True)
class OfficialPrototypeArtifact:
    feature_artifact_sha256: str
    supervision_provenance_sha256: str
    fit_uid_sha256: str
    expert_ids: tuple[str, ...]
    feature_dimension: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    prototypes: np.ndarray
    source_sha256: str
    fit_diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, digest in (
            ("feature", self.feature_artifact_sha256),
            ("supervision provenance", self.supervision_provenance_sha256),
            ("fit UID", self.fit_uid_sha256),
            ("source", self.source_sha256),
        ):
            _require_sha256(digest, name=name)
        if not self.expert_ids or len(self.expert_ids) != len(set(self.expert_ids)):
            raise ArtifactIntegrityError("official prototype expert order is invalid")
        mean = np.asarray(self.scaler_mean, dtype=np.float64)
        scale = np.asarray(self.scaler_scale, dtype=np.float64)
        prototypes = np.asarray(self.prototypes, dtype=np.float64)
        if (
            self.feature_dimension <= 0
            or mean.shape != (self.feature_dimension,)
            or scale.shape != (self.feature_dimension,)
            or prototypes.shape != (len(self.expert_ids), self.feature_dimension)
            or (scale <= 0).any()
            or not all(np.isfinite(value).all() for value in (mean, scale, prototypes))
        ):
            raise ArtifactIntegrityError(
                "official prototype parameter shape/value mismatch"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": PROTOTYPE_KIND,
            "fidelity": "faithful_to_hash_pinned_repository_algorithm",
            "routing_mode": "query_level",
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "supervision_provenance_sha256": self.supervision_provenance_sha256,
            "fit_uid_sha256": self.fit_uid_sha256,
            "expert_ids": list(self.expert_ids),
            "feature_dimension": self.feature_dimension,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "prototypes": self.prototypes.tolist(),
            "source_sha256": self.source_sha256,
            "fit_diagnostics": dict(self.fit_diagnostics),
            "online_inputs": "input_features_only",
            "lambda_cost_participates_in_prediction": False,
            "semantic_subject_partition": False,
            "raw_supervision_or_outcome_matrix_in_deploy_artifact": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> None:
        _atomic_envelope(Path(path), self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OfficialPrototypeArtifact":
        return cls(
            feature_artifact_sha256=str(value["feature_artifact_sha256"]),
            supervision_provenance_sha256=str(value["supervision_provenance_sha256"]),
            fit_uid_sha256=str(value["fit_uid_sha256"]),
            expert_ids=tuple(str(expert) for expert in value["expert_ids"]),
            feature_dimension=int(value["feature_dimension"]),
            scaler_mean=np.asarray(value["scaler_mean"], dtype=np.float64),
            scaler_scale=np.asarray(value["scaler_scale"], dtype=np.float64),
            prototypes=np.asarray(value["prototypes"], dtype=np.float64),
            source_sha256=str(value["source_sha256"]),
            fit_diagnostics=dict(value["fit_diagnostics"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OfficialPrototypeArtifact":
        return cls.from_dict(_read_envelope(Path(path), expected_kind=PROTOTYPE_KIND))

    def predict(
        self, features: np.ndarray, *, feature_artifact_sha256: str
    ) -> tuple[str, ...]:
        if str(feature_artifact_sha256) != self.feature_artifact_sha256:
            raise ArtifactIntegrityError(
                "official prototype runtime feature hash mismatch"
            )
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_dimension:
            raise ArtifactIntegrityError(
                "official prototype feature dimension mismatch"
            )
        transformed = (values - self.scaler_mean) / self.scaler_scale
        transformed = l2_normalize(transformed)
        scores = transformed @ np.asarray(self.prototypes, dtype=np.float64).T
        selected = scores.argmax(axis=1)
        return tuple(self.expert_ids[int(index)] for index in selected)


def _walk_prediction_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_PREDICTION_KEYS:
                raise ArtifactIntegrityError(f"outcome key in prediction bundle: {key}")
            _walk_prediction_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _walk_prediction_keys(nested)


def validate_prediction_bundle(payload: Mapping[str, Any]) -> None:
    if payload.get("kind") != PREDICTION_KIND:
        raise ArtifactIntegrityError("unexpected prediction bundle kind")
    _walk_prediction_keys(payload)
    split_records = payload.get("splits")
    if not isinstance(split_records, dict) or set(split_records) != {
        "profile_train",
        "dev",
    }:
        raise ArtifactIntegrityError(
            "prediction bundle must contain profile_train and dev only"
        )
    expert_ids = tuple(str(expert) for expert in payload.get("expert_ids", ()))
    if not expert_ids or len(expert_ids) != len(set(expert_ids)):
        raise ArtifactIntegrityError("prediction bundle expert order is invalid")
    split_uids: dict[str, tuple[str, ...]] = {}
    for split_name, record in split_records.items():
        if (
            "test" in split_name.casefold()
            or "test" in str(record.get("role", "")).casefold()
        ):
            raise ArtifactIntegrityError(
                "test split cannot enter a dev prediction bundle"
            )
        uids = tuple(str(uid) for uid in record.get("uids", ()))
        if not uids or len(uids) != len(set(uids)):
            raise ArtifactIntegrityError("prediction split UID order is invalid")
        if record.get("uid_sha256") != _uid_sha256(uids):
            raise ArtifactIntegrityError("prediction split UID hash mismatch")
        split_uids[split_name] = uids
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ArtifactIntegrityError("prediction bundle contains no methods")
    method_ids: set[str] = set()
    pool = set(expert_ids)
    for method in methods:
        method_id = str(method.get("method_id", ""))
        if not method_id or method_id in method_ids:
            raise ArtifactIntegrityError("prediction method IDs must be unique")
        method_ids.add(method_id)
        routing_mode = str(method.get("routing_mode"))
        if routing_mode not in {
            "strict_subject",
            "hybrid_task_query",
            "query_level",
        }:
            raise ArtifactIntegrityError("unsupported prediction routing mode")
        predictions = method.get("predictions")
        if not isinstance(predictions, dict) or set(predictions) != set(split_uids):
            raise ArtifactIntegrityError("method prediction splits are incomplete")
        for split_name, prediction in predictions.items():
            states_raw = prediction.get("states")
            states = (
                None
                if states_raw is None
                else tuple(str(state) for state in states_raw)
            )
            if routing_mode in {"strict_subject", "hybrid_task_query"} and (
                states is None or len(states) != len(split_uids[split_name])
            ):
                raise ArtifactIntegrityError(
                    "strict/hybrid task method requires one state per row"
                )
            if routing_mode == "query_level" and states is not None:
                raise ArtifactIntegrityError(
                    "query-level comparator cannot masquerade as states"
                )
            routes = prediction.get("routes")
            if not isinstance(routes, list) or not routes:
                raise ArtifactIntegrityError("method prediction has no routes")
            setting_ids: set[tuple[str, str]] = set()
            for route in routes:
                setting = (str(route.get("strategy_id")), str(route.get("budget_id")))
                if setting in setting_ids:
                    raise ArtifactIntegrityError("duplicate prediction strategy/budget")
                setting_ids.add(setting)
                experts = tuple(
                    str(expert) for expert in route.get("predicted_experts", ())
                )
                if (
                    len(experts) != len(split_uids[split_name])
                    or not set(experts) <= pool
                ):
                    raise ArtifactIntegrityError(
                        "prediction expert rows/order are invalid"
                    )
                cap = float(route.get("budget_cap"))
                if not np.isfinite(cap) or cap < 0:
                    raise ArtifactIntegrityError("prediction budget cap is invalid")
                if routing_mode == "strict_subject":
                    state_expert: dict[str, str] = {}
                    for state, expert in zip(states, experts):
                        previous = state_expert.setdefault(state, expert)
                        if previous != expert:
                            raise ArtifactIntegrityError(
                                "strict prediction changes expert within one state/budget"
                            )


def freeze_prediction_bundle(path: str | Path, payload: Mapping[str, Any]) -> str:
    validate_prediction_bundle(payload)
    _atomic_envelope(Path(path), payload)
    return canonical_sha256(payload)


def load_prediction_bundle(path: str | Path) -> dict[str, Any]:
    payload = _read_envelope(Path(path), expected_kind=PREDICTION_KIND)
    validate_prediction_bundle(payload)
    return payload


def freeze_dev_prediction_receipt(
    path: str | Path,
    *,
    prediction_path: str | Path,
    config_sha256: str,
    target_access_attempt_count: int,
) -> dict[str, Any]:
    target = Path(prediction_path)
    bundle = load_prediction_bundle(target)
    if target_access_attempt_count != 0:
        raise ArtifactIntegrityError("prediction producer attempted target access")
    payload = {
        "schema_version": 1,
        "kind": "subject_routing_dev_prediction_freeze_receipt",
        "target_role": "dev_seen",
        "predictions_frozen": True,
        "prediction_path": str(target),
        "prediction_file_sha256": _file_sha256(target),
        "prediction_payload_sha256": canonical_sha256(bundle),
        "config_sha256": _require_sha256(config_sha256, name="B08 config hash"),
        "producer_target_matrix_access_attempts": 0,
        "dev_outcomes_accessed_by_producer": False,
        "test_inputs_or_outcomes_accessed": False,
    }
    receipt_path = Path(path)
    if receipt_path.exists():
        raise FileExistsError(
            f"refusing to overwrite dev prediction receipt: {receipt_path}"
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["payload_sha256"] = canonical_sha256(payload)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", dir=receipt_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value


def verify_dev_prediction_receipt(
    receipt_path: str | Path, *, expected_config_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(receipt_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = value.get("payload_sha256")
    payload = dict(value)
    payload.pop("payload_sha256", None)
    required = {
        "kind": "subject_routing_dev_prediction_freeze_receipt",
        "target_role": "dev_seen",
        "predictions_frozen": True,
        "config_sha256": expected_config_sha256,
        "producer_target_matrix_access_attempts": 0,
        "dev_outcomes_accessed_by_producer": False,
        "test_inputs_or_outcomes_accessed": False,
    }
    if recorded != canonical_sha256(payload) or any(
        payload.get(key) != expected for key, expected in required.items()
    ):
        raise ArtifactIntegrityError("dev prediction receipt is invalid")
    prediction_path = Path(str(payload["prediction_path"]))
    if _file_sha256(prediction_path) != payload["prediction_file_sha256"]:
        raise ArtifactIntegrityError("frozen dev prediction file hash changed")
    bundle = load_prediction_bundle(prediction_path)
    if canonical_sha256(bundle) != payload["prediction_payload_sha256"]:
        raise ArtifactIntegrityError("frozen dev prediction payload hash changed")
    return bundle, value


def evaluate_prediction_split(
    bundle: Mapping[str, Any],
    *,
    split_name: str,
    labels: Any,
    dataset_by_uid: Mapping[str, str],
    prediction_file_sha256: str,
    cost_min: float,
    cost_max: float,
) -> list[dict[str, Any]]:
    """Evaluate a frozen split; callers control train/dev label capability."""

    validate_prediction_bundle(bundle)
    split = bundle["splits"][split_name]
    uids = tuple(str(uid) for uid in split["uids"])
    matrix = labels.matrix
    if matrix.uids != uids or matrix.expert_ids != tuple(bundle["expert_ids"]):
        raise ArtifactIntegrityError(
            "evaluation labels differ from frozen prediction order"
        )
    datasets = tuple(str(dataset_by_uid[uid]) for uid in uids)
    dataset_order = tuple(dict.fromkeys(datasets))
    row_classes = matrix.row_classes()
    output = []
    for method in bundle["methods"]:
        prediction = method["predictions"][split_name]
        states = prediction["states"]
        fallback_count = (
            0
            if states is None
            else sum(state == method["fallback_state_id"] for state in states)
        )
        for route in prediction["routes"]:
            experts = tuple(str(expert) for expert in route["predicted_experts"])
            quality, cost = matrix.selected_outcomes(experts)
            per_dataset = []
            for dataset in dataset_order:
                rows = np.asarray(
                    [index for index, value in enumerate(datasets) if value == dataset],
                    dtype=np.int64,
                )
                per_dataset.append(
                    {
                        "dataset": dataset,
                        "rows": int(len(rows)),
                        "accuracy": float(quality[rows].mean()),
                        "average_cost_usd_proxy": float(cost[rows].mean()),
                    }
                )
            micro_accuracy = float(quality.mean())
            average_cost = float(cost.mean())
            macro_accuracy = float(
                np.mean([record["accuracy"] for record in per_dataset])
            )
            macro_cost = float(
                np.mean([record["average_cost_usd_proxy"] for record in per_dataset])
            )
            cap = float(route["budget_cap"])
            latency = float(
                method["routing_overhead"][split_name]["latency_ms_per_row"]
            )
            output.append(
                {
                    "schema_version": 1,
                    "kind": "subject_routing_baseline_split_metric",
                    "environment": bundle["environment"],
                    "split_name": split_name,
                    "split_role": split["role"],
                    "rows": len(uids),
                    "uid_sha256": split["uid_sha256"],
                    "method_id": method["method_id"],
                    "routing_mode": method["routing_mode"],
                    "strategy_id": route["strategy_id"],
                    "budget_id": route["budget_id"],
                    "budget_cap_usd_proxy": cap,
                    "routed_accuracy_micro": micro_accuracy,
                    "routed_accuracy_macro_dataset": macro_accuracy,
                    "average_expert_cost_usd_proxy": average_cost,
                    "average_expert_cost_macro_dataset": macro_cost,
                    "dev_average_cost_constraint_met": bool(
                        average_cost <= cap + 1e-15
                    ),
                    "per_request_cost_above_train_cap_rate": float(np.mean(cost > cap)),
                    "router_latency_ms_per_row": latency,
                    "router_gpu_seconds": float(
                        method["routing_overhead"][split_name]["gpu_seconds"]
                    ),
                    "effective_K": int(method["effective_K"]),
                    "observed_assigned_states": (
                        None
                        if states is None
                        else len(set(states) - {method["fallback_state_id"]})
                    ),
                    "minimum_train_support": method["minimum_train_support"],
                    "fallback_rows": int(fallback_count),
                    "fallback_rate": float(fallback_count / len(uids)),
                    "all_wrong_rows_retained": int(row_classes["all_wrong"].sum()),
                    "row_drop_count": 0,
                    "per_dataset": per_dataset,
                    "official_full_split_metrics": {
                        "status": "NOT_COMPARABLE_SMOKE_SUBSET",
                        "Avg.Acc": None,
                        "Avg.Cost": None,
                        "RankScore": None,
                    },
                    "smoke_subset_analogs": {
                        "Avg.Acc_macro_dataset": macro_accuracy,
                        "Avg.Cost_macro_dataset": macro_cost,
                        "RankScore_micro": float(
                            rank_score(
                                micro_accuracy,
                                average_cost,
                                float(cost_min),
                                float(cost_max),
                                beta=0.1,
                            )
                        ),
                    },
                    "artifacts": dict(method["artifacts"]),
                    "prediction_file_sha256": prediction_file_sha256,
                    "prediction_route_sha256": canonical_sha256(
                        {
                            "method_id": method["method_id"],
                            "split_name": split_name,
                            "strategy_id": route["strategy_id"],
                            "budget_id": route["budget_id"],
                            "predicted_experts": list(experts),
                        }
                    ),
                    "evaluated_on": "dev" if split_name == "dev" else "train_profile",
                    "selected_on": None,
                    "test_accessed": False,
                }
            )
    return output


def select_dev_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    eligible_method_order: Sequence[str],
    strategy_order: Sequence[str],
) -> list[dict[str, Any]]:
    """Select one feasible smoke candidate per budget using frozen dev tie rules."""

    methods = tuple(str(method) for method in eligible_method_order)
    strategies = tuple(str(strategy) for strategy in strategy_order)
    method_position = {method: index for index, method in enumerate(methods)}
    strategy_position = {strategy: index for index, strategy in enumerate(strategies)}
    dev_records = [
        dict(record)
        for record in records
        if record["split_name"] == "dev" and record["method_id"] in method_position
    ]
    budgets = tuple(dict.fromkeys(record["budget_id"] for record in dev_records))
    selections = []
    for budget in budgets:
        feasible = [
            record
            for record in dev_records
            if record["budget_id"] == budget
            and record["dev_average_cost_constraint_met"]
        ]
        if not feasible:
            selections.append(
                {
                    "budget_id": budget,
                    "status": "FAILED_NO_CANDIDATE_MEETS_DEV_AVERAGE_COST_CAP",
                    "selected_on": "dev",
                    "selection_role": "dev_seen",
                    "test_feedback_used": False,
                }
            )
            continue
        selected = min(
            feasible,
            key=lambda record: (
                -float(record["routed_accuracy_micro"]),
                float(record["average_expert_cost_usd_proxy"]),
                float(record["router_latency_ms_per_row"]),
                int(record["effective_K"]),
                method_position[record["method_id"]],
                strategy_position[record["strategy_id"]],
            ),
        )
        selections.append(
            {
                "budget_id": budget,
                "status": "SELECTED_WITHIN_B08_DEV_SMOKE_ONLY",
                "method_id": selected["method_id"],
                "strategy_id": selected["strategy_id"],
                "routed_accuracy_micro": selected["routed_accuracy_micro"],
                "average_expert_cost_usd_proxy": selected[
                    "average_expert_cost_usd_proxy"
                ],
                "router_latency_ms_per_row": selected["router_latency_ms_per_row"],
                "effective_K": selected["effective_K"],
                "artifacts": selected["artifacts"],
                "prediction_route_sha256": selected["prediction_route_sha256"],
                "selected_on": "dev",
                "selection_role": "dev_seen",
                "selection_scope": "first_256_rows_single_seed_smoke_not_final_candidate",
                "test_feedback_used": False,
            }
        )
    return selections
