"""UID-scope, label-store, routing, and final-test access guards."""

from __future__ import annotations

import contextvars
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .data_contracts import (
    DataContractError,
    EvaluationLabels,
    InputRecord,
    TrainingLabels,
    UnknownExpertError,
)


class LeakageGuardError(PermissionError):
    """Base class for a disallowed phase or UID access."""


class UIDScopeError(LeakageGuardError):
    """Raised when fit/transform/assign exceeds its registered UID universe."""


class LabelAccessError(LeakageGuardError):
    """Raised when labels are requested in a route/assignment context."""


class FinalEvaluationLocked(LabelAccessError):
    """Raised before a valid frozen-prediction receipt unlocks test labels."""


class Operation(str, Enum):
    FIT = "fit"
    TRAIN = "train"
    TRANSFORM = "transform"
    ASSIGN = "assign"
    DEV_EVALUATE = "dev_evaluate"
    FINAL_EVALUATE = "final_evaluate"


@dataclass(slots=True)
class UIDAccessAudit:
    """A mandatory allowed-UID capability with a trace of every check."""

    operation: Operation
    allowed_uids: frozenset[str]
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, operation: Operation | str, allowed_uids: Sequence[str]) -> "UIDAccessAudit":
        normalized = tuple(str(uid) for uid in allowed_uids)
        if not normalized or len(normalized) != len(set(normalized)):
            raise UIDScopeError("allowed_uid must be non-empty and unique")
        return cls(Operation(operation), frozenset(normalized))

    def require(self, requested_uids: Sequence[str], *, component: str) -> None:
        requested = tuple(str(uid) for uid in requested_uids)
        outside = sorted(set(requested).difference(self.allowed_uids))
        event = {
            "component": str(component),
            "operation": self.operation.value,
            "requested_count": len(requested),
            "outside_count": len(outside),
            "passed": not outside,
        }
        self.events.append(event)
        if outside:
            raise UIDScopeError(
                f"{component} requested UID outside allowed_uid scope: {outside[0]}"
            )


@dataclass(frozen=True, slots=True)
class TrainingLabelStore:
    labels: TrainingLabels

    def read(
        self, requested_uids: Sequence[str], *, audit: UIDAccessAudit
    ) -> TrainingLabels:
        if audit.operation not in {Operation.FIT, Operation.TRAIN}:
            raise LabelAccessError(f"training labels unavailable during {audit.operation.value}")
        audit.require(requested_uids, component="training_label_store")
        return TrainingLabels(matrix=self.labels.matrix.subset(requested_uids))


@dataclass(frozen=True, slots=True)
class EvaluationLabelStore:
    labels: EvaluationLabels

    def read(
        self, requested_uids: Sequence[str], *, audit: UIDAccessAudit
    ) -> EvaluationLabels:
        if self.labels.split == "dev":
            if audit.operation is not Operation.DEV_EVALUATE:
                raise LabelAccessError("dev labels are available only to the dev evaluator")
        elif audit.operation is not Operation.FINAL_EVALUATE:
            raise FinalEvaluationLocked("test labels require final-evaluate operation")
        elif _ACTIVE_FINAL_UNLOCK.get() is None:
            raise FinalEvaluationLocked("test labels require evaluate_final_unlock")
        audit.require(requested_uids, component=f"{self.labels.split}_evaluation_label_store")
        return EvaluationLabels(
            matrix=self.labels.matrix.subset(requested_uids), split=self.labels.split
        )


@dataclass(frozen=True, slots=True)
class RouteItem:
    uid: str
    inputs: InputRecord


def route_online(
    router: Callable[[Sequence[Mapping[str, Any]]], Sequence[str]],
    items: Sequence[RouteItem],
    *,
    model_ids: Sequence[str],
    audit: UIDAccessAudit,
) -> dict[str, str]:
    """Route an audited batch without passing UID/source/labels to the model."""

    if audit.operation is not Operation.ASSIGN:
        raise UIDScopeError("route_online requires an assign UID scope")
    ids = tuple(str(item.uid) for item in items)
    if len(ids) != len(set(ids)):
        raise UIDScopeError("route batch contains duplicate UIDs")
    audit.require(ids, component="route_online")
    pool = tuple(str(model) for model in model_ids)
    if not pool or len(pool) != len(set(pool)):
        raise DataContractError("frozen model pool must be non-empty and unique")
    # UID exists only in `ids`; router sees InputRecord's allowlisted payload.
    payloads = tuple(item.inputs.feature_payload() for item in items)
    predictions = tuple(str(model) for model in router(payloads))
    if len(predictions) != len(items):
        raise DataContractError("router prediction count does not match input count")
    unknown = [model for model in predictions if model not in pool]
    if unknown:
        raise UnknownExpertError(f"router selected expert outside frozen pool: {unknown[0]}")
    return dict(zip(ids, predictions))


@dataclass(frozen=True, slots=True)
class _FinalUnlock:
    receipt_path: str
    receipt_sha256: str
    prediction_sha256: str


_ACTIVE_FINAL_UNLOCK: contextvars.ContextVar[_FinalUnlock | None] = contextvars.ContextVar(
    "subject_routing_final_unlock", default=None
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def evaluate_final_unlock(receipt_path: str | Path) -> Iterator[Mapping[str, str]]:
    """Temporarily unlock test labels after validating a frozen-prediction receipt."""

    path = Path(receipt_path).resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalEvaluationLocked(f"invalid final-evaluation receipt: {path}") from exc
    if not isinstance(receipt, dict):
        raise FinalEvaluationLocked("final-evaluation receipt is missing required attestations")
    if receipt.get("kind") == "subject_routing_final_evaluation_unlock":
        required = {
            "target_split": "test",
            "predictions_frozen": True,
            "method_selection_complete": True,
            "approved": True,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise FinalEvaluationLocked(
                "final-evaluation receipt is missing required attestations"
            )
        prediction_sha256 = receipt.get("prediction_sha256")
    elif receipt.get("kind") == "subject_routing_v2_frozen_prediction_receipt":
        payload = dict(receipt)
        claimed = payload.pop("payload_sha256", None)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        required = {
            "status": "FROZEN_LABEL_FREE_ROUTES_BEFORE_E04_UNLOCK",
            "receipt_precedes_first_v2_test_outcome_access": True,
            "first_permitted_outcome_access_step": "E04",
            "labels_or_costs_used_in_prediction": False,
            "test_fit_performed": False,
            "postfreeze_candidate_change": False,
            "test_outcome_rows_opened": 0,
        }
        routes = receipt.get("routes")
        if (
            claimed != hashlib.sha256(encoded).hexdigest()
            or any(receipt.get(key) != value for key, value in required.items())
            or not isinstance(routes, Mapping)
            or routes.get("exists") is not True
        ):
            raise FinalEvaluationLocked(
                "V2 final-evaluation receipt is missing required attestations"
            )
        prediction_sha256 = routes.get("sha256")
    else:
        raise FinalEvaluationLocked("final-evaluation receipt has an unknown kind")
    if not isinstance(prediction_sha256, str) or len(prediction_sha256) != 64:
        raise FinalEvaluationLocked("final-evaluation receipt has no prediction SHA256")
    capability = _FinalUnlock(str(path), _sha256(path), prediction_sha256)
    token = _ACTIVE_FINAL_UNLOCK.set(capability)
    try:
        yield {
            "receipt_path": capability.receipt_path,
            "receipt_sha256": capability.receipt_sha256,
            "prediction_sha256": capability.prediction_sha256,
        }
    finally:
        _ACTIVE_FINAL_UNLOCK.reset(token)


LEGACY_AUTO_TEST_ENTRYPOINTS = frozenset(
    f"routers.{name}.train_and_eval"
    for name in ("cosinecls", "kmeans", "knn", "linear", "mlp", "ovr", "routerdc", "zooter")
)


def authorize_legacy_train_and_eval(module_name: str) -> Mapping[str, str]:
    """Guard, but do not import/run, a legacy entry point that auto-evaluates test.

    Subject-routing launchers must call this wrapper before invoking one of the
    listed legacy modules.  The main subject pipeline will use split-specific
    trainers/evaluators instead; this adapter exists to fail closed if a legacy
    module is accidentally selected.
    """

    name = str(module_name)
    if name not in LEGACY_AUTO_TEST_ENTRYPOINTS:
        raise LeakageGuardError(f"unregistered legacy train_and_eval module: {name}")
    capability = _ACTIVE_FINAL_UNLOCK.get()
    if capability is None:
        raise FinalEvaluationLocked(
            f"legacy entry point {name} auto-traverses test and is locked"
        )
    return {
        "module": name,
        "receipt_sha256": capability.receipt_sha256,
        "prediction_sha256": capability.prediction_sha256,
    }
