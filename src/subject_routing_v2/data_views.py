"""Leakage-safe input, outcome, grouping, and sparse-label views.

The classes in this module are intentionally small capability boundaries.  A
router receives an :class:`InputView`; UID/source/group bookkeeping lives in a
separate :class:`RoutingSidecar`; outcome values are available only through a
training view, an evaluator capability, or the budgeted :class:`RevealService`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class DataIsolationError(ValueError):
    """Base class for a V2 data-boundary violation."""


class InputContaminationError(DataIsolationError):
    """Raised when offline or post-outcome information enters InputView."""


class OutcomeAccessError(PermissionError):
    """Raised when outcomes are requested outside their scoped capability."""


class EvaluationLockedError(OutcomeAccessError):
    """Raised when an evaluation view has not been legitimately unlocked."""


class RevealBudgetError(OutcomeAccessError):
    """Raised when a sparse-label action exceeds the frozen label budget."""


class MissingObservationError(DataIsolationError):
    """Raised when code tries to consume an unobserved outcome cell."""


_FORBIDDEN_INPUT_TOKENS = frozenset(
    {
        "answer",
        "actualcost",
        "actualoutputlength",
        "actualoutputtokens",
        "c",
        "correct",
        "cost",
        "dataset",
        "expertanswer",
        "expertoutput",
        "generation",
        "gold",
        "label",
        "modelanswer",
        "modeloutput",
        "outputlength",
        "outputtokens",
        "quality",
        "response",
        "rowindex",
        "sampleid",
        "source",
        "sourceid",
        "split",
        "target",
        "uid",
        "y",
    }
)
_QUALITY_KINDS = frozenset({"binary_correctness", "bounded_quality"})
_EVALUATION_ROLES = frozenset(
    {
        "SOURCE_LODO_EVALUATION",
        "SOURCE_POSTSELECTION_DEVELOPMENT_EVALUATION",
        "LEGACY_EXPLORATORY_REPLAY",
        "NEW_UNEXPOSED_CONFIRMATION",
    }
)


def _field_token(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _check_feature_names(names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in names)
    if not normalized or len(normalized) != len(set(normalized)):
        raise InputContaminationError("feature names must be non-empty and unique")
    for name in normalized:
        token = _field_token(name)
        if token in _FORBIDDEN_INPUT_TOKENS or any(
            forbidden in token
            for forbidden in (
                "answer",
                "actualcost",
                "expertoutput",
                "goldlabel",
                "sourceid",
                "sampleid",
            )
        ):
            raise InputContaminationError(f"forbidden InputView feature: {name}")
    return normalized


def _immutable_array(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.flags.writeable = False
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class InputView:
    """A batch of pre-expert numeric inputs with no routing side-channel."""

    features: np.ndarray
    observed_mask: np.ndarray
    feature_names: tuple[str, ...]
    encoder_id: str
    encoder_payload_sha256: str
    modality: str

    @classmethod
    def create(
        cls,
        features: Any,
        *,
        feature_names: Sequence[str],
        encoder_id: str,
        encoder_payload_sha256: str,
        modality: str,
        observed_mask: Any | None = None,
    ) -> "InputView":
        names = _check_feature_names(feature_names)
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names) or matrix.shape[0] == 0:
            raise InputContaminationError(
                "InputView features must be a non-empty rows x feature_names matrix"
            )
        if observed_mask is None:
            mask = np.ones(matrix.shape, dtype=bool)
        else:
            mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != matrix.shape:
            raise InputContaminationError("InputView observed mask shape mismatch")
        if not np.isfinite(matrix[mask]).all():
            raise InputContaminationError("observed InputView features must be finite")
        if np.isfinite(matrix[~mask]).any():
            raise InputContaminationError("unobserved InputView features must be NaN")
        encoder = str(encoder_id)
        if not encoder or _field_token(encoder) in _FORBIDDEN_INPUT_TOKENS:
            raise InputContaminationError("invalid encoder identity")
        digest = str(encoder_payload_sha256)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise InputContaminationError("encoder payload must have a lowercase SHA256")
        mode = str(modality)
        if mode not in {"text", "vision", "multimodal", "metadata_public"}:
            raise InputContaminationError(f"unsupported input modality: {mode}")
        return cls(
            features=_immutable_array(matrix, dtype=np.float64),
            observed_mask=_immutable_array(mask, dtype=bool),
            feature_names=names,
            encoder_id=encoder,
            encoder_payload_sha256=digest,
            modality=mode,
        )

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    def model_payload(self) -> Mapping[str, Any]:
        """Return the entire payload visible to an assigner."""

        return MappingProxyType(
            {
                "features": self.features,
                "feature_observed_mask": self.observed_mask,
                "feature_names": self.feature_names,
                "encoder_id": self.encoder_id,
                "encoder_payload_sha256": self.encoder_payload_sha256,
                "modality": self.modality,
            }
        )

    def to_serializable(self) -> dict[str, Any]:
        """Serialize only allowlisted model inputs, never UID/source metadata."""

        return {
            "schema_version": 1,
            "kind": "subject_routing_v2_input_view",
            "features": self.features.tolist(),
            "feature_observed_mask": self.observed_mask.tolist(),
            "feature_names": list(self.feature_names),
            "encoder_id": self.encoder_id,
            "encoder_payload_sha256": self.encoder_payload_sha256,
            "modality": self.modality,
        }


@dataclass(frozen=True, slots=True)
class RoutingSidecar:
    """Offline alignment fields that are never part of InputView."""

    uids: tuple[str, ...]
    source_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    split_roles: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        uids: Sequence[str],
        source_ids: Sequence[str],
        group_ids: Sequence[str],
        split_roles: Sequence[str],
    ) -> "RoutingSidecar":
        columns = tuple(
            tuple(str(value) for value in column)
            for column in (uids, source_ids, group_ids, split_roles)
        )
        if not columns[0] or len(columns[0]) != len(set(columns[0])):
            raise DataIsolationError("sidecar UIDs must be non-empty and unique")
        if len({len(column) for column in columns}) != 1:
            raise DataIsolationError("sidecar columns must have equal length")
        return cls(*columns)

    def assert_aligned(self, view: InputView) -> None:
        if len(self.uids) != view.rows:
            raise DataIsolationError("sidecar and InputView row counts differ")

    def audit_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "rows": len(self.uids),
                "uid_sha256": _canonical_sha256(self.uids),
                "source_counts": dict(
                    sorted({source: self.source_ids.count(source) for source in set(self.source_ids)}.items())
                ),
                "group_count": len(set(self.group_ids)),
                "split_counts": dict(
                    sorted({role: self.split_roles.count(role) for role in set(self.split_roles)}.items())
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class OutcomeMatrix:
    """Immutable Y/C matrix with explicit observation semantics."""

    uids: tuple[str, ...]
    expert_ids: tuple[str, ...]
    quality: np.ndarray
    cost: np.ndarray
    quality_observed_mask: np.ndarray
    cost_observed_mask: np.ndarray
    quality_kind: str
    cost_unit: str

    @classmethod
    def create(
        cls,
        *,
        uids: Sequence[str],
        expert_ids: Sequence[str],
        quality: Any,
        cost: Any,
        quality_observed_mask: Any | None,
        cost_observed_mask: Any | None,
        quality_kind: str,
        cost_unit: str,
        complete_matrix: bool = False,
    ) -> "OutcomeMatrix":
        ids = tuple(str(value) for value in uids)
        experts = tuple(str(value) for value in expert_ids)
        if not ids or len(ids) != len(set(ids)):
            raise DataIsolationError("outcome UIDs must be non-empty and unique")
        if not experts or len(experts) != len(set(experts)):
            raise DataIsolationError("expert IDs must be non-empty and unique")
        y = np.asarray(quality, dtype=np.float64)
        c = np.asarray(cost, dtype=np.float64)
        shape = (len(ids), len(experts))
        if y.shape != shape or c.shape != shape:
            raise DataIsolationError("outcome matrix shape differs from UID x expert order")
        if quality_observed_mask is None or cost_observed_mask is None:
            if not complete_matrix:
                raise DataIsolationError(
                    "observation masks are mandatory unless complete_matrix is attested"
                )
            y_mask = np.ones(shape, dtype=bool)
            c_mask = np.ones(shape, dtype=bool)
        else:
            y_mask = np.asarray(quality_observed_mask, dtype=bool)
            c_mask = np.asarray(cost_observed_mask, dtype=bool)
        if y_mask.shape != shape or c_mask.shape != shape:
            raise DataIsolationError("outcome mask shape mismatch")
        if np.isfinite(y[~y_mask]).any() or np.isfinite(c[~c_mask]).any():
            raise DataIsolationError("unobserved cells must be NaN, never zero placeholders")
        if not np.isfinite(y[y_mask]).all():
            raise DataIsolationError("observed quality cells must be finite")
        if not np.isfinite(c[c_mask]).all() or (c[c_mask] < 0).any():
            raise DataIsolationError("observed costs must be finite and non-negative")
        kind = str(quality_kind)
        if kind not in _QUALITY_KINDS:
            raise DataIsolationError(f"unsupported quality semantics: {kind}")
        if kind == "binary_correctness" and not np.isin(y[y_mask], (0.0, 1.0)).all():
            raise DataIsolationError("binary correctness must contain only observed 0/1")
        if kind == "bounded_quality" and (
            (y[y_mask] < 0).any() or (y[y_mask] > 1).any()
        ):
            raise DataIsolationError("bounded quality must lie in [0, 1]")
        unit = str(cost_unit)
        if unit != "usd_proxy_per_request":
            raise DataIsolationError(f"ambiguous or unsupported cost unit: {unit}")
        return cls(
            uids=ids,
            expert_ids=experts,
            quality=_immutable_array(y, dtype=np.float64),
            cost=_immutable_array(c, dtype=np.float64),
            quality_observed_mask=_immutable_array(y_mask, dtype=bool),
            cost_observed_mask=_immutable_array(c_mask, dtype=bool),
            quality_kind=kind,
            cost_unit=unit,
        )

    def row_classes(self) -> Mapping[str, np.ndarray]:
        observed_any = self.quality_observed_mask.any(axis=1)
        all_missing = ~observed_any
        partially_missing = observed_any & ~self.quality_observed_mask.all(axis=1)
        all_wrong = observed_any & np.all(
            (~self.quality_observed_mask) | (self.quality == 0.0), axis=1
        )
        return MappingProxyType(
            {
                "all_wrong": _immutable_array(all_wrong, dtype=bool),
                "partially_missing": _immutable_array(partially_missing, dtype=bool),
                "all_missing": _immutable_array(all_missing, dtype=bool),
            }
        )

    def subset(self, requested_uids: Sequence[str]) -> "OutcomeMatrix":
        requested = tuple(str(value) for value in requested_uids)
        positions = {uid: index for index, uid in enumerate(self.uids)}
        unknown = [uid for uid in requested if uid not in positions]
        if unknown:
            raise OutcomeAccessError(f"UID is outside outcome view: {unknown[0]}")
        rows = [positions[uid] for uid in requested]
        return OutcomeMatrix.create(
            uids=requested,
            expert_ids=self.expert_ids,
            quality=self.quality[rows],
            cost=self.cost[rows],
            quality_observed_mask=self.quality_observed_mask[rows],
            cost_observed_mask=self.cost_observed_mask[rows],
            quality_kind=self.quality_kind,
            cost_unit=self.cost_unit,
        )

    def selected(
        self, predicted_experts: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = tuple(str(value) for value in predicted_experts)
        if len(predictions) != len(self.uids):
            raise DataIsolationError("prediction count differs from outcome rows")
        columns = {expert: index for index, expert in enumerate(self.expert_ids)}
        unknown = [expert for expert in predictions if expert not in columns]
        if unknown:
            raise DataIsolationError(f"predicted expert outside frozen pool: {unknown[0]}")
        rows = np.arange(len(self.uids), dtype=np.int64)
        cols = np.asarray([columns[expert] for expert in predictions], dtype=np.int64)
        if not self.quality_observed_mask[rows, cols].all():
            raise MissingObservationError("selected quality contains an unobserved cell")
        if not self.cost_observed_mask[rows, cols].all():
            raise MissingObservationError("selected cost contains an unobserved cell")
        return (
            _immutable_array(self.quality[rows, cols], dtype=np.float64),
            _immutable_array(self.cost[rows, cols], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class TrainingOutcomeView:
    """Outcome matrix that is legal only for an explicit source-fit UID scope."""

    matrix: OutcomeMatrix
    scope_id: str

    @classmethod
    def create(
        cls,
        matrix: OutcomeMatrix,
        *,
        allowed_source_uids: Sequence[str],
        forbidden_test_uids: Sequence[str],
        scope_id: str,
    ) -> "TrainingOutcomeView":
        allowed = set(str(value) for value in allowed_source_uids)
        test = set(str(value) for value in forbidden_test_uids)
        actual = set(matrix.uids)
        if not actual.issubset(allowed):
            raise OutcomeAccessError("training outcome escaped its source-fit UID scope")
        if actual.intersection(test):
            raise OutcomeAccessError("test UID entered TrainingOutcomeView")
        if not scope_id:
            raise DataIsolationError("training outcome scope_id is required")
        return cls(matrix=matrix, scope_id=str(scope_id))

    def subset(self, requested_uids: Sequence[str]) -> "TrainingOutcomeView":
        matrix = self.matrix.subset(requested_uids)
        return TrainingOutcomeView(matrix=matrix, scope_id=self.scope_id)


_ACCESS_SENTINEL = object()


class EvaluationAccess:
    """Unforgeable-by-public-constructor evaluator capability."""

    __slots__ = ("_allowed_uids", "_role", "_sentinel", "receipt_sha256")

    def __init__(
        self,
        sentinel: object,
        *,
        role: str,
        allowed_uids: Iterable[str],
        receipt_sha256: str | None,
    ) -> None:
        if sentinel is not _ACCESS_SENTINEL:
            raise EvaluationLockedError("EvaluationAccess must be created by a gate")
        self._sentinel = sentinel
        self._role = str(role)
        self._allowed_uids = frozenset(str(value) for value in allowed_uids)
        self.receipt_sha256 = receipt_sha256

    def require(self, *, role: str, uids: Sequence[str]) -> None:
        if self._sentinel is not _ACCESS_SENTINEL or self._role != role:
            raise EvaluationLockedError("evaluation capability role mismatch")
        outside = set(str(value) for value in uids).difference(self._allowed_uids)
        if outside:
            raise EvaluationLockedError(
                f"evaluation capability does not cover UID: {sorted(outside)[0]}"
            )


def source_evaluation_access(
    *, heldout_uids: Sequence[str], fold_id: str
) -> EvaluationAccess:
    """Create a source-LODO evaluator capability (never a test unlock)."""

    if not str(fold_id).startswith("source_lodo_"):
        raise EvaluationLockedError("source evaluation requires a registered LODO fold")
    return EvaluationAccess(
        _ACCESS_SENTINEL,
        role="SOURCE_LODO_EVALUATION",
        allowed_uids=heldout_uids,
        receipt_sha256=None,
    )


def source_development_evaluation_access(
    receipt_path: str | Path,
    *,
    frozen_bundle_path: str | Path,
    ordered_uids: Sequence[str],
) -> EvaluationAccess:
    """Unlock an explicitly post-selection source check after bundle freeze.

    This gate is intentionally distinct from source LODO and new-test gates:
    it can support decomposition/development evidence only.  The receipt binds
    the exact frozen runtime bundle and ordered evaluation UID scope.
    """

    receipt_file = Path(receipt_path)
    bundle_file = Path(frozen_bundle_path)
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationLockedError("invalid source-development receipt") from exc
    payload = dict(receipt) if isinstance(receipt, dict) else {}
    claimed = payload.pop("payload_sha256", None)
    required = {
        "kind": "subject_routing_v2_source_development_evaluation_unlock",
        "evaluation_role": "SOURCE_POSTSELECTION_DEVELOPMENT_EVALUATION",
        "predictions_or_bundles_frozen": True,
        "approved": True,
    }
    if (
        not payload
        or claimed != _canonical_sha256(payload)
        or any(payload.get(key) != value for key, value in required.items())
    ):
        raise EvaluationLockedError(
            "source-development receipt is missing freeze attestations"
        )
    ids = tuple(map(str, ordered_uids))
    if not ids or len(ids) != len(set(ids)):
        raise EvaluationLockedError("source-development UID scope is invalid")
    if not bundle_file.is_file() or payload.get("bundle_sha256") != _sha256_file(
        bundle_file
    ):
        raise EvaluationLockedError("source-development bundle hash mismatch")
    if payload.get("ordered_uid_sha256") != _canonical_sha256(list(ids)):
        raise EvaluationLockedError("source-development UID hash mismatch")
    return EvaluationAccess(
        _ACCESS_SENTINEL,
        role="SOURCE_POSTSELECTION_DEVELOPMENT_EVALUATION",
        allowed_uids=ids,
        receipt_sha256=_sha256_file(receipt_file),
    )


def legacy_evaluation_access(
    receipt_path: str | Path, *, prediction_path: str | Path
) -> EvaluationAccess:
    """Unlock an exposed test only after an E03 frozen-prediction receipt."""

    receipt_file = Path(receipt_path)
    prediction_file = Path(prediction_path)
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationLockedError("invalid legacy evaluation receipt") from exc
    required = {
        "kind": "subject_routing_v2_evaluation_unlock",
        "evaluation_role": "LEGACY_EXPLORATORY_REPLAY",
        "predictions_frozen": True,
        "method_selection_complete": True,
        "approved": True,
    }
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in required.items()):
        raise EvaluationLockedError("legacy evaluation receipt is missing attestations")
    if not prediction_file.is_file():
        raise EvaluationLockedError("frozen prediction file is missing")
    if receipt.get("prediction_sha256") != _sha256_file(prediction_file):
        raise EvaluationLockedError("frozen prediction hash mismatch")
    uids = receipt.get("ordered_uids")
    if not isinstance(uids, list) or not uids:
        raise EvaluationLockedError("legacy receipt has no ordered UID scope")
    return EvaluationAccess(
        _ACCESS_SENTINEL,
        role="LEGACY_EXPLORATORY_REPLAY",
        allowed_uids=uids,
        receipt_sha256=_sha256_file(receipt_file),
    )


class EvaluationOutcomeView:
    """Outcome values that can only be consumed by a matching evaluator."""

    __slots__ = ("__matrix", "role")

    def __init__(self, matrix: OutcomeMatrix, *, role: str) -> None:
        normalized = str(role)
        if normalized not in _EVALUATION_ROLES:
            raise DataIsolationError(f"unknown evaluation role: {normalized}")
        self.__matrix = matrix
        self.role = normalized

    @property
    def uids(self) -> tuple[str, ...]:
        return self.__matrix.uids

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return self.__matrix.expert_ids

    def selected_outcomes(
        self,
        predicted_experts: Sequence[str],
        *,
        access: EvaluationAccess | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if access is None:
            raise EvaluationLockedError("evaluation outcomes are locked")
        access.require(role=self.role, uids=self.__matrix.uids)
        return self.__matrix.selected(predicted_experts)

    def diagnostic_matrix(self, *, access: EvaluationAccess | None) -> OutcomeMatrix:
        """Expose a read-only matrix only inside a matching diagnostic evaluator."""

        if access is None:
            raise EvaluationLockedError("evaluation outcomes are locked")
        access.require(role=self.role, uids=self.__matrix.uids)
        return self.__matrix


@dataclass(frozen=True, slots=True)
class RevealedCell:
    uid: str
    expert_id: str
    quality: float
    cost: float


class RevealService:
    """The only label access path for sparse-label experiments."""

    __slots__ = (
        "__cost",
        "__cost_mask",
        "__quality",
        "__quality_mask",
        "_allowed_cells",
        "_expert_position",
        "_label_budget_cells",
        "_ledger",
        "_revealed",
        "_uid_position",
    )

    def __init__(
        self,
        matrix: OutcomeMatrix,
        *,
        label_budget_cells: int,
        allowed_uids: Sequence[str],
    ) -> None:
        budget = int(label_budget_cells)
        if budget < 0:
            raise RevealBudgetError("label budget must be non-negative")
        allowed = set(str(value) for value in allowed_uids)
        if not set(matrix.uids).issubset(allowed):
            raise OutcomeAccessError("hidden sparse labels escaped allowed source UID scope")
        self.__quality = matrix.quality
        self.__cost = matrix.cost
        self.__quality_mask = matrix.quality_observed_mask
        self.__cost_mask = matrix.cost_observed_mask
        self._uid_position = {uid: row for row, uid in enumerate(matrix.uids)}
        self._expert_position = {
            expert: column for column, expert in enumerate(matrix.expert_ids)
        }
        self._allowed_cells = frozenset(
            (uid, expert)
            for uid in matrix.uids
            for expert in matrix.expert_ids
        )
        self._label_budget_cells = budget
        self._revealed: dict[tuple[str, str], RevealedCell] = {}
        self._ledger: list[dict[str, Any]] = []

    @property
    def label_budget_cells(self) -> int:
        return self._label_budget_cells

    @property
    def spent_cells(self) -> int:
        return len(self._revealed)

    @property
    def remaining_cells(self) -> int:
        return self._label_budget_cells - self.spent_cells

    @property
    def revealed_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._revealed)

    def remaining_candidates(
        self, candidates: Iterable[tuple[str, str]]
    ) -> tuple[tuple[str, str], ...]:
        result: list[tuple[str, str]] = []
        for raw_uid, raw_expert in candidates:
            key = (str(raw_uid), str(raw_expert))
            if key not in self._allowed_cells or key in self._revealed:
                continue
            row = self._uid_position[key[0]]
            column = self._expert_position[key[1]]
            if self.__quality_mask[row, column] and self.__cost_mask[row, column]:
                result.append(key)
        return tuple(result)

    def reveal(
        self, requests: Iterable[tuple[str, str]], *, reason: str
    ) -> tuple[RevealedCell, ...]:
        normalized = tuple((str(uid), str(expert)) for uid, expert in requests)
        if not str(reason).strip():
            raise OutcomeAccessError("every reveal requires a recorded reason")
        if len(normalized) != len(set(normalized)):
            raise OutcomeAccessError("a reveal action contains duplicate cells")
        unknown = [key for key in normalized if key not in self._allowed_cells]
        if unknown:
            raise OutcomeAccessError(f"reveal cell outside hidden source store: {unknown[0]}")
        newly_requested = [key for key in normalized if key not in self._revealed]
        if self.spent_cells + len(newly_requested) > self._label_budget_cells:
            raise RevealBudgetError("reveal action exceeds the frozen label budget")
        output: list[RevealedCell] = []
        for key in normalized:
            if key not in self._revealed:
                row = self._uid_position[key[0]]
                column = self._expert_position[key[1]]
                if not (
                    self.__quality_mask[row, column]
                    and self.__cost_mask[row, column]
                ):
                    raise MissingObservationError(f"cannot reveal missing cell: {key}")
                self._revealed[key] = RevealedCell(
                    uid=key[0],
                    expert_id=key[1],
                    quality=float(self.__quality[row, column]),
                    cost=float(self.__cost[row, column]),
                )
            output.append(self._revealed[key])
        self._ledger.append(
            {
                "action_index": len(self._ledger),
                "reason": str(reason),
                "requested_cells": len(normalized),
                "new_cells": len(newly_requested),
                "spent_after": self.spent_cells,
                "request_sha256": _canonical_sha256(normalized),
            }
        )
        return tuple(output)

    def revealed_snapshot(self) -> Mapping[str, Any]:
        """Return only already revealed values and a label-budget ledger."""

        return MappingProxyType(
            {
                "label_budget_cells": self.label_budget_cells,
                "spent_cells": self.spent_cells,
                "remaining_cells": self.remaining_cells,
                "revealed": tuple(self._revealed[key] for key in sorted(self._revealed)),
                "ledger": tuple(MappingProxyType(dict(row)) for row in self._ledger),
            }
        )


def select_public_priority_requests(
    *,
    service: RevealService,
    candidates: Sequence[tuple[str, str]],
    public_priorities: Sequence[float],
    count: int,
) -> tuple[tuple[str, str], ...]:
    """Select sparse queries using public scores and revealed-key state only."""

    if len(candidates) != len(public_priorities):
        raise DataIsolationError("candidate and public-priority lengths differ")
    requested = int(count)
    if requested < 0:
        raise DataIsolationError("requested query count must be non-negative")
    available = set(service.remaining_candidates(candidates))
    scored = [
        ((str(uid), str(expert)), float(priority))
        for (uid, expert), priority in zip(candidates, public_priorities, strict=True)
        if (str(uid), str(expert)) in available
    ]
    if not all(np.isfinite(priority) for _, priority in scored):
        raise DataIsolationError("public acquisition priority must be finite")
    scored.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    return tuple(key for key, _ in scored[:requested])


def binarize_quality(
    matrix: OutcomeMatrix,
    *,
    threshold: float,
    reason: str,
) -> tuple[OutcomeMatrix, dict[str, Any]]:
    """Explicitly convert bounded utility to binary targets with an audit record."""

    if matrix.quality_kind != "bounded_quality":
        raise DataIsolationError("only bounded_quality may be explicitly binarized")
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise DataIsolationError("binarization threshold must lie in [0, 1]")
    if not str(reason).strip():
        raise DataIsolationError("binarization requires a scientific reason")
    quality = np.full(matrix.quality.shape, np.nan, dtype=np.float64)
    quality[matrix.quality_observed_mask] = (
        matrix.quality[matrix.quality_observed_mask] >= value
    ).astype(np.float64)
    converted = OutcomeMatrix.create(
        uids=matrix.uids,
        expert_ids=matrix.expert_ids,
        quality=quality,
        cost=matrix.cost,
        quality_observed_mask=matrix.quality_observed_mask,
        cost_observed_mask=matrix.cost_observed_mask,
        quality_kind="binary_correctness",
        cost_unit=matrix.cost_unit,
    )
    return converted, {
        "conversion": "bounded_quality_to_binary_correctness",
        "threshold": value,
        "reason": str(reason),
        "missing_cells_preserved": True,
    }


@dataclass(frozen=True, slots=True)
class PublicGroupRecord:
    uid: str
    source_id: str
    structural_asset_digest: str
    exact_prompt_sha256: str


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while value != root:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def build_dependency_groups(
    records: Sequence[PublicGroupRecord],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Union exact-prompt and structural-asset duplicates into atomic groups."""

    ids = tuple(record.uid for record in records)
    if not ids or len(ids) != len(set(ids)):
        raise DataIsolationError("group records must contain unique non-empty UIDs")
    union = _UnionFind(ids)
    first: dict[tuple[str, str], str] = {}
    for record in records:
        if not record.source_id:
            raise DataIsolationError("source_id is required only in the grouping sidecar")
        for kind, digest in (
            ("asset", record.structural_asset_digest),
            ("prompt", record.exact_prompt_sha256),
        ):
            if len(str(digest)) != 64:
                raise DataIsolationError(f"invalid public {kind} digest")
            key = (kind, str(digest))
            prior = first.setdefault(key, record.uid)
            union.union(prior, record.uid)
    by_root: dict[str, list[str]] = {}
    for uid in ids:
        by_root.setdefault(union.find(uid), []).append(uid)
    groups: dict[str, tuple[str, ...]] = {}
    uid_to_group: dict[str, str] = {}
    for members in by_root.values():
        ordered = tuple(sorted(members))
        group_id = f"grp-{_canonical_sha256(ordered)[:24]}"
        groups[group_id] = ordered
        for uid in ordered:
            uid_to_group[uid] = group_id
    return uid_to_group, dict(sorted(groups.items()))


@dataclass(frozen=True, slots=True)
class FoldFitScope:
    """A group-disjoint fit/evaluation boundary for one source LODO fold."""

    fold_id: str
    fit_uids: frozenset[str]
    selection_uids: frozenset[str]
    evaluation_uids: frozenset[str]
    uid_to_group: Mapping[str, str]

    @classmethod
    def create(
        cls,
        *,
        fold_id: str,
        fit_uids: Sequence[str],
        selection_uids: Sequence[str],
        evaluation_uids: Sequence[str],
        uid_to_group: Mapping[str, str],
    ) -> "FoldFitScope":
        fit = frozenset(str(value) for value in fit_uids)
        selection = frozenset(str(value) for value in selection_uids)
        evaluation = frozenset(str(value) for value in evaluation_uids)
        if not fit or not evaluation:
            raise DataIsolationError("fold requires non-empty fit and evaluation scopes")
        if fit & selection or fit & evaluation or selection & evaluation:
            raise DataIsolationError("fold UID roles overlap")
        all_uids = fit | selection | evaluation
        missing = all_uids.difference(uid_to_group)
        if missing:
            raise DataIsolationError(f"fold UID lacks group_id: {sorted(missing)[0]}")
        fit_groups = {uid_to_group[uid] for uid in fit}
        selection_groups = {uid_to_group[uid] for uid in selection}
        evaluation_groups = {uid_to_group[uid] for uid in evaluation}
        if (
            fit_groups.intersection(selection_groups)
            or fit_groups.intersection(evaluation_groups)
            or selection_groups.intersection(evaluation_groups)
        ):
            raise DataIsolationError("dependency group crosses fit/evaluation boundary")
        return cls(
            fold_id=str(fold_id),
            fit_uids=fit,
            selection_uids=selection,
            evaluation_uids=evaluation,
            uid_to_group=MappingProxyType(dict(uid_to_group)),
        )

    def require_fit_uids(
        self, requested_uids: Sequence[str], *, component: str
    ) -> None:
        """Require normalization/PCA/teacher/profile inputs to be fold-local fit."""

        requested = set(str(value) for value in requested_uids)
        outside = requested.difference(self.fit_uids)
        if outside:
            raise OutcomeAccessError(
                f"{component} requested UID outside fold-local fit: {sorted(outside)[0]}"
            )

    def require_selection_uids(
        self, requested_uids: Sequence[str], *, component: str
    ) -> None:
        """Require model-selection access to stay in the fold-local dev scope."""

        requested = set(str(value) for value in requested_uids)
        outside = requested.difference(self.selection_uids)
        if outside:
            raise OutcomeAccessError(
                f"{component} requested UID outside fold-local selection: {sorted(outside)[0]}"
            )


def purge_group_overlap(
    candidate_fit_uids: Sequence[str],
    evaluation_uids: Sequence[str],
    *,
    uid_to_group: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Remove candidate fit rows sharing an atomic dependency group with eval."""

    evaluation = tuple(str(value) for value in evaluation_uids)
    missing = set(evaluation).difference(uid_to_group)
    if missing:
        raise DataIsolationError(f"evaluation UID lacks group_id: {sorted(missing)[0]}")
    blocked_groups = {uid_to_group[uid] for uid in evaluation}
    kept: list[str] = []
    purged: list[str] = []
    for raw_uid in candidate_fit_uids:
        uid = str(raw_uid)
        if uid not in uid_to_group:
            raise DataIsolationError(f"candidate fit UID lacks group_id: {uid}")
        if uid_to_group[uid] in blocked_groups:
            purged.append(uid)
        else:
            kept.append(uid)
    return tuple(kept), tuple(purged)
