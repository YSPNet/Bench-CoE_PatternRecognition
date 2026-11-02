"""Strict online-input and split-scoped label contracts.

``InputRecord`` has no sample ID, dataset, split, path, answer, output, or
target field.  Bookkeeping IDs live in the leakage guard and are removed
before a router callable receives its payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


class DataContractError(ValueError):
    """Base class for a fail-closed subject-routing contract violation."""


class InputContaminationError(DataContractError):
    """Raised when offline-only metadata enters an online input record."""


class MissingObservationError(DataContractError):
    """Raised when a selected outcome cell was not observed."""


class UnknownExpertError(DataContractError):
    """Raised when a route names an expert outside the frozen model pool."""


class UnexplainedCostError(DataContractError):
    """Raised when an unexplained zero cost would enter a primary result."""


_ONLINE_FIELDS = frozenset({"question", "options", "images"})
_FORBIDDEN_FIELD_TOKENS = frozenset(
    {
        "answer",
        "actualoutputtokens",
        "c",
        "cost",
        "dataset",
        "expertanswer",
        "expertoutput",
        "gold",
        "label",
        "modelid",
        "outputtokens",
        "path",
        "quality",
        "rowindex",
        "sampleid",
        "source",
        "split",
        "target",
        "uid",
        "uri",
        "y",
    }
)


def _field_token(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _reject_nested_forbidden_keys(value: Any, location: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            token = _field_token(key)
            if token in _FORBIDDEN_FIELD_TOKENS:
                raise InputContaminationError(f"forbidden online input field at {location}: {key}")
            _reject_nested_forbidden_keys(nested, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_nested_forbidden_keys(nested, f"{location}[{index}]")


@dataclass(frozen=True, slots=True)
class InputRecord:
    """Only information available before an expert is selected."""

    question: str
    options: tuple[str, ...]
    images: tuple[bytes, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InputRecord":
        if not isinstance(value, Mapping):
            raise InputContaminationError("online input must be a mapping")
        _reject_nested_forbidden_keys(value)
        fields = set(value)
        unknown = fields.difference(_ONLINE_FIELDS)
        if unknown:
            raise InputContaminationError(f"unsupported online input fields: {sorted(unknown)}")
        question = value.get("question")
        if not isinstance(question, str) or not question.strip():
            raise InputContaminationError("question must be a non-empty string")
        raw_options = value.get("options", ())
        if not isinstance(raw_options, (list, tuple)) or any(
            not isinstance(option, str) for option in raw_options
        ):
            raise InputContaminationError("options must be a sequence of strings")
        raw_images = value.get("images", ())
        if not isinstance(raw_images, (list, tuple)) or any(
            not isinstance(image, (bytes, bytearray, memoryview)) for image in raw_images
        ):
            raise InputContaminationError(
                "images must be in-memory bytes; filesystem paths/URIs are forbidden features"
            )
        return cls(
            question=question,
            options=tuple(raw_options),
            images=tuple(bytes(image) for image in raw_images),
        )

    def feature_payload(self) -> Mapping[str, Any]:
        """Return the complete and only payload a routing model may consume."""

        return MappingProxyType(
            {"question": self.question, "options": self.options, "images": self.images}
        )


def _immutable_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class _LabelMatrix:
    uids: tuple[str, ...]
    expert_ids: tuple[str, ...]
    y: np.ndarray
    c: np.ndarray
    observed_mask: np.ndarray
    cost_observed_mask: np.ndarray
    quality_kind: str
    cost_unit: str
    zero_cost_semantics: str

    @classmethod
    def create(
        cls,
        *,
        uids: Sequence[str],
        expert_ids: Sequence[str],
        y: Any,
        c: Any,
        observed_mask: Any | None,
        cost_observed_mask: Any | None = None,
        complete_matrix: bool = False,
        quality_kind: str = "binary_correctness",
        cost_unit: str = "usd_proxy_per_request",
        zero_cost_semantics: str = "unexplained_placeholder",
    ) -> "_LabelMatrix":
        uid_order = tuple(str(item) for item in uids)
        expert_order = tuple(str(item) for item in expert_ids)
        if not uid_order or len(uid_order) != len(set(uid_order)):
            raise DataContractError("label UIDs must be non-empty and unique")
        if not expert_order or len(expert_order) != len(set(expert_order)):
            raise DataContractError("expert IDs must be non-empty and unique")
        quality = np.asarray(y, dtype=np.float64)
        cost = np.asarray(c, dtype=np.float64)
        expected = (len(uid_order), len(expert_order))
        if quality.shape != expected or cost.shape != expected:
            raise DataContractError(
                f"Y/C shape must equal UID x expert order {expected}; got {quality.shape}/{cost.shape}"
            )
        if observed_mask is None:
            if not complete_matrix:
                raise DataContractError(
                    "observed_mask is required unless complete_matrix=True is explicitly attested"
                )
            y_observed = np.ones(expected, dtype=bool)
        else:
            y_observed = np.asarray(observed_mask, dtype=bool)
        c_observed = (
            y_observed.copy()
            if cost_observed_mask is None
            else np.asarray(cost_observed_mask, dtype=bool)
        )
        if y_observed.shape != expected or c_observed.shape != expected:
            raise DataContractError("observation masks must have the same shape as Y/C")

        if np.isfinite(quality[~y_observed]).any():
            raise DataContractError("missing Y cells must be NaN, never a 0/wrong placeholder")
        if np.isfinite(cost[~c_observed]).any():
            raise DataContractError("missing C cells must be NaN, never a 0-cost placeholder")
        if not np.isfinite(quality[y_observed]).all():
            raise DataContractError("observed Y cells must be finite")
        if not np.isfinite(cost[c_observed]).all() or (cost[c_observed] < 0).any():
            raise DataContractError("observed C cells must be finite and non-negative")
        if quality_kind == "binary_correctness":
            if not np.isin(quality[y_observed], (0.0, 1.0)).all():
                raise DataContractError("binary correctness Y must contain only observed 0/1 values")
        elif quality_kind == "bounded_quality":
            if ((quality[y_observed] < 0) | (quality[y_observed] > 1)).any():
                raise DataContractError("bounded quality Y must lie in [0, 1]")
        else:
            raise DataContractError(f"unsupported quality kind: {quality_kind}")
        if cost_unit != "usd_proxy_per_request":
            raise DataContractError(f"unsupported or ambiguous cost unit: {cost_unit}")

        return cls(
            uids=uid_order,
            expert_ids=expert_order,
            y=_immutable_array(quality),
            c=_immutable_array(cost),
            observed_mask=_immutable_array(y_observed, dtype=bool),
            cost_observed_mask=_immutable_array(c_observed, dtype=bool),
            quality_kind=quality_kind,
            cost_unit=cost_unit,
            zero_cost_semantics=zero_cost_semantics,
        )

    def row_classes(self) -> Mapping[str, np.ndarray]:
        all_missing = ~self.observed_mask.any(axis=1)
        partially_missing = self.observed_mask.any(axis=1) & ~self.observed_mask.all(axis=1)
        all_wrong = self.observed_mask.all(axis=1) & np.all(self.y == 0, axis=1)
        return MappingProxyType(
            {
                "all_wrong": _immutable_array(all_wrong, dtype=bool),
                "partially_missing": _immutable_array(partially_missing, dtype=bool),
                "all_missing": _immutable_array(all_missing, dtype=bool),
            }
        )

    def subset(self, requested_uids: Sequence[str]) -> "_LabelMatrix":
        requested = tuple(str(item) for item in requested_uids)
        position = {uid: row for row, uid in enumerate(self.uids)}
        missing = [uid for uid in requested if uid not in position]
        if missing:
            raise DataContractError(f"label store does not contain requested UID: {missing[0]}")
        rows = [position[uid] for uid in requested]
        return type(self).create(
            uids=requested,
            expert_ids=self.expert_ids,
            y=self.y[rows],
            c=self.c[rows],
            observed_mask=self.observed_mask[rows],
            cost_observed_mask=self.cost_observed_mask[rows],
            quality_kind=self.quality_kind,
            cost_unit=self.cost_unit,
            zero_cost_semantics=self.zero_cost_semantics,
        )

    def reorder_experts(self, target_order: Sequence[str]) -> "_LabelMatrix":
        target = tuple(str(item) for item in target_order)
        if len(target) != len(set(target)) or set(target) != set(self.expert_ids):
            raise DataContractError("target expert order is not a permutation of the frozen pool")
        positions = {expert: column for column, expert in enumerate(self.expert_ids)}
        columns = [positions[expert] for expert in target]
        return type(self).create(
            uids=self.uids,
            expert_ids=target,
            y=self.y[:, columns],
            c=self.c[:, columns],
            observed_mask=self.observed_mask[:, columns],
            cost_observed_mask=self.cost_observed_mask[:, columns],
            quality_kind=self.quality_kind,
            cost_unit=self.cost_unit,
            zero_cost_semantics=self.zero_cost_semantics,
        )

    def selected_outcomes(
        self, predicted_experts: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = tuple(str(item) for item in predicted_experts)
        if len(predictions) != len(self.uids):
            raise DataContractError("prediction count does not match label rows")
        model_to_column = {expert: column for column, expert in enumerate(self.expert_ids)}
        unknown = [expert for expert in predictions if expert not in model_to_column]
        if unknown:
            raise UnknownExpertError(f"selected expert is outside frozen pool: {unknown[0]}")
        rows = np.arange(len(self.uids), dtype=np.int64)
        columns = np.asarray([model_to_column[expert] for expert in predictions], dtype=np.int64)
        if not self.observed_mask[rows, columns].all():
            raise MissingObservationError("selected expert has an unobserved quality cell")
        if not self.cost_observed_mask[rows, columns].all():
            raise MissingObservationError("selected expert has an unobserved cost cell")
        selected_cost = self.c[rows, columns]
        if (selected_cost == 0).any() and self.zero_cost_semantics != "legitimate_zero":
            raise UnexplainedCostError(
                "selected expert has zero cost with no legitimate-zero attestation"
            )
        return (
            _immutable_array(self.y[rows, columns]),
            _immutable_array(selected_cost),
        )


@dataclass(frozen=True, slots=True)
class TrainingLabels:
    """Labels that may only be opened through a train/fit UID scope."""

    matrix: _LabelMatrix
    split: str = "train"

    @classmethod
    def create(cls, **values: Any) -> "TrainingLabels":
        split = str(values.pop("split", "train"))
        if split != "train":
            raise DataContractError("TrainingLabels must be scoped to train")
        return cls(matrix=_LabelMatrix.create(**values))


@dataclass(frozen=True, slots=True)
class EvaluationLabels:
    """Dev/test labels that require evaluator-only access contexts."""

    matrix: _LabelMatrix
    split: str

    @classmethod
    def create(cls, *, split: str, **values: Any) -> "EvaluationLabels":
        if split not in {"dev", "test"}:
            raise DataContractError("EvaluationLabels split must be dev or test")
        return cls(matrix=_LabelMatrix.create(**values), split=split)
