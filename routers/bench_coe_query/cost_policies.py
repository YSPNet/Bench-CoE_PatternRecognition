"""Target-free cost/quality decisions for the three cost-reduction routes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "y",
        "Y",
        "c",
        "C",
        "gold",
        "gold_answer",
        "answer",
        "expert_answers",
        "expert_responses",
        "correctness",
        "actual_cost",
        "actual_output_tokens",
        "actual_latency",
    }
)


@dataclass(frozen=True)
class InferenceRequest:
    """Minimal generation-time view; it intentionally contains no targets."""

    question: str
    images: tuple[str, ...]
    available_experts: tuple[str, ...]
    sample_id: str | None = None
    dataset: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InferenceRequest":
        forbidden = FORBIDDEN_INFERENCE_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(f"inference request contains forbidden target fields: {sorted(forbidden)}")
        question = value.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        images = value.get("images", ())
        experts = value.get("available_experts", ())
        if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
            raise ValueError("images must be a sequence")
        if isinstance(experts, (str, bytes)) or not isinstance(experts, Sequence):
            raise ValueError("available_experts must be a sequence")
        return cls(
            question=question,
            images=tuple(str(item) for item in images),
            available_experts=tuple(str(item) for item in experts),
            sample_id=None if value.get("sample_id") is None else str(value["sample_id"]),
            dataset=None if value.get("dataset") is None else str(value["dataset"]),
        )


def _validated_predictions(
    quality: np.ndarray,
    decision_cost: np.ndarray,
    availability: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(quality, dtype=np.float64)
    c = np.asarray(decision_cost, dtype=np.float64)
    if q.ndim != 2 or c.shape != q.shape:
        raise ValueError(f"quality/cost shape mismatch: {q.shape}/{c.shape}")
    if q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("policy batch and expert dimensions must be non-empty")
    if not np.isfinite(q).all():
        raise ValueError("quality predictions must be finite")
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("decision costs must be finite and non-negative")
    allowed = np.ones(q.shape, dtype=bool) if availability is None else np.asarray(availability, dtype=bool)
    if allowed.shape != q.shape:
        raise ValueError("availability shape mismatch")
    if not allowed.any(axis=1).all():
        raise ValueError("every request must have at least one available expert")
    return q, c, allowed


def quality_only(
    quality: np.ndarray,
    decision_cost: np.ndarray | None = None,
    *,
    availability: np.ndarray | None = None,
) -> np.ndarray:
    q = np.asarray(quality, dtype=np.float64)
    costs = np.zeros_like(q) if decision_cost is None else decision_cost
    q, _c, allowed = _validated_predictions(q, costs, availability)
    masked = np.where(allowed, q, -np.inf)
    return np.argmax(masked, axis=1).astype(np.int64)


def utility_policy(
    quality: np.ndarray,
    decision_cost: np.ndarray,
    *,
    lambda_cost: float,
    kappa: float,
    incremental_router_cost: float | np.ndarray = 0.0,
    availability: np.ndarray | None = None,
) -> np.ndarray:
    if not np.isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be finite and non-negative")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    q, c, allowed = _validated_predictions(quality, decision_cost, availability)
    extra = np.asarray(incremental_router_cost, dtype=np.float64)
    if not np.isfinite(extra).all() or (extra < 0).any():
        raise ValueError("incremental router cost must be finite and non-negative")
    # A request-constant extra cost does not alter argmax, but keeping it in the
    # explicit full-cost expression prevents it disappearing from cost audits.
    score = q - float(lambda_cost) * (c + extra) / float(kappa)
    score = np.where(allowed, score, -np.inf)
    return np.argmax(score, axis=1).astype(np.int64)


def predicted_gap_policy(
    quality: np.ndarray,
    decision_cost: np.ndarray,
    reference: np.ndarray,
    *,
    max_quality_gap: float,
    minimum_saving: float = 0.0,
    incremental_router_cost: float = 0.0,
    availability: np.ndarray | None = None,
) -> np.ndarray:
    """Choose the cheapest eligible predicted-safe downgrade or return pi0."""

    if not np.isfinite(max_quality_gap) or max_quality_gap < 0:
        raise ValueError("max_quality_gap must be finite and non-negative")
    if not np.isfinite(minimum_saving) or minimum_saving < 0:
        raise ValueError("minimum_saving must be finite and non-negative")
    if not np.isfinite(incremental_router_cost) or incremental_router_cost < 0:
        raise ValueError("incremental_router_cost must be finite and non-negative")
    q, c, allowed = _validated_predictions(quality, decision_cost, availability)
    baseline = np.asarray(reference, dtype=np.int64)
    if baseline.shape != (q.shape[0],) or (baseline < 0).any() or (baseline >= q.shape[1]).any():
        raise ValueError("reference indices are out of range")
    if not allowed[np.arange(len(baseline)), baseline].all():
        raise ValueError("reference expert unavailable; this is a new deployment condition")

    result = baseline.copy()
    for row, base in enumerate(baseline):
        reference_quality = q[row, base]
        reference_cost = c[row, base]
        saving = reference_cost - c[row] - float(incremental_router_cost)
        eligible = (
            allowed[row]
            & (q[row] >= reference_quality - float(max_quality_gap))
            & (saving >= float(minimum_saving))
            & (c[row] < reference_cost)
        )
        candidates = np.flatnonzero(eligible)
        if len(candidates):
            # Lowest predicted cost, then highest quality, then canonical index.
            order = np.lexsort((candidates, -q[row, candidates], c[row, candidates]))
            result[row] = int(candidates[order[0]])
    return result


def compatibility_digest(payload: Mapping[str, Any]) -> str:
    """Stable digest for the components whose change invalidates a policy/certificate."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def assert_compatible(expected_digest: str, payload: Mapping[str, Any]) -> None:
    observed = compatibility_digest(payload)
    if observed != expected_digest:
        raise ValueError(
            f"deployment components do not match the frozen policy: expected={expected_digest}, observed={observed}"
        )
