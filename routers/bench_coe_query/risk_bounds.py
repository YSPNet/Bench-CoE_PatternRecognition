"""Exact one-sided harm-risk bounds for a pre-frozen downgrade policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from scipy.stats import beta


@dataclass(frozen=True)
class RiskCertificate:
    candidate_id: str
    harms: int
    samples: int
    epsilon: float
    delta_local: float
    upper_bound: float | None
    passed: bool
    status: str


def clopper_pearson_upper(harms: int, samples: int, delta: float) -> float | None:
    """One-sided exact upper confidence bound for a Bernoulli mean."""

    if not isinstance(harms, int) or not isinstance(samples, int):
        raise TypeError("harms and samples must be integers")
    if samples < 0 or harms < 0 or harms > samples:
        raise ValueError("require 0 <= harms <= samples")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if samples == 0:
        return None
    if harms == samples:
        return 1.0
    if harms == 0:
        return float(-math.expm1(math.log(delta) / samples))
    return float(beta.ppf(1.0 - delta, harms + 1, samples - harms))


def certify_candidate(
    candidate_id: str,
    harm_values: Iterable[int | bool],
    *,
    epsilon: float,
    delta_local: float,
) -> RiskCertificate:
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0, 1]")
    values = tuple(int(value) for value in harm_values)
    if any(value not in (0, 1) for value in values):
        raise ValueError("harm values must be Bernoulli 0/1")
    upper = clopper_pearson_upper(sum(values), len(values), delta_local)
    passed = upper is not None and upper <= epsilon
    return RiskCertificate(
        candidate_id=str(candidate_id),
        harms=sum(values),
        samples=len(values),
        epsilon=float(epsilon),
        delta_local=float(delta_local),
        upper_bound=upper,
        passed=bool(passed),
        status="PASSED" if passed else ("NO_DATA" if upper is None else "FAILED"),
    )


def certify_bonferroni_family(
    harms_by_candidate: Mapping[str, Iterable[int | bool]],
    *,
    epsilon: float,
    delta: float,
) -> dict[str, RiskCertificate]:
    if not harms_by_candidate:
        raise ValueError("candidate family must not be empty")
    local = float(delta) / len(harms_by_candidate)
    return {
        str(name): certify_candidate(
            str(name), values, epsilon=epsilon, delta_local=local
        )
        for name, values in harms_by_candidate.items()
    }


def zero_harm_sample_requirement(epsilon: float, delta_local: float) -> int | None:
    """Smallest n for which zero observed harms yields U <= epsilon."""

    if not math.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must lie in [0, 1)")
    if not math.isfinite(delta_local) or not 0 < delta_local < 1:
        raise ValueError("delta_local must lie strictly between zero and one")
    if epsilon == 0:
        return None
    return int(math.ceil(math.log(delta_local) / math.log1p(-epsilon)))

