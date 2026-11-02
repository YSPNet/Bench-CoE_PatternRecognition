"""Paired, sample-ID-aware statistical utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairedBootstrapResult:
    observed_delta: float
    ci_low: float
    ci_high: float
    standard_error: float
    resamples: int
    seed: int


def _validate_paired(
    sample_ids: Sequence[str],
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sample_ids) == 0:
        raise ValueError("paired statistics require at least one sample")
    if len(set(str(item) for item in sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique for paired statistics")
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != (len(sample_ids),) or right_values.shape != (len(sample_ids),):
        raise ValueError("paired value vectors must match sample ID length")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("paired values must be finite")
    return left_values, right_values


def paired_bootstrap_delta(
    sample_ids: Sequence[str],
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedBootstrapResult:
    left_values, right_values = _validate_paired(sample_ids, left, right)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    differences = left_values - right_values
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    # Draw paired positions, never independently resample the two systems.
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        samples[start:stop] = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return PairedBootstrapResult(
        observed_delta=float(differences.mean()),
        ci_low=float(np.quantile(samples, alpha / 2.0)),
        ci_high=float(np.quantile(samples, 1.0 - alpha / 2.0)),
        standard_error=float(samples.std(ddof=1)) if resamples > 1 else 0.0,
        resamples=resamples,
        seed=seed,
    )


def mcnemar_exact(
    sample_ids: Sequence[str],
    left_correct: Sequence[int] | np.ndarray,
    right_correct: Sequence[int] | np.ndarray,
) -> dict[str, float | int]:
    left, right = _validate_paired(sample_ids, left_correct, right_correct)
    if not np.isin(left, (0, 1)).all() or not np.isin(right, (0, 1)).all():
        raise ValueError("McNemar inputs must be binary")
    left_only = int(((left == 1) & (right == 0)).sum())
    right_only = int(((left == 0) & (right == 1)).sum())
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        # Evaluate the exact Binomial(n, 0.5) tail in log space. Directly
        # forming 2**n and converting it to float overflows for full test sets.
        upper = min(left_only, right_only)
        log_terms = [
            math.lgamma(discordant + 1)
            - math.lgamma(value + 1)
            - math.lgamma(discordant - value + 1)
            - discordant * math.log(2.0)
            for value in range(upper + 1)
        ]
        maximum = max(log_terms)
        tail_probability = math.exp(maximum) * sum(math.exp(item - maximum) for item in log_terms)
        p_value = min(1.0, 2.0 * tail_probability)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value_two_sided_exact": float(p_value),
    }


def paired_permutation_mean_delta(
    sample_ids: Sequence[str],
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    permutations: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    left_values, right_values = _validate_paired(sample_ids, left, right)
    differences = left_values - right_values
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, permutations, 256):
        stop = min(start + 256, permutations)
        signs = rng.choice(
            np.asarray((-1.0, 1.0)),
            size=(stop - start, len(differences)),
        )
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int((permuted >= observed).sum())
    return {
        "observed_delta": float(differences.mean()),
        "permutations": permutations,
        "seed": seed,
        "p_value_two_sided": float((extreme + 1) / (permutations + 1)),
    }
