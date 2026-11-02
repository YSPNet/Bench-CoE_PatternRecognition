"""Dimension-safe L2 normalization and multimodal concatenation."""

from __future__ import annotations

import math

import numpy as np


class FeatureValidationError(ValueError):
    """Raised for invalid, zero, or misaligned feature matrices."""


def l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise FeatureValidationError("features must be a finite 2-D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise FeatureValidationError("cannot L2-normalize a zero feature row")
    result = np.asarray(values / norms, dtype=np.float32)
    result.flags.writeable = False
    return result


def weighted_l2_concat(
    text_features: np.ndarray,
    vision_features: np.ndarray,
    *,
    text_weight: float,
) -> np.ndarray:
    """Fuse different vector spaces without adding their coordinates."""

    weight = float(text_weight)
    if not 0.0 <= weight <= 1.0:
        raise FeatureValidationError("text_weight must lie in [0, 1]")
    text = l2_normalize(text_features)
    vision = l2_normalize(vision_features)
    if text.shape[0] != vision.shape[0]:
        raise FeatureValidationError("text and vision row counts differ")
    result = np.concatenate(
        (math.sqrt(weight) * text, math.sqrt(1.0 - weight) * vision), axis=1
    ).astype(np.float32, copy=False)
    # The two blocks are orthogonal by construction, including w=0 and w=1.
    expected_norm = np.ones(result.shape[0], dtype=np.float32)
    if not np.allclose(np.linalg.norm(result, axis=1), expected_norm, atol=1e-6):
        raise FeatureValidationError("weighted concatenation did not preserve unit norm")
    result.flags.writeable = False
    return result
