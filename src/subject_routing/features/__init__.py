"""Reproducible public-input feature utilities."""

from .cache import (
    ExistingFeatureCache,
    FeatureBatch,
    FeatureCacheSpec,
    input_content_sha256,
)
from .fusion import l2_normalize, weighted_l2_concat
from .inputs import FeatureInputFailure, validate_modality_inputs
from .qwen3_vl_embedding import Qwen3VLEmbeddingAdapter
from .transforms import TrainOnlyLinearTransform

__all__ = [
    "ExistingFeatureCache",
    "FeatureBatch",
    "FeatureCacheSpec",
    "FeatureInputFailure",
    "Qwen3VLEmbeddingAdapter",
    "TrainOnlyLinearTransform",
    "input_content_sha256",
    "l2_normalize",
    "validate_modality_inputs",
    "weighted_l2_concat",
]
