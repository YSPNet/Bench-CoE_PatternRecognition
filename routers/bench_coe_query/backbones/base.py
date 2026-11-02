"""Base interface for frozen multimodal query representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..schemas import StrictSample


@dataclass(frozen=True)
class FeatureBatchOutput:
    features: torch.Tensor
    metadata: tuple[Mapping[str, Any], ...]


class BackboneFeatureExtractor(ABC):
    name: str

    @abstractmethod
    def encode_batch(self, sample_batch: Sequence[StrictSample]) -> FeatureBatchOutput:
        """Return one fixed-width feature per sample without generating answers."""

    @abstractmethod
    def feature_signature_payload(self) -> Mapping[str, Any]:
        """Return every semantic input needed to invalidate a feature cache."""

    @abstractmethod
    def parameter_checksum(self) -> str:
        """Return a stable checksum used to prove the frozen backbone did not change."""
