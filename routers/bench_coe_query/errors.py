"""Typed, fail-closed errors used throughout the Bench-CoE query pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class BenchCoEError(RuntimeError):
    """Base error for a protocol or runtime violation."""


class ContractError(BenchCoEError):
    """The frozen experiment contract is missing or has been violated."""


class DataIntegrityError(BenchCoEError):
    """A frozen data artifact is missing, malformed, or misaligned."""


class PredictionIntegrityError(BenchCoEError):
    """A prediction artifact is incomplete, duplicated, or inconsistent."""


class FeatureCacheError(BenchCoEError):
    """A feature shard/cache does not match its signature or index."""


@dataclass(frozen=True)
class AssetResolutionError(BenchCoEError):
    """Structured image-resolution failure that is safe to serialize."""

    code: str
    sample_id: str
    message: str
    details: Mapping[str, Any]

    def __str__(self) -> str:
        return f"{self.code} for {self.sample_id}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error_type": self.code,
            "sample_id": self.sample_id,
            "message": self.message,
            "details": dict(self.details),
        }
