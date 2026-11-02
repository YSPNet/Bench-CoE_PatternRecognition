"""Content-addressed feature cache contracts and existing-cache adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from routers.bench_coe_query.config import sha256_file

from ..data_contracts import InputRecord
from .fusion import l2_normalize


class FeatureCacheError(RuntimeError):
    """Raised when a cache is stale, incomplete, or silently drops rows."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def input_content_sha256(records: Sequence[InputRecord]) -> str:
    """Hash online content and image order without source names or UIDs."""

    value = [
        {
            "question": record.question,
            "options": list(record.options),
            "ordered_image_sha256": [hashlib.sha256(image).hexdigest() for image in record.images],
        }
        for record in records
    ]
    return _canonical_sha256(value)


@dataclass(frozen=True, slots=True)
class FeatureCacheSpec:
    raw_input_sha256: str
    encoder: Mapping[str, Any]
    processor: Mapping[str, Any]
    pooling: str
    template: str
    dtype: str
    normalization: str
    environment_id: str
    transform_fit_uid_sha256: str | None = None

    @property
    def key(self) -> str:
        return _canonical_sha256(
            {
                "raw_input_sha256": self.raw_input_sha256,
                "encoder": dict(self.encoder),
                "processor": dict(self.processor),
                "pooling": self.pooling,
                "template": self.template,
                "dtype": self.dtype,
                "normalization": self.normalization,
                "environment_id": self.environment_id,
                "transform_fit_uid_sha256": self.transform_fit_uid_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    uids: tuple[str, ...]
    features: np.ndarray
    available: np.ndarray
    failures: Mapping[str, str]

    @classmethod
    def create(
        cls,
        *,
        uids: Sequence[str],
        features: np.ndarray,
        available: np.ndarray,
        failures: Mapping[str, str],
    ) -> "FeatureBatch":
        ids = tuple(str(uid) for uid in uids)
        values = np.asarray(features, dtype=np.float32)
        mask = np.asarray(available, dtype=bool)
        if len(ids) != len(set(ids)) or values.ndim != 2 or values.shape[0] != len(ids):
            raise FeatureCacheError("feature batch UID/row shape mismatch")
        if mask.shape != (len(ids),):
            raise FeatureCacheError("availability mask shape mismatch")
        missing_ids = {ids[row] for row in np.flatnonzero(~mask)}
        if set(failures) != missing_ids:
            raise FeatureCacheError("every unavailable row needs exactly one structured failure")
        if not np.isfinite(values[mask]).all():
            raise FeatureCacheError("available feature rows must be finite")
        if np.any(values[~mask] != 0):
            raise FeatureCacheError("unavailable rows must retain zero placeholders plus failure records")
        values = values.copy()
        mask = mask.copy()
        values.flags.writeable = False
        mask.flags.writeable = False
        return cls(ids, values, mask, MappingProxyType(dict(failures)))

    def require_no_silent_drop(self, expected_uids: Sequence[str]) -> None:
        if self.uids != tuple(str(uid) for uid in expected_uids):
            raise FeatureCacheError("feature rows do not preserve the requested denominator/order")
        if int((~self.available).sum()) != len(self.failures):
            raise FeatureCacheError("unavailable rows and failures differ")


class ExistingFeatureCache:
    """Read a hash-bound merged Bench-CoE cache without touching Y/C."""

    def __init__(
        self,
        root: str | Path,
        *,
        pointer_path: str | Path,
        expected_pointer_sha256: str,
        expected_manifest_sha256: str,
    ) -> None:
        self.root = Path(root).resolve()
        pointer = self.root / pointer_path
        if sha256_file(pointer) != expected_pointer_sha256:
            raise FeatureCacheError("feature pointer hash mismatch")
        value = json.loads(pointer.read_text(encoding="utf-8"))
        manifest_path = self.root / str(value["manifest"])
        if sha256_file(manifest_path) != expected_manifest_sha256:
            raise FeatureCacheError("merged feature manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged_root = manifest_path.parent
        features_path = merged_root / "features.npy"
        index_path = merged_root / "index.jsonl"
        if sha256_file(index_path) != str(manifest["index_sha256"]):
            raise FeatureCacheError("feature index hash mismatch")
        self.features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        if self.features.shape != (int(manifest["rows"]), int(manifest["feature_dim"])):
            raise FeatureCacheError("feature header differs from merged manifest")
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        self.position = {str(row["sample_id"]): index for index, row in enumerate(rows)}
        if len(self.position) != len(rows) or len(rows) != self.features.shape[0]:
            raise FeatureCacheError("feature index contains duplicates or has wrong length")
        self.manifest = MappingProxyType(manifest)
        self.provenance = MappingProxyType(
            {
                "pointer": str(pointer.relative_to(self.root)),
                "pointer_sha256": expected_pointer_sha256,
                "manifest": str(manifest_path.relative_to(self.root)),
                "manifest_sha256": expected_manifest_sha256,
                "features_payload_sha256": str(manifest["features_sha256"]),
            }
        )

    def project(
        self,
        uids: Sequence[str],
        *,
        branch_slice: tuple[int, int] | None = None,
        normalize: bool = False,
    ) -> FeatureBatch:
        ids = tuple(str(uid) for uid in uids)
        width = self.features.shape[1]
        if branch_slice is not None:
            start, stop = branch_slice
            if not 0 <= start < stop <= width:
                raise FeatureCacheError("invalid branch slice")
            width = stop - start
        result = np.zeros((len(ids), width), dtype=np.float32)
        available = np.zeros(len(ids), dtype=bool)
        failures: dict[str, str] = {}
        for row, uid in enumerate(ids):
            position = self.position.get(uid)
            if position is None:
                failures[uid] = "structured_cache_unavailable"
                continue
            vector = self.features[position]
            if branch_slice is not None:
                vector = vector[slice(*branch_slice)]
            result[row] = vector
            available[row] = True
        if normalize and available.any():
            result[available] = l2_normalize(result[available])
        batch = FeatureBatch.create(
            uids=ids, features=result, available=available, failures=failures
        )
        batch.require_no_silent_drop(ids)
        return batch
