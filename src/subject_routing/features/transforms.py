"""Train-UID-only standardization and PCA/SVD transforms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..leakage_guard import Operation, UIDAccessAudit, UIDScopeError


def _uid_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainOnlyLinearTransform:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    fit_uid_sha256: str
    environment_id: str

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        uids: Sequence[str],
        *,
        audit: UIDAccessAudit,
        environment_id: str,
        n_components: int | None = None,
        standardize: bool = False,
    ) -> "TrainOnlyLinearTransform":
        if audit.operation is not Operation.FIT:
            raise UIDScopeError("feature transform fit requires a fit UID scope")
        ids = tuple(str(uid) for uid in uids)
        audit.require(ids, component="feature_transform_fit")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(ids) or not np.isfinite(values).all():
            raise ValueError("fit features must be finite and aligned with fit UIDs")
        mean = values.mean(axis=0)
        centered = values - mean
        scale = centered.std(axis=0, ddof=0) if standardize else np.ones(values.shape[1])
        scale[scale == 0] = 1.0
        normalized = centered / scale
        maximum = min(normalized.shape)
        components_count = maximum if n_components is None else int(n_components)
        if not 1 <= components_count <= maximum:
            raise ValueError("PCA component count exceeds fit matrix rank bound")
        _u, _singular, vt = np.linalg.svd(normalized, full_matrices=False)
        components = vt[:components_count]
        # Remove SVD sign ambiguity using the largest-magnitude coordinate.
        for row in range(len(components)):
            pivot = int(np.argmax(np.abs(components[row])))
            if components[row, pivot] < 0:
                components[row] *= -1
        arrays = [mean.astype(np.float32), scale.astype(np.float32), components.astype(np.float32)]
        for array in arrays:
            array.flags.writeable = False
        return cls(*arrays, fit_uid_sha256=_uid_sha256(ids), environment_id=str(environment_id))

    def transform(
        self,
        features: np.ndarray,
        uids: Sequence[str],
        *,
        audit: UIDAccessAudit,
    ) -> np.ndarray:
        if audit.operation not in {Operation.TRANSFORM, Operation.ASSIGN}:
            raise UIDScopeError("feature transform requires transform/assign UID scope")
        ids = tuple(str(uid) for uid in uids)
        audit.require(ids, component="feature_transform_apply")
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (len(ids), len(self.mean)) or not np.isfinite(values).all():
            raise ValueError("transform features are invalid or misaligned")
        result = ((values - self.mean) / self.scale) @ self.components.T
        result = np.asarray(result, dtype=np.float32)
        result.flags.writeable = False
        return result
