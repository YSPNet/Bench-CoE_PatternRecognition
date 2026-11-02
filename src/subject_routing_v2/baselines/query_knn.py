"""Mask-aware neighbor capability estimation on an audited input embedding."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class KNNQualityError(ValueError):
    """Raised for malformed neighbor capability inputs."""


class KNNQualityEstimator:
    """Estimate every expert's quality from the same input-space neighbors."""

    def __init__(self, *, max_neighbors: int, n_jobs: int = 1) -> None:
        if int(max_neighbors) <= 0:
            raise KNNQualityError("max_neighbors must be positive")
        self.max_neighbors = int(max_neighbors)
        self.n_jobs = int(n_jobs)
        self.index: NearestNeighbors | None = None
        self.features: np.ndarray | None = None
        self.quality: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.fallback: np.ndarray | None = None

    def fit(
        self, features: Any, quality: Any, observed_mask: Any
    ) -> "KNNQualityEstimator":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(quality, dtype=np.float64)
        mask = np.asarray(observed_mask, dtype=bool)
        if x.ndim != 2 or y.ndim != 2 or y.shape != mask.shape or x.shape[0] != y.shape[0]:
            raise KNNQualityError("kNN fit shapes are inconsistent")
        if x.shape[0] <= self.max_neighbors:
            raise KNNQualityError("not enough fit rows for max_neighbors")
        if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
            raise KNNQualityError("kNN fit contains non-finite observed values")
        counts = mask.sum(axis=0)
        if (counts == 0).any():
            raise KNNQualityError("an expert has no observed neighbor capability")
        self.features = normalize(x, norm="l2", axis=1, copy=True)
        self.quality = y.copy()
        self.mask = mask.copy()
        self.fallback = np.divide(
            np.where(mask, y, 0.0).sum(axis=0), counts, dtype=np.float64
        )
        self.index = NearestNeighbors(
            n_neighbors=self.max_neighbors + 1,
            algorithm="brute",
            metric="cosine",
            n_jobs=self.n_jobs,
        ).fit(self.features)
        return self

    def _aggregate(self, neighbor_rows: np.ndarray, *, neighbors: int) -> np.ndarray:
        if self.quality is None or self.mask is None or self.fallback is None:
            raise KNNQualityError("kNN estimator is not fitted")
        selected = neighbor_rows[:, : int(neighbors)]
        values = self.quality[selected]
        observed = self.mask[selected]
        counts = observed.sum(axis=1)
        sums = np.where(observed, values, 0.0).sum(axis=1)
        return np.divide(
            sums,
            counts,
            out=np.broadcast_to(self.fallback, sums.shape).copy(),
            where=counts > 0,
        )

    def neighbor_rows(self, features: Any) -> np.ndarray:
        """Return the frozen maximum-k neighbor index once for many k values."""

        if self.index is None or self.features is None:
            raise KNNQualityError("kNN estimator is not fitted")
        x = normalize(np.asarray(features, dtype=np.float64), norm="l2", axis=1, copy=True)
        return self.index.kneighbors(
            x, n_neighbors=self.max_neighbors, return_distance=False
        )

    def training_leave_one_out_rows(self) -> np.ndarray:
        """Return maximum-k training neighbors with the exact row itself removed."""

        if self.index is None or self.features is None:
            raise KNNQualityError("kNN estimator is not fitted")
        indices = self.index.kneighbors(
            self.features, n_neighbors=self.max_neighbors + 1, return_distance=False
        )
        cleaned = np.empty((indices.shape[0], self.max_neighbors), dtype=np.int64)
        for row, candidates in enumerate(indices):
            retained = candidates[candidates != row]
            if retained.size < self.max_neighbors:
                raise KNNQualityError("could not remove training self-neighbor")
            cleaned[row] = retained[: self.max_neighbors]
        return cleaned

    def predict_from_neighbor_rows(
        self, neighbor_rows: Any, *, neighbors: int
    ) -> np.ndarray:
        """Aggregate a previously frozen neighbor ordering at one k."""

        rows = np.asarray(neighbor_rows, dtype=np.int64)
        k = int(neighbors)
        if rows.ndim != 2 or rows.shape[1] < k or not 1 <= k <= self.max_neighbors:
            raise KNNQualityError("invalid frozen neighbor rows or k")
        return np.clip(self._aggregate(rows, neighbors=k), 0.0, 1.0)

    def predict_quality(self, features: Any, *, neighbors: int) -> np.ndarray:
        if self.index is None or self.features is None:
            raise KNNQualityError("kNN estimator is not fitted")
        k = int(neighbors)
        if not 1 <= k <= self.max_neighbors:
            raise KNNQualityError("neighbors is outside the fitted range")
        indices = self.neighbor_rows(features)
        return self.predict_from_neighbor_rows(indices, neighbors=k)

    def predict_training_leave_one_out(self, *, neighbors: int) -> np.ndarray:
        if self.index is None or self.features is None:
            raise KNNQualityError("kNN estimator is not fitted")
        k = int(neighbors)
        if not 1 <= k <= self.max_neighbors:
            raise KNNQualityError("neighbors is outside the fitted range")
        indices = self.training_leave_one_out_rows()
        return self.predict_from_neighbor_rows(indices, neighbors=k)
