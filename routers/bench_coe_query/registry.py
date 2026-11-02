"""Strict, explicit sample/model registry loading."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from .errors import DataIntegrityError


def load_registry_list(path: str | Path, *, name: str) -> tuple[str, ...]:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise DataIntegrityError(f"missing {name} registry: {registry_path}")
    try:
        with registry_path.open("rb") as handle:
            values = pickle.load(handle)
    except Exception as exc:
        raise DataIntegrityError(f"failed to load {name} registry {registry_path}: {exc}") from exc
    if not isinstance(values, (list, tuple)):
        raise DataIntegrityError(f"{name} registry must be a list/tuple, got {type(values).__name__}")
    normalized = tuple(str(item) for item in values)
    if not normalized or any(not item for item in normalized):
        raise DataIntegrityError(f"{name} registry contains an empty value")
    if len(normalized) != len(set(normalized)):
        raise DataIntegrityError(f"{name} registry contains duplicate values")
    return normalized


def build_index(values: Iterable[str], *, name: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, raw in enumerate(values):
        value = str(raw)
        if value in result:
            raise DataIntegrityError(f"duplicate {name} ID: {value}")
        result[value] = index
    return result


def require_exact_order(observed: Iterable[str], expected: Iterable[str], *, name: str) -> tuple[str, ...]:
    observed_tuple = tuple(str(item) for item in observed)
    expected_tuple = tuple(str(item) for item in expected)
    if observed_tuple != expected_tuple:
        mismatch = next(
            (
                index,
                observed_tuple[index] if index < len(observed_tuple) else None,
                expected_tuple[index] if index < len(expected_tuple) else None,
            )
            for index in range(max(len(observed_tuple), len(expected_tuple)))
            if index >= len(observed_tuple)
            or index >= len(expected_tuple)
            or observed_tuple[index] != expected_tuple[index]
        )
        raise DataIntegrityError(
            f"{name} order mismatch at column/index {mismatch[0]}: observed={mismatch[1]!r}, expected={mismatch[2]!r}"
        )
    return observed_tuple
