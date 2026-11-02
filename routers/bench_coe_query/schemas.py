"""Immutable records shared by strict data, assets, labels, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AssetSpec:
    kind: str
    uri: str | None = None
    tsv_file: str | None = None
    row_number: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssetSpec":
        row_number = value.get("index")
        return cls(
            kind=str(value.get("type", "")),
            uri=None if value.get("uri") is None else str(value["uri"]),
            tsv_file=None if value.get("tsv_file") is None else str(value["tsv_file"]),
            row_number=None if row_number is None else int(row_number),
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.kind}
        if self.uri is not None:
            result["uri"] = self.uri
        if self.tsv_file is not None:
            result["tsv_file"] = self.tsv_file
        if self.row_number is not None:
            result["index"] = self.row_number
        return result


@dataclass(frozen=True)
class StrictSample:
    sample_id: str
    row_index: int
    dataset: str
    task_type: str
    text: str
    assets: tuple[AssetSpec, ...]
    split: str
    y_row: tuple[int, ...]
    c_row: tuple[float, ...]


@dataclass(frozen=True)
class StrictVLRouterData:
    dataset_root: Path
    samples: tuple[StrictSample, ...]
    model_order: tuple[str, ...]
    sample_to_row: Mapping[str, int]
    model_to_column: Mapping[str, int]
    split_to_rows: Mapping[str, tuple[int, ...]]
    y: np.ndarray
    c: np.ndarray
    contract_sha256: str

    @classmethod
    def create(
        cls,
        *,
        dataset_root: Path,
        samples: tuple[StrictSample, ...],
        model_order: tuple[str, ...],
        sample_to_row: dict[str, int],
        model_to_column: dict[str, int],
        split_to_rows: dict[str, tuple[int, ...]],
        y: np.ndarray,
        c: np.ndarray,
        contract_sha256: str,
    ) -> "StrictVLRouterData":
        y.flags.writeable = False
        c.flags.writeable = False
        return cls(
            dataset_root=dataset_root,
            samples=samples,
            model_order=model_order,
            sample_to_row=MappingProxyType(dict(sample_to_row)),
            model_to_column=MappingProxyType(dict(model_to_column)),
            split_to_rows=MappingProxyType(dict(split_to_rows)),
            y=y,
            c=c,
            contract_sha256=contract_sha256,
        )

    def sample(self, sample_id: str) -> StrictSample:
        return self.samples[self.sample_to_row[sample_id]]

    def split_samples(self, split: str) -> tuple[StrictSample, ...]:
        return tuple(self.samples[row] for row in self.split_to_rows[split])


@dataclass(frozen=True)
class ResolvedImage:
    sample_id: str
    image: Image.Image
    source: str
    source_key: str
    source_row: int
    original_size: tuple[int, int]
    mode: str
    metadata: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        sample_id: str,
        image: Image.Image,
        source: str,
        source_key: str,
        source_row: int,
        original_size: tuple[int, int],
        mode: str,
        metadata: dict[str, Any],
    ) -> "ResolvedImage":
        return cls(
            sample_id=sample_id,
            image=image,
            source=source,
            source_key=source_key,
            source_row=source_row,
            original_size=original_size,
            mode=mode,
            metadata=MappingProxyType(metadata),
        )
