"""Strict local-path and official TSV-packed image resolution."""

from __future__ import annotations

import ast
import base64
import binascii
import csv
import io
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .errors import AssetResolutionError
from .schemas import AssetSpec, ResolvedImage, StrictSample


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class TSVRowIndex:
    """A lazy byte-offset index for VLMEvalKit TSV logical records."""

    def __init__(self, path: str | Path, key_column: str):
        self.path = Path(path).resolve()
        self.key_column = key_column
        self.header: tuple[str, ...] = ()
        self._key_to_location: dict[str, tuple[int, int, int]] = {}
        self._row_offsets: list[tuple[int, int]] = []
        self._built = False

    @staticmethod
    def _aliases(value: str) -> tuple[str, ...]:
        stripped = value.strip()
        aliases = [stripped]
        try:
            number = float(stripped)
            if number.is_integer():
                aliases.append(str(int(number)))
        except ValueError:
            pass
        return tuple(dict.fromkeys(aliases))

    @staticmethod
    def _parse_record(raw: bytes) -> list[list[str]]:
        text = raw.decode("utf-8")
        # strict=True is essential here. Without it csv.reader accepts an
        # unterminated quoted final field at EOF, so an embedded newline in the
        # last TSV column can look like a complete row and corrupt all offsets
        # that follow it.
        return list(csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True))

    def _build(self) -> None:
        if self._built:
            return
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with self.path.open("rb") as handle:
            header_bytes = handle.readline()
            try:
                header = next(csv.reader([header_bytes.decode("utf-8-sig").rstrip("\r\n")], delimiter="\t"))
            except (UnicodeDecodeError, csv.Error) as exc:
                raise ValueError(f"invalid TSV header in {self.path}: {exc}") from exc
            if self.key_column not in header or "image" not in header:
                raise ValueError(
                    f"TSV {self.path} requires columns {self.key_column!r} and 'image'; got {header}"
                )
            self.header = tuple(header)
            key_position = header.index(self.key_column)
            row_number = 0
            while True:
                offset = handle.tell()
                record = handle.readline()
                if not record:
                    break
                if not record.strip():
                    continue
                while True:
                    try:
                        parsed_rows = self._parse_record(record)
                    except csv.Error as exc:
                        if str(exc) == "unexpected end of data":
                            continuation = handle.readline()
                            if not continuation:
                                raise ValueError(
                                    f"unterminated TSV logical record {row_number} in {self.path}"
                                ) from exc
                            record += continuation
                            continue
                        raise ValueError(f"invalid TSV record {row_number} in {self.path}: {exc}") from exc
                    except UnicodeDecodeError as exc:
                        raise ValueError(f"invalid TSV record {row_number} in {self.path}: {exc}") from exc
                    if len(parsed_rows) == 1 and len(parsed_rows[0]) == len(header):
                        fields = parsed_rows[0]
                        break
                    if len(parsed_rows) > 1 or (parsed_rows and len(parsed_rows[0]) > len(header)):
                        raise ValueError(
                            f"malformed TSV logical record {row_number} in {self.path}: field counts={[len(row) for row in parsed_rows]}"
                        )
                    continuation = handle.readline()
                    if not continuation:
                        raise ValueError(f"unterminated TSV logical record {row_number} in {self.path}")
                    record += continuation
                length = len(record)
                key = fields[key_position]
                self._row_offsets.append((offset, length))
                for alias in self._aliases(key):
                    if alias in self._key_to_location:
                        raise ValueError(f"duplicate TSV key {alias!r} in {self.path}")
                    self._key_to_location[alias] = (row_number, offset, length)
                row_number += 1
        self._built = True

    def contains(self, key: str) -> bool:
        self._build()
        return any(alias in self._key_to_location for alias in self._aliases(key))

    def _read_at(self, offset_and_length: tuple[int, int]) -> dict[str, str]:
        self._build()
        offset, length = offset_and_length
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.read(length)
        parsed_rows = self._parse_record(line)
        if len(parsed_rows) != 1 or len(parsed_rows[0]) != len(self.header):
            raise ValueError(f"TSV record changed while reading {self.path}")
        fields = parsed_rows[0]
        return dict(zip(self.header, fields))

    def row_by_key(self, key: str) -> tuple[int, dict[str, str]]:
        self._build()
        location = next(
            (
                self._key_to_location[alias]
                for alias in self._aliases(key)
                if alias in self._key_to_location
            ),
            None,
        )
        if location is None:
            raise KeyError(key)
        row_number, offset, length = location
        return row_number, self._read_at((offset, length))

    def row_by_number(self, row_number: int) -> tuple[int, dict[str, str]]:
        self._build()
        if row_number < 0 or row_number >= len(self._row_offsets):
            raise IndexError(row_number)
        return row_number, self._read_at(self._row_offsets[row_number])


class StrictAssetResolver:
    """Resolve exactly one local RGB image without resizing or network access."""

    def __init__(self, dataset_root: str | Path, tsv_root: str | Path | None = None):
        self.dataset_root = Path(dataset_root).resolve()
        self.tsv_root = (
            Path(tsv_root).resolve()
            if tsv_root is not None
            else self.dataset_root / "vlm_router_data/TSV_images"
        )
        self._indices: dict[str, TSVRowIndex] = {}

    def _error(self, code: str, sample_id: str, message: str, **details: Any) -> AssetResolutionError:
        return AssetResolutionError(code=code, sample_id=sample_id, message=message, details=details)

    def _index(self, dataset: str) -> TSVRowIndex:
        if dataset not in self._indices:
            path = self.tsv_root / f"{dataset}.tsv"
            key_column = "id" if dataset == "MMMU_DEV_VAL" else "index"
            self._indices[dataset] = TSVRowIndex(path, key_column)
        return self._indices[dataset]

    @staticmethod
    def _tail_id(sample_id: str) -> str:
        if "/" not in sample_id:
            return sample_id
        return sample_id.split("/", 1)[1]

    def _follow_reference(
        self,
        index: TSVRowIndex,
        image_value: str,
        *,
        sample_id: str,
        max_hops: int = 8,
    ) -> tuple[str, int]:
        value = image_value.strip()
        row_number = -1
        seen: set[str] = set()
        for _hop in range(max_hops + 1):
            if len(value) > 64 or not index.contains(value):
                return value, row_number
            if value in seen:
                raise self._error("cyclic_tsv_reference", sample_id, "cyclic TSV image reference", reference=value)
            seen.add(value)
            try:
                row_number, row = index.row_by_key(value)
            except (KeyError, ValueError) as exc:
                raise self._error("missing_tsv_reference", sample_id, str(exc), reference=value) from exc
            value = str(row.get("image", "")).strip()
        raise self._error("tsv_reference_depth", sample_id, "TSV reference chain exceeded limit")

    def _decode_base64(
        self,
        encoded: str,
        *,
        sample_id: str,
        source: str,
        source_key: str,
        source_row: int,
        metadata: dict[str, Any],
    ) -> ResolvedImage:
        payload = encoded.strip()
        if payload.startswith("data:"):
            if "," not in payload:
                raise self._error("invalid_base64_image", sample_id, "malformed data URI", source=source)
            payload = payload.split(",", 1)[1]
        if not payload or payload.lower() == "nan":
            raise self._error("empty_image", sample_id, "empty TSV image payload", source=source)
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise self._error("invalid_base64_image", sample_id, str(exc), source=source) from exc
        try:
            with Image.open(io.BytesIO(raw)) as loaded:
                loaded.load()
                original_size = loaded.size
                original_mode = loaded.mode
                image = loaded.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise self._error("corrupt_image", sample_id, str(exc), source=source) from exc
        return ResolvedImage.create(
            sample_id=sample_id,
            image=image,
            source=source,
            source_key=source_key,
            source_row=source_row,
            original_size=original_size,
            mode=original_mode,
            metadata=metadata,
        )

    def resolve_path(self, sample_id: str, asset: AssetSpec) -> ResolvedImage:
        if not asset.uri or not asset.uri.strip():
            raise self._error("empty_image", sample_id, "image asset has no URI")
        path = Path(asset.uri)
        if not path.is_absolute():
            path = self.dataset_root / path
        path = path.resolve()
        if not path.is_file():
            raise self._error("missing_image", sample_id, "image path is not readable", path=str(path))
        try:
            with Image.open(path) as loaded:
                loaded.load()
                original_size = loaded.size
                original_mode = loaded.mode
                image = loaded.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise self._error("corrupt_image", sample_id, str(exc), path=str(path)) from exc
        return ResolvedImage.create(
            sample_id=sample_id,
            image=image,
            source=str(path),
            source_key=path.name,
            source_row=-1,
            original_size=original_size,
            mode=original_mode,
            metadata={"resolver": "ordinary_path", "resized": False},
        )

    def resolve_tsv(self, sample: StrictSample) -> ResolvedImage:
        index = self._index(sample.dataset)
        tail = self._tail_id(sample.sample_id)
        try:
            row_number, row = index.row_by_key(tail)
        except FileNotFoundError as exc:
            raise self._error("missing_tsv", sample.sample_id, str(exc), dataset=sample.dataset) from exc
        except (KeyError, ValueError) as exc:
            raise self._error(
                "missing_tsv_row", sample.sample_id, str(exc), dataset=sample.dataset, key=tail
            ) from exc

        value = str(row.get("image", "")).strip()
        try:
            parsed = ast.literal_eval(value) if value.startswith("[") and value.endswith("]") else value
        except (SyntaxError, ValueError) as exc:
            raise self._error("invalid_image_list", sample.sample_id, str(exc), dataset=sample.dataset) from exc
        if isinstance(parsed, (list, tuple)):
            if len(parsed) == 0:
                raise self._error("empty_image", sample.sample_id, "TSV image list is empty")
            if len(parsed) != 1:
                raise self._error(
                    "multi_image",
                    sample.sample_id,
                    "strict single-image contract rejects an image list",
                    image_count=len(parsed),
                    dataset=sample.dataset,
                )
            value = str(parsed[0])
        elif not isinstance(parsed, str):
            raise self._error("invalid_image_value", sample.sample_id, "TSV image value is not a string")

        value, reference_row = self._follow_reference(index, value, sample_id=sample.sample_id)
        return self._decode_base64(
            value,
            sample_id=sample.sample_id,
            source=str(index.path),
            source_key=tail,
            source_row=row_number,
            metadata={
                "resolver": "official_tsv",
                "dataset": sample.dataset,
                "reference_row": reference_row,
                "resized": False,
            },
        )

    def resolve_sample(self, sample: StrictSample) -> ResolvedImage:
        if len(sample.assets) == 0:
            raise self._error("empty_image", sample.sample_id, "sample has no image asset")
        if len(sample.assets) != 1:
            raise self._error(
                "multi_image",
                sample.sample_id,
                "strict single-image contract rejects multiple assets",
                image_count=len(sample.assets),
                dataset=sample.dataset,
            )
        official_tsv = self.tsv_root / f"{sample.dataset}.tsv"
        if official_tsv.is_file():
            return self.resolve_tsv(sample)
        return self.resolve_path(sample.sample_id, sample.assets[0])
