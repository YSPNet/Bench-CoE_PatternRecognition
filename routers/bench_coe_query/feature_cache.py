"""Signature-bound, resumable, one-writer feature shards and exact-once merge."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import FeatureCacheError


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def feature_signature(payload: Mapping[str, Any]) -> str:
    if not payload:
        raise FeatureCacheError("feature signature payload may not be empty")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise FeatureCacheError(f"JSON artifact must contain an object: {path}")
    return value


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        records = [json.loads(line) for line in text.splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(f"invalid JSONL artifact: {path}") from exc
    if any(not isinstance(record, dict) for record in records):
        raise FeatureCacheError(f"JSONL records must be objects: {path}")
    return records, hashlib.sha256(payload).hexdigest()


def _manifest_internal_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    recorded = payload.pop("manifest_sha256", None)
    if not isinstance(recorded, str) or not recorded:
        raise FeatureCacheError("merged feature manifest lacks manifest_sha256")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_validated_shard(
    root: Path,
    *,
    rank: int,
    world_size: int,
    signature: str,
    expected_state_sha256: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prefix = f"rank-{rank:03d}-of-{world_size:03d}"
    state_path = root / "shards" / f"{prefix}.state.json"
    feature_path = root / "shards" / f"{prefix}.features.npy"
    index_path = root / "shards" / f"{prefix}.index.jsonl"
    if not all(path.is_file() for path in (state_path, feature_path, index_path)):
        raise FeatureCacheError(f"missing shard files for {prefix}")

    state_sha256 = file_sha256(state_path)
    if expected_state_sha256 is not None and state_sha256 != expected_state_sha256:
        raise FeatureCacheError(f"shard state hash mismatch: {prefix}")
    state = _load_json(state_path)
    expected_metadata = {
        "rank": rank,
        "world_size": world_size,
        "signature": signature,
    }
    for key, value in expected_metadata.items():
        if state.get(key) != value:
            raise FeatureCacheError(f"shard state mismatch for {prefix}.{key}")

    observed_features_sha256 = file_sha256(feature_path)
    records, observed_index_sha256 = _load_jsonl(index_path)
    if state.get("features_sha256") != observed_features_sha256:
        raise FeatureCacheError(f"shard feature hash mismatch: {prefix}")
    if state.get("index_sha256") != observed_index_sha256:
        raise FeatureCacheError(f"shard index hash mismatch: {prefix}")

    try:
        features = np.load(feature_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FeatureCacheError(f"invalid shard feature array: {prefix}") from exc
    feature_dim = int(state.get("feature_dim", -1))
    dtype = str(state.get("dtype", ""))
    if features.ndim != 2 or features.shape[1] != feature_dim or str(features.dtype) != dtype:
        raise FeatureCacheError(f"shard feature dimension/dtype mismatch: {prefix}")
    if not np.isfinite(features).all():
        raise FeatureCacheError(f"shard contains non-finite features: {prefix}")
    if features.shape[0] != len(records) or len(records) != int(state.get("count", -1)):
        raise FeatureCacheError(f"feature/index/state count mismatch: {prefix}")

    try:
        rows = [int(record["row_index"]) for record in records]
        sample_ids = [str(record["sample_id"]) for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureCacheError(f"invalid shard index record: {prefix}") from exc
    if len(rows) != len(set(rows)) or len(sample_ids) != len(set(sample_ids)):
        raise FeatureCacheError(f"shard index contains duplicate rows or sample IDs: {prefix}")
    if any(row % world_size != rank for row in rows):
        raise FeatureCacheError(f"row assigned to wrong shard: {prefix}")
    if state.get("completed_rows") != sorted(rows):
        raise FeatureCacheError(f"shard completed_rows mismatch: {prefix}")

    evidence = {
        "rank": rank,
        "count": len(records),
        "state_sha256": state_sha256,
        "features_sha256": observed_features_sha256,
        "index_sha256": observed_index_sha256,
        "complete": state.get("complete") is True,
    }
    return features, records, state, evidence


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FeatureShardWriter:
    def __init__(
        self,
        cache_root: str | Path,
        *,
        rank: int,
        world_size: int,
        signature: str,
        feature_dim: int,
        dtype: str | np.dtype = "float16",
        resume: bool = False,
    ):
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise FeatureCacheError(f"invalid rank/world_size: {rank}/{world_size}")
        self.root = Path(cache_root)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.signature = str(signature)
        self.feature_dim = int(feature_dim)
        self.dtype = np.dtype(dtype)
        prefix = f"rank-{rank:03d}-of-{world_size:03d}"
        self.features_path = self.root / "shards" / f"{prefix}.features.npy"
        self.index_path = self.root / "shards" / f"{prefix}.index.jsonl"
        self.state_path = self.root / "shards" / f"{prefix}.state.json"
        self._features = np.empty((0, self.feature_dim), dtype=self.dtype)
        self._records: list[dict[str, Any]] = []
        self._rows: set[int] = set()
        if resume:
            self._resume()
        elif self.state_path.exists() or self.features_path.exists() or self.index_path.exists():
            raise FeatureCacheError(f"shard already exists; pass resume=True: {prefix}")

    @property
    def completed_rows(self) -> frozenset[int]:
        return frozenset(self._rows)

    def _resume(self) -> None:
        required = (self.features_path, self.index_path, self.state_path)
        if not all(path.is_file() for path in required):
            if any(path.exists() for path in required):
                raise FeatureCacheError("partial shard cannot be resumed")
            return
        state = _load_json(self.state_path)
        expected = {
            "rank": self.rank,
            "world_size": self.world_size,
            "signature": self.signature,
            "feature_dim": self.feature_dim,
            "dtype": str(self.dtype),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise FeatureCacheError(f"resume state mismatch for {key}: {state.get(key)!r} != {value!r}")
        self._features, self._records, _validated_state, _evidence = _load_validated_shard(
            self.root,
            rank=self.rank,
            world_size=self.world_size,
            signature=self.signature,
        )
        if self._features.shape != (len(self._records), self.feature_dim):
            raise FeatureCacheError("resume feature/index shape mismatch")
        self._rows = {int(record["row_index"]) for record in self._records}
        if len(self._rows) != len(self._records):
            raise FeatureCacheError("resume index contains duplicate rows")

    def append(
        self,
        row_indices: Sequence[int],
        sample_ids: Sequence[str],
        features: np.ndarray,
        *,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        flush: bool = True,
    ) -> None:
        rows = [int(row) for row in row_indices]
        values = np.asarray(features, dtype=self.dtype)
        if values.shape != (len(rows), self.feature_dim) or len(sample_ids) != len(rows):
            raise FeatureCacheError("append rows/sample IDs/features shape mismatch")
        if not np.isfinite(values).all():
            raise FeatureCacheError("feature batch contains non-finite values")
        if metadata is not None and len(metadata) != len(rows):
            raise FeatureCacheError("feature metadata length mismatch")
        if len(rows) != len(set(rows)):
            raise FeatureCacheError("append batch contains duplicate rows")
        for row in rows:
            if row % self.world_size != self.rank:
                raise FeatureCacheError(f"row {row} does not belong to rank {self.rank}/{self.world_size}")
            if row in self._rows:
                raise FeatureCacheError(f"row {row} already exists in shard")
        new_records = []
        for index, (row, sample_id) in enumerate(zip(rows, sample_ids)):
            record = {"row_index": row, "sample_id": str(sample_id)}
            if metadata is not None:
                record["metadata"] = dict(metadata[index])
            new_records.append(record)
        self._features = np.concatenate((self._features, values), axis=0)
        self._records.extend(new_records)
        self._rows.update(rows)
        if flush:
            self.flush(complete=False)

    def flush(self, *, complete: bool) -> None:
        _atomic_npy(self.features_path, self._features)
        payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in self._records
        )
        _atomic_text(self.index_path, payload)
        state = {
            "schema_version": 1,
            "rank": self.rank,
            "world_size": self.world_size,
            "signature": self.signature,
            "feature_dim": self.feature_dim,
            "dtype": str(self.dtype),
            "count": len(self._records),
            "completed_rows": sorted(self._rows),
            "complete": bool(complete),
            "features_sha256": file_sha256(self.features_path),
            "index_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        _atomic_text(self.state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")

    def finalize(self) -> None:
        self.flush(complete=True)


def merge_feature_shards(
    cache_root: str | Path,
    *,
    expected_rows: Iterable[int],
    world_size: int,
    signature: str,
) -> dict[str, Any]:
    root = Path(cache_root)
    expected = tuple(sorted(int(row) for row in expected_rows))
    if len(expected) != len(set(expected)):
        raise FeatureCacheError("expected merge rows contain duplicates")
    all_features: list[np.ndarray] = []
    all_records: list[dict[str, Any]] = []
    feature_dim: int | None = None
    dtype: str | None = None
    shard_manifests = []
    for rank in range(world_size):
        prefix = f"rank-{rank:03d}-of-{world_size:03d}"
        state_path = root / "shards" / f"{prefix}.state.json"
        features, records, state, evidence = _load_validated_shard(
            root,
            rank=rank,
            world_size=world_size,
            signature=signature,
        )
        if not state.get("complete") or state.get("signature") != signature:
            raise FeatureCacheError(f"incomplete or signature-mismatched shard: {prefix}")
        if feature_dim is None:
            feature_dim = int(features.shape[1])
            dtype = str(features.dtype)
        if features.ndim != 2 or features.shape[1] != feature_dim or str(features.dtype) != dtype:
            raise FeatureCacheError(f"feature dimension/dtype mismatch: {prefix}")
        all_features.append(features)
        all_records.extend(records)
        shard_manifests.append(evidence)
    observed_rows = [int(record["row_index"]) for record in all_records]
    if len(observed_rows) != len(set(observed_rows)):
        raise FeatureCacheError("merge contains duplicate rows")
    if tuple(sorted(observed_rows)) != expected:
        raise FeatureCacheError(
            f"merge row coverage mismatch: expected={len(expected)}, observed={len(observed_rows)}"
        )
    order = np.argsort(np.asarray(observed_rows, dtype=np.int64))
    concatenated = np.concatenate(all_features, axis=0) if all_features else np.empty((0, 0))
    merged_features = concatenated[order]
    merged_records = [all_records[int(position)] for position in order]
    merged_dir = root / "merged"
    features_path = merged_dir / "features.npy"
    index_path = merged_dir / "index.jsonl"
    _atomic_npy(features_path, merged_features)
    index_payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in merged_records
    )
    _atomic_text(index_path, index_payload)
    manifest = {
        "schema_version": 1,
        "signature": signature,
        "world_size": world_size,
        "rows": len(expected),
        "feature_dim": feature_dim,
        "dtype": dtype,
        "features_sha256": file_sha256(features_path),
        "index_sha256": hashlib.sha256(index_payload.encode("utf-8")).hexdigest(),
        "shards": shard_manifests,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _atomic_text(merged_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_merged_features(
    cache_root: str | Path,
    *,
    expected_signature: str,
    expected_manifest_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    root = Path(cache_root) / "merged"
    manifest_path = root / "manifest.json"
    if expected_manifest_file_sha256 is not None and file_sha256(manifest_path) != expected_manifest_file_sha256:
        raise FeatureCacheError("merged feature manifest file hash mismatch")
    manifest = _load_json(manifest_path)
    if manifest.get("signature") != expected_signature:
        raise FeatureCacheError("merged feature signature mismatch")
    observed_internal_sha256 = _manifest_internal_sha256(manifest)
    if manifest.get("manifest_sha256") != observed_internal_sha256:
        raise FeatureCacheError("merged feature canonical manifest hash mismatch")
    if expected_manifest_sha256 is not None and observed_internal_sha256 != expected_manifest_sha256:
        raise FeatureCacheError("merged feature manifest hash mismatch")
    try:
        features = np.load(root / "features.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FeatureCacheError("invalid merged feature array") from exc
    records, observed_index_sha256 = _load_jsonl(root / "index.jsonl")
    if file_sha256(root / "features.npy") != manifest["features_sha256"]:
        raise FeatureCacheError("merged feature file hash mismatch")
    if observed_index_sha256 != manifest.get("index_sha256"):
        raise FeatureCacheError("merged feature index hash mismatch")
    if (
        features.ndim != 2
        or features.shape[0] != len(records)
        or features.shape[0] != int(manifest["rows"])
        or features.shape[1] != int(manifest["feature_dim"])
        or str(features.dtype) != str(manifest["dtype"])
    ):
        raise FeatureCacheError("merged feature/index/manifest count mismatch")
    if not np.isfinite(features).all():
        raise FeatureCacheError("merged feature file contains non-finite values")
    try:
        rows = [int(record["row_index"]) for record in records]
        sample_ids = [str(record["sample_id"]) for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureCacheError("invalid merged index record") from exc
    if rows != sorted(rows) or len(rows) != len(set(rows)) or len(sample_ids) != len(set(sample_ids)):
        raise FeatureCacheError("merged feature index is not unique and canonically ordered")
    features.flags.writeable = False
    return features, records, manifest


def _resolve_bound_path(dataset_root: Path, value: Any, *, field: str) -> Path:
    path = Path(str(value))
    resolved = (path if path.is_absolute() else dataset_root / path).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise FeatureCacheError(f"{field} escapes the dataset root") from exc
    return resolved


def validate_feature_shards(
    cache_root: str | Path,
    *,
    expected_signature: str,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recompute every shard state, feature, and index digest recorded by a merge."""

    cache = Path(cache_root).resolve()
    world_size = int(manifest.get("world_size", -1))
    manifest_shards = manifest.get("shards")
    if world_size <= 0 or not isinstance(manifest_shards, list) or len(manifest_shards) != world_size:
        raise FeatureCacheError("merged manifest shard inventory mismatch")
    by_rank = {int(item.get("rank", -1)): item for item in manifest_shards}
    if set(by_rank) != set(range(world_size)):
        raise FeatureCacheError("merged manifest shard ranks are incomplete")
    shard_evidence = []
    for rank in range(world_size):
        recorded = by_rank[rank]
        _features, _records, state, evidence = _load_validated_shard(
            cache,
            rank=rank,
            world_size=world_size,
            signature=expected_signature,
            expected_state_sha256=str(recorded.get("state_sha256")),
        )
        if state.get("complete") is not True:
            raise FeatureCacheError(f"full feature shard is incomplete: rank {rank}")
        for digest_name in ("features_sha256", "index_sha256"):
            if recorded.get(digest_name) is not None and recorded.get(digest_name) != evidence[digest_name]:
                raise FeatureCacheError(f"merged manifest {digest_name} mismatch: rank {rank}")
        shard_evidence.append(evidence)
    return shard_evidence


def load_full_feature_cache(
    cache_root: str | Path,
    *,
    dataset_root: str | Path,
    expected_contract_sha256: str,
    expected_input_content: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load a full cache through the externally bound pointer/gate trust chain."""

    dataset = Path(dataset_root).resolve()
    cache = Path(cache_root)
    cache = (cache if cache.is_absolute() else dataset / cache).resolve()
    pointer_path = cache.parent / "LATEST_FULL.json"
    pointer = _load_json(pointer_path)
    if pointer.get("contract_sha256") != expected_contract_sha256:
        raise FeatureCacheError("full feature pointer contract hash mismatch")
    if _resolve_bound_path(dataset, pointer.get("cache_root"), field="cache_root") != cache:
        raise FeatureCacheError("full feature pointer selects a different cache root")

    gate_path = _resolve_bound_path(dataset, pointer.get("extraction_gate"), field="extraction_gate")
    manifest_path = _resolve_bound_path(dataset, pointer.get("manifest"), field="manifest")
    if gate_path.parent != cache or manifest_path != cache / "merged/manifest.json":
        raise FeatureCacheError("full feature pointer paths do not belong to the selected cache")
    if file_sha256(gate_path) != pointer.get("gate_sha256"):
        raise FeatureCacheError("full extraction gate hash mismatch")
    if file_sha256(manifest_path) != pointer.get("manifest_sha256"):
        raise FeatureCacheError("full merged manifest file hash mismatch")

    gate = _load_json(gate_path)
    if gate.get("passed") is not True or gate.get("contract_sha256") != expected_contract_sha256:
        raise FeatureCacheError("full extraction gate is not passing for this contract")
    if gate.get("cache_root") != str(cache.relative_to(dataset)):
        raise FeatureCacheError("full extraction gate cache root mismatch")
    if gate.get("feature_manifest_sha256") != pointer.get("manifest_sha256"):
        raise FeatureCacheError("pointer/gate merged manifest hash mismatch")
    if any(check.get("passed") is not True for check in gate.get("checks", [])):
        raise FeatureCacheError("full extraction gate contains a failed check")

    signature_payload_path = cache / "feature_signature.json"
    signature_payload = _load_json(signature_payload_path)
    recorded_bindings = (
        pointer.get("input_content"),
        gate.get("input_content"),
        signature_payload.get("input_content"),
    )
    if expected_input_content is not None:
        expected_binding = dict(expected_input_content)
        if any(binding != expected_binding for binding in recorded_bindings):
            raise FeatureCacheError("full feature input-content binding mismatch")
    elif any(binding is not None for binding in recorded_bindings):
        raise FeatureCacheError(
            "an expected input-content binding is required for this full feature cache"
        )
    signature = feature_signature(signature_payload)
    if signature != gate.get("feature_signature"):
        raise FeatureCacheError("full feature signature payload hash mismatch")
    features, records, manifest = load_merged_features(
        cache,
        expected_signature=signature,
        expected_manifest_sha256=str(gate.get("feature_manifest_internal_sha256")),
        expected_manifest_file_sha256=str(pointer.get("manifest_sha256")),
    )

    world_size = int(manifest.get("world_size", -1))
    shard_evidence = validate_feature_shards(
        cache,
        expected_signature=signature,
        manifest=manifest,
    )

    integrity = {
        "cache_root": str(cache.relative_to(dataset)),
        "contract_sha256": expected_contract_sha256,
        "pointer": str(pointer_path.relative_to(dataset)),
        "pointer_sha256": file_sha256(pointer_path),
        "gate": str(gate_path.relative_to(dataset)),
        "gate_sha256": file_sha256(gate_path),
        "feature_signature": signature,
        "feature_signature_payload_sha256": file_sha256(signature_payload_path),
        "input_content": (
            None if expected_input_content is None else dict(expected_input_content)
        ),
        "manifest": str(manifest_path.relative_to(dataset)),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "features_sha256": manifest["features_sha256"],
        "index_sha256": manifest["index_sha256"],
        "rows": len(records),
        "feature_dim": int(features.shape[1]),
        "dtype": str(features.dtype),
        "world_size": world_size,
        "shards": shard_evidence,
        "passed": True,
    }
    return features, records, manifest, integrity
