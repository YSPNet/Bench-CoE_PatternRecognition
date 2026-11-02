from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentError, HELD_OUT, ROLES, canonical_hash, resolve


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    def reject(value):
        raise ExperimentError(f"nonfinite JSON value {value} in {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False) + "\n").encode())


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def indexed_names(path: Path) -> tuple[str, ...]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict):
        if (any(type(i) is not int for i in value.values())
                or set(value.values()) != set(range(len(value)))):
            raise ExperimentError(f"registry indices must be a bijection: {path}")
        value = [name for name, index in sorted(value.items(), key=lambda item: item[1])]
    if not isinstance(value, (list, tuple)) or not value:
        raise ExperimentError(f"registry must contain indexed names: {path}")
    if any(not isinstance(name, str) or not name for name in value) or len(set(value)) != len(value):
        raise ExperimentError(f"registry names must be unique nonempty strings: {path}")
    return tuple(value)


@dataclass
class Dataset:
    root: Path
    meta: pd.DataFrame
    models: tuple[str, ...]
    roles: dict[str, np.ndarray]
    y_rows: np.ndarray
    c_rows: np.ndarray
    c_columns: np.ndarray
    cost_bounds: tuple[float, float]
    hashes: dict[str, str]
    protocol: dict

    @classmethod
    def load(cls, root: str | Path, config: dict) -> "Dataset":
        root = Path(root).resolve()
        meta_path = root / "data/registry/meta.parquet"
        meta = pd.read_parquet(meta_path).reset_index(drop=True)
        if not {"sample_id", "dataset"}.issubset(meta.columns):
            raise ExperimentError("metadata requires sample_id and dataset")
        for column in ("sample_id", "dataset"):
            if any(not isinstance(x, str) or not x for x in meta[column]):
                raise ExperimentError(f"metadata {column} must contain nonempty strings")
        if not meta.sample_id.is_unique:
            raise ExperimentError("duplicate metadata sample_id")
        if any(sid.split("/", 1)[0] != ds for sid, ds in zip(meta.sample_id, meta.dataset)):
            raise ExperimentError("sample_id prefix disagrees with dataset registry")
        observed = set(meta.dataset)
        if len(observed) != 14 or not set(HELD_OUT).issubset(observed):
            raise ExperimentError("Table A2 requires 14 datasets including the fixed 3 held-out datasets")

        files = [meta_path, root / "data/matrices/Y.npz", root / "data/matrices/C.npy"]
        registry_paths = [root / path for path in (
            "data/registry/model_index.pkl", "data/registry/sample_index.pkl",
            "data/matrices/sample_ids.pkl", "data/matrices/model_names.pkl",
        )]
        models, y_ids, c_ids, c_models = [indexed_names(p) for p in registry_paths]
        sample_ids = meta.sample_id.tolist()
        if set(y_ids) != set(sample_ids) or set(c_ids) != set(sample_ids) or set(c_models) != set(models):
            raise ExperimentError("quality/cost registries do not exactly cover metadata and model pool")

        def positions(reference, desired):
            mapping = {name: i for i, name in enumerate(reference)}
            return np.array([mapping[name] for name in desired], dtype=np.int64)

        official = {}
        for split in ("train", "dev", "test"):
            path = root / "SPLITS" / f"{split}.jsonl"
            records = read_jsonl(path)
            ids = [record["sample_id"] for record in records]
            if not ids or len(ids) != len(set(ids)) or set(ids) - set(sample_ids):
                raise ExperimentError(f"empty, duplicate or unknown official {split} IDs")
            official[split] = set(ids)
            files.append(path)
        if (any(official[a] & official[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")))
                or set.union(*official.values()) != set(sample_ids)):
            raise ExperimentError("official splits overlap or fail to cover metadata")

        manifest_path = resolve(root, config["protocol_manifest"])
        protocol = read_json(manifest_path)
        held = protocol.get("held_out_datasets_by_domain", {})
        if set(held.values()) != set(HELD_OUT):
            raise ExperimentError("protocol held-out datasets disagree with Table A2")
        if protocol.get("official_splits_overwritten") is not False:
            raise ExperimentError("protocol must preserve the official splits")
        if set(protocol.get("seen_datasets", [])) != observed - set(HELD_OUT):
            raise ExperimentError("protocol seen datasets disagree with metadata")
        roles = {}
        for role in ROLES:
            view = protocol["views"][role]
            path = resolve(root, view["path"])
            if file_hash(path) != view["sha256"]:
                raise ExperimentError(f"protocol view hash mismatch: {role}")
            records = read_jsonl(path)
            rows = [record["row_index"] for record in records]
            if (not rows or len(rows) != len(set(rows))
                    or any(type(r) is not int or r < 0 or r >= len(meta) for r in rows)
                    or view["count"] != len(rows)):
                raise ExperimentError(f"invalid protocol row indices: {role}")
            ids = [record["sample_id"] for record in records]
            selected = meta.iloc[rows]
            if ids != selected.sample_id.tolist():
                raise ExperimentError(f"protocol sample_id/row_index mismatch: {role}")
            if any(record.get("dataset") != ds for record, ds in zip(records, selected.dataset)):
                raise ExperimentError(f"protocol dataset mismatch: {role}")
            split = "train" if role == "train_seen" else "dev" if role == "dev_seen" else "test"
            expected = {sid for sid in official[split]
                        if (sid.split("/", 1)[0] in HELD_OUT) == (role == "ood_test")}
            if set(ids) != expected or view["official_split"] != split:
                raise ExperimentError(f"protocol is not the exact official-split intersection: {role}")
            roles[role] = np.asarray(rows, dtype=np.int64)
            files.append(path)

        bounds_path = root / "data/matrices/cost_bounds.json"
        bounds = read_json(bounds_path)
        cmin, cmax = float(bounds["cmin"]), float(bounds["cmax"])
        if not np.isfinite([cmin, cmax]).all() or not 0 < cmin < cmax:
            raise ExperimentError("cost bounds must satisfy 0 < cmin < cmax")
        files.extend(registry_paths + [bounds_path, manifest_path])
        hashes = {str(path.relative_to(root)) if path.is_relative_to(root) else str(path): file_hash(path)
                  for path in files}
        return cls(root, meta, models, roles, positions(y_ids, sample_ids),
                   positions(c_ids, sample_ids), positions(c_models, models),
                   (cmin, cmax), hashes, protocol)

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.hashes)

    def matrices(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for relative in ("data/matrices/Y.npz", "data/matrices/C.npy"):
            if file_hash(self.root / relative) != self.hashes[relative]:
                raise ExperimentError("cached outcomes changed after the protocol was loaded")
        with np.load(self.root / "data/matrices/Y.npz", allow_pickle=False) as archive:
            y = archive["Y"]
        c = np.load(self.root / "data/matrices/C.npy", mmap_mode="r", allow_pickle=False)
        expected = (len(self.meta), len(self.models))
        if y.shape != expected or c.shape != expected:
            raise ExperimentError(f"matrix shapes must match registries {expected}: Y={y.shape}, C={c.shape}")
        y_selected = y[self.y_rows[rows]]
        c_selected = np.asarray(c[np.ix_(self.c_rows[rows], self.c_columns)])
        if not np.isfinite(y_selected).all() or not np.isin(y_selected, [0, 1]).all():
            raise ExperimentError("quality values must be binary cached correctness")
        if not np.isfinite(c_selected).all() or np.any(c_selected < 0):
            raise ExperimentError("costs must be finite and nonnegative; no imputation is allowed")
        return y_selected.copy(), c_selected.copy()


def feature_paths(data: Dataset, config: dict) -> tuple[Path, Path]:
    features = config["features"]
    root = resolve(data.root, features["root"])
    return (root / "text" / (features["text_encoder"].split("/")[-1] + ".parquet"),
            root / "vision" / (features["vision_encoder"].split("/")[-1] + ".parquet"))


def load_features(data: Dataset, config: dict) -> tuple[np.ndarray, np.ndarray, str]:
    arrays, hashes = [], {}
    required = set(np.concatenate(list(data.roles.values())).tolist())
    required_ids = set(data.meta.iloc[sorted(required)].sample_id)
    for path in feature_paths(data, config):
        if not path.is_file():
            raise ExperimentError(f"missing feature file: {path}; run extract-features first")
        frame = pd.read_parquet(path)
        if not {"sample_id", "embedding"}.issubset(frame.columns) or not frame.sample_id.is_unique:
            raise ExperimentError(f"invalid or duplicate feature IDs: {path}")
        if set(frame.sample_id) - set(data.meta.sample_id) or required_ids - set(frame.sample_id):
            raise ExperimentError(f"feature IDs are unknown or required rows are missing: {path}")
        vectors = {sid: np.asarray(vector, dtype=np.float32) for sid, vector in zip(frame.sample_id, frame.embedding)}
        dimensions = {vector.shape for vector in vectors.values()}
        if len(dimensions) != 1 or len(next(iter(dimensions))) != 1 or next(iter(dimensions))[0] == 0:
            raise ExperimentError(f"inconsistent feature dimensions: {path}")
        if any(not np.isfinite(vector).all() for vector in vectors.values()):
            raise ExperimentError(f"nonfinite features: {path}")
        dimension = next(iter(dimensions))[0]
        # Inactive held-out train/dev rows are never requested by the runner.
        aligned = np.full((len(data.meta), dimension), np.nan, dtype=np.float32)
        for row in required:
            aligned[row] = vectors[data.meta.iloc[row].sample_id]
        arrays.append(aligned)
        hashes[str(path)] = file_hash(path)
    return arrays[0], arrays[1], canonical_hash(hashes)
