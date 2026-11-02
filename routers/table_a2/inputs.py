from __future__ import annotations

import io
import hashlib
import ast
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentError, canonical_hash
from .data import Dataset, atomic_bytes, atomic_json, feature_paths, file_hash, read_json, read_jsonl


@contextmanager
def forbid_synthetic_fallback_images():
    """Fail if a native router tries to replace an input with a placeholder.

    The native end-to-end routers historically create a 224x224 black (and in
    one extraction path, white) image after a missing path or decode error.
    Table-A2 inputs are prepared and hashed ahead of time, so silently changing
    the input is never acceptable in this protocol.  Patching the shared PIL
    module here also covers router modules that imported ``Image`` earlier.
    """
    from PIL import Image

    original = Image.new

    def strict_new(mode, size, color=0):
        normalized = str(color).lower() if isinstance(color, str) else color
        is_placeholder = (
            mode == "RGB"
            and tuple(size) == (224, 224)
            and normalized in {"black", "white", 0, (0, 0, 0), (255, 255, 255)}
        )
        if is_placeholder:
            raise ExperimentError(
                "strict Table A2 input policy rejected a synthetic fallback image"
            )
        return original(mode, size, color)

    Image.new = strict_new
    try:
        yield
    finally:
        Image.new = original


def benchmark_samples(data: Dataset) -> dict[str, dict]:
    result = {}
    for path in sorted((data.root / "BENCHMARKS").glob("*/*_samples.jsonl")):
        for sample in read_jsonl(path):
            sid = sample["sample_id"]
            if sid in result:
                raise ExperimentError(f"duplicate BENCHMARKS sample: {sid}")
            result[sid] = sample
    if set(data.meta.sample_id) - result.keys():
        raise ExperimentError("BENCHMARKS does not cover the registered samples")
    for sid in data.meta.sample_id:
        sample = result[sid]
        if not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip():
            raise ExperimentError(f"missing text prompt: {sid}")
        if not isinstance(sample.get("assets"), list) or not sample["assets"]:
            raise ExperimentError(f"at least one image is required: {sid}")
    return result


def asset_context(data: Dataset):
    # Reuse only the local image decoder, independently of BenchCoE labels and
    # training contracts. It resolves canonical dataset TSV keys, not stale URIs.
    from routers.bench_coe_query.assets import StrictAssetResolver
    from routers.bench_coe_query.schemas import AssetSpec, StrictSample
    samples = benchmark_samples(data)
    resolver = StrictAssetResolver(data.root)
    sources = set((data.root / "BENCHMARKS").glob("*/*_samples.jsonl"))
    strict = {}
    for row, meta in data.meta.iterrows():
        record = samples[meta.sample_id]
        assets = [dict(asset) for asset in record["assets"]]
        for asset in assets:
            asset["uri"] = asset.get("uri") or asset.get("path")
        tsv = data.root / "vlm_router_data/TSV_images" / f"{meta.dataset}.tsv"
        if tsv.is_file():
            sources.add(tsv)
        else:
            for asset in assets:
                if not asset.get("uri"):
                    raise ExperimentError(f"no readable local image or official TSV: {meta.sample_id}")
                source = Path(asset["uri"])
                sources.add(source if source.is_absolute() else data.root / source)
        strict[row] = StrictSample(meta.sample_id, row, meta.dataset, str(meta.get("task", "")),
                                  record["prompt"], tuple(AssetSpec.from_mapping(asset) for asset in assets),
                                  str(meta.get("split", "")), (), ())
    signature = canonical_hash({str(path.resolve()): file_hash(path) for path in sorted(sources)})
    return samples, strict, resolver, signature


def resolve_images(sample, resolver):
    """Decode all canonical images without changing BenchCoE's single-image contract."""
    tsv = resolver.tsv_root / f"{sample.dataset}.tsv"
    if not tsv.is_file():
        return [resolver.resolve_path(sample.sample_id, asset) for asset in sample.assets]
    index = resolver._index(sample.dataset)
    key = resolver._tail_id(sample.sample_id)
    row_number, record = index.row_by_key(key)
    value = str(record.get("image", "")).strip()
    try:
        values = ast.literal_eval(value) if value.startswith("[") else [value]
    except (SyntaxError, ValueError) as error:
        raise ExperimentError(f"invalid canonical image list: {sample.sample_id}") from error
    if not isinstance(values, (list, tuple)) or not values or any(not isinstance(v, str) for v in values):
        raise ExperimentError(f"invalid canonical image values: {sample.sample_id}")
    result = []
    try:
        for i, item in enumerate(values):
            item, reference_row = resolver._follow_reference(index, item, sample_id=sample.sample_id)
            result.append(resolver._decode_base64(
                item, sample_id=sample.sample_id, source=str(index.path), source_key=key,
                source_row=row_number, metadata={"resolver": "table_a2_official_tsv",
                    "image_index": i, "image_count": len(values), "reference_row": reference_row,
                    "resized": False}))
    except Exception:
        for resolved in result:
            resolved.image.close()
        raise
    return result


def prepare_images(data: Dataset, output: Path, progress=print, read_only=False) -> tuple[pd.DataFrame, str]:
    """Cache lossless decoded RGB images so native datasets never use fallbacks."""
    if output == data.root or any(output.is_relative_to(data.root / directory) for directory in
                                  ("SPLITS", "BENCHMARKS", "ORACLE", "data", "models", "vlm_router_data")):
        raise ExperimentError("image cache must be independent of benchmark inputs")
    samples, strict, resolver, source_hash = asset_context(data)
    signature = canonical_hash({"data": data.fingerprint, "source": source_hash,
                                "format": "RGB PNG", "multi_image_policy": "native first image"})
    manifest_path = output / "images/IMAGES.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["signature"] != signature:
            raise ExperimentError("image cache inputs changed; choose a new output directory")
        records = manifest["images"]
    else:
        records = {}
    required = sorted(set(np.concatenate(list(data.roles.values())).tolist()))
    complete = set(records) == set(data.meta.iloc[required].sample_id)
    if read_only and not complete:
        raise ExperimentError("parallel workers require a complete image cache; run prepare-images first")
    meta = data.meta.copy()
    meta["text"] = [samples[sid]["prompt"] for sid in meta.sample_id]
    meta["assets"] = [[] for _ in range(len(meta))]
    for i, row in enumerate(required):
        sid = data.meta.iloc[row].sample_id
        image_path = output / "images" / (canonical_hash(sid) + ".png")
        if sid in records:
            if not image_path.is_file() or file_hash(image_path) != records[sid]["sha256"]:
                raise ExperimentError(f"image cache missing or modified: {sid}")
        else:
            resolved_images = resolve_images(strict[row], resolver)
            resolved = resolved_images[0]
            buffer = io.BytesIO()
            resolved.image.save(buffer, format="PNG")
            for item in resolved_images:
                item.image.close()
            atomic_bytes(image_path, buffer.getvalue())
            records[sid] = {"path": str(image_path), "sha256": file_hash(image_path),
                            "source": resolved.source, "source_key": resolved.source_key,
                            "image_count": len(resolved_images), "selected_image_index": 0}
        meta.at[row, "assets"] = [{"type": "image", "path": str(image_path), "uri": str(image_path)}]
        if (i + 1) % 128 == 0 or i + 1 == len(required):
            if not complete:
                atomic_json(manifest_path, {"signature": signature, "images": records})
            progress(f"images: {i + 1}/{len(required)}")
    return meta, file_hash(manifest_path)


def extract_features(data: Dataset, config: dict, output: Path, progress=print,
                     text_encoder=None, vision_encoder=None) -> dict:
    """Resume bounded feature batches, reading images without expert inference."""
    paths = feature_paths(data, config)
    samples, strict, resolver, source_hash = asset_context(data)
    signature = canonical_hash({"data": data.fingerprint, "source": source_hash,
                                "features": config["features"]})
    shard_root = output / "feature_shards" / signature
    export_path = shard_root / "EXPORT.json"
    pending_path = shard_root / "EXPORT_PENDING.json"
    if any(path.exists() for path in paths):
        if not export_path.is_file() and not pending_path.is_file():
            raise ExperimentError("feature exports already exist; reuse them for run, or select a new features.root")
        export = read_json(export_path if export_path.is_file() else pending_path)
        if any(path.is_file() and file_hash(path) != export["hashes"][str(path)] for path in paths):
            raise ExperimentError("feature exports were modified")
        if export_path.is_file():
            if not all(path.is_file() for path in paths):
                raise ExperimentError("feature exports are missing")
            return export
    shards = []
    size = config["features"]["batch_size"]
    for start in range(0, len(data.meta), size):
        end = min(start + size, len(data.meta))
        path = shard_root / f"{start:06d}-{end:06d}.parquet"
        state = path.with_suffix(".json")
        ids = data.meta.iloc[start:end].sample_id.tolist()
        if state.is_file():
            saved = read_json(state)
            if (saved["signature"] != signature or saved["sample_ids"] != ids
                    or not path.is_file() or saved["sha256"] != file_hash(path)):
                raise ExperimentError(f"invalid feature resume shard: {path}")
        else:
            if text_encoder is None or vision_encoder is None:
                from routers.features.encoders_registry import AVAILABLE_TEXT_ENCODERS
                from routers.features.text_encoder import TextEncoder
                from routers.features.vision_encoder import VisionEncoder
                feature = config["features"]
                info = AVAILABLE_TEXT_ENCODERS.get(feature["text_encoder"])
                if info is None or info.get("requires_api", False):
                    raise ExperimentError("feature extraction requires a registered local text encoder; API encoders are forbidden")
                text_encoder = TextEncoder(feature["text_encoder"], device=config["runtime"]["device"],
                                           batch_size=feature["batch_size"], show_progress=False)
                vision_encoder = VisionEncoder(feature["vision_encoder"], device=config["runtime"]["device"])
            text = np.asarray(text_encoder.extract([samples[sid]["prompt"] for sid in ids]))
            images, image_counts = [], []
            for row in range(start, end):
                resolved = resolve_images(strict[row], resolver)
                images.extend(item.image for item in resolved)
                image_counts.append(len(resolved))
            try:
                batch_extract = getattr(vision_encoder, "_extract_batch_from_images", None)
                all_vision = np.asarray(batch_extract(images, batch_size=size) if batch_extract else vision_encoder.extract(images))
            finally:
                for image in images:
                    image.close()
            if all_vision.ndim != 2 or len(all_vision) != len(images) or not np.isfinite(all_vision).all():
                raise ExperimentError("vision encoder returned invalid image features")
            vision, offset = [], 0
            for count in image_counts:
                feature = all_vision[offset:offset + count].mean(axis=0)
                if count > 1:
                    feature = feature / max(float(np.linalg.norm(feature)), 1e-12)
                vision.append(feature)
                offset += count
            vision = np.asarray(vision)
            if (text.ndim != 2 or vision.ndim != 2 or len(text) != len(ids) or len(vision) != len(ids)
                    or not np.isfinite(text).all() or not np.isfinite(vision).all()):
                raise ExperimentError("encoders returned invalid features")
            frame = pd.DataFrame({"sample_id": ids, "text": list(text.astype(np.float32)),
                                  "vision": list(vision.astype(np.float32))})
            buffer = io.BytesIO()
            frame.to_parquet(buffer, index=False)
            atomic_bytes(path, buffer.getvalue())
            atomic_json(state, {"signature": signature, "sample_ids": ids, "sha256": file_hash(path)})
        shards.append(path)
        progress(f"features: {end}/{len(data.meta)}")
    frame = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    if frame.sample_id.tolist() != data.meta.sample_id.tolist():
        raise ExperimentError("feature merge did not preserve exact-once registry order")
    exports = {}
    for column, path in zip(("text", "vision"), paths):
        buffer = io.BytesIO()
        frame[["sample_id", column]].rename(columns={column: "embedding"}).to_parquet(buffer, index=False)
        exports[path] = buffer.getvalue()
    export = {"signature": signature, "num_samples": len(frame),
              "hashes": {str(path): hashlib.sha256(value).hexdigest() for path, value in exports.items()},
              "features": config["features"]}
    if pending_path.exists() and read_json(pending_path) != export:
        raise ExperimentError("pending feature export does not match validated shards")
    atomic_json(pending_path, export)
    for path, value in exports.items():
        if not path.exists():
            atomic_bytes(path, value)
    atomic_json(export_path, export)
    return export
