"""Prepare strict E6 image inputs and the required SigLIP-L/16 features.

Historical image and model caches are read-only; new assets live in E6.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import OUTPUT, ROOT, read_json, write_json
from routers.table_a2.config import load_config
from routers.table_a2.data import Dataset
from routers.table_a2.inputs import asset_context, resolve_images

OUT = OUTPUT / "E6"
ARCHIVE = Path("/home/sm5/ys/ICASSP/archive/non_paper_research_20261011")
OLD_IMAGES = ARCHIVE / ".cache/table_a2/image_cache/images"
OLD_HF = ARCHIVE / ".cache/table_a2/huggingface/hub"
NEW_HF = OUT / "hf_cache"
IMAGE_DIR = OUT / "images"
IMAGE_MANIFEST = OUT / "IMAGES.json"
FEATURE_DIR = OUT / "siglip_shards"
FEATURE_FILE = OUT / "siglip-large-patch16-384.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inputs():
    data = Dataset.load(ROOT, load_config(ROOT / "configs/table_a2.yaml"))
    samples, strict, resolver, source_hash = asset_context(data)
    return data, samples, strict, resolver, source_hash


def link_backbones():
    NEW_HF.mkdir(parents=True, exist_ok=True)
    for name in ("unc-nlp--lxmert-base-uncased", "google--vit-base-patch16-224",
                 "bert-base-uncased", "BAAI--bge-m3"):
        cache_name = "models--" + name
        source, target = OLD_HF / cache_name, NEW_HF / cache_name
        if not source.is_dir():
            raise RuntimeError(f"historical pretrained cache missing: {source}")
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise RuntimeError(f"unexpected E6 model cache entry: {target}")
        else:
            target.symlink_to(source, target_is_directory=True)
    print(json.dumps({"stage": "backbone_cache_linked", "path": str(NEW_HF)}), flush=True)


def prepare_images():
    data, samples, strict, resolver, source_hash = inputs()
    old = read_json(OLD_IMAGES / "IMAGES.json")["images"]
    previous = read_json(IMAGE_MANIFEST) if IMAGE_MANIFEST.exists() else {}
    if previous and previous.get("source_hash") != source_hash:
        raise RuntimeError("source image fingerprint changed")
    new = previous.get("new", {})
    records = {}
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for row, sid in enumerate(data.meta.sample_id):
        if sid in old:
            item = old[sid]
            path = OLD_IMAGES / Path(item["path"]).name
            if not path.is_file():
                raise RuntimeError(f"missing archived image: {sid}")
            records[sid] = {"path": str(path), "image_count": item["image_count"],
                            "sha256": item["sha256"], "origin": "historical_cache"}
            continue
        path = IMAGE_DIR / (hashlib.sha256(sid.encode()).hexdigest() + ".png")
        if sid not in new:
            resolved = resolve_images(strict[row], resolver)
            buffer = io.BytesIO()
            resolved[0].image.save(buffer, format="PNG")
            for image in resolved:
                image.image.close()
            path.write_bytes(buffer.getvalue())
            new[sid] = {"path": str(path), "image_count": len(resolved),
                        "sha256": sha256(path), "origin": "journal_tsv_decode"}
            if len(new) % 64 == 0:
                write_json(IMAGE_MANIFEST, {"source_hash": source_hash,
                                             "archived_manifest_sha256": sha256(OLD_IMAGES / "IMAGES.json"),
                                             "new": new})
                print(json.dumps({"stage": "images", "decoded": len(new)}), flush=True)
        elif not path.is_file() or sha256(path) != new[sid]["sha256"]:
            raise RuntimeError(f"new image checkpoint changed: {sid}")
        records[sid] = new[sid]
    if len(records) != len(data.meta):
        raise RuntimeError("image coverage is incomplete")
    write_json(IMAGE_MANIFEST, {"source_hash": source_hash,
                                 "archived_manifest_sha256": sha256(OLD_IMAGES / "IMAGES.json"),
                                 "new": new, "n_all": len(records), "n_archived": len(old)})
    write_json(OUT / "IMAGE_PATHS.json", records)
    print(json.dumps({"stage": "images_complete", "n_all": len(records),
                      "new": len(new), "archived": len(old)}), flush=True)


def extract_siglip():
    from routers.features.vision_encoder import VisionEncoder

    data, _, strict, resolver, _ = inputs()
    records = read_json(OUT / "IMAGE_PATHS.json")
    if set(records) != set(data.meta.sample_id):
        raise RuntimeError("image record coverage differs from registry")
    encoder = VisionEncoder("google/siglip-large-patch16-384", device="cuda", batch_size=32)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    shards = []
    for start in range(0, len(data.meta), 128):
        end = min(start + 128, len(data.meta))
        path = FEATURE_DIR / f"{start:05d}-{end:05d}.npz"
        ids = data.meta.sample_id.iloc[start:end].tolist()
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if saved["sample_ids"].tolist() != ids or saved["features"].shape != (len(ids), 1024):
                    raise RuntimeError(f"invalid SigLIP checkpoint: {path}")
            shards.append(path)
            continue
        images, counts = [], []
        for row in range(start, end):
            sid = data.meta.sample_id.iloc[row]
            if records[sid]["image_count"] > 1:
                resolved = resolve_images(strict[row], resolver)
                batch_images = [item.image for item in resolved]
            else:
                batch_images = [Image.open(records[sid]["path"]).convert("RGB")]
            counts.append(len(batch_images))
            images.extend(batch_images)
        try:
            encoded = encoder._extract_batch_from_images(images, batch_size=32)
        finally:
            for image in images:
                image.close()
        features, offset = [], 0
        for n in counts:
            vector = encoded[offset:offset+n].mean(axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            features.append(vector)
            offset += n
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.shape != (len(ids), 1024) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid SigLIP features at {start}:{end}")
        np.savez_compressed(path, sample_ids=np.asarray(ids), features=matrix)
        shards.append(path)
        print(json.dumps({"stage": "siglip", "done": end, "total": len(data.meta)}), flush=True)
    ids, arrays = [], []
    for path in shards:
        with np.load(path, allow_pickle=False) as saved:
            ids.extend(saved["sample_ids"].tolist())
            arrays.append(saved["features"])
    features = np.concatenate(arrays, axis=0)
    if ids != data.meta.sample_id.tolist():
        raise RuntimeError("SigLIP feature sample order mismatch")
    pd.DataFrame({"sample_id": ids, "embedding": list(features)}).to_parquet(FEATURE_FILE, index=False)
    write_json(OUT / "SIGLIP_FEATURES.json", {"model": "google/siglip-large-patch16-384",
                "features_sha256": sha256(FEATURE_FILE), "n": len(ids), "dimension": 1024,
                "multi_image_pooling": "mean_then_L2", "batch_size": 32})
    print(json.dumps({"stage": "siglip_complete", "n": len(ids), "sha256": sha256(FEATURE_FILE)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("link-backbones", "images", "siglip"))
    args = parser.parse_args()
    {"link-backbones": link_backbones, "images": prepare_images,
     "siglip": extract_siglip}[args.stage]()
