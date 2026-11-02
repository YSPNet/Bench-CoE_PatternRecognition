"""Shared frozen-backbone safety helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def force_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def require_model_identity(path: Path, *, expected_model_type: str, expected_architecture: str) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", ""))
    architectures = [str(item) for item in config.get("architectures", [])]
    if model_type != expected_model_type or expected_architecture not in architectures:
        raise ValueError(
            f"wrong local model identity at {path}: model_type={model_type!r}, architectures={architectures}"
        )
    return config


def freeze_and_verify(module: torch.nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"backbone still has trainable parameters: {trainable[:3]}")


@torch.no_grad()
def parameter_summary_checksum(
    module: torch.nn.Module, *, chunk_bytes: int = 16 * 1024 * 1024
) -> str:
    """Return a streaming SHA-256 of the complete ordered ``state_dict``.

    The public name is retained for checkpoint/source compatibility.  Version 2
    deliberately replaces the legacy aggregate-moment summary: it covers every
    persistent parameter and buffer, state-dict metadata, logical ordering,
    dtype, shape, layout, stride, storage offset, and the raw bytes of each
    tensor.  Tensor payloads move to CPU in bounded chunks so multi-billion-
    parameter frozen backbones do not require a second full host copy.
    """

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    algorithm = "bench-coe-complete-ordered-state-dict-sha256-v2"
    digest.update((algorithm + "\0").encode("ascii"))
    state = module.state_dict(keep_vars=False)
    metadata = getattr(state, "_metadata", {})
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest.update(len(metadata_bytes).to_bytes(8, "big"))
    digest.update(metadata_bytes)

    for position, (name, item) in enumerate(state.items()):
        if not isinstance(item, torch.Tensor):
            payload = json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            header = {
                "position": position,
                "name": name,
                "kind": "extra_state",
                "payload_bytes": len(payload),
            }
            header_bytes = json.dumps(
                header, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            digest.update(len(header_bytes).to_bytes(8, "big"))
            digest.update(header_bytes)
            digest.update(payload)
            continue

        value = item.detach()
        if value.layout != torch.strided:
            raise TypeError(
                f"complete state digest requires strided tensors: {name} has {value.layout}"
            )
        logical = value.contiguous().view(-1)
        payload_bytes = logical.numel() * logical.element_size()
        header = {
            "position": position,
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "layout": str(value.layout),
            "stride": list(value.stride()),
            "storage_offset": int(value.storage_offset()),
            "numel": int(value.numel()),
            "payload_bytes": int(payload_bytes),
            "canonical_device": "cpu",
            "byte_order": sys.byteorder,
        }
        header_bytes = json.dumps(
            header, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(header_bytes).to_bytes(8, "big"))
        digest.update(header_bytes)
        elements_per_chunk = max(1, chunk_bytes // max(1, logical.element_size()))
        for start in range(0, logical.numel(), elements_per_chunk):
            cpu_chunk = logical[start : start + elements_per_chunk].to(
                device="cpu", non_blocking=False
            )
            raw = cpu_chunk.contiguous().view(torch.uint8).numpy().tobytes(order="C")
            digest.update(raw)
        digest.update(b"\0state-entry-end\0")
    return f"sha256-state-dict-v2:{digest.hexdigest()}"
