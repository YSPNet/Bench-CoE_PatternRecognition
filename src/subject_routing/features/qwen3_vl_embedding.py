"""Optional local-only adapter boundary for a dedicated Qwen3-VL-Embedding model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data_contracts import InputRecord


class DedicatedEmbeddingUnavailable(RuntimeError):
    """Raised when the optional checkpoint is not locally approved and pinned."""


class Qwen3VLEmbeddingAdapter:
    """Interface only; it never downloads or silently substitutes a chat VLM."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)
        path_value = self.spec.get("local_path")
        if self.spec.get("status") != "READY" or not path_value:
            raise DedicatedEmbeddingUnavailable(str(self.spec.get("reason", "checkpoint blocked")))
        self.path = Path(str(path_value)).resolve()
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            if not (self.path / name).is_file():
                raise DedicatedEmbeddingUnavailable(f"missing pinned local artifact: {name}")

    def encode(self, records: Sequence[InputRecord]) -> np.ndarray:
        raise NotImplementedError(
            "loader implementation remains disabled until a checkpoint revision and processor are pinned"
        )
