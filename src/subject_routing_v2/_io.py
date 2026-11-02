"""Small deterministic I/O helpers shared by V2 step runners."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    """Return the SHA256 of canonical UTF-8 JSON."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a V2 text artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def file_record(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    target = Path(path)
    display = target
    if root is not None:
        try:
            display = target.resolve().relative_to(Path(root).resolve())
        except ValueError:
            display = target.resolve()
    return {
        "path": display.as_posix(),
        "exists": target.is_file(),
        "bytes": target.stat().st_size if target.is_file() else None,
        "sha256": sha256_file(target) if target.is_file() else None,
    }


def with_payload_hash(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload
