"""Durable step receipts and immutable input fingerprints for the X round."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/next_round_x"
PACKAGE = ROOT / "docs/next_round_x"


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_text(path: str | Path, value: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(value)
    tmp.replace(p)


def write_json(path: str | Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields=None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def record(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    display = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    return {"path": display, "exists": p.is_file(),
            "bytes": p.stat().st_size if p.is_file() else None,
            "sha256": digest(p) if p.is_file() else None}


def receipt(step: int, *, summary: str, outputs: list[str | Path],
            command: str, started: float, status: str = "COMPLETED",
            scientific_status: str = "NOT_APPLICABLE", checks: dict | None = None,
            blockers: list[str] | None = None, resources: dict | None = None) -> dict:
    index = read_json(PACKAGE / "PROMPT_INDEX.json")
    prompt = PACKAGE / index[step]["file"]
    result = {"schema_version": 1, "step": step, "title": index[step]["title"],
              "status": status, "scientific_status": scientific_status,
              "summary": summary, "command": command, "exit_code": 0,
              "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "wall_seconds": time.perf_counter() - started,
              "prompt": record(prompt), "outputs": [record(p) for p in outputs],
              "checks": checks or {}, "blockers": blockers or [],
              "resources": resources or {"gpu_seconds": 0, "paid_api_calls": 0,
                                          "expert_regenerations": 0}}
    result["payload_sha256"] = canonical(result)
    write_json(OUT / f"steps/{step:02d}.json", result)
    render_status()
    return result


def render_status() -> None:
    index = read_json(PACKAGE / "PROMPT_INDEX.json")
    rows = []
    for item in index:
        p = OUT / f"steps/{item['step']:02d}.json"
        r = read_json(p) if p.exists() else {}
        rows.append((item, r))
    finished = sum(r.get("status") == "COMPLETED" for _, r in rows)
    body = ["# X1—X4 实际执行状态", "",
            f"已完成步骤：{finished}/28。工程状态与科学结论分别记录。", "",
            "用户已授权连续执行与自动验收，无需逐步确认。官方split、历史冻结结果保持独立；各分支仍按真实资源和机制门推进。", "",
            "| 步骤 | 任务 | 工程状态 | 科学结论 | 说明 |", "|---|---|---|---|---|"]
    for item, r in rows:
        body.append(f"| {item['step']:02d} | {item['title']} | {r.get('status', 'NOT_STARTED')} | "
                    f"{r.get('scientific_status', 'PENDING')} | {r.get('summary', '')} |")
    write_text(OUT / "STATUS.md", "\n".join(body) + "\n")
