"""Secret-safe, reproducible run manifests for Bench-CoE query experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import (
    FULL_RUN_APPROVAL_RELATIVE_PATH,
    ExperimentContract,
    load_contract,
    sha256_file,
)


DEFAULT_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "numpy",
    "pandas",
    "pyarrow",
    "Pillow",
    "PyYAML",
    "pytest",
)
_SECRET_NAME = re.compile(r"(token|secret|password|passwd|api[_-]?key|credential|authorization)", re.I)
_SECRET_VALUE = re.compile(r"^(hf_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{12,})$")


def _run_readonly(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def sanitize_command(arguments: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for argument in arguments:
        text = str(argument)
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if text.startswith("--") and "=" in text:
            name, _value = text.split("=", 1)
            result.append(f"{name}=<redacted>" if _SECRET_NAME.search(name) else text)
            continue
        if text.startswith("--") and _SECRET_NAME.search(text):
            result.append(text)
            redact_next = True
            continue
        if _SECRET_VALUE.match(text):
            result.append("<redacted>")
            continue
        result.append(text)
    return result


def safe_environment(whitelist: Iterable[str], environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    safe: dict[str, str] = {}
    for name in sorted(set(str(item) for item in whitelist)):
        if _SECRET_NAME.search(name):
            continue
        if name in source:
            value = str(source[name])
            safe[name] = "<redacted>" if _SECRET_VALUE.match(value) else value
    return safe


def package_versions(packages: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_state(repository_root: Path) -> dict[str, Any]:
    revision = _run_readonly(["git", "rev-parse", "HEAD"], repository_root)
    status = _run_readonly(["git", "status", "--porcelain=v1", "--untracked-files=all"], repository_root)
    if not revision["ok"]:
        return {
            "available": False,
            "reason": "git_metadata_unavailable",
            "error": revision.get("stderr") or revision.get("error"),
        }
    lines = status.get("stdout", "").splitlines() if status.get("ok") else []
    return {
        "available": True,
        "commit": revision["stdout"],
        "dirty": bool(lines),
        "status_entries": lines,
        "status_command_ok": bool(status.get("ok")),
    }


def gpu_inventory() -> dict[str, Any]:
    query = _run_readonly(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query["ok"]:
        return {"available": False, "error": query.get("stderr") or query.get("error")}
    gpus = []
    for line in query["stdout"].splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_total_mib": int(fields[3]),
                    "driver_version": fields[4],
                }
            )
    return {"available": True, "count": len(gpus), "gpus": gpus}


def frozen_artifact_hashes(contract: ExperimentContract) -> dict[str, dict[str, str]]:
    data = contract.values["data"]
    entries: dict[str, Mapping[str, Any]] = {}
    entries.update({f"split_{name}": value for name, value in data["splits"].items()})
    entries.update({f"matrix_{name}": value for name, value in data["matrices"].items()})
    entries.update({f"registry_{name}": value for name, value in data["registry"].items()})
    result: dict[str, dict[str, str]] = {}
    for name, entry in entries.items():
        path = contract.resolve(str(entry["path"]))
        result[name] = {
            "path": str(path),
            "expected_sha256": str(entry["sha256"]),
            "observed_sha256": sha256_file(path) if path.is_file() else "missing",
        }
    benchmark_root = contract.resolve(str(data["benchmarks"]["path"]))
    benchmark_files = sorted(benchmark_root.glob("*/*_samples.jsonl"), key=lambda item: item.as_posix())
    benchmark_payload = "".join(
        f"{path.relative_to(contract.dataset_root).as_posix()}\t{path.stat().st_size}\t{sha256_file(path)}\n"
        for path in benchmark_files
    ).encode("utf-8")
    result["benchmarks_manifest"] = {
        "path": str(benchmark_root),
        "expected_sha256": str(data["benchmarks"]["manifest_sha256"]),
        "observed_sha256": hashlib.sha256(benchmark_payload).hexdigest(),
    }
    return result


def model_config_hashes(contract: ExperimentContract) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, model in contract.values["models"].items():
        env_name = str(model["path_env"])
        model_root = Path(os.environ.get(env_name, str(model["path"]))).resolve()
        config_path = model_root / "config.json"
        result[str(name)] = {
            "path": str(model_root),
            "expected_config_sha256": str(model["config_sha256"]),
            "observed_config_sha256": sha256_file(config_path) if config_path.is_file() else "missing",
        }
    return result


def implementation_source_hashes(contract: ExperimentContract) -> dict[str, str]:
    """Bind runs to source bytes even when this release has no usable .git metadata."""

    roots = (
        contract.dataset_root / "routers/bench_coe_query",
        contract.dataset_root / "scripts/bench_coe_query",
    )
    paths = []
    for root in roots:
        if root.is_dir():
            paths.extend(root.rglob("*.py"))
            paths.extend(root.rglob("*.sh"))
            paths.extend(root.rglob("AGENTS.md"))
    return {
        path.relative_to(contract.dataset_root).as_posix(): sha256_file(path)
        for path in sorted(set(paths), key=lambda item: item.as_posix())
    }


def build_manifest(
    contract: ExperimentContract,
    *,
    seed: int | None = None,
    command: Sequence[str] | None = None,
    run_id: str | None = None,
    started_at: str | None = None,
    input_content_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .input_content import input_content_summary

    whitelist = contract.values["reproducibility"]["manifest_environment_whitelist"]
    now = datetime.now(timezone.utc).isoformat()
    approval_path = contract.dataset_root / FULL_RUN_APPROVAL_RELATIVE_PATH
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at or now,
        "manifested_at": now,
        "seed": seed,
        "command": sanitize_command(command if command is not None else sys.argv),
        "contract": {"path": str(contract.path), "sha256": contract.sha256},
        "repository": git_state(contract.dataset_root),
        "environment": safe_environment(whitelist),
        "runtime": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "gpu_inventory": gpu_inventory(),
        "model_configs": model_config_hashes(contract),
        "input_content": {
            "pinned": input_content_summary(contract),
            "verification": (
                None
                if input_content_verification is None
                else dict(input_content_verification)
            ),
        },
        "frozen_artifacts": frozen_artifact_hashes(contract),
        "implementation_sources": implementation_source_hashes(contract),
        "full_run_approval": {
            "path": str(FULL_RUN_APPROVAL_RELATIVE_PATH),
            "sha256": sha256_file(approval_path) if approval_path.is_file() else None,
        },
    }
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    if _SECRET_VALUE.search(serialized):
        raise ValueError("manifest secret-safety invariant failed")
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output_path


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path("configs/bench_coe_query/experiment_contract.yaml")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    contract = load_contract(args.config)
    write_manifest(args.output, build_manifest(contract, seed=args.seed, run_id=args.run_id))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
