"""Content-address every local model and visual asset consumed by the experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import ExperimentContract, effective_model_path, sha256_file
from .errors import ContractError


INPUT_CONTENT_MANIFEST_RELATIVE_PATH = Path(
    "outputs/bench_coe_query/audit/INPUT_CONTENT_MANIFEST.json"
)
MODEL_NAMES = ("bert", "tinyllava", "qwen3vl")
_MODEL_METADATA_SUFFIXES = {".json", ".txt", ".py", ".model", ".tiktoken"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _with_internal_digest(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise ContractError(f"content manifest payload unexpectedly contains {field}")
    result[field] = canonical_sha256(result)
    return result


def _validate_internal_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    recorded = value.get(field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ContractError(f"{label} lacks a valid {field}")
    payload = dict(value)
    payload.pop(field, None)
    observed = canonical_sha256(payload)
    if observed != recorded:
        raise ContractError(f"{label} internal digest mismatch")
    return observed


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _configured_model_path(contract: ExperimentContract, model_name: str) -> Path:
    configured = Path(str(contract.values["models"][model_name]["path"]))
    if not configured.is_absolute():
        configured = contract.dataset_root / configured
    return Path(os.path.abspath(configured))


def _discover_model_payload_files(model_root: Path) -> tuple[Path, ...]:
    if not model_root.is_dir():
        raise ContractError(f"model directory is missing: {model_root}")
    candidates = []
    safetensors = []
    fallback_pytorch_weights = []
    for path in model_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix == ".safetensors":
            safetensors.append(path)
            candidates.append(path)
        elif suffix in _MODEL_METADATA_SUFFIXES:
            candidates.append(path)
        elif name.startswith("pytorch_model") and suffix == ".bin":
            fallback_pytorch_weights.append(path)
    if not safetensors:
        candidates.extend(fallback_pytorch_weights)
    config = model_root / "config.json"
    if config not in candidates:
        raise ContractError(f"model payload selection lacks config.json: {model_root}")
    if not safetensors and not fallback_pytorch_weights:
        raise ContractError(f"model payload selection lacks PyTorch weights: {model_root}")

    # A sharded safetensors index is executable loader input. Require every
    # referenced shard and include it in the selected payload inventory.
    for index_path in sorted(model_root.rglob("*.safetensors.index.json")):
        try:
            document = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(document["weight_map"].values())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ContractError(f"invalid safetensors weight index: {index_path}") from exc
        for shard_name in shard_names:
            shard = (index_path.parent / str(shard_name)).resolve()
            try:
                shard.relative_to(model_root)
            except ValueError as exc:
                raise ContractError(f"weight index escapes model root: {index_path}") from exc
            if not shard.is_file():
                raise ContractError(f"weight index references a missing shard: {shard}")
            candidates.append(shard)
    return tuple(sorted(set(candidates), key=lambda item: item.relative_to(model_root).as_posix()))


def _file_records(paths: Iterable[Path], *, root: Path) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ContractError(f"content-addressed file escapes its root: {path}") from exc
        stat_before = resolved.stat()
        digest = sha256_file(resolved)
        stat_after = resolved.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns) != (
            stat_after.st_size,
            stat_after.st_mtime_ns,
        ):
            raise ContractError(f"input file changed while hashing: {resolved}")
        records.append(
            {
                "relative_path": relative,
                "bytes": int(stat_after.st_size),
                "sha256": digest,
            }
        )
    return records


def _model_entry(contract: ExperimentContract, model_name: str) -> dict[str, Any]:
    configured = _configured_model_path(contract, model_name)
    resolved = effective_model_path(contract, model_name)
    if not resolved.is_dir():
        raise ContractError(f"resolved model directory is missing: {resolved}")
    configured_is_symlink = configured.is_symlink()
    symlink_target = os.readlink(configured) if configured_is_symlink else None
    files = _file_records(_discover_model_payload_files(resolved), root=resolved)
    payload = {
        "model_name": model_name,
        "configured_path": str(configured),
        "configured_path_is_symlink": configured_is_symlink,
        "configured_symlink_target": symlink_target,
        "resolved_path": str(resolved),
        "selection": (
            "all local JSON/TXT/PY/MODEL/TIKTOKEN loader metadata plus every safetensors "
            "weight/index/shard; pytorch_model*.bin only when safetensors is absent"
        ),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    return _with_internal_digest(payload, field="model_manifest_sha256")


def _visual_asset_entry(contract: ExperimentContract) -> dict[str, Any]:
    relative_root = Path(str(contract.values["data"]["tsv_assets_root"]))
    root = contract.resolve(relative_root).resolve()
    if not root.is_dir():
        raise ContractError(f"TSV visual asset root is missing: {root}")
    paths = tuple(sorted(root.glob("*.tsv"), key=lambda item: item.name))
    if not paths:
        raise ContractError(f"TSV visual asset inventory is empty: {root}")
    files = _file_records(paths, root=root)
    payload = {
        "configured_root": relative_root.as_posix(),
        "resolved_root": str(root),
        "selection": "every regular *.tsv file directly under the frozen TSV asset root",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    return _with_internal_digest(payload, field="visual_assets_manifest_sha256")


def build_input_content_manifest(contract: ExperimentContract) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "bench_coe_query_input_content_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(contract.path),
            "sha256": contract.sha256,
            "immutable_base_contract": True,
        },
        "payload_selection_policy": {
            "models": "local Transformers from_pretrained payload superset; downloads and partial files excluded",
            "visual_assets": "whole-file SHA-256 for all TSV image containers",
            "network_allowed": False,
        },
        "models": {name: _model_entry(contract, name) for name in MODEL_NAMES},
        "visual_assets": _visual_asset_entry(contract),
    }
    return _with_internal_digest(payload, field="manifest_sha256")


def write_input_content_manifest(
    contract: ExperimentContract,
    path: str | Path = INPUT_CONTENT_MANIFEST_RELATIVE_PATH,
) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = contract.dataset_root / output
    _atomic_json(output, build_input_content_manifest(contract))
    return output


def _load_manifest_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid input content manifest: {path}") from exc
    if not isinstance(document, dict):
        raise ContractError("input content manifest must be a JSON object")
    _validate_internal_digest(document, field="manifest_sha256", label="input content manifest")
    for name in MODEL_NAMES:
        model = document.get("models", {}).get(name)
        if not isinstance(model, dict):
            raise ContractError(f"input content manifest lacks model {name}")
        _validate_internal_digest(
            model,
            field="model_manifest_sha256",
            label=f"input content model {name}",
        )
    visual = document.get("visual_assets")
    if not isinstance(visual, dict):
        raise ContractError("input content manifest lacks visual_assets")
    _validate_internal_digest(
        visual,
        field="visual_assets_manifest_sha256",
        label="input content visual assets",
    )
    return document


def validate_input_content_approval_binding(
    contract: ExperimentContract,
    approval: Mapping[str, Any],
) -> dict[str, Any]:
    binding = approval.get("input_content")
    if not isinstance(binding, Mapping):
        raise ContractError("full-run approval lacks the input_content binding")
    expected_path = INPUT_CONTENT_MANIFEST_RELATIVE_PATH.as_posix()
    if binding.get("manifest_path") != expected_path:
        raise ContractError("full-run approval input-content manifest path is not canonical")
    path = contract.dataset_root / INPUT_CONTENT_MANIFEST_RELATIVE_PATH
    if not path.is_file():
        raise ContractError(f"input content manifest is missing: {path}")
    observed_file_sha256 = sha256_file(path)
    if binding.get("manifest_file_sha256") != observed_file_sha256:
        raise ContractError("full-run approval input-content manifest file hash mismatch")
    document = _load_manifest_document(path)
    if document.get("contract", {}).get("sha256") != contract.sha256:
        raise ContractError("input content manifest is bound to another experiment contract")
    if binding.get("manifest_sha256") != document.get("manifest_sha256"):
        raise ContractError("full-run approval input-content internal hash mismatch")
    expected_models = {
        name: document["models"][name]["model_manifest_sha256"] for name in MODEL_NAMES
    }
    if dict(binding.get("model_manifest_sha256", {})) != expected_models:
        raise ContractError("full-run approval model-content hashes do not match the manifest")
    expected_visual = document["visual_assets"]["visual_assets_manifest_sha256"]
    if binding.get("visual_assets_manifest_sha256") != expected_visual:
        raise ContractError("full-run approval visual-content hash does not match the manifest")
    return {
        "manifest_path": expected_path,
        "manifest_file_sha256": observed_file_sha256,
        "manifest_sha256": document["manifest_sha256"],
        "model_manifest_sha256": expected_models,
        "visual_assets_manifest_sha256": expected_visual,
    }


def load_pinned_input_content_manifest(
    contract: ExperimentContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Imported lazily to keep config -> validation -> this module acyclic.
    from .config import load_full_run_approval

    approval, approval_sha256 = load_full_run_approval(contract)
    assert approval is not None and approval_sha256 is not None
    binding = validate_input_content_approval_binding(contract, approval)
    document = _load_manifest_document(
        contract.dataset_root / INPUT_CONTENT_MANIFEST_RELATIVE_PATH
    )
    binding["approval_sha256"] = approval_sha256
    return document, binding


def input_content_summary(contract: ExperimentContract) -> dict[str, Any]:
    document, binding = load_pinned_input_content_manifest(contract)
    return {
        **binding,
        "models": {
            name: {
                key: document["models"][name][key]
                for key in (
                    "configured_path",
                    "configured_path_is_symlink",
                    "configured_symlink_target",
                    "resolved_path",
                    "file_count",
                    "total_bytes",
                    "model_manifest_sha256",
                )
            }
            for name in MODEL_NAMES
        },
        "visual_assets": {
            key: document["visual_assets"][key]
            for key in (
                "configured_root",
                "resolved_root",
                "file_count",
                "total_bytes",
                "visual_assets_manifest_sha256",
            )
        },
    }


def feature_input_content_fields(
    contract: ExperimentContract,
    model_name: str,
) -> dict[str, Any]:
    if model_name not in {"tinyllava", "qwen3vl"}:
        raise ContractError(f"feature content binding does not support {model_name!r}")
    summary = input_content_summary(contract)
    model = summary["models"][model_name]
    visual = summary["visual_assets"]
    return {
        "schema_version": 1,
        "manifest_path": summary["manifest_path"],
        "manifest_file_sha256": summary["manifest_file_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
        "approval_sha256": summary["approval_sha256"],
        "model_name": model_name,
        "model_manifest_sha256": model["model_manifest_sha256"],
        "model_configured_path": model["configured_path"],
        "model_configured_path_is_symlink": model["configured_path_is_symlink"],
        "model_configured_symlink_target": model["configured_symlink_target"],
        "model_resolved_path": model["resolved_path"],
        "model_file_count": model["file_count"],
        "model_total_bytes": model["total_bytes"],
        "visual_assets_manifest_sha256": visual["visual_assets_manifest_sha256"],
        "visual_assets_file_count": visual["file_count"],
        "visual_assets_total_bytes": visual["total_bytes"],
    }


def verify_pinned_input_content(
    contract: ExperimentContract,
    *,
    model_names: Sequence[str] = MODEL_NAMES,
    include_visual_assets: bool = True,
) -> dict[str, Any]:
    document, binding = load_pinned_input_content_manifest(contract)
    requested = tuple(str(name) for name in model_names)
    if not requested or any(name not in MODEL_NAMES for name in requested):
        raise ContractError(f"invalid model content verification selection: {requested}")
    models: dict[str, Any] = {}
    for name in requested:
        observed = _model_entry(contract, name)
        expected = document["models"][name]
        if observed != expected:
            raise ContractError(f"model input content changed after approval: {name}")
        models[name] = {
            "model_manifest_sha256": observed["model_manifest_sha256"],
            "files_verified": observed["file_count"],
            "bytes_verified": observed["total_bytes"],
            "resolved_path": observed["resolved_path"],
            "configured_path_is_symlink": observed["configured_path_is_symlink"],
            "configured_symlink_target": observed["configured_symlink_target"],
        }
    visual: dict[str, Any] | None = None
    if include_visual_assets:
        observed_visual = _visual_asset_entry(contract)
        expected_visual = document["visual_assets"]
        if observed_visual != expected_visual:
            raise ContractError("TSV visual input content changed after approval")
        visual = {
            "visual_assets_manifest_sha256": observed_visual[
                "visual_assets_manifest_sha256"
            ],
            "files_verified": observed_visual["file_count"],
            "bytes_verified": observed_visual["total_bytes"],
            "resolved_root": observed_visual["resolved_root"],
        }
    return {
        "schema_version": 1,
        **binding,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "visual_assets": visual,
        "files_verified": sum(item["files_verified"] for item in models.values())
        + (0 if visual is None else int(visual["files_verified"])),
        "bytes_verified": sum(item["bytes_verified"] for item in models.values())
        + (0 if visual is None else int(visual["bytes_verified"])),
        "passed": True,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "summary"))
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--output", default=str(INPUT_CONTENT_MANIFEST_RELATIVE_PATH))
    parser.add_argument("--model", action="append", choices=MODEL_NAMES)
    parser.add_argument("--skip-visual-assets", action="store_true")
    args = parser.parse_args()
    from .config import load_contract

    contract = load_contract(args.config)
    if args.command == "build":
        output = write_input_content_manifest(contract, args.output)
        result: Any = {
            "path": str(output),
            "file_sha256": sha256_file(output),
            "manifest_sha256": _load_manifest_document(output)["manifest_sha256"],
        }
    elif args.command == "verify":
        result = verify_pinned_input_content(
            contract,
            model_names=tuple(args.model or MODEL_NAMES),
            include_visual_assets=not args.skip_visual_assets,
        )
    else:
        result = input_content_summary(contract)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
