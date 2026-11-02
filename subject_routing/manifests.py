"""Build and validate the immutable P01 sample/split/model manifests.

The builder deliberately uses the target-free Bench-CoE adapter.  It may read
public prompts and structural asset references to compute duplicate digests,
but it cannot represent or open Y/C while it constructs sample membership.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from routers.bench_coe_query.config import load_contract, sha256_file
from routers.bench_coe_query.target_free import (
    deny_target_matrix_access,
    load_target_free_vlrb,
)


SPLITS = ("train", "dev", "test")
HELD_OUT_DATASETS = ("MMMU_DEV_VAL", "MathVision_MINI", "ChartQA_TEST")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sequence_sha256(values: Iterable[Any]) -> str:
    return canonical_sha256(list(values))


def _with_payload_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["payload_sha256"] = canonical_sha256(payload)
    return result


def load_frozen_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"frozen JSON must contain an object: {source}")
    recorded = value.get("payload_sha256")
    payload = dict(value)
    payload.pop("payload_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(payload):
        raise ValueError(f"frozen JSON payload digest mismatch: {source}")
    return value


def write_frozen_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write a manifest, refusing an existing target by default."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    value = _with_payload_digest(payload)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def reindex_model_columns(
    values: np.ndarray,
    source_models: Sequence[str],
    target_models: Sequence[str],
) -> np.ndarray:
    """Return columns in ``target_models`` order, failing on any ID mismatch."""

    source = tuple(str(item) for item in source_models)
    target = tuple(str(item) for item in target_models)
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError("model orders must contain unique expert IDs")
    if set(source) != set(target):
        raise ValueError("source and target model orders are not permutations")
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != len(source):
        raise ValueError("last matrix dimension does not match source model order")
    source_index = {model: column for column, model in enumerate(source)}
    return array[..., [source_index[model] for model in target]]


def _jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid split record {path}:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise ValueError(f"missing sample_id in {path}:{line_number}")
            ids.append(row["sample_id"])
    return ids


def _source_dataset_catalog(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "config/datasets.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    records = value.get("datasets") if isinstance(value, dict) else None
    if not isinstance(records, list):
        raise ValueError("config/datasets.yaml has no datasets list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        name = str(record["name"])
        if name in result:
            raise ValueError(f"duplicate dataset definition: {name}")
        result[name] = {
            "source_benchmark_split": str(record["split"]),
            "task": str(record["task"]),
            "modality": [str(item) for item in record["modality"]],
        }
    return result


def _artifact(path: Path, root: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            f"frozen artifact hash mismatch for {path}: expected={expected_sha256}, observed={observed}"
        )
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def _benchmark_files(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted((root / "BENCHMARKS").glob("*/*_samples.jsonl")):
        record = _artifact(path, root)
        with path.open("r", encoding="utf-8") as handle:
            record["records"] = sum(1 for line in handle if line.strip())
        result.append(record)
    return result


def _visual_asset_provenance(root: Path, datasets: Sequence[str]) -> dict[str, Any]:
    manifest_path = root / "outputs/bench_coe_query/audit/INPUT_CONTENT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("visual_assets", {}).get("files", [])
    by_name = {Path(str(item["relative_path"])).stem: item for item in files}
    selected = []
    for dataset in sorted(datasets):
        entry = by_name.get(dataset)
        if entry is None:
            raise ValueError(f"input-content manifest has no TSV record for {dataset}")
        path = root / "vlm_router_data/TSV_images" / str(entry["relative_path"])
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"visual TSV missing or size changed: {path}")
        selected.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(entry["bytes"]),
                "sha256": str(entry["sha256"]),
            }
        )
    return {
        "content_manifest": _artifact(manifest_path, root),
        "content_manifest_internal_sha256": manifest.get("manifest_sha256"),
        "verification_scope": "existence_and_size_rechecked; content digests inherited from frozen input-content manifest",
        "dataset_tsv_files": selected,
    }


def _group_summary(groups: Mapping[str, list[tuple[str, str]]]) -> dict[str, Any]:
    repeated = []
    cross_split = []
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        split_counts = Counter(split for split, _dataset in members)
        dataset_counts = Counter(dataset for _split, dataset in members)
        record = {
            "digest": digest,
            "count": len(members),
            "split_counts": dict(sorted(split_counts.items())),
            "dataset_counts": dict(sorted(dataset_counts.items())),
        }
        repeated.append(record)
        if len(split_counts) > 1:
            cross_split.append(record)
    return {
        "unique_groups": len(groups),
        "repeated_groups": len(repeated),
        "rows_in_repeated_groups": sum(item["count"] for item in repeated),
        "cross_split_groups": len(cross_split),
        "rows_in_cross_split_groups": sum(item["count"] for item in cross_split),
        "cross_split_group_records": cross_split,
    }


def _duplicate_audit(samples: Sequence[Any]) -> dict[str, Any]:
    asset_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    prompt_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    asset_count_histogram: Counter[int] = Counter()
    for sample in samples:
        structural_assets = [asset.as_dict() for asset in sample.assets]
        asset_digest = canonical_sha256(
            {"dataset": sample.dataset, "assets": structural_assets}
        )
        prompt_digest = hashlib.sha256(sample.text.encode("utf-8")).hexdigest()
        member = (sample.split, sample.dataset)
        asset_groups[asset_digest].append(member)
        prompt_groups[prompt_digest].append(member)
        asset_count_histogram[len(sample.assets)] += 1
    return {
        "scope": "all public BENCHMARKS records; no Y/C/expert outputs",
        "export_policy": "aggregate counts and SHA-256 group digests only; no prompts, assets, or test performance",
        "structural_asset_groups": _group_summary(asset_groups),
        "exact_prompt_groups": _group_summary(prompt_groups),
        "asset_count_histogram": {
            str(key): value for key, value in sorted(asset_count_histogram.items())
        },
        "decision": {
            "primary_protocol": "preserve immutable official router splits",
            "silent_repartition": False,
            "group_safe_use": "registered as a separate sensitivity analysis; never substitutes for the primary split",
            "known_strict_single_image_exclusions": sum(
                count for asset_count, count in asset_count_histogram.items() if asset_count != 1
            ),
        },
    }


def _role(
    ids: Sequence[str],
    sample_to_row: Mapping[str, int],
    sample_to_dataset: Mapping[str, str],
) -> dict[str, Any]:
    rows = [sample_to_row[sample_id] for sample_id in ids]
    return {
        "count": len(ids),
        "ordered_sample_ids": list(ids),
        "row_indices": rows,
        "ordered_uid_sha256": sequence_sha256(ids),
        "ordered_row_sha256": sequence_sha256(rows),
        "dataset_counts": dict(sorted(Counter(sample_to_dataset[item] for item in ids).items())),
    }


def build_frozen_manifests(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root_path = Path(root).resolve()
    contract_path = root_path / "configs/bench_coe_query/experiment_contract.yaml"
    contract = load_contract(contract_path)
    data_cfg = contract.values["data"]

    # Byte-level matrix provenance is verified here; no array is deserialized.
    matrices: dict[str, Any] = {}
    for name in ("y", "c", "cost_bounds"):
        entry = data_cfg["matrices"][name]
        record = _artifact(
            contract.resolve(str(entry["path"])),
            root_path,
            expected_sha256=str(entry["sha256"]),
        )
        if "shape" in entry:
            record["declared_shape"] = [int(item) for item in entry["shape"]]
            record["declared_dtype"] = str(entry["dtype"])
        matrices[name] = record

    with deny_target_matrix_access(contract) as guard:
        data = load_target_free_vlrb(root_path, contract=contract)
        dataset_catalog = _source_dataset_catalog(root_path)
        datasets = sorted({sample.dataset for sample in data.samples})
        if set(datasets) != set(dataset_catalog):
            raise ValueError("dataset catalog differs from target-free registry")

        official_ids = {
            split: _jsonl_ids(root_path / f"SPLITS/{split}.jsonl") for split in SPLITS
        }
        sample_to_dataset = {sample.sample_id: sample.dataset for sample in data.samples}
        sample_to_split = {sample.sample_id: sample.split for sample in data.samples}
        registry_ids = [sample.sample_id for sample in data.samples]
        sample_to_row = dict(data.sample_to_row)

        with (root_path / "data/matrices/sample_ids.pkl").open("rb") as handle:
            matrix_sample_ids = [str(item) for item in pickle.load(handle)]
        with (root_path / "data/matrices/model_names.pkl").open("rb") as handle:
            matrix_model_names = [str(item) for item in pickle.load(handle)]
        if matrix_sample_ids != registry_ids:
            raise ValueError("matrix sample_ids order differs from the canonical registry")
        if matrix_model_names != list(data.model_order):
            raise ValueError("matrix model_names order differs from the canonical registry")

        held = set(HELD_OUT_DATASETS)
        seen = set(datasets).difference(held)
        id14_roles = {
            split: _role(official_ids[split], sample_to_row, sample_to_dataset)
            for split in SPLITS
        }
        ood_role_ids = {
            "train_seen": [uid for uid in official_ids["train"] if sample_to_dataset[uid] in seen],
            "dev_seen": [uid for uid in official_ids["dev"] if sample_to_dataset[uid] in seen],
            "id_test": [uid for uid in official_ids["test"] if sample_to_dataset[uid] in seen],
            "ood_test": [uid for uid in official_ids["test"] if sample_to_dataset[uid] in held],
            "excluded_held_out_train": [uid for uid in official_ids["train"] if sample_to_dataset[uid] in held],
            "excluded_held_out_dev": [uid for uid in official_ids["dev"] if sample_to_dataset[uid] in held],
        }

        split_sources = {}
        for split in SPLITS:
            entry = data_cfg["splits"][split]
            split_sources[split] = _artifact(
                contract.resolve(str(entry["path"])),
                root_path,
                expected_sha256=str(entry["sha256"]),
            )
            split_sources[split]["records"] = len(official_ids[split])
            split_sources[split]["ordered_uid_sha256"] = sequence_sha256(official_ids[split])

        registry_rows = [
            {
                "row_index": row,
                "sample_id": uid,
                "dataset": sample_to_dataset[uid],
                "router_split": sample_to_split[uid],
                "source_benchmark_split": dataset_catalog[sample_to_dataset[uid]][
                    "source_benchmark_split"
                ],
            }
            for row, uid in enumerate(registry_ids)
        ]

        duplicate_audit = _duplicate_audit(data.samples)
        visual_assets = _visual_asset_provenance(root_path, datasets)

    if not guard["passed"]:
        raise RuntimeError("target-free manifest construction attempted to open Y/C")

    benchmark_files = _benchmark_files(root_path)
    registry_provenance = {}
    for name in ("meta", "models", "samples"):
        entry = data_cfg["registry"][name]
        registry_provenance[name] = _artifact(
            contract.resolve(str(entry["path"])),
            root_path,
            expected_sha256=str(entry["sha256"]),
        )

    split_manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "subject_routing_frozen_split_and_sample_manifest",
        "contract": {"path": contract_path.relative_to(root_path).as_posix(), "sha256": contract.sha256},
        "policy": {
            "split_origin": "existing immutable VL-RouterBench SPLITS; no reconstruction or random split",
            "router_split_is_distinct_from_source_benchmark_split": True,
            "test_content_exported": False,
            "test_targets_or_expert_performance_exported": False,
            "target_matrix_values_loaded": False,
        },
        "counts": {"samples": len(registry_ids), "datasets": len(datasets)},
        "dataset_catalog": {
            name: {
                **dataset_catalog[name],
                "registry_count": sum(sample_to_dataset[uid] == name for uid in registry_ids),
            }
            for name in sorted(dataset_catalog)
        },
        "sample_registry": {
            "ordered_uid_sha256": sequence_sha256(registry_ids),
            "rows": registry_rows,
            "duplicate_matrix_sample_ids": {
                **_artifact(root_path / "data/matrices/sample_ids.pkl", root_path),
                "exact_order_verified": True,
            },
        },
        "official_router_splits": {
            split: {**split_sources[split], **id14_roles[split]} for split in SPLITS
        },
        "environments": {
            "id14": {
                "fit_datasets": datasets,
                "held_out_datasets": [],
                "roles": id14_roles,
                "fit_isolation": "all transforms/partitions/capability tables fit train only",
            },
            "ood11_3": {
                "claim": "reconstructed held-out protocol; not exact Table A2 reproduction",
                "fit_datasets": sorted(seen),
                "held_out_datasets": list(HELD_OUT_DATASETS),
                "roles": {
                    name: _role(ids, sample_to_row, sample_to_dataset)
                    for name, ids in ood_role_ids.items()
                },
                "fit_isolation": "all held-out train/dev inputs and labels excluded from every fitted component",
            },
        },
        "duplicate_audit": duplicate_audit,
        "provenance": {
            "splits": split_sources,
            "registries": registry_provenance,
            "matrices_byte_hash_only": matrices,
            "benchmarks": {
                "manifest_sha256": str(data_cfg["benchmarks"]["manifest_sha256"]),
                "files": benchmark_files,
            },
            "visual_assets": visual_assets,
        },
        "target_matrix_guard": {
            "active": True,
            "attempt_count": guard["attempt_count"],
            "passed": guard["passed"],
        },
    }

    models = list(data.model_order)
    matrix_names_path = root_path / "data/matrices/model_names.pkl"
    model_pool: dict[str, Any] = {
        "schema_version": 1,
        "kind": "subject_routing_frozen_model_pool",
        "contract": {"path": contract_path.relative_to(root_path).as_posix(), "sha256": contract.sha256},
        "count": len(models),
        "canonical_order": models,
        "ordered_expert_sha256": sequence_sha256(models),
        "expert_to_column": {expert: column for column, expert in enumerate(models)},
        "experts": [
            {
                "expert_id": expert,
                "column_index": column,
                "availability": "cached_evaluation_only",
                "interface": "remote_api_cache" if expert in {"GPT4o", "GeminiFlash2-5"} else "local_expert_cache",
                "regeneration_allowed": False,
            }
            for column, expert in enumerate(models)
        ],
        "validation": {
            "unique_ids": len(models) == len(set(models)),
            "contract_order_exact": models == list(contract.model_order),
            "matrix_model_names_duplicate_registry": {
                **_artifact(matrix_names_path, root_path),
                "exact_order_verified": True,
            },
            "matrix_shape": [int(item) for item in data_cfg["matrices"]["y"]["shape"]],
            "column_permutation_must_be_explicit": True,
        },
        "provenance": {
            "model_registry": registry_provenance["models"],
            "y": matrices["y"],
            "c": matrices["c"],
        },
    }
    return split_manifest, model_pool
