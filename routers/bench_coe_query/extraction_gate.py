"""Prompt-11 full feature-cache accounting and extraction quality gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentContract, load_full_run_approval, sha256_file
from .data import load_vlrb_strict
from .full_policy import EXPECTED_MULTI_IMAGE_SPLIT_COUNTS, partition_feature_rows
from .feature_cache import feature_signature, load_merged_features, validate_feature_shards
from .manifest import write_manifest
from .input_content import feature_input_content_fields, verify_pinned_input_content


MINIMUM_COSINE_SIMILARITY = 0.9999
MAXIMUM_RELATIVE_L2 = 0.01


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _reference_comparison(
    contract: ExperimentContract,
    backbone: str,
    cache_root: Path,
) -> dict[str, Any]:
    reference_root = (
        contract.dataset_root
        / "outputs/bench_coe_query/smoke"
        / backbone
        / "v2-single32"
    )
    full_features = np.load(cache_root / "merged/features.npy", allow_pickle=False).astype(np.float32)
    full_records = _jsonl(cache_root / "merged/index.jsonl")
    reference_features = np.load(reference_root / "merged/features.npy", allow_pickle=False).astype(np.float32)
    reference_records = _jsonl(reference_root / "merged/index.jsonl")
    full_position = {int(record["row_index"]): position for position, record in enumerate(full_records)}
    reference_rows = [int(record["row_index"]) for record in reference_records[:32]]
    missing = [row for row in reference_rows if row not in full_position]
    if missing:
        return {"passed": False, "missing_reference_rows": missing}
    left = reference_features[: len(reference_rows)]
    right = full_features[np.asarray([full_position[row] for row in reference_rows], dtype=np.int64)]
    difference = np.abs(left - right)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, np.finfo(np.float32).tiny)
    cosine = (left * right).sum(axis=1) / denominator
    relative_l2 = np.linalg.norm(left - right, axis=1) / np.maximum(
        left_norm, np.finfo(np.float32).tiny
    )
    sample_ids_equal = [record["sample_id"] for record in reference_records[:32]] == [
        full_records[full_position[row]]["sample_id"] for row in reference_rows
    ]
    passed = bool(
        len(reference_rows) >= 32
        and sample_ids_equal
        and float(cosine.min()) >= MINIMUM_COSINE_SIMILARITY
        and float(relative_l2.max(initial=0.0)) <= MAXIMUM_RELATIVE_L2
    )
    return {
        "passed": passed,
        "reference_root": str(reference_root.relative_to(contract.dataset_root)),
        "rows": len(reference_rows),
        "sample_ids_equal": sample_ids_equal,
        "bitwise_equal": bool(np.array_equal(left, right)),
        "maximum_absolute_difference": float(difference.max(initial=0.0)),
        "minimum_cosine_similarity": float(cosine.min()),
        "maximum_relative_l2": float(relative_l2.max(initial=0.0)),
        "acceptance": {
            "minimum_cosine_similarity": MINIMUM_COSINE_SIMILARITY,
            "maximum_relative_l2": MAXIMUM_RELATIVE_L2,
        },
    }


def build_full_extraction_gate(
    contract: ExperimentContract,
    *,
    backbone: str,
    cache_root: str | Path,
    world_size: int,
) -> dict[str, Any]:
    if backbone not in {"tinyllava", "qwen3vl"}:
        raise ValueError(backbone)
    root = Path(cache_root).resolve()
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    eligible_rows, expected_exclusions = partition_feature_rows(
        data,
        range(len(data.samples)),
        contract=contract,
        require_release_counts=True,
    )
    feature_manifest_path = root / "merged/manifest.json"
    payload = _json(root / "feature_signature.json")
    expected_signature = feature_signature(payload)
    expected_input_content = feature_input_content_fields(contract, backbone)
    input_content_verification = verify_pinned_input_content(
        contract,
        model_names=(backbone,),
        include_visual_assets=True,
    )
    feature_manifest_document = _json(feature_manifest_path)
    features, records, feature_manifest = load_merged_features(
        root,
        expected_signature=expected_signature,
        expected_manifest_sha256=str(feature_manifest_document.get("manifest_sha256")),
        expected_manifest_file_sha256=sha256_file(feature_manifest_path),
    )
    shard_evidence = validate_feature_shards(
        root,
        expected_signature=expected_signature,
        manifest=feature_manifest,
    )
    _approval, current_approval_sha256 = load_full_run_approval(contract)
    resources = [_json(root / f"resource_rank_{rank:03d}.json") for rank in range(world_size)]
    exclusion_documents = [
        _json(root / f"exclusions_rank_{rank:03d}.json") for rank in range(world_size)
    ]
    exclusions = [record for document in exclusion_documents for record in document["records"]]
    observed_rows = [int(record["row_index"]) for record in records]
    observed_excluded_rows = [int(record["row_index"]) for record in exclusions]
    expected_excluded_rows = [int(record["row_index"]) for record in expected_exclusions]

    checks = [
        _check(
            "feature_exact_once_coverage",
            observed_rows == list(eligible_rows)
            and len(observed_rows) == len(set(observed_rows))
            and int(feature_manifest["rows"]) == len(eligible_rows),
            {
                "expected_feature_rows": len(eligible_rows),
                "observed_feature_rows": len(observed_rows),
                "duplicates": len(observed_rows) - len(set(observed_rows)),
            },
        ),
        _check(
            "structured_exclusion_exact_once_coverage",
            sorted(observed_excluded_rows) == sorted(expected_excluded_rows)
            and len(observed_excluded_rows) == len(set(observed_excluded_rows)),
            {
                "expected_exclusions": len(expected_excluded_rows),
                "observed_exclusions": len(observed_excluded_rows),
                "split_counts": {
                    split: sum(record["split"] == split for record in exclusions)
                    for split in EXPECTED_MULTI_IMAGE_SPLIT_COUNTS
                },
            },
        ),
        _check(
            "frozen_denominator_accounting",
            len(observed_rows) + len(observed_excluded_rows) == len(data.samples)
            and not set(observed_rows).intersection(observed_excluded_rows),
            {
                "official_rows": len(data.samples),
                "feature_rows": len(observed_rows),
                "structured_fallback_rows": len(observed_excluded_rows),
            },
        ),
        _check(
            "features_finite_and_fixed_width",
            features.ndim == 2
            and features.shape[0] == len(eligible_rows)
            and np.isfinite(features).all()
            and int(feature_manifest["feature_dim"]) == features.shape[1]
            and sha256_file(root / "merged/features.npy") == feature_manifest["features_sha256"],
            {
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "finite": bool(np.isfinite(features).all()),
                "features_sha256_matches": sha256_file(root / "merged/features.npy")
                == feature_manifest["features_sha256"],
            },
        ),
        _check(
            "sample_ids_rows_and_split_match",
            all(
                data.samples[int(record["row_index"])].sample_id == str(record["sample_id"])
                for record in records
            ),
            {"rows_checked": len(records)},
        ),
        _check(
            "signature_contract_and_policy_match",
            bool(
                feature_manifest.get("signature") == expected_signature
                and
                payload.get("contract_sha256") == contract.sha256
                and payload.get("full_multi_image_policy")
                == "exclude_with_frozen_denominator_report"
                and payload.get("full_multi_image_fallback") == "train_strongest_global"
                and payload.get("approval_sha256") == current_approval_sha256
                and payload.get("input_content") == expected_input_content
            ),
            {
                "contract_sha256": payload.get("contract_sha256"),
                "feature_signature": feature_manifest.get("signature"),
                "signature_payload_sha256": expected_signature,
                "approval_sha256": payload.get("approval_sha256"),
                "current_approval_sha256": current_approval_sha256,
                "input_content": payload.get("input_content"),
            },
        ),
        _check(
            "model_tokenizer_and_visual_inputs_content_addressed",
            input_content_verification.get("passed") is True,
            input_content_verification,
        ),
        _check(
            "all_shard_payload_digests_match_merge_manifest",
            len(shard_evidence) == world_size,
            {"ranks": len(shard_evidence), "shards": shard_evidence},
        ),
        _check(
            "all_rank_backbones_unchanged",
            len(resources) == world_size
            and all(
                item.get("backbone_unchanged") is True
                and item.get("backbone_checksum_before") == item.get("backbone_checksum_after")
                and int(item.get("optimizer_parameters", -1)) == 0
                and item.get("feature_signature") == expected_signature
                and item.get("feature_signature_payload") == payload
                and item.get("input_content") == expected_input_content
                for item in resources
            ),
            {
                "ranks": len(resources),
                "checksums": sorted({item.get("backbone_checksum_before") for item in resources}),
                "optimizer_parameters": [item.get("optimizer_parameters") for item in resources],
                "feature_signatures": sorted(
                    {item.get("feature_signature") for item in resources}
                ),
            },
        ),
    ]
    reference = _reference_comparison(contract, backbone, root)
    checks.append(_check("single_gpu_32_row_reextraction_consistency", reference["passed"], reference))
    gate = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone,
        "world_size": world_size,
        "contract_sha256": contract.sha256,
        "cache_root": str(root.relative_to(contract.dataset_root)),
        "feature_manifest": str(feature_manifest_path.relative_to(contract.dataset_root)),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_manifest_internal_sha256": feature_manifest["manifest_sha256"],
        "feature_signature": feature_manifest["signature"],
        "input_content": expected_input_content,
        "input_content_verification": input_content_verification,
        "feature_rows": len(eligible_rows),
        "structured_exclusions": len(expected_excluded_rows),
        "official_denominator_rows": len(data.samples),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    gate_path = root / "EXTRACTION_GATE.json"
    write_manifest(gate_path, gate)
    pointer_path = root.parent / "LATEST_FULL.json"
    pointer = {
        "schema_version": 2,
        "backbone": backbone,
        "cache_root": str(root.relative_to(contract.dataset_root)),
        "manifest": str(feature_manifest_path.relative_to(contract.dataset_root)),
        "manifest_sha256": sha256_file(feature_manifest_path),
        "extraction_gate": str(gate_path.relative_to(contract.dataset_root)),
        "gate_sha256": sha256_file(gate_path),
        "contract_sha256": contract.sha256,
        "input_content": expected_input_content,
    }
    write_manifest(pointer_path, pointer)
    return gate
