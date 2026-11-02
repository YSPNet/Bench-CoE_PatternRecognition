"""Generate and verify the AF-004/AF-006 provenance remediation record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import ExperimentContract, load_contract, sha256_file


DEFAULT_REPORT = Path("outputs/bench_coe_query/audit/AF004_AF006_REMEDIATION.json")
DEFAULT_PROBE_ROOT = Path("outputs/bench_coe_query/audit/af006_full_state_probe")
ROUND3_ISSUES = Path(
    "outputs/bench_coe_query/audit_history/prompt14_round3/ISSUES.json"
)
ROUND4_ISSUES = Path(
    "outputs/bench_coe_query/audit_history/prompt14_round4/ISSUES.json"
)
INITIAL_SANITATION = Path(
    "outputs/bench_coe_query/audit_history/prompt14_round4/AF010_TARGET_FREE_REFREEZE.json"
)
STATE_DIGEST_PREFIX = "sha256-state-dict-v2:"
EXPECTED_LEGACY_NULL_LINES = {
    0: (1, 7),
    1: (1,),
    2: (1,),
    3: (1, 2),
    4: (1, 3),
}
GRADIENT_FIELDS = ("gradient_norm_mean", "gradient_norm_max")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(
            line,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _relative(contract: ExperimentContract, path: Path) -> str:
    return path.resolve().relative_to(contract.dataset_root).as_posix()


def _run_dir(contract: ExperimentContract, run_id: str, seed: int) -> Path:
    return (
        contract.dataset_root
        / "outputs/bench_coe_query/runs/bert/vlrb_adapted_v1"
        / run_id
        / f"seed-{seed}"
    )


def _checkpoint_scaler_evidence(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    resume = checkpoint.get("extra", {}).get("resume_state", {})
    scaler = resume.get("grad_scaler_state")
    if not isinstance(scaler, dict):
        raise RuntimeError(f"checkpoint lacks GradScaler state: {path}")
    required = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(scaler) != required:
        raise RuntimeError(f"unexpected GradScaler state fields: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "grad_scaler_state": {
            key: float(value) if key not in {"growth_interval", "_growth_tracker"} else int(value)
            for key, value in scaler.items()
        },
    }


def _find_issue(document: Mapping[str, Any], issue_id: str) -> Mapping[str, Any]:
    for issue in document.get("issues", []):
        if issue.get("id") == issue_id:
            return issue
    raise RuntimeError(f"archived audit does not contain {issue_id}")


def _legacy_amp_ledger(
    contract: ExperimentContract,
    run_id: str,
    referenced_files: dict[str, str],
) -> dict[str, Any]:
    round3_path = contract.dataset_root / ROUND3_ISSUES
    round4_path = contract.dataset_root / ROUND4_ISSUES
    sanitation_path = contract.dataset_root / INITIAL_SANITATION
    round3 = _strict_json(round3_path)
    round4 = _strict_json(round4_path)
    sanitation = _strict_json(sanitation_path)
    original_issue = _find_issue(round3, "AF-004")
    current_issue = _find_issue(round4, "AF-004")
    if "Infinity" not in str(original_issue.get("evidence", "")):
        raise RuntimeError("round-3 AF-004 does not preserve the original Infinity category")
    if sanitation.get("bert_history_nonfinite_values_replaced_with_null") != 16:
        raise RuntimeError("initial sanitation report does not record 16 replacements")

    for source in (round3_path, round4_path, sanitation_path):
        referenced_files[_relative(contract, source)] = sha256_file(source)

    entries: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    checkpoint_evidence: list[dict[str, Any]] = []
    for seed, expected_lines in EXPECTED_LEGACY_NULL_LINES.items():
        run = _run_dir(contract, run_id, seed)
        history_path = run / "training_history.jsonl"
        records = _strict_jsonl(history_path)
        observed_nulls = []
        for line_number, record in enumerate(records, 1):
            for field in GRADIENT_FIELDS:
                if record.get(field) is None:
                    observed_nulls.append((line_number, field))
        expected_nulls = [
            (line_number, field)
            for line_number in expected_lines
            for field in GRADIENT_FIELDS
        ]
        if observed_nulls != expected_nulls:
            raise RuntimeError(
                f"legacy BERT null inventory changed for seed {seed}: "
                f"observed={observed_nulls}, expected={expected_nulls}"
            )
        history_relative = _relative(contract, history_path)
        history_sha256 = sha256_file(history_path)
        referenced_files[history_relative] = history_sha256
        histories.append(
            {
                "seed": seed,
                "path": history_relative,
                "sha256": history_sha256,
                "records": len(records),
                "legacy_null_fields": len(observed_nulls),
                "strict_finite_json": True,
            }
        )
        for line_number, field in expected_nulls:
            entries.append(
                {
                    "seed": seed,
                    "history_path": history_relative,
                    "line": line_number,
                    "epoch": int(records[line_number - 1]["epoch"]),
                    "json_pointer": f"/{field}",
                    "original_token": "Infinity",
                    "original_value_class": "positive_infinity",
                    "current_value": None,
                    "sanitation_reason": (
                        "RFC-8259 JSON forbids non-finite numeric literals; preserve an "
                        "explicit null plus this immutable typed migration entry"
                    ),
                    "legacy_amp_optimizer_step_skipped_status": (
                        "unknown_not_recoverable_from_retained_epoch_summary"
                    ),
                }
            )
        for kind in ("best", "last"):
            checkpoint_path = run / f"checkpoints/{kind}/checkpoint.pt"
            evidence = _checkpoint_scaler_evidence(checkpoint_path)
            evidence["path"] = _relative(contract, checkpoint_path)
            evidence["seed"] = seed
            evidence["kind"] = kind
            referenced_files[evidence["path"]] = evidence["sha256"]
            checkpoint_evidence.append(evidence)

    bert_source = contract.dataset_root / "routers/bench_coe_query/bert_router.py"
    bert_text = bert_source.read_text(encoding="utf-8")
    required_future_fields = (
        '"gradient_norm_status"',
        '"gradient_norm_nonfinite_steps"',
        '"optimizer_steps_skipped"',
        '"scale_start"',
        '"scale_end"',
        '"skipped_step_events"',
    )
    future_instrumentation = all(field in bert_text for field in required_future_fields)
    referenced_files[_relative(contract, bert_source)] = sha256_file(bert_source)
    passed = bool(
        len(entries) == 16
        and sum(item["legacy_null_fields"] for item in histories) == 16
        and len(checkpoint_evidence) == 10
        and future_instrumentation
    )
    return {
        "issue_id": "AF-004",
        "passed": passed,
        "migration_entries": entries,
        "migration_entries_sha256": _canonical_sha256({"entries": entries}),
        "histories": histories,
        "retained_checkpoint_scaler_states": checkpoint_evidence,
        "original_category_evidence": {
            "round3_issue_path": _relative(contract, round3_path),
            "round3_issue_file_sha256": sha256_file(round3_path),
            "round3_issue_status": original_issue.get("status"),
            "initial_sanitation_path": _relative(contract, sanitation_path),
            "initial_sanitation_file_sha256": sha256_file(sanitation_path),
            "replacement_count": 16,
        },
        "round4_finding_evidence": {
            "path": _relative(contract, round4_path),
            "sha256": sha256_file(round4_path),
            "status": current_issue.get("status"),
        },
        "legacy_uncertainty": {
            "exact_skipped_step_events_recoverable": False,
            "fabricated_status_used": False,
            "interpretation": (
                "Retained best/last scaler states are recorded verbatim, but they do not "
                "uniquely recover every historical per-step transition. No no-skip claim is made."
            ),
        },
        "future_training_instrumentation": {
            "passed": future_instrumentation,
            "source": _relative(contract, bert_source),
            "source_sha256": sha256_file(bert_source),
            "per_epoch_skipped_step_count": True,
            "scale_start_end_min_max": True,
            "typed_gradient_norm_status": True,
            "skipped_step_event_global_steps_and_scale_transitions": True,
            "strict_json_allow_nan_false": True,
        },
    }


def _load_index(path: Path) -> list[dict[str, Any]]:
    return _strict_jsonl(path)


def _feature_comparison(probe_root: Path, full_root: Path) -> dict[str, Any]:
    probe = np.load(probe_root / "merged/features.npy", allow_pickle=False).astype(np.float32)
    probe_index = _load_index(probe_root / "merged/index.jsonl")
    full = np.load(full_root / "merged/features.npy", mmap_mode="r", allow_pickle=False)
    full_index = _load_index(full_root / "merged/index.jsonl")
    full_position = {
        int(record["row_index"]): position for position, record in enumerate(full_index)
    }
    rows = [int(record["row_index"]) for record in probe_index]
    if len(rows) != 32 or len(set(rows)) != 32 or any(row not in full_position for row in rows):
        raise RuntimeError("full-state probe does not cover 32 unique full-cache rows")
    reference = np.asarray(
        full[np.asarray([full_position[row] for row in rows], dtype=np.int64)],
        dtype=np.float32,
    )
    difference = probe - reference
    probe_norm = np.linalg.norm(probe, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    denominator = np.maximum(
        probe_norm * reference_norm, np.finfo(np.float32).tiny
    )
    cosine = (probe * reference).sum(axis=1) / denominator
    relative_l2 = np.linalg.norm(difference, axis=1) / np.maximum(
        reference_norm, np.finfo(np.float32).tiny
    )
    ids_equal = [record["sample_id"] for record in probe_index] == [
        full_index[full_position[row]]["sample_id"] for row in rows
    ]
    passed = bool(
        ids_equal
        and float(cosine.min()) >= 0.9999
        and float(relative_l2.max(initial=0.0)) <= 0.01
    )
    return {
        "passed": passed,
        "rows": len(rows),
        "sample_ids_equal": ids_equal,
        "bitwise_equal": bool(np.array_equal(probe, reference)),
        "maximum_absolute_difference": float(np.abs(difference).max(initial=0.0)),
        "minimum_cosine_similarity": float(cosine.min()),
        "maximum_relative_l2": float(relative_l2.max(initial=0.0)),
        "acceptance": {
            "minimum_cosine_similarity": 0.9999,
            "maximum_relative_l2": 0.01,
        },
    }


def _full_state_attestation(
    contract: ExperimentContract,
    probe_root: Path,
    referenced_files: dict[str, str],
) -> dict[str, Any]:
    backbones: dict[str, Any] = {}
    for backbone in ("tinyllava", "qwen3vl"):
        probe = contract.dataset_root / probe_root / backbone
        pointer_path = (
            contract.dataset_root
            / f"outputs/bench_coe_query/features/{backbone}/LATEST_FULL.json"
        )
        pointer = _strict_json(pointer_path)
        full_root = contract.dataset_root / str(pointer["cache_root"])
        resources = []
        comparisons: dict[str, Any] = {}
        for wave_name in ("train", "test"):
            wave = probe / f"wave-{wave_name}"
            for local_rank in range(4):
                path = wave / f"resource_rank_{local_rank:03d}.json"
                resource = _strict_json(path)
                before = str(resource.get("backbone_checksum_before", ""))
                after = str(resource.get("backbone_checksum_after", ""))
                resource_passed = bool(
                    resource.get("rank") == local_rank
                    and resource.get("world_size") == 4
                    and resource.get("backbone") == backbone
                    and resource.get("backbone_unchanged") is True
                    and resource.get("optimizer_parameters") == 0
                    and before.startswith(STATE_DIGEST_PREFIX)
                    and len(before) == len(STATE_DIGEST_PREFIX) + 64
                    and before == after
                )
                resources.append(
                    {
                        "rank_process_index": len(resources),
                        "wave": wave_name,
                        "local_rank": local_rank,
                        "world_size": 4,
                        "path": _relative(contract, path),
                        "sha256": sha256_file(path),
                        "rows_processed": int(resource.get("rows_processed_this_attempt", -1)),
                        "complete_state_dict_sha256_before": before,
                        "complete_state_dict_sha256_after": after,
                        "passed": resource_passed,
                    }
                )
                referenced_files[_relative(contract, path)] = sha256_file(path)
            comparisons[wave_name] = _feature_comparison(wave, full_root)
            for path in (
                wave / "merged/features.npy",
                wave / "merged/index.jsonl",
                wave / "merged/manifest.json",
                wave / "feature_signature.json",
            ):
                referenced_files[_relative(contract, path)] = sha256_file(path)
        full_files = (
            pointer_path,
            full_root / "merged/manifest.json",
            full_root / "merged/index.jsonl",
            full_root / "merged/features.npy",
        )
        for path in full_files:
            referenced_files[_relative(contract, path)] = sha256_file(path)
        legacy_resources = []
        for rank in range(8):
            path = full_root / f"resource_rank_{rank:03d}.json"
            resource = _strict_json(path)
            legacy_resources.append(
                {
                    "rank": rank,
                    "path": _relative(contract, path),
                    "sha256": sha256_file(path),
                    "legacy_checksum_before": resource.get("backbone_checksum_before"),
                    "legacy_checksum_after": resource.get("backbone_checksum_after"),
                }
            )
            referenced_files[_relative(contract, path)] = sha256_file(path)
        resource_passed = all(item["passed"] for item in resources)
        distinct = sorted(
            {item["complete_state_dict_sha256_before"] for item in resources}
        )
        backbones[backbone] = {
            "passed": bool(
                resource_passed
                and sum(item["rows_processed"] for item in resources) == 64
                and len(distinct) == 1
                and all(item["passed"] for item in comparisons.values())
            ),
            "probe_root": _relative(contract, probe),
            "execution_geometry": "two_independent_four_gpu_waves",
            "rank_processes": 8,
            "actual_forward_rows": sum(item["rows_processed"] for item in resources),
            "rank_attestations": resources,
            "cross_rank_complete_state_digests": distinct,
            "probe_to_archived_cache_comparisons": comparisons,
            "legacy_full_cache_resources_retained_unmodified": legacy_resources,
        }

    utils_path = contract.dataset_root / "routers/bench_coe_query/backbones/utils.py"
    tiny_path = contract.dataset_root / "routers/bench_coe_query/backbones/tinyllava.py"
    qwen_path = contract.dataset_root / "routers/bench_coe_query/backbones/qwen3vl.py"
    input_manifest = contract.dataset_root / "outputs/bench_coe_query/audit/INPUT_CONTENT_MANIFEST.json"
    for path in (utils_path, tiny_path, qwen_path, input_manifest):
        referenced_files[_relative(contract, path)] = sha256_file(path)
    return {
        "issue_id": "AF-006",
        "passed": all(value["passed"] for value in backbones.values()),
        "algorithm": {
            "id": STATE_DIGEST_PREFIX.removesuffix(":"),
            "source": _relative(contract, utils_path),
            "source_sha256": sha256_file(utils_path),
            "ordered_state_dict": True,
            "persistent_parameters_and_buffers": True,
            "state_dict_metadata": True,
            "dtype_shape_layout_stride_storage_offset": True,
            "raw_tensor_bytes": True,
            "bounded_cpu_transfer_chunks": True,
        },
        "input_content_manifest": {
            "path": _relative(contract, input_manifest),
            "sha256": sha256_file(input_manifest),
        },
        "backbones": backbones,
        "scope": {
            "actual_rank_process_forward_probes": 16,
            "per_backbone_execution": "two independent four-GPU waves over train/test rows",
            "archived_full_cache_rewritten": False,
            "legacy_moment_summaries_misrepresented_as_full_state": False,
            "interpretation": (
                "The original 30,493-row cache bytes and legacy resources remain immutable. "
                "New four-GPU train/test waves (eight independent processes and 64 rows per "
                "backbone) with identical pinned model/adapter inputs cryptographically attest "
                "every ordered state_dict byte before and after use and compare the resulting "
                "features to the archived full cache."
            ),
        },
    }


def build_remediation(
    contract: ExperimentContract,
    *,
    run_id: str,
    probe_root: Path = DEFAULT_PROBE_ROOT,
) -> dict[str, Any]:
    referenced_files: dict[str, str] = {}
    amp = _legacy_amp_ledger(contract, run_id, referenced_files)
    state = _full_state_attestation(contract, probe_root, referenced_files)
    this_source = Path(__file__).resolve()
    referenced_files[_relative(contract, this_source)] = sha256_file(this_source)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "af004_af006_provenance_remediation",
        "contract_sha256": contract.sha256,
        "run_id": run_id,
        "legacy_amp_diagnostics": amp,
        "complete_backbone_state_attestation": state,
        "referenced_files": dict(sorted(referenced_files.items())),
        "passed": amp["passed"] is True and state["passed"] is True,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    if payload["passed"] is not True:
        raise RuntimeError("AF-004/AF-006 remediation did not pass")
    return payload


def verify_remediation_report(
    contract: ExperimentContract,
    path: str | Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = contract.dataset_root / report_path
    report = _strict_json(report_path)
    expected_manifest = str(report.get("manifest_sha256", ""))
    unsigned = dict(report)
    unsigned.pop("manifest_sha256", None)
    observed_manifest = _canonical_sha256(unsigned)
    file_mismatches = []
    for relative, expected in report.get("referenced_files", {}).items():
        current = contract.dataset_root / str(relative)
        observed = sha256_file(current) if current.is_file() else None
        if observed != expected:
            file_mismatches.append(
                {"path": relative, "expected_sha256": expected, "observed_sha256": observed}
            )
    passed = bool(
        report.get("contract_sha256") == contract.sha256
        and report.get("passed") is True
        and report.get("legacy_amp_diagnostics", {}).get("passed") is True
        and report.get("complete_backbone_state_attestation", {}).get("passed") is True
        and expected_manifest == observed_manifest
        and not file_mismatches
    )
    return {
        "passed": passed,
        "path": _relative(contract, report_path),
        "file_sha256": sha256_file(report_path),
        "manifest_sha256": expected_manifest,
        "manifest_verified": expected_manifest == observed_manifest,
        "referenced_files_verified": len(report.get("referenced_files", {})),
        "referenced_file_mismatches": file_mismatches,
        "legacy_amp_migration_entries": len(
            report.get("legacy_amp_diagnostics", {}).get("migration_entries", [])
        ),
        "full_state_backbones_verified": sum(
            item.get("passed") is True
            for item in report.get("complete_backbone_state_attestation", {})
            .get("backbones", {})
            .values()
        ),
        "full_state_rank_processes_verified": sum(
            len(item.get("rank_attestations", []))
            for item in report.get("complete_backbone_state_attestation", {})
            .get("backbones", {})
            .values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "verify"))
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--run-id", default="full-vlrb-adapted-v1")
    parser.add_argument("--probe-root", default=str(DEFAULT_PROBE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    contract = load_contract(args.config)
    output = Path(args.output)
    if not output.is_absolute():
        output = contract.dataset_root / output
    if args.phase == "build":
        result = build_remediation(
            contract,
            run_id=args.run_id,
            probe_root=Path(args.probe_root),
        )
        _atomic_json(output, result)
        summary = verify_remediation_report(contract, output)
    else:
        summary = verify_remediation_report(contract, output)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
