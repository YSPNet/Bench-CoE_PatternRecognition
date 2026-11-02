"""E02 source-only OOD manifest and fit-provenance validation.

This module intentionally consumes only frozen registries and artifact metadata.
It resolves dataset membership through the canonical sample registry rather than
UID substrings and exposes a strict schema for future OOD fit artifacts.  Test
UIDs may be listed, but test labels and metrics have no representation here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from subject_routing.manifests import canonical_sha256, load_frozen_json


class OODScopeError(ValueError):
    """Raised when reconstructed OOD membership or provenance is unsafe."""


CANONICAL_HELD_OUT_BY_GROUP = {
    "general": "MMMU_DEV_VAL",
    "stem": "MathVision_MINI",
    "charts_ocr": "ChartQA_TEST",
}
CANONICAL_HELD_OUT = tuple(CANONICAL_HELD_OUT_BY_GROUP.values())
FIT_ROLES = ("train_seen", "dev_seen")
TEST_UID_ROLES = ("id_test", "ood_test")
EXCLUDED_ROLES = ("excluded_held_out_train", "excluded_held_out_dev")
ALL_OOD_ROLES = (*FIT_ROLES, *TEST_UID_ROLES, *EXCLUDED_ROLES)

# Exact canonical dataset-to-container bindings.  E02 uses these bindings only
# after resolving dataset membership through the frozen sample registry.  In
# particular, filenames and UID prefixes are never used to decide whether a
# source is held out.
BENCHMARK_FILE_BY_DATASET = {
    "AI2D_TEST": "BENCHMARKS/vqa_mc/ai2d_test_samples.jsonl",
    "ChartQA_TEST": "BENCHMARKS/ocr_qa/chartqa_test_samples.jsonl",
    "DocVQA_VAL": "BENCHMARKS/ocr_qa/docvqa_val_samples.jsonl",
    "HallusionBench": "BENCHMARKS/vqa_mc/hallusionbench_samples.jsonl",
    "InfoVQA_VAL": "BENCHMARKS/ocr_qa/infovqa_val_samples.jsonl",
    "MMBench_DEV_EN_V11": "BENCHMARKS/vqa_mc/mmbench_dev_en_v11_samples.jsonl",
    "MMMU_DEV_VAL": "BENCHMARKS/vqa_mc/mmmu_dev_val_samples.jsonl",
    "MMStar": "BENCHMARKS/vqa_mc/mmstar_samples.jsonl",
    "MathVerse_MINI": "BENCHMARKS/math/mathverse_mini_samples.jsonl",
    "MathVision_MINI": "BENCHMARKS/math/mathvision_mini_samples.jsonl",
    "MathVista_MINI": "BENCHMARKS/math/mathvista_mini_samples.jsonl",
    "OCRBench": "BENCHMARKS/ocr_qa/ocrbench_samples.jsonl",
    "RealWorldQA": "BENCHMARKS/vqa_oe/realworldqa_samples.jsonl",
    "TextVQA_VAL": "BENCHMARKS/vqa_oe/textvqa_val_samples.jsonl",
}
_SAMPLE_ID_PREFIX = re.compile(
    rb'^\{"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")'
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scoped_benchmark_prompts(
    root: Path,
    ordered_uids: Sequence[str],
    *,
    uid_to_dataset: Mapping[str, str],
    allowed_datasets: Sequence[str],
    forbidden_uids: Sequence[str] = (),
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Load only registry-authorized prompts from source JSONL containers.

    The JSONL container must be scanned to find requested rows, but a line is
    fully decoded/deserialized only after its leading ``sample_id`` matches the
    authorized UID set.  Containers for datasets outside ``allowed_datasets``
    are never opened.  This gives E02 an auditable source-only input path
    without constructing the repository-wide public view (which includes test
    and held-out records).
    """

    repository = root.resolve()
    ids = tuple(str(uid) for uid in ordered_uids)
    allowed = tuple(str(dataset) for dataset in allowed_datasets)
    if not ids or len(ids) != len(set(ids)):
        raise OODScopeError("scoped prompt UIDs must be non-empty and unique")
    if not allowed or len(allowed) != len(set(allowed)):
        raise OODScopeError("allowed prompt datasets must be non-empty and unique")
    if set(allowed).difference(BENCHMARK_FILE_BY_DATASET):
        raise OODScopeError("allowed prompt dataset lacks an exact container binding")
    if set(ids).intersection(map(str, forbidden_uids)):
        raise OODScopeError("forbidden UID requested from scoped prompt loader")

    requested_by_dataset: dict[str, set[str]] = {dataset: set() for dataset in allowed}
    for uid in ids:
        dataset = uid_to_dataset.get(uid)
        if dataset is None:
            raise OODScopeError(f"scoped prompt UID is absent from registry: {uid}")
        dataset = str(dataset)
        if dataset not in requested_by_dataset:
            raise OODScopeError(
                f"scoped prompt UID resolves outside allowed datasets: {uid} -> {dataset}"
            )
        requested_by_dataset[dataset].add(uid)

    found: dict[str, str] = {}
    opened: list[dict[str, Any]] = []
    for dataset in allowed:
        requested = requested_by_dataset[dataset]
        if not requested:
            continue
        relative = BENCHMARK_FILE_BY_DATASET[dataset]
        path = _resolve_inside(repository, relative)
        if not path.is_file():
            raise OODScopeError(f"scoped benchmark container is missing: {relative}")
        digest = hashlib.sha256()
        lines_scanned = 0
        selected_lines = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                digest.update(line)
                lines_scanned += 1
                match = _SAMPLE_ID_PREFIX.match(line)
                if match is None:
                    raise OODScopeError(
                        f"sample_id is not the first JSONL field: {relative}:{line_number}"
                    )
                try:
                    sample_id = str(json.loads(match.group(1).decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise OODScopeError(
                        f"invalid scoped sample_id: {relative}:{line_number}"
                    ) from exc
                if sample_id not in requested:
                    # Deliberately do not decode the rest of an unrequested row.
                    continue
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise OODScopeError(
                        f"invalid requested benchmark row: {relative}:{line_number}"
                    ) from exc
                if str(record.get("dataset")) != dataset:
                    raise OODScopeError(
                        f"requested benchmark row disagrees with registry dataset: {sample_id}"
                    )
                prompt = record.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise OODScopeError(f"requested benchmark prompt is empty: {sample_id}")
                if sample_id in found:
                    raise OODScopeError(f"duplicate requested benchmark UID: {sample_id}")
                found[sample_id] = prompt
                selected_lines += 1
        opened.append(
            {
                "dataset": dataset,
                "path": relative,
                "sha256": digest.hexdigest(),
                "lines_scanned": lines_scanned,
                "requested_rows_deserialized": selected_lines,
                "unrequested_rows_deserialized": 0,
            }
        )
    missing = [uid for uid in ids if uid not in found]
    if missing:
        raise OODScopeError(f"requested benchmark UID was not found: {missing[0]}")
    prompts = tuple(found[uid] for uid in ids)
    return prompts, {
        "kind": "subject_routing_scoped_source_prompt_read_audit",
        "selection": "canonical_registry_UID_and_dataset_exact_lookup",
        "requested_rows": len(ids),
        "requested_uid_sha256": canonical_sha256(list(ids)),
        "allowed_datasets": list(allowed),
        "opened_containers": opened,
        "opened_dataset_count": len(opened),
        "outside_allowed_container_open_count": 0,
        "unrequested_rows_deserialized": 0,
        "forbidden_UIDs_requested": 0,
        "represented_fields": ["sample_id", "dataset", "prompt"],
        "gold_labels_expert_outputs_or_costs_represented": False,
    }


def _resolve_inside(root: Path, relative: str | Path) -> Path:
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OODScopeError(f"path escapes repository: {relative}") from exc
    return path


def _registry_rows(split_manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    rows = split_manifest.get("sample_registry", {}).get("rows")
    if not isinstance(rows, list) or not rows:
        raise OODScopeError("split manifest has no canonical sample registry rows")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = str(row["sample_id"])
        if uid in result:
            raise OODScopeError(f"duplicate canonical sample ID: {uid}")
        result[uid] = {
            "dataset": str(row["dataset"]),
            "router_split": str(row["router_split"]),
        }
    return result


def _role_audit(
    *,
    role: str,
    declaration: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, str]],
    allowed_datasets: set[str],
    expected_split: str,
) -> dict[str, Any]:
    uids = tuple(str(item) for item in declaration["ordered_sample_ids"])
    if not uids or len(uids) != len(set(uids)):
        raise OODScopeError(f"{role} UIDs must be non-empty and unique")
    if len(uids) != int(declaration["count"]):
        raise OODScopeError(f"{role} count differs from UID list")
    if canonical_sha256(list(uids)) != str(declaration["ordered_uid_sha256"]):
        raise OODScopeError(f"{role} UID digest mismatch")
    unknown = set(uids).difference(registry)
    if unknown:
        raise OODScopeError(f"{role} contains unknown registry UIDs")

    observed_datasets = Counter(registry[uid]["dataset"] for uid in uids)
    observed_splits = {registry[uid]["router_split"] for uid in uids}
    if set(observed_datasets).difference(allowed_datasets):
        raise OODScopeError(f"{role} crosses its canonical dataset partition")
    if observed_splits != {expected_split}:
        raise OODScopeError(f"{role} crosses its official router split")
    declared_counts = {
        str(key): int(value) for key, value in declaration["dataset_counts"].items()
    }
    if dict(sorted(observed_datasets.items())) != dict(sorted(declared_counts.items())):
        raise OODScopeError(f"{role} dataset counts disagree with canonical registry")
    return {
        "role": role,
        "official_router_split": expected_split,
        "count": len(uids),
        "dataset_counts": dict(sorted(observed_datasets.items())),
        "ordered_uid_sha256": canonical_sha256(list(uids)),
        "ordered_sample_ids": list(uids),
        "membership_resolution": "canonical_sample_registry_exact_ID_lookup",
        "labels_included": False,
        "metrics_included": False,
    }


def build_ood11_3_manifest(
    root: Path,
    *,
    split_manifest_path: str | Path,
    pretest_registry_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic, label-free OOD protocol manifest."""

    root = root.resolve()
    split_path = _resolve_inside(root, split_manifest_path)
    pretest_path = _resolve_inside(root, pretest_registry_path)
    split_manifest = load_frozen_json(split_path)
    pretest = load_frozen_json(pretest_path)
    if pretest.get("registry_version") != "e01-v2":
        raise OODScopeError("E02 requires the archived-history-aware E01-v2 registry")
    if pretest.get("test_access_log") != []:
        raise OODScopeError("E01 pretest registry has test access")

    environment = split_manifest.get("environments", {}).get("ood11_3")
    if not isinstance(environment, dict):
        raise OODScopeError("split manifest has no ood11_3 environment")
    held_out = tuple(str(item) for item in environment["held_out_datasets"])
    if held_out != CANONICAL_HELD_OUT:
        raise OODScopeError(
            f"held-out IDs are not exact canonical names: {held_out!r}"
        )
    seen = tuple(str(item) for item in environment["fit_datasets"])
    if len(seen) != 11 or len(set(seen)) != 11 or set(seen).intersection(held_out):
        raise OODScopeError("seen11/heldout3 partition is invalid")

    registry = _registry_rows(split_manifest)
    observed_datasets = {row["dataset"] for row in registry.values()}
    if set(seen).union(held_out) != observed_datasets:
        raise OODScopeError("seen11/heldout3 do not exactly cover registry datasets")
    roles = environment["roles"]
    if set(roles) != set(ALL_OOD_ROLES):
        raise OODScopeError("ood11_3 roles differ from the frozen six-role protocol")

    role_specs = {
        "train_seen": (set(seen), "train"),
        "dev_seen": (set(seen), "dev"),
        "id_test": (set(seen), "test"),
        "ood_test": (set(held_out), "test"),
        "excluded_held_out_train": (set(held_out), "train"),
        "excluded_held_out_dev": (set(held_out), "dev"),
    }
    audited_roles = {
        role: _role_audit(
            role=role,
            declaration=roles[role],
            registry=registry,
            allowed_datasets=allowed,
            expected_split=official_split,
        )
        for role, (allowed, official_split) in role_specs.items()
    }

    role_uid_sets = {
        role: set(record["ordered_sample_ids"])
        for role, record in audited_roles.items()
    }
    if role_uid_sets["train_seen"] & role_uid_sets["excluded_held_out_train"]:
        raise OODScopeError("seen and held-out official train roles overlap")
    if role_uid_sets["dev_seen"] & role_uid_sets["excluded_held_out_dev"]:
        raise OODScopeError("seen and held-out official dev roles overlap")
    if role_uid_sets["id_test"] & role_uid_sets["ood_test"]:
        raise OODScopeError("ID and OOD test UID roles overlap")
    official = split_manifest["official_router_splits"]
    partitions = {
        "train": ("train_seen", "excluded_held_out_train"),
        "dev": ("dev_seen", "excluded_held_out_dev"),
        "test": ("id_test", "ood_test"),
    }
    for split, (left, right) in partitions.items():
        expected = set(map(str, official[split]["ordered_sample_ids"]))
        if role_uid_sets[left].union(role_uid_sets[right]) != expected:
            raise OODScopeError(f"OOD roles do not exactly partition official {split}")

    selected_methods = list(
        map(str, pretest["e02_rebuild_plan"]["mandatory_candidate_order"])
    )
    return {
        "schema_version": 1,
        "kind": "subject_routing_reconstructed_ood11_3_manifest",
        "environment_id": "ood11_3",
        "status": "FROZEN_SCOPE_TEST_UIDS_LABEL_FREE",
        "claim": "reconstructed_Appendix_I_4_dataset_holdout_not_author_sample_manifest",
        "paper": {
            "title": "VL-RouterBench: A Benchmark for Vision-Language Model Routing",
            "arxiv": "https://arxiv.org/abs/2512.23562",
            "appendix": "I.4 Generalization on Held-out Dataset",
            "role_mapping": dict(CANONICAL_HELD_OUT_BY_GROUP),
        },
        "sources": {
            "split_manifest": {
                "path": Path(split_manifest_path).as_posix(),
                "sha256": sha256_file(split_path),
                "payload_sha256": split_manifest["payload_sha256"],
            },
            "pretest_registry": {
                "path": Path(pretest_registry_path).as_posix(),
                "sha256": sha256_file(pretest_path),
                "payload_sha256": pretest["payload_sha256"],
                "registry_version": pretest["registry_version"],
            },
        },
        "dataset_partition": {
            "seen11": list(seen),
            "heldout3": list(held_out),
            "resolved_by": "sample_id_to_dataset_from_frozen_sample_registry",
            "UID_prefix_or_fuzzy_string_filter_used": False,
        },
        "roles": audited_roles,
        "fit_scope": {
            "all_fit_components_allowed_UID_universe": "train_seen",
            "source_dev_selection_allowed_UID_universe": "dev_seen",
            "heldout_train_or_dev_input_label_teacher_statistics_allowed": False,
            "test_fit_allowed": False,
            "selected_method_ids": selected_methods,
        },
        "cache_reuse": {
            "allowed": "frozen_per_input_pretrained_raw_vectors_only",
            "fit_operation_on_allowed_cache": False,
            "id14_fitted_scaler_PCA_graph_taxonomy_student_profile_policy_allowed": False,
        },
        "test_boundary": {
            "test_UIDs_listed": True,
            "test_inputs_loaded": False,
            "test_labels_included": False,
            "test_metrics_included": False,
            "test_selection_allowed": False,
            "test_access_log": [],
        },
    }


def validate_ood11_3_manifest(
    root: Path,
    manifest_path: str | Path,
    *,
    split_manifest_path: str | Path,
    pretest_registry_path: str | Path,
) -> dict[str, Any]:
    """Rebuild the deterministic payload and compare it with the frozen file."""

    path = _resolve_inside(root.resolve(), manifest_path)
    frozen = load_frozen_json(path)
    expected = build_ood11_3_manifest(
        root,
        split_manifest_path=split_manifest_path,
        pretest_registry_path=pretest_registry_path,
    )
    payload = dict(frozen)
    payload.pop("payload_sha256", None)
    if canonical_sha256(payload) != canonical_sha256(expected):
        raise OODScopeError("frozen ood11_3 manifest differs from canonical rebuild")
    return frozen


def test_uid_list_payload(manifest: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Return an explicitly label-free final-test UID list payload."""

    if role not in TEST_UID_ROLES:
        raise OODScopeError(f"not a test UID role: {role}")
    record = manifest["roles"][role]
    return {
        "schema_version": 1,
        "kind": "subject_routing_label_free_test_UID_list",
        "environment_id": "ood11_3",
        "role": role,
        "official_router_split": "test",
        "count": int(record["count"]),
        "dataset_counts": dict(record["dataset_counts"]),
        "ordered_uid_sha256": str(record["ordered_uid_sha256"]),
        "ordered_sample_ids": list(record["ordered_sample_ids"]),
        "labels_included": False,
        "metrics_included": False,
        "test_access_log": [],
    }


def validate_ood_fit_artifact(
    artifact: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one future E02 fit artifact against exact source UID universes."""

    if artifact.get("kind") != "subject_routing_ood11_3_fit_artifact":
        raise OODScopeError("unexpected OOD fit artifact kind")
    if artifact.get("environment_id") != "ood11_3" or artifact.get("source_only") is not True:
        raise OODScopeError("OOD fit artifact must be source-only ood11_3")
    allowed_methods = set(manifest["fit_scope"]["selected_method_ids"])
    if str(artifact.get("method_id")) not in allowed_methods:
        raise OODScopeError("OOD fit artifact method was not frozen in E01")
    stages = artifact.get("fit_provenance")
    if not isinstance(stages, list) or not stages:
        raise OODScopeError("OOD fit artifact needs explicit per-stage UID provenance")

    train = set(manifest["roles"]["train_seen"]["ordered_sample_ids"])
    dev = set(manifest["roles"]["dev_seen"]["ordered_sample_ids"])
    held = set(manifest["roles"]["excluded_held_out_train"]["ordered_sample_ids"])
    held.update(manifest["roles"]["excluded_held_out_dev"]["ordered_sample_ids"])
    tests = set(manifest["roles"]["id_test"]["ordered_sample_ids"])
    tests.update(manifest["roles"]["ood_test"]["ordered_sample_ids"])
    audited: list[dict[str, Any]] = []
    for stage in stages:
        component = str(stage.get("component", ""))
        role = str(stage.get("role", ""))
        uids = tuple(str(item) for item in stage.get("ordered_uids", []))
        if not component or not uids or len(uids) != len(set(uids)):
            raise OODScopeError("fit provenance component/UIDs are invalid")
        if canonical_sha256(list(uids)) != str(stage.get("ordered_uid_sha256")):
            raise OODScopeError(f"fit UID digest mismatch: {component}")
        universe = dev if component == "dev_selection" else train
        expected_role = "dev_seen" if component == "dev_selection" else "train_seen"
        if role != expected_role or set(uids).difference(universe):
            raise OODScopeError(f"fit stage exceeds its source-only scope: {component}")
        if set(uids).intersection(held) or set(uids).intersection(tests):
            raise OODScopeError(f"held-out/test UID entered fit stage: {component}")
        audited.append(
            {
                "component": component,
                "role": role,
                "rows": len(uids),
                "ordered_uid_sha256": canonical_sha256(list(uids)),
                "heldout_or_test_rows": 0,
            }
        )

    for parent in artifact.get("parent_artifacts", []):
        fit_operation = bool(parent.get("fit_operation", True))
        environment = str(parent.get("environment_id", ""))
        layer = str(parent.get("layer", ""))
        raw_cache = layer == "pretrained_per_input_raw_vectors" and not fit_operation
        if environment == "id14" and not raw_cache:
            raise OODScopeError("id14-fitted parent artifact contaminates OOD rebuild")
        if not fit_operation and not raw_cache:
            raise OODScopeError("only raw pretrained per-input vectors may bypass refit")

    return {
        "method_id": str(artifact["method_id"]),
        "stage_count": len(audited),
        "stages": audited,
        "heldout_fit_UIDs": 0,
        "test_fit_UIDs": 0,
        "id14_fitted_parent_artifacts": 0,
        "status": "PASS_SOURCE_ONLY_SCOPE",
    }


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _walk_strings(item)


def audit_pre_e02_candidate_artifacts(
    root: Path,
    pretest_registry: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Inventory bounded predecessors without promoting them to full E02 fits."""

    uid_to_role: dict[str, str] = {}
    for role, record in manifest["roles"].items():
        for uid in record["ordered_sample_ids"]:
            uid_to_role[str(uid)] = str(role)
    held_roles = set(EXCLUDED_ROLES) | set(TEST_UID_ROLES)
    records: list[dict[str, Any]] = []
    for candidate in pretest_registry["candidate_audit"]["records"]:
        if candidate["comparison_table"] != "confirmatory_strict":
            continue
        for artifact in candidate["artifacts"]:
            relative = str(artifact["path"])
            path = _resolve_inside(root, relative)
            value = json.loads(path.read_text(encoding="utf-8"))
            strings = set(_walk_strings(value))
            explicit = sorted(strings.intersection(uid_to_role))
            role_counts = Counter(uid_to_role[uid] for uid in explicit)
            held_count = sum(role_counts[role] for role in held_roles)
            if held_count:
                raise OODScopeError(f"bounded predecessor contains held-out UID: {relative}")
            records.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "path": relative,
                    "sha256": sha256_file(path),
                    "explicit_registry_UIDs": len(explicit),
                    "explicit_UID_role_counts": dict(sorted(role_counts.items())),
                    "explicit_heldout_UIDs": held_count,
                    "status": "PRE_E02_BOUNDED_EVIDENCE_NOT_FULL_REBUILD",
                }
            )
    return records
