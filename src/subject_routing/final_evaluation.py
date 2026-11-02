"""Frozen, label-separated final evaluation utilities for subject routing."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from routers.utils.rank_score import rank_score
from subject_routing.manifests import canonical_sha256, load_frozen_json

from .clustering.semantic_tree import (
    PrunedTreeArtifact,
    SemanticTreeArtifact,
    assign_frozen_tree,
)
from .clustering.spherical_kmeans import SphericalKMeansArtifact, assign_frozen
from .data_contracts import EvaluationLabels
from .folds import PublicGroupingRecord, connected_groups
from .leakage_guard import (
    EvaluationLabelStore,
    Operation,
    UIDAccessAudit,
    UIDScopeError,
)
from .ood_rebuild import BENCHMARK_FILE_BY_DATASET


class FinalEvaluationError(ValueError):
    """Raised when the frozen final protocol or its outputs are inconsistent."""


FINAL_CANDIDATES = (
    "b02_kmeans_k4",
    "b03_hierarchical_semantic_tree",
    "n1_matched_fixed_k2",
    "n1_routing_gain_tree_dev_frozen",
    "pi0_global_strict",
)
PARTITION_SEEDS = (1013, 1021, 1042, 1087, 3026)
BUDGET_IDS = ("q000", "q010", "q025", "q050", "q075", "q090", "q100")
_SAMPLE_ID_PREFIX = re.compile(rb'^\{"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")')


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_sha256(uids: Sequence[str]) -> str:
    return canonical_sha256([str(uid) for uid in uids])


def require_sha256(value: Any, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FinalEvaluationError(f"{name} is not a lowercase SHA256")
    return digest


def load_envelope(path: str | Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    payload = value.get("payload") if isinstance(value, dict) else None
    if not isinstance(payload, dict):
        raise FinalEvaluationError(f"artifact is not a frozen envelope: {source}")
    if value.get("payload_sha256") != canonical_sha256(payload):
        raise FinalEvaluationError(f"artifact payload hash mismatch: {source}")
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise FinalEvaluationError(f"artifact kind mismatch: {source}")
    return payload


def _verify_reference(root: Path, reference: Mapping[str, Any], *, envelope: bool) -> dict[str, Any]:
    relative = reference.get("path")
    if relative is None:
        return {
            "path": None,
            "payload_sha256": require_sha256(reference.get("sha256"), name="null artifact"),
            "file_sha256": None,
            "bytes": 0,
        }
    path = root / str(relative)
    if not path.is_file():
        raise FinalEvaluationError(f"frozen artifact is missing: {relative}")
    file_sha = sha256_file(path)
    expected_file = reference.get("file_sha256")
    if expected_file is not None and file_sha != expected_file:
        raise FinalEvaluationError(f"frozen artifact file hash changed: {relative}")
    payload_sha = reference.get("payload_sha256", reference.get("sha256"))
    payload_sha = require_sha256(payload_sha, name=f"artifact payload {relative}")
    if envelope:
        payload = load_envelope(path)
        if canonical_sha256(payload) != payload_sha:
            raise FinalEvaluationError(f"frozen artifact payload changed: {relative}")
    return {
        "path": str(relative),
        "payload_sha256": payload_sha,
        "file_sha256": file_sha,
        "bytes": path.stat().st_size,
    }


def validate_final_pool(
    root: str | Path, pool_path: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the exact E04 pool and every deploy artifact without test data."""

    repository = Path(root).resolve()
    source = repository / Path(pool_path)
    pool = load_frozen_json(source)
    if (
        pool.get("kind") != "subject_routing_E04_final_dev_candidate_pool"
        or pool.get("status") != "FROZEN_AFTER_ALL_DEV_CHOICES_BEFORE_FIRST_TEST_ACCESS"
        or pool.get("retained_candidates") != list(FINAL_CANDIDATES)
        or pool.get("locked_route_count") != 175
        or pool.get("test_inputs_loaded_or_routed") != 0
        or pool.get("test_labels_read") != 0
        or pool.get("test_metrics_computed") != 0
    ):
        raise FinalEvaluationError("E04 final candidate pool is not test-closed and complete")
    routes = pool.get("locked_routes")
    if not isinstance(routes, list) or len(routes) != 175:
        raise FinalEvaluationError("E04 final route list must contain exactly 175 routes")

    expected = {
        (candidate, seed, budget)
        for candidate in FINAL_CANDIDATES
        for seed in PARTITION_SEEDS
        for budget in BUDGET_IDS
    }
    observed: set[tuple[str, int, str]] = set()
    inventory: dict[str, dict[str, Any]] = {}
    budget_signature: str | None = None
    for route in routes:
        key = (
            str(route.get("candidate_id")),
            int(route.get("partition_seed")),
            str(route.get("budget_id")),
        )
        if key in observed or key not in expected or route.get("test_feedback_used") is not False:
            raise FinalEvaluationError(f"invalid, duplicate, or test-informed final route: {key}")
        observed.add(key)
        policy_ref = _verify_reference(repository, route["policy_artifact"], envelope=True)
        profile_ref = _verify_reference(repository, route["profile_artifact"], envelope=True)
        subject_ref = _verify_reference(repository, route["subject_artifact"], envelope=True)
        assigner_ref = _verify_reference(repository, route["assigner_artifact"], envelope=True)
        for record in (policy_ref, profile_ref, subject_ref, assigner_ref):
            if record["path"] is not None:
                inventory.setdefault(record["path"], record)

        policy = load_envelope(repository / policy_ref["path"])
        if policy.get("kind") != "subject_routing_strict_train_profile_policy_tables":
            raise FinalEvaluationError("final policy is not strict subject routing")
        if policy.get("method_id") != route.get("variant_id"):
            raise FinalEvaluationError("final policy method/variant link differs")
        strategy = str(route.get("strategy_id"))
        budget = str(route.get("budget_id"))
        decisions = policy.get("decisions", {}).get(strategy, {}).get(budget)
        if not isinstance(decisions, dict) or not decisions:
            raise FinalEvaluationError("final route is absent from its policy table")
        if any(
            not isinstance(record, dict) or not isinstance(record.get("expert_id"), str)
            for record in decisions.values()
        ):
            raise FinalEvaluationError("final policy contains an invalid state decision")
        signature = str(policy.get("budget_signature", ""))
        if budget_signature is None:
            budget_signature = signature
        elif signature != budget_signature:
            raise FinalEvaluationError("final policies do not share one budget signature")
    if observed != expected:
        raise FinalEvaluationError("final route Cartesian product is incomplete")
    return pool, [inventory[path] for path in sorted(inventory)]


def load_route_policy(root: str | Path, route: Mapping[str, Any]) -> dict[str, Any]:
    repository = Path(root).resolve()
    reference = route["policy_artifact"]
    path = repository / str(reference["path"])
    if sha256_file(path) != str(reference["file_sha256"]):
        raise FinalEvaluationError("policy file changed after E04")
    payload = load_envelope(path, expected_kind="subject_routing_strict_train_profile_policy_tables")
    if canonical_sha256(payload) != str(reference["payload_sha256"]):
        raise FinalEvaluationError("policy payload changed after E04")
    return payload


def assign_frozen_variant(
    root: str | Path,
    route: Mapping[str, Any],
    features: np.ndarray,
    *,
    feature_artifact_sha256: str,
) -> tuple[str, ...]:
    """Assign test inputs with a frozen input-only artifact; never accepts labels."""

    repository = Path(root).resolve()
    assigner_ref = route["assigner_artifact"]
    subject_ref = route["subject_artifact"]
    if assigner_ref.get("path") is None:
        if subject_ref.get("path") is not None or int(route["effective_K"]) != 1:
            raise FinalEvaluationError("global route has inconsistent artifact metadata")
        return tuple("root" for _ in range(np.asarray(features).shape[0]))
    assigner_path = repository / str(assigner_ref["path"])
    payload = load_envelope(assigner_path)
    if canonical_sha256(payload) != str(assigner_ref["sha256"]):
        raise FinalEvaluationError("assigner payload differs from E04 route")
    kind = str(payload.get("kind"))
    if kind == "subject_routing_input_only_spherical_kmeans":
        artifact = SphericalKMeansArtifact.load(assigner_path)
        batch = assign_frozen(
            artifact,
            features,
            feature_artifact_sha256=feature_artifact_sha256,
        )
    elif kind == "subject_routing_frozen_semantic_tree_pruning":
        subject_path = repository / str(subject_ref["path"])
        tree_payload = load_envelope(subject_path)
        if canonical_sha256(tree_payload) != str(subject_ref["sha256"]):
            raise FinalEvaluationError("tree payload differs from E04 route")
        tree = SemanticTreeArtifact.load(subject_path)
        pruning = PrunedTreeArtifact.load(assigner_path, tree)
        batch = assign_frozen_tree(
            tree,
            pruning,
            features,
            feature_artifact_sha256=feature_artifact_sha256,
        )
    else:
        raise FinalEvaluationError(f"unsupported final assigner kind: {kind}")
    states = tuple(str(state) for state in batch.states)
    if len(states) != np.asarray(features).shape[0]:
        raise FinalEvaluationError("frozen assigner returned the wrong row count")
    return states


def route_frozen_policy(
    states: Sequence[str],
    policy: Mapping[str, Any],
    *,
    strategy_id: str,
    budget_id: str,
    expert_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Perform a strict state-table lookup with an explicit train-global fallback."""

    decisions = policy.get("decisions", {}).get(str(strategy_id), {}).get(str(budget_id))
    if not isinstance(decisions, dict) or not decisions:
        raise FinalEvaluationError("requested route is absent from frozen policy")
    global_state = str(policy.get("global_state_id", "__global__"))
    fallback = decisions.get(global_state)
    pool = tuple(str(expert) for expert in expert_ids)
    if not pool or len(pool) != len(set(pool)):
        raise FinalEvaluationError("expert pool must be unique and non-empty")
    budget_records = {str(record["budget_id"]): record for record in policy.get("budgets", [])}
    budget = budget_records.get(str(budget_id))
    if budget is None:
        raise FinalEvaluationError("budget is absent from frozen policy")
    allowed = {str(expert) for expert in budget["allowed_experts"]}
    selected: list[str] = []
    reasons: list[str] = []
    state_to_expert: dict[str, str] = {}
    for raw_state in states:
        state = str(raw_state)
        direct = decisions.get(state)
        record = direct if direct is not None else fallback
        if not isinstance(record, Mapping):
            raise FinalEvaluationError(f"state has no policy or global fallback: {state}")
        expert = str(record.get("expert_id"))
        if expert not in pool or expert not in allowed:
            raise FinalEvaluationError("policy selected an unknown or over-budget expert")
        previous = state_to_expert.setdefault(state, expert)
        if previous != expert:
            raise FinalEvaluationError("strict route changed expert within one state")
        selected.append(expert)
        reasons.append("none" if direct is not None else "unknown_state_to_train_global")
    return tuple(selected), tuple(reasons)


def load_scoped_grouping_records(
    root: str | Path,
    ordered_uids: Sequence[str],
    *,
    uid_to_dataset: Mapping[str, str],
    allowed_datasets: Sequence[str],
) -> tuple[dict[str, PublicGroupingRecord], dict[str, Any]]:
    """Read only requested public input rows and derive label-free dependency groups."""

    repository = Path(root).resolve()
    ids = tuple(str(uid) for uid in ordered_uids)
    if not ids or len(ids) != len(set(ids)):
        raise FinalEvaluationError("grouping UIDs must be non-empty and unique")
    allowed = tuple(str(dataset) for dataset in allowed_datasets)
    requested_by_dataset = {dataset: set() for dataset in allowed}
    for uid in ids:
        dataset = str(uid_to_dataset.get(uid, ""))
        if dataset not in requested_by_dataset:
            raise FinalEvaluationError(f"grouping UID escaped allowed datasets: {uid}")
        requested_by_dataset[dataset].add(uid)
    records: dict[str, PublicGroupingRecord] = {}
    opened: list[dict[str, Any]] = []
    for dataset in allowed:
        requested = requested_by_dataset[dataset]
        if not requested:
            continue
        relative = BENCHMARK_FILE_BY_DATASET.get(dataset)
        if relative is None:
            raise FinalEvaluationError(f"no benchmark binding for {dataset}")
        path = repository / relative
        digest = hashlib.sha256()
        selected = 0
        scanned = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                digest.update(line)
                scanned += 1
                match = _SAMPLE_ID_PREFIX.match(line)
                if match is None:
                    raise FinalEvaluationError(f"sample_id is not first: {relative}:{line_number}")
                sample_id = str(json.loads(match.group(1).decode("utf-8")))
                if sample_id not in requested:
                    continue
                row = json.loads(line)
                if str(row.get("dataset")) != dataset:
                    raise FinalEvaluationError("benchmark dataset disagrees with registry")
                prompt = row.get("prompt")
                assets = row.get("assets")
                if not isinstance(prompt, str) or not isinstance(assets, list):
                    raise FinalEvaluationError("requested benchmark row lacks prompt/assets")
                structural = canonical_sha256({"dataset": dataset, "assets": assets})
                records[sample_id] = PublicGroupingRecord(
                    uid=sample_id,
                    dataset=dataset,
                    structural_asset_digest=structural,
                    exact_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                )
                selected += 1
        opened.append(
            {
                "dataset": dataset,
                "path": relative,
                "sha256": digest.hexdigest(),
                "lines_scanned": scanned,
                "requested_rows_deserialized": selected,
                "unrequested_rows_deserialized": 0,
            }
        )
    if set(records) != set(ids):
        missing = sorted(set(ids).difference(records))
        raise FinalEvaluationError(f"grouping input is incomplete: {missing[0]}")
    uid_to_group, groups = connected_groups(ids, records)
    return records, {
        "kind": "subject_routing_final_input_dependency_groups",
        "unit": "union_of_exact_prompt_and_structural_asset_components",
        "rows": len(ids),
        "groups": len(groups),
        "repeated_groups": sum(len(members) > 1 for members in groups.values()),
        "maximum_group_size": max(map(len, groups.values())),
        "uid_to_group": uid_to_group,
        "uid_to_group_sha256": canonical_sha256(uid_to_group),
        "opened_containers": opened,
        "labels_costs_or_expert_outputs_read": False,
    }


def materialize_scoped_test_labels(
    requested_uids: Sequence[str],
    *,
    allowed_test_uids: Sequence[str],
    expert_ids: Sequence[str],
    oracle_score_root: str | Path,
    cost_matrix_path: str | Path,
    sample_registry_path: str | Path,
    model_registry_path: str | Path,
    expected_cost_sha256: str,
    expected_sample_registry_sha256: str,
    expected_model_registry_sha256: str,
    expected_quality_sha256: Mapping[str, str],
    prediction_sha256: str,
    prediction_receipt_sha256: str,
    audit: UIDAccessAudit,
    unlock_capability: Mapping[str, str] | None = None,
) -> tuple[EvaluationLabels, dict[str, Any]]:
    """Materialize explicit test rows only, after the active prediction unlock."""

    prediction_sha = require_sha256(prediction_sha256, name="frozen prediction")
    # Probe the central guard with a one-cell in-memory sentinel before any
    # supplied filesystem path is checked or opened.
    probe = EvaluationLabels.create(
        split="test",
        uids=("__final_unlock_probe__",),
        expert_ids=("__probe_expert__",),
        y=np.asarray([[0.0]]),
        c=np.asarray([[1.0]]),
        observed_mask=None,
        cost_observed_mask=None,
        complete_matrix=True,
        quality_kind="binary_correctness",
        cost_unit="usd_proxy_per_request",
        zero_cost_semantics="unexplained_placeholder",
    )
    EvaluationLabelStore(probe).read(
        ("__final_unlock_probe__",),
        audit=UIDAccessAudit.create(Operation.FINAL_EVALUATE, ("__final_unlock_probe__",)),
    )
    if (
        not isinstance(unlock_capability, Mapping)
        or unlock_capability.get("prediction_sha256") != prediction_sha
    ):
        raise FinalEvaluationError("active unlock capability differs from frozen prediction")
    receipt_sha = require_sha256(prediction_receipt_sha256, name="prediction receipt")
    if unlock_capability.get("receipt_sha256") != receipt_sha:
        raise FinalEvaluationError("active unlock receipt hash differs from evaluator input")
    if audit.operation is not Operation.FINAL_EVALUATE:
        raise UIDScopeError("test materialization requires final_evaluate operation")
    ids = tuple(str(uid) for uid in requested_uids)
    allowed = frozenset(str(uid) for uid in allowed_test_uids)
    if not ids or len(ids) != len(set(ids)):
        raise FinalEvaluationError("test UIDs must be non-empty and unique")
    outside = sorted(set(ids).difference(allowed))
    if outside:
        raise UIDScopeError(f"test label request escaped allowed test UIDs: {outside[0]}")
    audit.require(ids, component="scoped_final_test_label_materializer")

    sample_path = Path(sample_registry_path)
    model_path = Path(model_registry_path)
    cost_path = Path(cost_matrix_path)
    for path, expected, name in (
        (sample_path, expected_sample_registry_sha256, "sample registry"),
        (model_path, expected_model_registry_sha256, "model registry"),
        (cost_path, expected_cost_sha256, "cost matrix"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise FinalEvaluationError(f"{name} is missing or changed")
    samples = tuple(str(uid) for uid in pickle.loads(sample_path.read_bytes()))
    models = tuple(str(model) for model in pickle.loads(model_path.read_bytes()))
    canonical_experts = tuple(str(expert) for expert in expert_ids)
    if models != canonical_experts:
        raise FinalEvaluationError("model registry differs from frozen expert order")
    row_by_uid = {uid: row for row, uid in enumerate(samples)}
    if any(uid not in row_by_uid for uid in ids):
        raise FinalEvaluationError("test UID is absent from frozen sample registry")

    import pyarrow.parquet as pq

    quality = np.full((len(ids), len(models)), np.nan, dtype=np.float64)
    output_row = {uid: row for row, uid in enumerate(ids)}
    output_column = {model: column for column, model in enumerate(models)}
    by_source: dict[str, list[str]] = {}
    for uid in ids:
        by_source.setdefault(uid.split("/", 1)[0], []).append(uid)
    quality_records: list[dict[str, Any]] = []
    score_root = Path(oracle_score_root)
    for source, source_uids in sorted(by_source.items()):
        path = score_root / f"{source.casefold()}.parquet"
        expected = expected_quality_sha256.get(source)
        if expected is None or not path.is_file() or sha256_file(path) != expected:
            raise FinalEvaluationError(f"test quality source is missing or changed: {source}")
        table = pq.read_table(
            path,
            columns=["sample_id", "model_id", "quality"],
            filters=[("sample_id", "in", source_uids)],
        )
        rows = table.to_pylist()
        for record in rows:
            uid = str(record["sample_id"])
            expert = str(record["model_id"])
            if uid not in output_row or expert not in output_column:
                raise FinalEvaluationError("predicate-filtered quality escaped test scope")
            target = (output_row[uid], output_column[expert])
            if np.isfinite(quality[target]):
                raise FinalEvaluationError("duplicate test quality cell")
            quality[target] = float(record["quality"])
        quality_records.append(
            {
                "source": source,
                "path": str(path),
                "sha256": expected,
                "requested_samples": len(source_uids),
                "returned_cells": len(rows),
                "read_mode": "predicate_filtered_requested_test_sample_ids_only",
            }
        )
    if not np.isfinite(quality).all() or not np.isin(quality, (0.0, 1.0)).all():
        raise FinalEvaluationError("scoped test quality matrix is incomplete or non-binary")

    source_cost = np.load(cost_path, mmap_mode="r", allow_pickle=False)
    if source_cost.shape != (len(samples), len(models)):
        raise FinalEvaluationError("cost matrix shape differs from frozen registries")
    selected_rows = np.asarray([row_by_uid[uid] for uid in ids], dtype=np.int64)
    cost = np.asarray(source_cost[selected_rows], dtype=np.float64)
    if cost.shape != quality.shape or not np.isfinite(cost).all() or (cost <= 0).any():
        raise FinalEvaluationError("scoped test cost matrix is invalid")
    labels = EvaluationLabels.create(
        split="test",
        uids=ids,
        expert_ids=models,
        y=quality,
        c=cost,
        observed_mask=None,
        cost_observed_mask=None,
        complete_matrix=True,
        quality_kind="binary_correctness",
        cost_unit="usd_proxy_per_request",
        zero_cost_semantics="unexplained_placeholder",
    )
    scoped = EvaluationLabelStore(labels).read(ids, audit=audit)
    return scoped, {
        "requested_rows": len(ids),
        "requested_uid_sha256": uid_sha256(ids),
        "expert_ids": list(models),
        "quality_files": quality_records,
        "cost_matrix_path": str(cost_path),
        "cost_matrix_sha256": expected_cost_sha256,
        "cost_access": "read_only_memmap_explicit_test_rows_only",
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha,
        "global_y_npz_opened": False,
        "all_wrong_rows": int(np.all(quality == 0, axis=1).sum()),
        "row_drop_count": 0,
    }


def summarize_route(
    *,
    uids: Sequence[str],
    datasets: Sequence[str],
    states: Sequence[str],
    experts: Sequence[str],
    fallback_reasons: Sequence[str],
    selected_quality: np.ndarray,
    selected_cost: np.ndarray,
    all_wrong: np.ndarray,
    budget_cap: float,
    allowed_experts: Sequence[str],
    cost_bounds: Mapping[str, float],
    routing_overhead: Mapping[str, float],
) -> dict[str, Any]:
    count = len(uids)
    sequences = (datasets, states, experts, fallback_reasons)
    if not count or any(len(values) != count for values in sequences):
        raise FinalEvaluationError("route metric inputs do not share one denominator")
    quality = np.asarray(selected_quality, dtype=np.float64)
    cost = np.asarray(selected_cost, dtype=np.float64)
    wrong = np.asarray(all_wrong, dtype=bool)
    if any(values.shape != (count,) for values in (quality, cost, wrong)):
        raise FinalEvaluationError("route outcome vector shape mismatch")
    per_dataset = []
    for dataset in sorted(set(map(str, datasets))):
        rows = np.asarray([index for index, value in enumerate(datasets) if value == dataset])
        per_dataset.append(
            {
                "dataset": dataset,
                "rows": int(len(rows)),
                "accuracy": float(quality[rows].mean()),
                "average_cost_usd_proxy": float(cost[rows].mean()),
                "all_wrong_rows_retained": int(wrong[rows].sum()),
            }
        )
    micro = float(quality.mean())
    macro = float(np.mean([record["accuracy"] for record in per_dataset]))
    average_cost = float(cost.mean())
    macro_cost = float(np.mean([record["average_cost_usd_proxy"] for record in per_dataset]))
    fallback_count = sum(reason != "none" for reason in fallback_reasons)
    state_support = Counter(map(str, states))
    allowed = set(map(str, allowed_experts))
    return {
        "rows": count,
        "uid_sha256": uid_sha256(uids),
        "routed_accuracy_micro": micro,
        "routed_accuracy_macro_dataset": macro,
        "average_expert_cost_usd_proxy": average_cost,
        "average_expert_cost_macro_dataset": macro_cost,
        "official_Avg.Acc": macro,
        "official_Avg.Acc_definition": "unweighted_mean_of_dataset_accuracies",
        "official_Avg.Cost": macro_cost,
        "official_Avg.Cost_definition": "unweighted_mean_of_dataset_mean_usd_proxy_costs",
        "official_RankScore": float(
            rank_score(
                macro,
                macro_cost,
                float(cost_bounds["cmin"]),
                float(cost_bounds["cmax"]),
                beta=0.1,
            )
        ),
        "rank_score_micro": float(
            rank_score(
                micro,
                average_cost,
                float(cost_bounds["cmin"]),
                float(cost_bounds["cmax"]),
                beta=0.1,
            )
        ),
        "train_budget_cap_usd_proxy": float(budget_cap),
        "budget_allowed_expert_violation_count": sum(expert not in allowed for expert in experts),
        "per_request_actual_C_above_train_cap_rate": float(np.mean(cost > budget_cap)),
        "state_support": dict(sorted(state_support.items())),
        "observed_states": len(state_support),
        "minimum_test_state_support": min(state_support.values()),
        "fallback_rows": fallback_count,
        "fallback_rate": float(fallback_count / count),
        "all_wrong_rows_retained": int(wrong.sum()),
        "row_drop_count": 0,
        "per_dataset": per_dataset,
        "routing_overhead": dict(routing_overhead),
    }


def paired_group_bootstrap_matrix(
    deltas: np.ndarray,
    datasets: Sequence[str],
    group_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Dataset-stratified dependency-group bootstrap for several paired contrasts."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(datasets) or len(datasets) != len(group_ids):
        raise FinalEvaluationError("bootstrap arrays do not share one item denominator")
    if replicates <= 0 or not np.isfinite(values).all():
        raise FinalEvaluationError("bootstrap inputs are invalid")
    group_rows: dict[str, list[int]] = {}
    for row, group in enumerate(group_ids):
        group_rows.setdefault(str(group), []).append(row)
    strata: dict[str, list[str]] = {}
    cross_dataset_groups = 0
    for group, rows in group_rows.items():
        members = sorted({str(datasets[row]) for row in rows})
        if len(members) > 1:
            cross_dataset_groups += 1
        stratum = "|".join(members)
        strata.setdefault(stratum, []).append(group)

    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty((values.shape[0], replicates), dtype=np.float64)
    batch_size = min(256, replicates)
    group_index = {group: index for index, group in enumerate(group_rows)}
    ordered_groups = list(group_rows)
    sums = np.zeros((values.shape[0], len(ordered_groups)), dtype=np.float64)
    sizes = np.zeros(len(ordered_groups), dtype=np.float64)
    for group, rows in group_rows.items():
        position = group_index[group]
        sums[:, position] = values[:, np.asarray(rows, dtype=np.int64)].sum(axis=1)
        sizes[position] = len(rows)
    for start in range(0, replicates, batch_size):
        width = min(batch_size, replicates - start)
        numerator = np.zeros((values.shape[0], width), dtype=np.float64)
        denominator = np.zeros(width, dtype=np.float64)
        for groups in strata.values():
            positions = np.asarray([group_index[group] for group in groups], dtype=np.int64)
            probabilities = np.full(len(positions), 1.0 / len(positions))
            counts = rng.multinomial(len(positions), probabilities, size=width)
            numerator += sums[:, positions] @ counts.T
            denominator += counts @ sizes[positions]
        bootstrap[:, start : start + width] = numerator / denominator[None, :]
    return {
        "point": values.mean(axis=1),
        "lower_95": np.quantile(bootstrap, 0.025, axis=1),
        "upper_95": np.quantile(bootstrap, 0.975, axis=1),
        "p_superiority_one_sided": (
            1.0 + np.sum(bootstrap <= 0.0, axis=1)
        )
        / (replicates + 1.0),
        "replicates": bootstrap,
        "dependency_groups": len(group_rows),
        "strata": len(strata),
        "cross_dataset_groups": cross_dataset_groups,
    }


def holm_adjust(p_values: Sequence[float], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise FinalEvaluationError("Holm p-values must be a finite vector")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    continue_reject = True
    rejected = np.zeros(len(values), dtype=bool)
    for rank, index in enumerate(order):
        multiplier = len(values) - rank
        running = max(running, min(1.0, float(values[index]) * multiplier))
        adjusted[index] = running
        if continue_reject and values[index] <= alpha / multiplier:
            rejected[index] = True
        else:
            continue_reject = False
    return [
        {
            "raw_p": float(values[index]),
            "holm_adjusted_p": float(adjusted[index]),
            "reject_at_FWER_0p05": bool(rejected[index]),
        }
        for index in range(len(values))
    ]


def pareto_frontier_flags(accuracies: Sequence[float], costs: Sequence[float]) -> list[bool]:
    accuracy = np.asarray(accuracies, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64)
    if accuracy.shape != cost.shape or accuracy.ndim != 1:
        raise FinalEvaluationError("frontier arrays must be aligned vectors")
    flags = []
    for index in range(len(accuracy)):
        dominates = (
            (accuracy >= accuracy[index])
            & (cost <= cost[index])
            & ((accuracy > accuracy[index]) | (cost < cost[index]))
        )
        flags.append(not bool(dominates.any()))
    return flags
