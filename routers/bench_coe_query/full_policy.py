"""Approved full-run handling for the release's 47 genuine multi-image rows."""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (
    FULL_MULTI_IMAGE_FALLBACK,
    FULL_MULTI_IMAGE_POLICY,
    ExperimentContract,
    load_full_run_approval,
)
from .errors import ContractError
from .schemas import StrictVLRouterData


EXPECTED_MULTI_IMAGE_SPLIT_COUNTS = {"train": 32, "dev": 5, "test": 10}


def approved_full_policy(contract: ExperimentContract) -> dict[str, Any]:
    approval, _sha256 = load_full_run_approval(contract)
    assert approval is not None
    decisions = dict(approval["decisions"])
    if decisions["multi_image_policy"] != FULL_MULTI_IMAGE_POLICY:
        raise ContractError("unsupported full multi-image policy")
    if decisions["multi_image_fallback"] != FULL_MULTI_IMAGE_FALLBACK:
        raise ContractError("unsupported full multi-image fallback")
    return decisions


def frozen_train_strongest_column(
    contract: ExperimentContract,
    model_order: Sequence[str],
) -> int:
    """Read the already-frozen train-only statistic without consulting Y/C."""

    path = contract.dataset_root / "outputs/bench_coe_query/labels/label_manifest.meta.json"
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("statistics_source") != "train_only":
        raise ContractError("label manifest statistic source is not train_only")
    if tuple(metadata.get("model_order", ())) != tuple(model_order):
        raise ContractError("label manifest model order differs from the frozen contract")
    accuracy = np.asarray(metadata.get("train_expert_accuracy"), dtype=np.float64)
    if accuracy.shape != (len(model_order),) or not np.isfinite(accuracy).all():
        raise ContractError("label manifest train expert accuracy is invalid")
    return int(np.argmax(accuracy))


def partition_feature_rows(
    data: StrictVLRouterData,
    rows: Iterable[int],
    *,
    contract: ExperimentContract,
    require_release_counts: bool = False,
) -> tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
    """Return single-image feature rows and explicit structured exclusions."""

    approved_full_policy(contract)
    selected = tuple(int(row) for row in rows)
    eligible: list[int] = []
    exclusions: list[dict[str, Any]] = []
    for row in selected:
        sample = data.samples[row]
        if len(sample.assets) == 1:
            eligible.append(row)
            continue
        exclusions.append(
            {
                "status": "error",
                "error_type": "multi_image",
                "sample_id": sample.sample_id,
                "row_index": row,
                "split": sample.split,
                "dataset": sample.dataset,
                "image_count": len(sample.assets),
                "feature_policy": "excluded",
                "prediction_fallback": FULL_MULTI_IMAGE_FALLBACK,
                "evaluation_denominator": "official_split_frozen",
            }
        )
    if require_release_counts:
        observed = {
            split: sum(item["split"] == split for item in exclusions)
            for split in EXPECTED_MULTI_IMAGE_SPLIT_COUNTS
        }
        expected_total = int(contract.values["data"]["asset_policy"]["known_multi_image_samples"])
        if len(selected) != int(contract.values["data"]["expected_samples"]):
            raise ContractError("release-count validation requires the complete official row set")
        if len(exclusions) != expected_total or observed != EXPECTED_MULTI_IMAGE_SPLIT_COUNTS:
            raise ContractError(
                f"multi-image release counts changed: total={len(exclusions)}, split_counts={observed}"
            )
    return tuple(eligible), tuple(exclusions)


def materialize_frozen_denominator_logits(
    *,
    data: StrictVLRouterData,
    official_rows: Sequence[int],
    feature_rows: Sequence[int],
    feature_logits: np.ndarray,
    strongest_train_column: int,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...], tuple[int, ...]]:
    """Fill excluded rows with a train-only StrongestGlobal routing decision.

    This helper deliberately has no Y/C argument. The caller must compute the
    strongest column from train-only statistics before prediction. The emitted
    finite logits keep the ordinary frozen prediction/evaluator protocol intact.
    """

    rows = tuple(int(row) for row in official_rows)
    cached_rows = tuple(int(row) for row in feature_rows)
    values = np.asarray(feature_logits, dtype=np.float32)
    if values.shape != (len(cached_rows), len(data.model_order)):
        raise ValueError("feature logits do not match cached rows/model order")
    if len(cached_rows) != len(set(cached_rows)) or len(rows) != len(set(rows)):
        raise ValueError("prediction row lists contain duplicates")
    if strongest_train_column < 0 or strongest_train_column >= len(data.model_order):
        raise ValueError("StrongestGlobal column is out of range")
    position_by_row = {row: position for position, row in enumerate(cached_rows)}
    extra = set(cached_rows).difference(rows)
    if extra:
        raise ValueError(f"feature logits contain {len(extra)} rows outside the requested split")

    output = np.full((len(rows), len(data.model_order)), -1.0, dtype=np.float32)
    output[:, strongest_train_column] = 1.0
    annotations: list[dict[str, Any]] = []
    fallback_rows: list[int] = []
    for output_position, row in enumerate(rows):
        cached_position = position_by_row.get(row)
        sample = data.samples[row]
        if cached_position is not None:
            output[output_position] = values[cached_position]
            annotations.append({"routing_input_status": "vlm_feature"})
            continue
        if len(sample.assets) == 1:
            raise ValueError(f"single-image row {row} is unexpectedly absent from the feature cache")
        fallback_rows.append(row)
        annotations.append(
            {
                "routing_input_status": "structured_asset_error_fallback",
                "asset_error_type": "multi_image",
                "image_count": len(sample.assets),
                "fallback_policy": FULL_MULTI_IMAGE_FALLBACK,
                "evaluation_denominator": "official_split_frozen",
            }
        )
    if not np.isfinite(output).all():
        raise ValueError("materialized full-denominator logits are non-finite")
    return output, tuple(annotations), tuple(fallback_rows)
