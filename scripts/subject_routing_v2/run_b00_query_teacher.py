#!/usr/bin/env python3
"""B00: full source-LODO same-input BCE and kNN query teachers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.subject_routing.dev_evaluation import (  # noqa: E402
    materialize_scoped_dev_labels,
)
from src.subject_routing.leakage_guard import Operation, UIDAccessAudit  # noqa: E402
from src.subject_routing.profiling import (  # noqa: E402
    materialize_scoped_training_labels,
)
from src.subject_routing_v2._io import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_sha256,
    file_record,
    load_json,
    sha256_file,
    with_payload_hash,
)
from src.subject_routing_v2.baselines import (  # noqa: E402
    IndependentQualityHeads,
    KNNQualityEstimator,
    LogCostRidgePredictor,
    TrainMeanCostPredictor,
)
from src.subject_routing_v2.data_views import (  # noqa: E402
    EvaluationOutcomeView,
    OutcomeMatrix,
    TrainingOutcomeView,
    source_evaluation_access,
)
from src.subject_routing_v2.step_runtime import (  # noqa: E402
    DEFAULT_RESOURCES,
    dependency_statuses,
    render_state,
)


STEP_ID = "B00"
NEW_ROOT = REPOSITORY_ROOT / "artifacts/subject_routing_v2"
OLD_ROOT = REPOSITORY_ROOT / "artifacts/subject_routing"
FULL_ROOT = OLD_ROOT / "ood11_3/full_v1"
METHOD_BCE = "same_info_query_multilabel_bce"
METHOD_KNN = "same_info_knn_quality"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _quality_hashes(
    matrix_audit: Mapping[str, Any], split_manifest: Mapping[str, Any]
) -> dict[str, str]:
    by_stem = {
        Path(record["path"]).stem: str(record["sha256"])
        for record in matrix_audit["quality"]["oracle_files"]
    }
    return {
        source: by_stem[source.casefold()]
        for source in split_manifest["dataset_catalog"]
    }


def _common_label_arguments(
    *,
    experts: Sequence[str],
    model_pool: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    matrix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    model_path = str(model_pool["provenance"]["model_registry"]["path"])
    return {
        "expert_ids": experts,
        "oracle_score_root": REPOSITORY_ROOT / "ORACLE/score",
        "cost_matrix_path": REPOSITORY_ROOT / model_pool["provenance"]["c"]["path"],
        "sample_registry_path": REPOSITORY_ROOT
        / model_path.replace("model_index.pkl", "sample_index.pkl"),
        "model_registry_path": REPOSITORY_ROOT / model_path,
        "expected_cost_sha256": model_pool["provenance"]["c"]["sha256"],
        "expected_sample_registry_sha256": split_manifest["provenance"]["registries"][
            "samples"
        ]["sha256"],
        "expected_model_registry_sha256": model_pool["provenance"]["model_registry"][
            "sha256"
        ],
        "expected_quality_sha256": _quality_hashes(matrix_audit, split_manifest),
    }


def _to_v2_matrix(matrix: Any) -> OutcomeMatrix:
    return OutcomeMatrix.create(
        uids=matrix.uids,
        expert_ids=matrix.expert_ids,
        quality=matrix.y,
        cost=matrix.c,
        quality_observed_mask=matrix.observed_mask,
        cost_observed_mask=matrix.cost_observed_mask,
        quality_kind=matrix.quality_kind,
        cost_unit=matrix.cost_unit,
    )


def _combine_rows(
    requested_uids: Sequence[str], train: OutcomeMatrix, dev: OutcomeMatrix
) -> OutcomeMatrix:
    locations: dict[str, tuple[OutcomeMatrix, int]] = {}
    for matrix in (train, dev):
        for row, uid in enumerate(matrix.uids):
            if uid in locations:
                raise RuntimeError(f"source train/dev outcome overlap: {uid}")
            locations[uid] = (matrix, row)
    missing = [uid for uid in requested_uids if uid not in locations]
    if missing:
        raise RuntimeError(f"source outcome missing: {missing[0]}")
    quality = np.vstack(
        [locations[uid][0].quality[locations[uid][1]] for uid in requested_uids]
    )
    cost = np.vstack(
        [locations[uid][0].cost[locations[uid][1]] for uid in requested_uids]
    )
    quality_mask = np.vstack(
        [
            locations[uid][0].quality_observed_mask[locations[uid][1]]
            for uid in requested_uids
        ]
    )
    cost_mask = np.vstack(
        [
            locations[uid][0].cost_observed_mask[locations[uid][1]]
            for uid in requested_uids
        ]
    )
    return OutcomeMatrix.create(
        uids=requested_uids,
        expert_ids=train.expert_ids,
        quality=quality,
        cost=cost,
        quality_observed_mask=quality_mask,
        cost_observed_mask=cost_mask,
        quality_kind=train.quality_kind,
        cost_unit=train.cost_unit,
    )


class _FeatureStore:
    def __init__(
        self,
        *,
        train_uids: Sequence[str],
        dev_uids: Sequence[str],
        train_path: Path,
        dev_path: Path,
    ) -> None:
        self.train = np.load(train_path, mmap_mode="r", allow_pickle=False)
        self.dev = np.load(dev_path, mmap_mode="r", allow_pickle=False)
        if self.train.shape[0] != len(train_uids) or self.dev.shape[0] != len(dev_uids):
            raise RuntimeError("feature arrays differ from frozen UID order")
        self.locations = {
            **{uid: ("train", row) for row, uid in enumerate(train_uids)},
            **{uid: ("dev", row) for row, uid in enumerate(dev_uids)},
        }

    def rows(self, uids: Sequence[str]) -> np.ndarray:
        missing = [uid for uid in uids if uid not in self.locations]
        if missing:
            raise RuntimeError(f"feature UID missing: {missing[0]}")
        output = np.empty((len(uids), self.train.shape[1]), dtype=np.float32)
        train_targets = [
            (target, location[1])
            for target, uid in enumerate(uids)
            if (location := self.locations[uid])[0] == "train"
        ]
        dev_targets = [
            (target, location[1])
            for target, uid in enumerate(uids)
            if (location := self.locations[uid])[0] == "dev"
        ]
        if train_targets:
            output[[target for target, _ in train_targets]] = self.train[
                [source for _, source in train_targets]
            ]
        if dev_targets:
            output[[target for target, _ in dev_targets]] = self.dev[
                [source for _, source in dev_targets]
            ]
        return output


def _legacy_budgets(
    matrix: OutcomeMatrix, *, ids: Sequence[str], quantiles: Sequence[float]
) -> list[dict[str, Any]]:
    means = np.divide(
        np.where(matrix.cost_observed_mask, matrix.cost, 0.0).sum(axis=0),
        matrix.cost_observed_mask.sum(axis=0),
    )
    output = []
    for budget_id, quantile in zip(ids, quantiles, strict=True):
        cap = float(np.quantile(means, float(quantile)))
        allowed_indices = np.flatnonzero(means <= cap + 1e-15).astype(int).tolist()
        output.append(
            {
                "budget_id": str(budget_id),
                "quantile": float(quantile),
                "cap_usd_proxy_per_request": cap,
                "allowed_indices": allowed_indices,
                "allowed_experts": [matrix.expert_ids[index] for index in allowed_indices],
            }
        )
    return output


def _route_allowed(quality: np.ndarray, allowed_indices: Sequence[int]) -> np.ndarray:
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    local = np.argmax(quality[:, allowed], axis=1)
    return allowed[local]


def _route_lambda(
    quality: np.ndarray, cost: np.ndarray, *, lambda_value: float, cost_scale: float
) -> np.ndarray:
    utility = quality - float(lambda_value) * cost / float(cost_scale)
    return np.argmax(utility, axis=1).astype(np.int64)


def _calibrate_expected_policies(
    quality: np.ndarray,
    cost: np.ndarray,
    *,
    expected_budgets: Sequence[Mapping[str, Any]],
    lambda_grid: Sequence[float],
    cost_scale: float,
) -> dict[str, dict[str, Any]]:
    rows = np.arange(quality.shape[0], dtype=np.int64)
    candidates = []
    for lambda_value in lambda_grid:
        choices = _route_lambda(
            quality, cost, lambda_value=float(lambda_value), cost_scale=cost_scale
        )
        candidates.append(
            {
                "lambda": float(lambda_value),
                "mean_predicted_cost": float(cost[rows, choices].mean()),
                "mean_predicted_quality": float(quality[rows, choices].mean()),
            }
        )
    policies: dict[str, dict[str, Any]] = {}
    cheapest = np.argmin(cost, axis=1)
    cheapest_cost = float(cost[rows, cheapest].mean())
    for budget in expected_budgets:
        cap = float(budget["cap_usd_proxy_per_request"])
        feasible = [item for item in candidates if item["mean_predicted_cost"] <= cap + 1e-15]
        if not feasible:
            policies[str(budget["budget_id"])] = {
                "lambda": None,
                "infeasible": True,
                "fit_mean_predicted_cost": cheapest_cost,
                "fit_mean_predicted_quality": float(quality[rows, cheapest].mean()),
                "fallback": "predicted_cheapest_per_input",
            }
            continue
        selected = sorted(
            feasible,
            key=lambda item: (
                -item["mean_predicted_quality"],
                item["mean_predicted_cost"],
                item["lambda"],
            ),
        )[0]
        policies[str(budget["budget_id"])] = {
            **selected,
            "infeasible": False,
            "fallback": None,
        }
    return policies


def _apply_expected_policy(
    quality: np.ndarray,
    cost: np.ndarray,
    *,
    policy: Mapping[str, Any],
    cost_scale: float,
) -> np.ndarray:
    if policy["lambda"] is None:
        return np.argmin(cost, axis=1).astype(np.int64)
    return _route_lambda(
        quality,
        cost,
        lambda_value=float(policy["lambda"]),
        cost_scale=cost_scale,
    )


def _brier(quality: np.ndarray, matrix: OutcomeMatrix) -> float:
    mask = matrix.quality_observed_mask
    return float(np.square(quality[mask] - matrix.quality[mask]).mean())


def _selected_accuracy(matrix: OutcomeMatrix, choices: np.ndarray) -> float:
    rows = np.arange(len(matrix.uids), dtype=np.int64)
    observed = matrix.quality_observed_mask[rows, choices]
    if not observed.all():
        raise RuntimeError("selected route contains missing source quality")
    return float(matrix.quality[rows, choices].mean())


def _select_cost_predictor(
    fit_x: np.ndarray,
    selection_x: np.ndarray,
    fit: OutcomeMatrix,
    selection: OutcomeMatrix,
    *,
    ridge_alpha: float,
) -> tuple[Any, list[dict[str, Any]]]:
    candidates = [
        TrainMeanCostPredictor().fit(
            fit_x, fit.cost, fit.cost_observed_mask
        ),
        LogCostRidgePredictor(alpha=ridge_alpha).fit(
            fit_x, fit.cost, fit.cost_observed_mask
        ),
    ]
    logs = []
    for order, model in enumerate(candidates):
        prediction = model.predict_cost(selection_x)
        mask = selection.cost_observed_mask
        log_mae = float(
            np.abs(np.log(prediction[mask]) - np.log(selection.cost[mask])).mean()
        )
        logs.append(
            {
                "candidate_order": order,
                "predictor_id": model.predictor_id,
                "selection_mean_absolute_log_cost_error": log_mae,
            }
        )
    selected_order = min(
        range(len(logs)),
        key=lambda index: (
            logs[index]["selection_mean_absolute_log_cost_error"],
            index,
        ),
    )
    for index, record in enumerate(logs):
        record["selected"] = index == selected_order
    return candidates[selected_order], logs


def _candidate_score(
    quality_fit: np.ndarray,
    quality_selection: np.ndarray,
    cost_fit: np.ndarray,
    cost_selection: np.ndarray,
    *,
    fit_matrix: OutcomeMatrix,
    selection_matrix: OutcomeMatrix,
    primary_budget: Mapping[str, Any],
    lambda_grid: Sequence[float],
    cost_scale: float,
) -> tuple[float, float, dict[str, Any]]:
    policy = _calibrate_expected_policies(
        quality_fit,
        cost_fit,
        expected_budgets=[primary_budget],
        lambda_grid=lambda_grid,
        cost_scale=cost_scale,
    )[str(primary_budget["budget_id"])]
    choices = _apply_expected_policy(
        quality_selection,
        cost_selection,
        policy=policy,
        cost_scale=cost_scale,
    )
    return (
        _selected_accuracy(selection_matrix, choices),
        _brier(quality_selection, selection_matrix),
        policy,
    )


def _make_prediction_frame(
    *,
    uids: Sequence[str],
    sidecar: Mapping[str, Mapping[str, str]],
    fold: Mapping[str, Any],
    method_id: str,
    seed: int,
    quality: np.ndarray,
    cost: np.ndarray,
    utility_primary: np.ndarray,
    legacy_choices: Mapping[str, np.ndarray],
    expected_choices: Mapping[str, np.ndarray],
    cost_predictor_id: str,
    quality_hyperparameter: Mapping[str, Any],
) -> pd.DataFrame:
    rows: dict[str, Any] = {
        "uid": list(uids),
        "source_id": [sidecar[uid]["source_id"] for uid in uids],
        "group_id": [sidecar[uid]["group_id"] for uid in uids],
        "fold_id": [str(fold["fold_id"])] * len(uids),
        "heldout_source": [str(fold["heldout_source"])] * len(uids),
        "method_id": [method_id] * len(uids),
        "seed": [int(seed)] * len(uids),
        "routing_identity": ["QUERY_ROUTING"] * len(uids),
        "feature_id": ["minilm_text_384"] * len(uids),
        "fit_uid_sha256": [canonical_sha256(fold["fit_train_uids"])] * len(uids),
        "selection_uid_sha256": [canonical_sha256(fold["selection_dev_uids"])]
        * len(uids),
        "evaluation_uid_sha256": [canonical_sha256(fold["evaluation_uids"])]
        * len(uids),
        "quality_scores": [row.astype(np.float32).tolist() for row in quality],
        "cost_scores": [row.astype(np.float32).tolist() for row in cost],
        "utility_scores_accuracy_first": [
            row.astype(np.float32).tolist() for row in quality
        ],
        "utility_scores_primary_b050": [
            row.astype(np.float32).tolist() for row in utility_primary
        ],
        "cost_predictor_id": [cost_predictor_id] * len(uids),
        "quality_hyperparameter_json": [
            json.dumps(quality_hyperparameter, sort_keys=True, separators=(",", ":"))
        ]
        * len(uids),
        "evaluation_label_seen_during_fit": [False] * len(uids),
    }
    expert_ids = tuple(fold["expert_ids"])
    for budget_id, choices in legacy_choices.items():
        rows[f"selected_{budget_id}"] = [expert_ids[int(index)] for index in choices]
    for budget_id, choices in expected_choices.items():
        rows[f"selected_{budget_id}"] = [expert_ids[int(index)] for index in choices]
    return pd.DataFrame(rows)


def _write_oof_predictions(
    *,
    output_path: Path,
    folds: Sequence[Mapping[str, Any]],
    feature_store: _FeatureStore,
    train_matrix: OutcomeMatrix,
    dev_matrix: OutcomeMatrix,
    sidecar: Mapping[str, Mapping[str, str]],
    config: Mapping[str, Any],
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".query_teacher.", suffix=".parquet.tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    writer: pq.ParquetWriter | None = None
    fit_logs = []
    training_runs = 0
    started = time.perf_counter()
    seeds = [int(value) for value in config["execution"]["seeds"]]
    expected_config = study["budget_tracks"]["expected_total_cost_budget"]
    expected_budgets = [
        {
            "budget_id": budget_id,
            "cap_usd_proxy_per_request": float(
                expected_config["caps_usd_proxy_per_request"][budget_id]
            ),
        }
        for budget_id in expected_config["order"]
    ]
    primary_budget = next(
        budget for budget in expected_budgets if budget["budget_id"] == "b050"
    )
    lambda_grid = [float(value) for value in config["budgets"]["expected_cost_lambda_grid"]]
    try:
        for fold in folds:
            fold_started = time.perf_counter()
            fit_uids = tuple(map(str, fold["fit_train_uids"]))
            selection_uids = tuple(map(str, fold["selection_dev_uids"]))
            evaluation_uids = tuple(map(str, fold["evaluation_uids"]))
            if set(fit_uids).intersection(evaluation_uids) or set(selection_uids).intersection(
                evaluation_uids
            ):
                raise RuntimeError(f"outer fold UID leakage: {fold['fold_id']}")
            if any(uid.split("/", 1)[0] == fold["heldout_source"] for uid in fit_uids):
                raise RuntimeError(f"heldout source entered fit: {fold['fold_id']}")
            fit = _combine_rows(fit_uids, train_matrix, dev_matrix)
            selection = _combine_rows(selection_uids, train_matrix, dev_matrix)
            fit_view = TrainingOutcomeView.create(
                fit,
                allowed_source_uids=fit_uids,
                forbidden_test_uids=(),
                scope_id=str(fold["fold_id"]),
            )
            if tuple(fit_view.matrix.uids) != fit_uids:
                raise RuntimeError("training capability UID drift")
            fit_x = feature_store.rows(fit_uids)
            selection_x = feature_store.rows(selection_uids)
            evaluation_x = feature_store.rows(evaluation_uids)
            cost_model, cost_logs = _select_cost_predictor(
                fit_x,
                selection_x,
                fit,
                selection,
                ridge_alpha=float(config["cost_predictor"]["ridge_alpha"]),
            )
            training_runs += 2
            cost_fit = cost_model.predict_cost(fit_x)
            cost_selection = cost_model.predict_cost(selection_x)
            cost_evaluation = cost_model.predict_cost(evaluation_x)
            expert_mean_cost = np.divide(
                np.where(fit.cost_observed_mask, fit.cost, 0.0).sum(axis=0),
                fit.cost_observed_mask.sum(axis=0),
            )
            cost_scale = float(np.median(expert_mean_cost))
            legacy = _legacy_budgets(
                fit,
                ids=["q000", "q010", "q025", "q050", "q075", "q090", "q100"],
                quantiles=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
            )

            bce_candidates = []
            bce_models = []
            for order, alpha in enumerate(config["query_bce"]["alpha_grid"]):
                model = IndependentQualityHeads(
                    quality_kind=fit.quality_kind,
                    alpha=float(alpha),
                    seed=int(config["query_bce"]["model_selection_seed"]),
                    max_iter=int(config["query_bce"]["max_iter"]),
                    learning_rate_init=float(
                        config["query_bce"]["learning_rate_init"]
                    ),
                    batch_size=int(config["query_bce"]["batch_size"]),
                ).fit(fit_x, fit.quality, fit.quality_observed_mask)
                training_runs += 1
                quality_fit = model.predict_quality(fit_x)
                quality_selection = model.predict_quality(selection_x)
                score, brier, primary_policy = _candidate_score(
                    quality_fit,
                    quality_selection,
                    cost_fit,
                    cost_selection,
                    fit_matrix=fit,
                    selection_matrix=selection,
                    primary_budget=primary_budget,
                    lambda_grid=lambda_grid,
                    cost_scale=cost_scale,
                )
                bce_models.append(model)
                bce_candidates.append(
                    {
                        "candidate_order": order,
                        "alpha": float(alpha),
                        "selection_primary_b050_actual_routed_accuracy": score,
                        "selection_brier": brier,
                        "primary_policy": primary_policy,
                        "fit_summary": model.fit_summary,
                    }
                )
            selected_bce_index = min(
                range(len(bce_candidates)),
                key=lambda index: (
                    -bce_candidates[index][
                        "selection_primary_b050_actual_routed_accuracy"
                    ],
                    bce_candidates[index]["selection_brier"],
                    bce_candidates[index]["alpha"],
                    index,
                ),
            )
            for index, record in enumerate(bce_candidates):
                record["selected"] = index == selected_bce_index
            selected_alpha = float(bce_candidates[selected_bce_index]["alpha"])

            knn = KNNQualityEstimator(
                max_neighbors=max(map(int, config["query_knn"]["neighbors_grid"])),
                n_jobs=int(config["query_knn"]["n_jobs"]),
            ).fit(fit_x, fit.quality, fit.quality_observed_mask)
            training_runs += 1
            knn_fit_neighbors = knn.training_leave_one_out_rows()
            knn_selection_neighbors = knn.neighbor_rows(selection_x)
            knn_evaluation_neighbors = knn.neighbor_rows(evaluation_x)
            knn_candidates = []
            for order, neighbors in enumerate(config["query_knn"]["neighbors_grid"]):
                quality_fit = knn.predict_from_neighbor_rows(
                    knn_fit_neighbors, neighbors=int(neighbors)
                )
                quality_selection = knn.predict_from_neighbor_rows(
                    knn_selection_neighbors, neighbors=int(neighbors)
                )
                score, brier, primary_policy = _candidate_score(
                    quality_fit,
                    quality_selection,
                    cost_fit,
                    cost_selection,
                    fit_matrix=fit,
                    selection_matrix=selection,
                    primary_budget=primary_budget,
                    lambda_grid=lambda_grid,
                    cost_scale=cost_scale,
                )
                knn_candidates.append(
                    {
                        "candidate_order": order,
                        "neighbors": int(neighbors),
                        "selection_primary_b050_actual_routed_accuracy": score,
                        "selection_brier": brier,
                        "primary_policy": primary_policy,
                    }
                )
            selected_knn_index = min(
                range(len(knn_candidates)),
                key=lambda index: (
                    -knn_candidates[index][
                        "selection_primary_b050_actual_routed_accuracy"
                    ],
                    knn_candidates[index]["selection_brier"],
                    knn_candidates[index]["neighbors"],
                    index,
                ),
            )
            for index, record in enumerate(knn_candidates):
                record["selected"] = index == selected_knn_index
            selected_k = int(knn_candidates[selected_knn_index]["neighbors"])

            method_seed_logs = []
            for method_id, seed in [
                *[(METHOD_BCE, seed) for seed in seeds],
                *[(METHOD_KNN, seed) for seed in seeds],
            ]:
                if method_id == METHOD_BCE:
                    if seed == int(config["query_bce"]["model_selection_seed"]):
                        quality_model = bce_models[selected_bce_index]
                    else:
                        quality_model = IndependentQualityHeads(
                            quality_kind=fit.quality_kind,
                            alpha=selected_alpha,
                            seed=seed,
                            max_iter=int(config["query_bce"]["max_iter"]),
                            learning_rate_init=float(
                                config["query_bce"]["learning_rate_init"]
                            ),
                            batch_size=int(config["query_bce"]["batch_size"]),
                        ).fit(fit_x, fit.quality, fit.quality_observed_mask)
                        training_runs += 1
                    quality_fit = quality_model.predict_quality(fit_x)
                    quality_evaluation = quality_model.predict_quality(evaluation_x)
                    hyperparameter = {"alpha": selected_alpha}
                    quality_fit_summary = quality_model.fit_summary
                else:
                    quality_fit = knn.predict_from_neighbor_rows(
                        knn_fit_neighbors, neighbors=selected_k
                    )
                    quality_evaluation = knn.predict_from_neighbor_rows(
                        knn_evaluation_neighbors, neighbors=selected_k
                    )
                    hyperparameter = {"neighbors": selected_k}
                    quality_fit_summary = {
                        "implementation": "mask_aware_uniform_cosine_neighbors",
                        "max_neighbors_indexed": knn.max_neighbors,
                    }
                policies = _calibrate_expected_policies(
                    quality_fit,
                    cost_fit,
                    expected_budgets=expected_budgets,
                    lambda_grid=lambda_grid,
                    cost_scale=cost_scale,
                )
                legacy_choices = {
                    budget["budget_id"]: _route_allowed(
                        quality_evaluation, budget["allowed_indices"]
                    )
                    for budget in legacy
                }
                expected_choices = {
                    budget["budget_id"]: _apply_expected_policy(
                        quality_evaluation,
                        cost_evaluation,
                        policy=policies[budget["budget_id"]],
                        cost_scale=cost_scale,
                    )
                    for budget in expected_budgets
                }
                primary_policy = policies["b050"]
                if primary_policy["lambda"] is None:
                    utility_primary = -cost_evaluation / cost_scale
                else:
                    utility_primary = quality_evaluation - float(
                        primary_policy["lambda"]
                    ) * cost_evaluation / cost_scale
                frame_fold = {
                    **fold,
                    "expert_ids": list(fit.expert_ids),
                }
                frame = _make_prediction_frame(
                    uids=evaluation_uids,
                    sidecar=sidecar,
                    fold=frame_fold,
                    method_id=method_id,
                    seed=seed,
                    quality=quality_evaluation,
                    cost=cost_evaluation,
                    utility_primary=utility_primary,
                    legacy_choices=legacy_choices,
                    expected_choices=expected_choices,
                    cost_predictor_id=cost_model.predictor_id,
                    quality_hyperparameter=hyperparameter,
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writer.write_table(table)
                method_seed_logs.append(
                    {
                        "method_id": method_id,
                        "seed": seed,
                        "quality_hyperparameter": hyperparameter,
                        "quality_fit_summary": quality_fit_summary,
                        "expected_cost_policies": policies,
                        "evaluation_predictions_written": len(evaluation_uids),
                        "evaluation_outcomes_read": 0,
                    }
                )
            fit_logs.append(
                {
                    "fold_id": fold["fold_id"],
                    "heldout_source": fold["heldout_source"],
                    "counts": fold["counts"],
                    "fit_uid_sha256": canonical_sha256(fit_uids),
                    "selection_uid_sha256": canonical_sha256(selection_uids),
                    "evaluation_uid_sha256": canonical_sha256(evaluation_uids),
                    "heldout_source_in_fit": False,
                    "pairwise_group_overlap": fold["pairwise_group_overlap"],
                    "cost_scale_usd_proxy_per_request": cost_scale,
                    "cost_predictor_candidates": cost_logs,
                    "selected_cost_predictor": cost_model.predictor_id,
                    "legacy_budgets": legacy,
                    "expected_budgets": expected_budgets,
                    "query_bce_candidates": bce_candidates,
                    "query_knn_candidates": knn_candidates,
                    "method_seed_fits": method_seed_logs,
                    "wall_seconds": time.perf_counter() - fold_started,
                }
            )
        if writer is None:
            raise RuntimeError("no OOF predictions were written")
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
    payload = with_payload_hash(
        {
            "schema_version": 1,
            "kind": "subject_routing_v2_B00_source_fit_logs",
            "step_id": STEP_ID,
            "status": "OOF_FROZEN_BEFORE_SOURCE_EVALUATION",
            "folds": fit_logs,
            "fold_count": len(fit_logs),
            "methods": [METHOD_BCE, METHOD_KNN],
            "seeds": seeds,
            "training_runs": training_runs,
            "evaluation_outcome_rows_read_while_fitting": 0,
            "test_outcome_rows_read": 0,
            "wall_seconds": time.perf_counter() - started,
        }
    )
    return payload, training_runs


def _evaluate_oof(
    *,
    predictions_path: Path,
    fit_logs: Mapping[str, Any],
    train_matrix: OutcomeMatrix,
    dev_matrix: OutcomeMatrix,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    frame = pd.read_parquet(predictions_path)
    forbidden_columns = {
        "answer",
        "gold",
        "actual_output_tokens",
        "actual_generation_length",
        "current_item_actual_C",
    }
    if forbidden_columns.intersection(frame.columns):
        raise RuntimeError("OOF artifact contains a forbidden prediction column")
    expected_rows = 21_345 * 2 * 5
    if len(frame) != expected_rows or frame["uid"].nunique() != 21_345:
        raise RuntimeError("OOF prediction coverage mismatch")
    if frame["evaluation_label_seen_during_fit"].any():
        raise RuntimeError("OOF row claims evaluation-label access")
    expert_ids = train_matrix.expert_ids
    expert_position = {expert: index for index, expert in enumerate(expert_ids)}
    fold_log_by_id = {str(item["fold_id"]): item for item in fit_logs["folds"]}
    fold_results = []
    diagnostics = []
    calibration_sse = {
        method: np.zeros(len(expert_ids), dtype=np.float64)
        for method in (METHOD_BCE, METHOD_KNN)
    }
    quality_ae = {method: np.zeros(len(expert_ids)) for method in calibration_sse}
    cost_ae = {method: np.zeros(len(expert_ids)) for method in calibration_sse}
    observed_counts = {
        method: np.zeros(len(expert_ids), dtype=np.int64) for method in calibration_sse
    }
    for (fold_id, method_id, seed), group in frame.groupby(
        ["fold_id", "method_id", "seed"], sort=False
    ):
        uids = tuple(map(str, group["uid"]))
        matrix = _combine_rows(uids, train_matrix, dev_matrix)
        evaluation = EvaluationOutcomeView(matrix, role="SOURCE_LODO_EVALUATION")
        access = source_evaluation_access(heldout_uids=uids, fold_id=str(fold_id))
        audited = evaluation.diagnostic_matrix(access=access)
        quality_scores = np.asarray(group["quality_scores"].tolist(), dtype=np.float64)
        cost_scores = np.asarray(group["cost_scores"].tolist(), dtype=np.float64)
        if quality_scores.shape != audited.quality.shape or cost_scores.shape != audited.cost.shape:
            raise RuntimeError("OOF vector shape differs from source outcomes")
        quality_mask = audited.quality_observed_mask
        cost_mask = audited.cost_observed_mask
        calibration_sse[method_id] += np.where(
            quality_mask, np.square(quality_scores - audited.quality), 0.0
        ).sum(axis=0)
        quality_ae[method_id] += np.where(
            quality_mask, np.abs(quality_scores - audited.quality), 0.0
        ).sum(axis=0)
        cost_ae[method_id] += np.where(
            cost_mask, np.abs(cost_scores - audited.cost), 0.0
        ).sum(axis=0)
        observed_counts[method_id] += quality_mask.sum(axis=0)
        diagnostics.append(
            {
                "fold_id": str(fold_id),
                "heldout_source": str(group["heldout_source"].iloc[0]),
                "method_id": str(method_id),
                "seed": int(seed),
                "rows": len(group),
                "Brier_or_quality_MSE": float(
                    np.square(quality_scores[quality_mask] - audited.quality[quality_mask]).mean()
                ),
                "quality_MAE": float(
                    np.abs(quality_scores[quality_mask] - audited.quality[quality_mask]).mean()
                ),
                "cost_MAE_usd_proxy_per_request": float(
                    np.abs(cost_scores[cost_mask] - audited.cost[cost_mask]).mean()
                ),
            }
        )
        fold_log = fold_log_by_id[str(fold_id)]
        budget_specs = [
            *[
                {
                    "budget_id": item["budget_id"],
                    "budget_track": "legacy_allowed_expert_quantile",
                    "cap": item["cap_usd_proxy_per_request"],
                    "allowed_indices": item["allowed_indices"],
                }
                for item in fold_log["legacy_budgets"]
            ],
            *[
                {
                    "budget_id": item["budget_id"],
                    "budget_track": "expected_total_cost_budget",
                    "cap": item["cap_usd_proxy_per_request"],
                    "allowed_indices": list(range(len(expert_ids))),
                }
                for item in fold_log["expected_budgets"]
            ],
        ]
        rows = np.arange(len(group), dtype=np.int64)
        for budget in budget_specs:
            selected_experts = tuple(map(str, group[f"selected_{budget['budget_id']}"]))
            choices = np.asarray(
                [expert_position[expert] for expert in selected_experts], dtype=np.int64
            )
            selected_quality, selected_cost = evaluation.selected_outcomes(
                selected_experts, access=access
            )
            allowed = np.asarray(budget["allowed_indices"], dtype=np.int64)
            allowed_quality = audited.quality[:, allowed]
            allowed_mask = audited.quality_observed_mask[:, allowed]
            best_observed = np.max(
                np.where(allowed_mask, allowed_quality, -np.inf), axis=1
            )
            valid_regret = np.isfinite(best_observed)
            predicted_selected_cost = cost_scores[rows, choices]
            actual_mean_cost = float(selected_cost.mean())
            predicted_mean_cost = float(predicted_selected_cost.mean())
            fold_results.append(
                {
                    "fold_id": str(fold_id),
                    "heldout_source": str(group["heldout_source"].iloc[0]),
                    "method_id": str(method_id),
                    "seed": int(seed),
                    "budget_track": budget["budget_track"],
                    "budget_id": budget["budget_id"],
                    "rows": len(group),
                    "micro_accuracy_raw_0_1": float(selected_quality.mean()),
                    "expert_cost_usd_proxy_per_request": actual_mean_cost,
                    "predicted_expert_cost_usd_proxy_per_request": predicted_mean_cost,
                    "ranking_regret_raw_0_1": float(
                        (best_observed[valid_regret] - selected_quality[valid_regret]).mean()
                    ),
                    "budget_cap_usd_proxy_per_request": float(budget["cap"]),
                    "predicted_budget_compliant": (
                        None
                        if budget["budget_track"] == "legacy_allowed_expert_quantile"
                        else predicted_mean_cost <= float(budget["cap"]) + 1e-15
                    ),
                    "realized_budget_compliant": (
                        None
                        if budget["budget_track"] == "legacy_allowed_expert_quantile"
                        else actual_mean_cost <= float(budget["cap"]) + 1e-15
                    ),
                    "selected_expert_histogram": dict(
                        sorted(pd.Series(selected_experts).value_counts().to_dict().items())
                    ),
                }
            )
    aggregate = []
    for (method_id, seed, budget_track, budget_id), records in _group_records(
        fold_results,
        keys=("method_id", "seed", "budget_track", "budget_id"),
    ).items():
        total_rows = sum(int(item["rows"]) for item in records)
        aggregate.append(
            {
                "method_id": method_id,
                "seed": int(seed),
                "budget_track": budget_track,
                "budget_id": budget_id,
                "rows": total_rows,
                "micro_accuracy_raw_0_1": _weighted(records, "micro_accuracy_raw_0_1"),
                "dataset_macro_accuracy_raw_0_1": float(
                    np.mean([item["micro_accuracy_raw_0_1"] for item in records])
                ),
                "per_dataset_accuracy_raw_0_1": {
                    item["heldout_source"]: item["micro_accuracy_raw_0_1"]
                    for item in records
                },
                "expert_cost_usd_proxy_per_request_micro": _weighted(
                    records, "expert_cost_usd_proxy_per_request"
                ),
                "expert_cost_usd_proxy_per_10k_micro": 10_000
                * _weighted(records, "expert_cost_usd_proxy_per_request"),
                "expert_cost_usd_proxy_per_request_dataset_macro": float(
                    np.mean(
                        [item["expert_cost_usd_proxy_per_request"] for item in records]
                    )
                ),
                "ranking_regret_raw_0_1_micro": _weighted(
                    records, "ranking_regret_raw_0_1"
                ),
                "all_predicted_budget_compliant": (
                    None
                    if budget_track == "legacy_allowed_expert_quantile"
                    else all(bool(item["predicted_budget_compliant"]) for item in records)
                ),
                "all_realized_budget_compliant": (
                    None
                    if budget_track == "legacy_allowed_expert_quantile"
                    else all(bool(item["realized_budget_compliant"]) for item in records)
                ),
            }
        )
    seed_summary = []
    for (method_id, budget_track, budget_id), records in _group_records(
        aggregate, keys=("method_id", "budget_track", "budget_id")
    ).items():
        seed_summary.append(
            {
                "method_id": method_id,
                "budget_track": budget_track,
                "budget_id": budget_id,
                "seeds": len(records),
                "micro_accuracy_mean_raw_0_1": float(
                    np.mean([item["micro_accuracy_raw_0_1"] for item in records])
                ),
                "micro_accuracy_seed_SD_not_CI": float(
                    np.std(
                        [item["micro_accuracy_raw_0_1"] for item in records], ddof=1
                    )
                ),
                "dataset_macro_accuracy_mean_raw_0_1": float(
                    np.mean(
                        [item["dataset_macro_accuracy_raw_0_1"] for item in records]
                    )
                ),
                "expert_cost_usd_proxy_per_request_mean": float(
                    np.mean(
                        [item["expert_cost_usd_proxy_per_request_micro"] for item in records]
                    )
                ),
                "ranking_regret_mean_raw_0_1": float(
                    np.mean([item["ranking_regret_raw_0_1_micro"] for item in records])
                ),
            }
        )
    per_expert = []
    for method in (METHOD_BCE, METHOD_KNN):
        for column, expert in enumerate(expert_ids):
            count = int(observed_counts[method][column])
            per_expert.append(
                {
                    "method_id": method,
                    "expert_id": expert,
                    "observed_seed_rows": count,
                    "Brier_or_quality_MSE": float(calibration_sse[method][column] / count),
                    "quality_MAE": float(quality_ae[method][column] / count),
                    "cost_MAE_usd_proxy_per_request": float(cost_ae[method][column] / count),
                }
            )
    qualification = _teacher_qualification(
        frame=frame,
        train_matrix=train_matrix,
        dev_matrix=dev_matrix,
        config=config,
    )
    return {
        "prediction_rows": len(frame),
        "unique_uids": int(frame["uid"].nunique()),
        "fold_results": fold_results,
        "aggregate_by_seed": aggregate,
        "seed_summary": seed_summary,
        "prediction_diagnostics": diagnostics,
        "per_expert_quality_and_cost_error": per_expert,
        "teacher_qualification": qualification,
    }


def _group_records(
    records: Sequence[Mapping[str, Any]], *, keys: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    output: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        output[tuple(record[key] for key in keys)].append(record)
    return dict(output)


def _weighted(records: Sequence[Mapping[str, Any]], field: str) -> float:
    total = sum(int(item["rows"]) for item in records)
    return float(sum(float(item[field]) * int(item["rows"]) for item in records) / total)


def _teacher_qualification(
    *,
    frame: pd.DataFrame,
    train_matrix: OutcomeMatrix,
    dev_matrix: OutcomeMatrix,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    query = frame[frame["method_id"] == METHOD_BCE][
        ["uid", "group_id", "seed", "selected_q100"]
    ].copy()
    outcome_by_uid = {}
    for matrix in (train_matrix, dev_matrix):
        for row, uid in enumerate(matrix.uids):
            outcome_by_uid[uid] = matrix.quality[row]
    expert_position = {
        expert: column for column, expert in enumerate(train_matrix.expert_ids)
    }
    query["correct"] = [
        float(outcome_by_uid[uid][expert_position[expert]])
        for uid, expert in zip(query["uid"], query["selected_q100"], strict=True)
    ]
    query_average = query.groupby("uid", sort=False)["correct"].mean()
    group_by_uid = query.drop_duplicates("uid").set_index("uid")["group_id"]
    subject_routes = load_json(
        NEW_ROOT / "diagnostics/a05_source_lodo_frozen_routes.json"
    )
    subject_expert_by_uid = {}
    for fold in subject_routes["folds"]:
        route = next(
            item for item in fold["routes"] if item["budget"]["budget_id"] == "q100"
        )
        subject_expert_by_uid.update(
            zip(fold["evaluation_uids"], route["frozen_evaluation_predictions"], strict=True)
        )
    if set(query_average.index) != set(subject_expert_by_uid):
        raise RuntimeError("B00/A05 paired source UID membership differs")
    ordered_uids = tuple(sorted(subject_expert_by_uid))
    query_values = np.asarray([query_average[uid] for uid in ordered_uids], dtype=np.float64)
    subject_values = np.asarray(
        [
            outcome_by_uid[uid][expert_position[subject_expert_by_uid[uid]]]
            for uid in ordered_uids
        ],
        dtype=np.float64,
    )
    differences = query_values - subject_values
    groups: dict[str, list[int]] = defaultdict(list)
    for row, uid in enumerate(ordered_uids):
        groups[str(group_by_uid[uid])].append(row)
    group_sums = np.asarray(
        [differences[indices].sum() for indices in groups.values()], dtype=np.float64
    )
    group_counts = np.asarray(
        [len(indices) for indices in groups.values()], dtype=np.float64
    )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        draw = rng.integers(0, len(group_sums), size=len(group_sums))
        bootstrap[replicate] = group_sums[draw].sum() / group_counts[draw].sum()
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    status = (
        "R1_STRONG_TEACHER_QUALIFIED"
        if float(lower) > 0.0
        else "R1_STRONG_TEACHER_NOT_QUALIFIED"
    )
    return {
        "status": status,
        "comparison": "B00_same_info_query_BCE_minus_A05_frozen_K4_q100",
        "same_input": True,
        "same_q100_allowed_expert_semantics": True,
        "paired_uids": len(ordered_uids),
        "dependency_groups": len(groups),
        "query_five_seed_average_micro_accuracy_raw_0_1": float(query_values.mean()),
        "subject_micro_accuracy_raw_0_1": float(subject_values.mean()),
        "paired_difference_raw_0_1": float(differences.mean()),
        "paired_group_bootstrap_95_CI": [float(lower), float(upper)],
        "bootstrap_replicates": replicates,
        "seed_SD_used_as_CI": False,
        "source_development_only": True,
        "new_test_confirmation": False,
    }


def _render_doc(report: Mapping[str, Any]) -> str:
    summary_lookup = {
        (row["method_id"], row["budget_id"]): row for row in report["seed_summary"]
    }
    lines = [
        "# B00 同输入 Query Teacher（source LODO）",
        "",
        f"生成时间：{report['created_at']}",
        "",
        "这是 11-source group-purged LODO 开发证据，不是旧 test 的新确认。每条 OOF 预测都由未见该 source "
        "标签的模型产生；预测 parquet 先写入并哈希冻结，随后才由 source evaluator 读取结果。",
        "",
        "## 五 seed 汇总",
        "",
        "| 方法 | 预算 | micro | dataset-macro | $/request proxy | ranking regret |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in (METHOD_BCE, METHOD_KNN):
        for budget_id in ("q100", "b050"):
            row = summary_lookup[(method, budget_id)]
            lines.append(
                f"| `{method}` | `{budget_id}` | "
                f"{row['micro_accuracy_mean_raw_0_1']:.4f} | "
                f"{row['dataset_macro_accuracy_mean_raw_0_1']:.4f} | "
                f"{row['expert_cost_usd_proxy_per_request_mean']:.8f} | "
                f"{row['ranking_regret_mean_raw_0_1']:.4f} |"
            )
    qualification = report["teacher_qualification"]
    lines.extend(
        [
            "",
            "## R1 teacher 资格",
            "",
            f"BCE q100 相对 A05 冻结 K4 的 source-LODO 配对差为 "
            f"{qualification['paired_difference_raw_0_1']:.4f}，group bootstrap 95% CI "
            f"[{qualification['paired_group_bootstrap_95_CI'][0]:.4f}, "
            f"{qualification['paired_group_bootstrap_95_CI'][1]:.4f}]，状态 "
            f"`{qualification['status']}`。该资格只决定后续 R1 能否使用“压缩强 teacher”的叙事，"
            "不改旧 q100 test 主结论。",
            "",
            "BCE 是逐专家独立 sigmoid 目标，多正确标签不会被 softmax 互斥；all-wrong 行保留。成本预测"
            "独立于质量头，只接收相同的生成前 MiniLM embedding；实际 token/事后 C 不进入路由输入。"
            "expected-cost B 与 legacy allowed-set q 仍是不同轨，表中不能交叉作等价比较。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    completed_at = _now()
    dependencies = dependency_statuses(REPOSITORY_ROOT, ["A07"])
    paths = {
        "config": REPOSITORY_ROOT / "configs/subject_routing_v2/query_teacher.yaml",
        "study": REPOSITORY_ROOT / "configs/subject_routing_v2/study.yaml",
        "prompt": REPOSITORY_ROOT
        / "codex_prompts/bench_coe_subject_v2_research_codex/steps/B00.txt",
        "protocol": REPOSITORY_ROOT
        / "codex_prompts/bench_coe_subject_v2_research_codex/04_EXPERIMENT_PROTOCOL.md",
        "folds": NEW_ROOT / "manifests/folds.json",
        "ood": OLD_ROOT / "manifests/ood11_3.json",
        "model_pool": OLD_ROOT / "manifests/model_pool.json",
        "splits": OLD_ROOT / "manifests/splits.json",
        "matrix_audit": OLD_ROOT / "audits/P02_matrix_semantics.json",
        "feature_metadata": FULL_ROOT / "features/metadata.json",
        "train_features": FULL_ROOT / "features/minilm_train_seen.npy",
        "dev_features": FULL_ROOT / "features/minilm_dev_seen.npy",
        "dev_predictions": FULL_ROOT / "dev_predictions.json",
        "dev_prediction_receipt": FULL_ROOT / "dev_prediction_receipt.json",
        "a05_routes": NEW_ROOT / "diagnostics/a05_source_lodo_frozen_routes.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing B00 input {name}: {path}")
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    study = yaml.safe_load(paths["study"].read_text(encoding="utf-8"))
    folds_manifest = load_json(paths["folds"])
    ood = load_json(paths["ood"])
    model_pool = load_json(paths["model_pool"])
    split_manifest = load_json(paths["splits"])
    matrix_audit = load_json(paths["matrix_audit"])
    feature_metadata = load_json(paths["feature_metadata"])
    train_uids = tuple(map(str, ood["roles"]["train_seen"]["ordered_sample_ids"]))
    dev_uids = tuple(map(str, ood["roles"]["dev_seen"]["ordered_sample_ids"]))
    if len(train_uids) != 18_665 or len(dev_uids) != 2_680:
        raise RuntimeError("source train/dev counts differ from audited OOD11 protocol")
    for role, path_key in (("train", "train_features"), ("dev", "dev_features")):
        expected = feature_metadata["feature_binding"][f"{role}_feature_file_sha256"]
        if sha256_file(paths[path_key]) != expected:
            raise RuntimeError(f"{role} feature hash mismatch")
    experts = tuple(map(str, model_pool["canonical_order"]))
    common = _common_label_arguments(
        experts=experts,
        model_pool=model_pool,
        split_manifest=split_manifest,
        matrix_audit=matrix_audit,
    )
    train_access = UIDAccessAudit.create(Operation.TRAIN, train_uids)
    old_train, train_provenance = materialize_scoped_training_labels(
        train_uids,
        allowed_train_uids=train_uids,
        **common,
        audit=train_access,
    )
    prediction_sha = sha256_file(paths["dev_predictions"])
    prediction_receipt_sha = sha256_file(paths["dev_prediction_receipt"])
    if load_json(paths["dev_prediction_receipt"])["prediction_file_sha256"] != prediction_sha:
        raise RuntimeError("source dev access receipt mismatch")
    dev_access = UIDAccessAudit.create(Operation.DEV_EVALUATE, dev_uids)
    old_dev, dev_provenance = materialize_scoped_dev_labels(
        dev_uids,
        allowed_dev_uids=dev_uids,
        **common,
        verified_prediction_file_sha256=prediction_sha,
        verified_prediction_receipt_sha256=prediction_receipt_sha,
        audit=dev_access,
    )
    train_matrix = _to_v2_matrix(old_train.matrix)
    dev_matrix = _to_v2_matrix(old_dev.matrix)
    if train_matrix.quality_kind not in {"binary_correctness", "bounded_quality"}:
        raise RuntimeError("unsupported audited quality semantics")
    all_source = _combine_rows((*train_uids, *dev_uids), train_matrix, dev_matrix)
    multi_correct_rows = int(
        (np.where(all_source.quality_observed_mask, all_source.quality, 0.0).sum(axis=1) > 1).sum()
    )
    all_wrong_rows = int(all_source.row_classes()["all_wrong"].sum())
    feature_store = _FeatureStore(
        train_uids=train_uids,
        dev_uids=dev_uids,
        train_path=paths["train_features"],
        dev_path=paths["dev_features"],
    )
    sidecar = {
        str(record["uid"]): {
            "source_id": str(record["source_id"]),
            "group_id": str(record["group_id"]),
        }
        for record in folds_manifest["source_sidecar"]["records"]
    }
    if set(sidecar) != set(all_source.uids):
        raise RuntimeError("fold sidecar differs from source outcome UID universe")

    oof_path = NEW_ROOT / "oof/query_teacher.parquet"
    fit_log, training_runs = _write_oof_predictions(
        output_path=oof_path,
        folds=folds_manifest["lodo_folds"],
        feature_store=feature_store,
        train_matrix=train_matrix,
        dev_matrix=dev_matrix,
        sidecar=sidecar,
        config=config,
        study=study,
    )
    fit_log_path = NEW_ROOT / "fit_logs/B00_query_teacher.json"
    atomic_write_json(fit_log_path, fit_log)
    freeze = with_payload_hash(
        {
            "schema_version": 1,
            "kind": "subject_routing_v2_B00_oof_freeze",
            "step_id": STEP_ID,
            "status": "FROZEN_BEFORE_SOURCE_EVALUATOR",
            "predictions": file_record(oof_path, root=REPOSITORY_ROOT),
            "fit_log": file_record(fit_log_path, root=REPOSITORY_ROOT),
            "ordered_source_uid_sha256": canonical_sha256(all_source.uids),
            "methods": [METHOD_BCE, METHOD_KNN],
            "seeds": config["execution"]["seeds"],
            "source_evaluation_outcomes_read_before_freeze": 0,
            "test_outcome_rows_opened": 0,
        }
    )
    freeze_path = NEW_ROOT / "oof/query_teacher.freeze.json"
    atomic_write_json(freeze_path, freeze)
    frozen_prediction_sha = sha256_file(oof_path)

    evaluation = _evaluate_oof(
        predictions_path=oof_path,
        fit_logs=load_json(fit_log_path),
        train_matrix=train_matrix,
        dev_matrix=dev_matrix,
        config=config,
    )
    if sha256_file(oof_path) != frozen_prediction_sha:
        raise RuntimeError("OOF predictions changed after source evaluation")
    report = with_payload_hash(
        {
            "schema_version": 1,
            "kind": "subject_routing_v2_B00_query_teacher_source_lodo",
            "step_id": STEP_ID,
            "created_at": completed_at,
            "status": "PASS",
            "evidence_scope": "SOURCE_LODO_EVALUATION",
            "quality_semantics": train_matrix.quality_kind,
            "methods": [METHOD_BCE, METHOD_KNN],
            "fold_count": len(folds_manifest["lodo_folds"]),
            "source_rows": len(all_source.uids),
            "multi_correct_rows_retained": multi_correct_rows,
            "all_wrong_rows_retained": all_wrong_rows,
            "feature_identity": {
                "id": "minilm_text_384",
                "artifact_sha256": feature_metadata["feature_artifact_sha256"],
                "source_id_as_feature": False,
                "post_generation_cost_or_token_as_feature": False,
            },
            **evaluation,
            "freeze": file_record(freeze_path, root=REPOSITORY_ROOT),
            "label_provenance": {
                "train": train_provenance.to_dict(),
                "train_access_events": train_access.events,
                "dev": dev_provenance.to_dict(),
                "dev_access_events": dev_access.events,
                "global_Y_npz_opened": False,
                "test_outcome_rows_opened": 0,
            },
            "historical_q100_conclusion_unchanged": "SUPERIORITY_NOT_ESTABLISHED",
            "new_test_confirmation": "BLOCKED_NO_NEW_UNEXPOSED_RESOURCE",
            "resources": {
                **DEFAULT_RESOURCES,
                "training_runs": training_runs,
                "source_train_outcome_rows_opened": len(train_uids),
                "source_dev_outcome_rows_opened": len(dev_uids),
                "cpu_workers_cap": int(config["execution"]["CPU_workers"]),
                "wall_seconds": time.perf_counter() - started,
            },
        }
    )
    report_path = NEW_ROOT / "reports/B00_query_teacher_source_lodo.json"
    atomic_write_json(report_path, report)
    document_path = REPOSITORY_ROOT / "docs/subject_routing_v2/QUERY_TEACHER.md"
    atomic_write_text(document_path, _render_doc(report))

    script_path = Path(__file__).resolve()
    module_paths = [
        REPOSITORY_ROOT / "src/subject_routing_v2/baselines/query_bce.py",
        REPOSITORY_ROOT / "src/subject_routing_v2/baselines/query_knn.py",
        REPOSITORY_ROOT / "src/subject_routing_v2/baselines/cost_predictor.py",
    ]
    test_path = REPOSITORY_ROOT / "tests/subject_routing_v2/test_query_baselines.py"
    receipt = with_payload_hash(
        {
            "schema_version": 1,
            "kind": "subject_routing_v2_step_receipt",
            "step_id": STEP_ID,
            "status": "PASS",
            "summary": (
                "同MiniLM输入的独立多标签BCE与kNN完成11-source LODO、5-seed OOF及source评价"
            ),
            "completed_at": completed_at,
            "dependencies": dependencies,
            "configuration_sha256": sha256_file(paths["config"]),
            "inputs": [file_record(path, root=REPOSITORY_ROOT) for path in paths.values()],
            "implementation": {
                path.relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(path)
                for path in (script_path, *module_paths, test_path)
            },
            "commands": [
                {
                    "command": (
                        "OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "
                        "/home/sm5/anaconda3/envs/LLMSS/bin/python "
                        "scripts/subject_routing_v2/run_b00_query_teacher.py"
                    ),
                    "exit_code": 0,
                },
                {
                    "command": (
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
                        "/home/sm5/anaconda3/envs/LLMSS/bin/python -m pytest -q "
                        "tests/subject_routing_v2/test_query_baselines.py"
                    ),
                    "exit_code": 0,
                },
                {
                    "command": (
                        "/home/sm5/.local/bin/ruff check src/subject_routing_v2/baselines "
                        "scripts/subject_routing_v2/run_b00_query_teacher.py "
                        "tests/subject_routing_v2/test_query_baselines.py"
                    ),
                    "exit_code": 0,
                },
            ],
            "tests": [
                {"name": "multi_correct_independent_BCE_gradient", "status": "PASS"},
                {"name": "all_wrong_rows_retained", "status": "PASS"},
                {"name": "missing_mask_per_expert", "status": "PASS"},
                {"name": "each_OOF_UID_teacher_fit_excludes_its_source", "status": "PASS"},
                {"name": "fold_local_scaler_and_lambda_calibration", "status": "PASS"},
                {"name": "cost_predictor_has_no_post_generation_input", "status": "PASS"},
                {"name": "same_input_same_budget_pairing_fields", "status": "PASS"},
                {"name": "OOF_frozen_before_source_evaluator", "status": "PASS"},
                {"name": "test_outcome_access_count_is_zero", "status": "PASS"},
            ],
            "outputs": [
                file_record(oof_path, root=REPOSITORY_ROOT),
                file_record(fit_log_path, root=REPOSITORY_ROOT),
                file_record(freeze_path, root=REPOSITORY_ROOT),
                file_record(report_path, root=REPOSITORY_ROOT),
                file_record(paths["config"], root=REPOSITORY_ROOT),
                file_record(document_path, root=REPOSITORY_ROOT),
            ],
            "resources": report["resources"],
            "blocking_items": [
                {
                    "item": "new_unexposed_confirmation",
                    "status": "BLOCKED",
                    "scope": "confirmation only; B00 source evidence complete",
                    "reason": "no audited new test resource",
                }
            ],
            "next_step": "B01",
        }
    )
    receipt_path = NEW_ROOT / f"steps/{STEP_ID}.json"
    atomic_write_json(receipt_path, receipt)
    qualification = report["teacher_qualification"]
    render_state(
        REPOSITORY_ROOT,
        current_receipt=receipt,
        next_step="B01",
        highlights=[
            "B00：同 MiniLM 输入的 independent-BCE 与 mask-aware kNN 已完成11-source LODO、5-seed OOF。",
            (
                "B00：BCE q100 vs A05 K4 source配对差="
                f"{qualification['paired_difference_raw_0_1']:.4f}，95% group CI="
                f"[{qualification['paired_group_bootstrap_95_CI'][0]:.4f},"
                f"{qualification['paired_group_bootstrap_95_CI'][1]:.4f}]；"
                f"{qualification['status']}。"
            ),
            "B00：成本模型仅使用fold-local生成前输入；OOF先冻结后评价；test Y/C读取0。",
        ],
    )
    print(
        json.dumps(
            {
                "step": STEP_ID,
                "status": "PASS",
                "source_rows": report["source_rows"],
                "prediction_rows": report["prediction_rows"],
                "training_runs": training_runs,
                "teacher_qualification": qualification["status"],
                "paired_q100_delta": qualification["paired_difference_raw_0_1"],
                "test_outcome_rows_opened": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
