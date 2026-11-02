"""Independent Prompt-15 evaluator, unlocked only after every prediction is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from routers.utils.rank_score import rank_score

from .checkpoints import checkpoint_sha256
from .config import ExperimentContract, load_contract, sha256_file
from .data import load_vlrb_strict
from .errors import PredictionIntegrityError
from .evaluator import (
    FALLBACK_METADATA_FIELDS,
    _cost_bounds,
    _usage,
    expected_multi_image_fallback_sample_ids,
    read_frozen_predictions,
    validate_predictions,
)
from .labels import build_diagnostic_labels_for_split, compute_train_expert_accuracy
from .ood_common import (
    OOD_BACKBONES,
    OOD_CONFIG_PATH,
    OOD_EVAL_ROLES,
    OOD_GATE_PATH,
    OOD_PROTOCOL_PATH,
    OOD_SEEDS,
    atomic_json,
    atomic_text,
    load_strict_json,
    ood_run_dir,
    ood_training_binding,
)
from .ood_protocol import load_and_validate_protocol
from .target_free import (
    deny_target_matrix_access,
    load_and_validate_target_free_receipt,
    load_target_free_vlrb,
    prediction_receipt_path,
)


BARRIER_PATH = Path("outputs/bench_coe_query/ood/PREDICTION_BARRIER.json")


def _relative(contract: ExperimentContract, path: str | Path) -> str:
    return Path(path).resolve().relative_to(contract.dataset_root).as_posix()


def _validate_receipt_binding(
    receipt: Mapping[str, Any],
    *,
    contract: ExperimentContract,
    protocol: Mapping[str, Any],
    role: str,
    binding_sha256: str,
    gate_file_sha256: str,
) -> None:
    binding = receipt.get("input_binding", {})
    prompt15 = binding.get("prompt15_protocol", {})
    expected = {
        "claim": "reconstructed_held_out_protocol",
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "protocol_file_sha256": sha256_file(contract.dataset_root / OOD_PROTOCOL_PATH),
        "protocol_role": role,
        "role_manifest_sha256": protocol["views"][role]["sha256"],
        "training_binding_sha256": binding_sha256,
        "execution_gate_sha256": gate_file_sha256,
        "official_split": "test",
    }
    if prompt15 != expected:
        raise PredictionIntegrityError(f"OOD receipt protocol binding mismatch for {role}")
    if binding.get("kind") not in {
        "verified_local_bert_payload",
        "verified_frozen_feature_cache",
    }:
        raise PredictionIntegrityError("OOD receipt input kind is not approved")


def build_prediction_barrier(
    contract: ExperimentContract,
    *,
    dataset_dir: str | Path = ".",
    config_path: str | Path = OOD_CONFIG_PATH,
    protocol_path: str | Path = OOD_PROTOCOL_PATH,
    gate_path: str | Path = OOD_GATE_PATH,
    output_path: str | Path = BARRIER_PATH,
) -> dict[str, Any]:
    """Validate all 30 frozen files without Y/C, after all 15 producers exited."""

    gate_file = contract.dataset_root / gate_path
    gate_file_sha256 = sha256_file(gate_file)
    gate = load_strict_json(gate_file)
    if gate.get("passed") is not True:
        raise PredictionIntegrityError("OOD execution gate is not passing")
    artifacts: list[dict[str, Any]] = []
    producer_pids: set[int] = set()
    with deny_target_matrix_access(contract) as guard:
        data = load_target_free_vlrb(dataset_dir, contract=contract)
        protocol, view = load_and_validate_protocol(
            contract,
            data,
            config_path=config_path,
            protocol_path=protocol_path,
            validate_training_statistics=False,
        )
        binding_sha256, _binding = ood_training_binding(contract, protocol)
        for backbone in OOD_BACKBONES:
            for seed in OOD_SEEDS:
                run_dir = ood_run_dir(contract, backbone, seed)
                completion_path = run_dir / "OOD_TRAINING_COMPLETE.json"
                completion = load_strict_json(completion_path)
                checkpoint = run_dir / "checkpoints/best/checkpoint.pt"
                checkpoint_id = checkpoint_sha256(checkpoint)
                if (
                    completion.get("passed") is not True
                    or completion.get("training_binding_sha256") != binding_sha256
                    or completion.get("best_checkpoint_sha256") != checkpoint_id
                ):
                    raise PredictionIntegrityError(
                        f"invalid OOD training completion: {backbone}/seed-{seed}"
                    )
                freeze_path = run_dir / "PREDICTION_FREEZE.json"
                freeze = load_strict_json(freeze_path)
                if (
                    freeze.get("passed") is not True
                    or freeze.get("evaluation_performed") is not False
                    or freeze.get("checkpoint_sha256") != checkpoint_id
                    or freeze.get("training_binding_sha256") != binding_sha256
                ):
                    raise PredictionIntegrityError(
                        f"invalid OOD prediction freeze: {backbone}/seed-{seed}"
                    )
                for role in OOD_EVAL_ROLES:
                    prediction = run_dir / f"predictions_{role}.jsonl"
                    receipt_path = prediction_receipt_path(prediction)
                    receipt = load_and_validate_target_free_receipt(
                        receipt_path,
                        contract=contract,
                        data=data,
                        prediction_path=prediction,
                        checkpoint_path=checkpoint,
                        expected_backbone=backbone,
                        expected_split="test",
                        expected_seed=seed,
                    )
                    _validate_receipt_binding(
                        receipt,
                        contract=contract,
                        protocol=protocol,
                        role=role,
                        binding_sha256=binding_sha256,
                        gate_file_sha256=gate_file_sha256,
                    )
                    expected_ids = {
                        data.samples[row].sample_id for row in view.rows[role]
                    }
                    records, digest = read_frozen_predictions(prediction)
                    observed_ids = {str(record["sample_id"]) for record in records}
                    if observed_ids != expected_ids:
                        raise PredictionIntegrityError(
                            f"target-free OOD coverage mismatch for {backbone}/{seed}/{role}"
                        )
                    producer_pid = int(receipt.get("producer_pid", -1))
                    if producer_pid <= 0:
                        raise PredictionIntegrityError("invalid OOD producer PID")
                    producer_pids.add(producer_pid)
                    artifacts.append(
                        {
                            "backbone": backbone,
                            "seed": seed,
                            "role": role,
                            "prediction": _relative(contract, prediction),
                            "prediction_sha256": digest,
                            "records": len(records),
                            "receipt": _relative(contract, receipt_path),
                            "receipt_file_sha256": receipt["receipt_file_sha256"],
                            "checkpoint_sha256": checkpoint_id,
                            "producer_pid": producer_pid,
                        }
                    )
    if guard.get("passed") is not True:
        raise PredictionIntegrityError("prediction barrier attempted target-matrix access")
    alive = [pid for pid in sorted(producer_pids) if Path(f"/proc/{pid}").exists()]
    if alive:
        raise PredictionIntegrityError(f"OOD producer processes have not exited: {alive}")
    if len(artifacts) != 30 or len(producer_pids) != 15:
        raise PredictionIntegrityError(
            f"OOD prediction barrier expected 30 files/15 producers, got {len(artifacts)}/{len(producer_pids)}"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "prompt15_prediction_before_evaluation_barrier",
        "passed": True,
        "contract_sha256": contract.sha256,
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "execution_gate_file_sha256": gate_file_sha256,
        "training_binding_sha256": binding_sha256,
        "producer_processes": len(producer_pids),
        "producer_pids": sorted(producer_pids),
        "all_producers_exited_before_barrier": True,
        "prediction_files": len(artifacts),
        "prediction_records": sum(int(item["records"]) for item in artifacts),
        "target_matrix_guard": guard,
        "artifacts": artifacts,
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["barrier_manifest_sha256"] = hashlib.sha256(unsigned.encode("utf-8")).hexdigest()
    output = contract.dataset_root / output_path
    atomic_json(output, payload)
    return payload


def _load_barrier(
    contract: ExperimentContract,
    *,
    prediction: Path,
    role: str,
    backbone: str,
    seed: int,
    prediction_sha256: str,
    barrier_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = contract.dataset_root / barrier_path
    barrier = load_strict_json(path)
    recorded = str(barrier.get("barrier_manifest_sha256", ""))
    unsigned = dict(barrier)
    unsigned.pop("barrier_manifest_sha256", None)
    observed = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    if recorded != observed or barrier.get("passed") is not True:
        raise PredictionIntegrityError("invalid OOD prediction barrier")
    relative = _relative(contract, prediction)
    matches = [
        item
        for item in barrier.get("artifacts", [])
        if item.get("backbone") == backbone
        and item.get("seed") == seed
        and item.get("role") == role
        and item.get("prediction") == relative
    ]
    if len(matches) != 1 or matches[0].get("prediction_sha256") != prediction_sha256:
        raise PredictionIntegrityError("prediction is absent or changed since global barrier")
    return barrier, matches[0]


def _validate_ood_fallback(
    records: list[dict[str, Any]],
    data: Any,
    rows: np.ndarray,
    predictions: np.ndarray,
    *,
    backbone: str,
    strongest_column: int,
) -> np.ndarray:
    expected = (
        set()
        if backbone == "bert"
        else set(
            expected_multi_image_fallback_sample_ids(
                data, (str(record["sample_id"]) for record in records)
            )
        )
    )
    fallback_mask = np.asarray(
        [record.get("routing_input_status") == "structured_asset_error_fallback" for record in records],
        dtype=bool,
    )
    observed = {
        str(records[position]["sample_id"]) for position in np.flatnonzero(fallback_mask)
    }
    if observed != expected:
        raise PredictionIntegrityError(
            f"OOD structured fallback set mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}"
        )
    expected_logits = np.full(len(data.model_order), -1.0, dtype=np.float64)
    expected_logits[strongest_column] = 1.0
    for position, record in enumerate(records):
        if not fallback_mask[position]:
            if FALLBACK_METADATA_FIELDS.intersection(record):
                raise PredictionIntegrityError("non-fallback OOD row carries fallback metadata")
            continue
        sample = data.samples[int(rows[position])]
        required = {
            "asset_error_type": "multi_image",
            "image_count": len(sample.assets),
            "fallback_policy": "train_strongest_global",
            "evaluation_denominator": "official_split_frozen",
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise PredictionIntegrityError("invalid OOD fallback metadata")
        if int(predictions[position]) != strongest_column:
            raise PredictionIntegrityError("OOD fallback does not use train_seen strongest expert")
        if not np.array_equal(np.asarray(record["logits"], dtype=np.float64), expected_logits):
            raise PredictionIntegrityError("OOD fallback logits differ from deterministic policy")
    return fallback_mask


def _evaluate_records(
    records: list[dict[str, Any]],
    data: Any,
    *,
    role: str,
    view: Any,
    protocol: Mapping[str, Any],
    backbone: str,
    seed: int,
    checkpoint_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_ids = tuple(data.samples[row].sample_id for row in view.rows[role])
    rows, predictions, split = validate_predictions(
        records,
        data,
        expected_split="test",
        expected_sample_ids=expected_ids,
        expected_seed=seed,
        expected_checkpoint_id=checkpoint_id,
    )
    if split != "test" or any(record.get("protocol_role") != role for record in records):
        raise PredictionIntegrityError("OOD prediction protocol-role annotation mismatch")
    train_accuracy = compute_train_expert_accuracy(
        data.y,
        view.rows["train_seen"],
        data.model_order,
        expected_model_order=data.model_order,
    )
    strongest = int(np.argmax(train_accuracy))
    if strongest != int(protocol["training_statistics"]["strongest_expert_index"]):
        raise PredictionIntegrityError("live train_seen strongest expert differs from protocol")
    fallback_mask = _validate_ood_fallback(
        records,
        data,
        rows,
        predictions,
        backbone=backbone,
        strongest_column=strongest,
    )
    selected_correct = data.y[rows, predictions].astype(np.int8)
    selected_cost = data.c[rows, predictions].astype(np.float64)
    routed_accuracy = float(selected_correct.mean())
    average_cost = float(selected_cost.mean())
    cmin, cmax, beta = _cost_bounds(data)
    labels = build_diagnostic_labels_for_split(
        data.y,
        rows,
        train_accuracy,
        data.model_order,
        expected_model_order=data.model_order,
    )
    trainable = labels.is_trainable
    dataset_values = np.asarray([data.samples[row].dataset for row in rows], dtype=object)
    domain_by_dataset = {
        dataset: domain
        for domain, dataset in protocol["held_out_datasets_by_domain"].items()
    }
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted(set(dataset_values.tolist())):
        mask = dataset_values == dataset
        accuracy = float(selected_correct[mask].mean())
        cost = float(selected_cost[mask].mean())
        per_dataset[dataset] = {
            "domain": domain_by_dataset.get(dataset, "seen"),
            "samples": int(mask.sum()),
            "routed_accuracy": accuracy,
            "average_cost": cost,
            "rank_score": float(rank_score(accuracy, cost, cmin, cmax, beta=beta)),
        }
    worst_dataset = min(
        per_dataset,
        key=lambda name: (per_dataset[name]["routed_accuracy"], name),
    )
    usage = _usage(predictions, data.model_order)
    dominant = max(
        data.model_order,
        key=lambda model_id: (usage["shares"][model_id], -data.model_order.index(model_id)),
    )
    usage["dominant_expert_id"] = dominant
    usage["dominant_expert_index"] = data.model_order.index(dominant)
    correct_counts = labels.correct_expert_count
    strata_masks = {
        "all_wrong": correct_counts == 0,
        "single_correct": correct_counts == 1,
        "multi_correct": correct_counts > 1,
    }
    strongest_correct = data.y[rows, strongest].astype(np.int8)
    oracle_correct = (data.y[rows].sum(axis=1) > 0).astype(np.int8)
    selected = [
        {
            "sample_id": str(records[position]["sample_id"]),
            "row_index": int(rows[position]),
            "official_split": "test",
            "protocol_role": role,
            "dataset": data.samples[int(rows[position])].dataset,
            "domain": domain_by_dataset.get(data.samples[int(rows[position])].dataset, "seen"),
            "predicted_expert_index": int(predictions[position]),
            "predicted_expert_id": data.model_order[int(predictions[position])],
            "selected_correct": int(selected_correct[position]),
            "selected_cost": float(selected_cost[position]),
            "hard_label_index": int(labels.labels[position]),
            "correct_expert_count": int(correct_counts[position]),
            "routing_input_status": str(records[position].get("routing_input_status")),
        }
        for position in range(len(records))
    ]
    metrics = {
        "schema_version": 1,
        "kind": "prompt15_ood_role_metrics",
        "claim": "reconstructed held-out protocol; not exact Table A2 membership",
        "backbone": backbone,
        "seed": seed,
        "protocol_role": role,
        "official_split": "test",
        "num_samples": len(records),
        "routed_accuracy": routed_accuracy,
        "average_cost": average_cost,
        "rank_score": float(rank_score(routed_accuracy, average_cost, cmin, cmax, beta=beta)),
        "rank_score_beta": beta,
        "hard_label_accuracy_non_all_wrong": (
            float((predictions[trainable] == labels.labels[trainable]).mean())
            if trainable.any()
            else None
        ),
        "hard_label_samples": int(trainable.sum()),
        "per_dataset": per_dataset,
        "dataset_macro": {
            "routed_accuracy": float(
                np.mean([value["routed_accuracy"] for value in per_dataset.values()])
            ),
            "average_cost": float(
                np.mean([value["average_cost"] for value in per_dataset.values()])
            ),
        },
        "worst_dataset": {
            "dataset": worst_dataset,
            "domain": per_dataset[worst_dataset]["domain"],
            "routed_accuracy": per_dataset[worst_dataset]["routed_accuracy"],
            "samples": per_dataset[worst_dataset]["samples"],
        },
        "expert_usage": usage,
        "expert_collapse": {
            "max_share": usage["max_share"],
            "normalized_entropy": usage["normalized_entropy"],
            "gini": usage["gini"],
            "dominant_expert_id": dominant,
        },
        "input_coverage": {
            "native_or_feature_rows": int((~fallback_mask).sum()),
            "structured_fallback_rows": int(fallback_mask.sum()),
            "structured_fallback_fraction": float(fallback_mask.mean()),
            "fallback_policy": "train_seen_strongest_global" if fallback_mask.any() else None,
            "denominator": "protocol_role_complete",
        },
        "difficulty_strata": {
            name: {
                "samples": int(mask.sum()),
                "routed_accuracy": float(selected_correct[mask].mean()) if mask.any() else None,
            }
            for name, mask in strata_masks.items()
        },
        "strongest_global": {
            "selected_from": "train_seen_only",
            "expert_index": strongest,
            "expert_id": data.model_order[strongest],
            "role_accuracy": float(strongest_correct.mean()),
            "delta_routed_accuracy": routed_accuracy - float(strongest_correct.mean()),
        },
        "oracle": {"role_accuracy": float(oracle_correct.mean())},
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "held_out_y_used_for_training_or_tuning": False,
    }
    return metrics, selected


def evaluate_ood_prediction_file(
    contract: ExperimentContract,
    *,
    backbone: str,
    seed: int,
    role: str,
    dataset_dir: str | Path = ".",
    config_path: str | Path = OOD_CONFIG_PATH,
    protocol_path: str | Path = OOD_PROTOCOL_PATH,
    gate_path: str | Path = OOD_GATE_PATH,
    barrier_path: str | Path = BARRIER_PATH,
    write_outputs: bool = True,
) -> dict[str, Any]:
    if backbone not in OOD_BACKBONES or seed not in OOD_SEEDS or role not in OOD_EVAL_ROLES:
        raise ValueError(f"invalid OOD evaluation target: {backbone}/{seed}/{role}")
    run_dir = ood_run_dir(contract, backbone, seed)
    prediction = run_dir / f"predictions_{role}.jsonl"
    records, prediction_sha256 = read_frozen_predictions(prediction)
    barrier, barrier_record = _load_barrier(
        contract,
        prediction=prediction,
        role=role,
        backbone=backbone,
        seed=seed,
        prediction_sha256=prediction_sha256,
        barrier_path=barrier_path,
    )
    checkpoint = run_dir / "checkpoints/best/checkpoint.pt"
    checkpoint_id = checkpoint_sha256(checkpoint)
    receipt_path = prediction_receipt_path(prediction)
    with deny_target_matrix_access(contract) as guard:
        target_free_data = load_target_free_vlrb(dataset_dir, contract=contract)
        target_protocol, _target_view = load_and_validate_protocol(
            contract,
            target_free_data,
            config_path=config_path,
            protocol_path=protocol_path,
            validate_training_statistics=False,
        )
        binding_sha256, _binding = ood_training_binding(contract, target_protocol)
        receipt = load_and_validate_target_free_receipt(
            receipt_path,
            contract=contract,
            data=target_free_data,
            prediction_path=prediction,
            checkpoint_path=checkpoint,
            expected_backbone=backbone,
            expected_split="test",
            expected_seed=seed,
        )
        _validate_receipt_binding(
            receipt,
            contract=contract,
            protocol=target_protocol,
            role=role,
            binding_sha256=binding_sha256,
            gate_file_sha256=sha256_file(contract.dataset_root / gate_path),
        )
    if guard.get("passed") is not True:
        raise PredictionIntegrityError("OOD receipt validation accessed Y/C")

    data = load_vlrb_strict(dataset_dir, contract=contract)
    protocol, view = load_and_validate_protocol(
        contract,
        data,
        config_path=config_path,
        protocol_path=protocol_path,
        validate_training_statistics=True,
    )
    metrics, selected = _evaluate_records(
        records,
        data,
        role=role,
        view=view,
        protocol=protocol,
        backbone=backbone,
        seed=seed,
        checkpoint_id=checkpoint_id,
    )
    if sha256_file(prediction) != prediction_sha256:
        raise PredictionIntegrityError("OOD prediction changed during evaluation")
    metrics["prediction_file"] = _relative(contract, prediction)
    metrics["prediction_sha256"] = prediction_sha256
    metrics["prediction_unchanged"] = True
    metrics["contract_sha256"] = contract.sha256
    metrics["prediction_provenance"] = {
        "checkpoint_sha256": checkpoint_id,
        "checkpoint_bytes_verified": checkpoint_id == barrier_record["checkpoint_sha256"],
        "receipt": _relative(contract, receipt_path),
        "receipt_file_sha256": receipt["receipt_file_sha256"],
        "receipt_verified_before_y_c_load": True,
        "producer_loaded_y_or_c": False,
        "producer_evaluated_predictions": False,
        "global_barrier": _relative(contract, contract.dataset_root / barrier_path),
        "global_barrier_file_sha256": sha256_file(contract.dataset_root / barrier_path),
        "global_barrier_manifest_sha256": barrier["barrier_manifest_sha256"],
        "all_15_producers_exited_before_any_evaluation": barrier[
            "all_producers_exited_before_barrier"
        ],
        "training_binding_sha256": binding_sha256,
    }
    if write_outputs:
        atomic_json(run_dir / f"metrics_{role}.json", metrics)
        atomic_text(
            run_dir / f"selected_{role}.jsonl",
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
                for record in selected
            ),
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    barrier = subparsers.add_parser("barrier")
    barrier.add_argument("--dataset-dir", default=".")
    barrier.add_argument("--contract", default="configs/bench_coe_query/experiment_contract.yaml")
    barrier.add_argument("--ood-config", default=str(OOD_CONFIG_PATH))
    barrier.add_argument("--protocol", default=str(OOD_PROTOCOL_PATH))
    barrier.add_argument("--gate", default=str(OOD_GATE_PATH))
    barrier.add_argument("--output", default=str(BARRIER_PATH))
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--backbone", required=True, choices=OOD_BACKBONES)
    evaluate.add_argument("--seed", required=True, type=int, choices=OOD_SEEDS)
    evaluate.add_argument("--role", required=True, choices=OOD_EVAL_ROLES)
    evaluate.add_argument("--dataset-dir", default=".")
    evaluate.add_argument("--contract", default="configs/bench_coe_query/experiment_contract.yaml")
    evaluate.add_argument("--ood-config", default=str(OOD_CONFIG_PATH))
    evaluate.add_argument("--protocol", default=str(OOD_PROTOCOL_PATH))
    evaluate.add_argument("--gate", default=str(OOD_GATE_PATH))
    evaluate.add_argument("--barrier", default=str(BARRIER_PATH))
    args = parser.parse_args()
    contract = load_contract(args.contract)
    if args.command == "barrier":
        result = build_prediction_barrier(
            contract,
            dataset_dir=args.dataset_dir,
            config_path=args.ood_config,
            protocol_path=args.protocol,
            gate_path=args.gate,
            output_path=args.output,
        )
    else:
        result = evaluate_ood_prediction_file(
            contract,
            backbone=args.backbone,
            seed=args.seed,
            role=args.role,
            dataset_dir=args.dataset_dir,
            config_path=args.ood_config,
            protocol_path=args.protocol,
            gate_path=args.gate,
            barrier_path=args.barrier,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
