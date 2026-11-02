"""Freeze ID/OOD predictions in a process that cannot access Y or C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .backbones.utils import force_offline_environment, require_model_identity
from .checkpoints import checkpoint_sha256, load_checkpoint
from .config import ExperimentContract, effective_model_path, load_contract, sha256_file
from .feature_cache import load_full_feature_cache
from .full_policy import materialize_frozen_denominator_logits
from .heads import AuditedBenchCoEHead
from .input_content import feature_input_content_fields, verify_pinned_input_content
from .ood_common import (
    OOD_CONFIG_PATH,
    OOD_GATE_PATH,
    OOD_PROTOCOL_PATH,
    atomic_json,
    full_feature_cache_root,
    load_strict_json,
    ood_run_dir,
    ood_training_binding,
)
from .ood_gate import assert_ood_execution_gate
from .ood_protocol import load_and_validate_protocol
from .predictor import write_logits_predictions
from .reproducibility import seed_everything
from .runtime_predict import _predict_bert_logits, _predict_head_logits
from .target_free import (
    TargetFreeVLRouterData,
    deny_target_matrix_access,
    load_target_free_vlrb,
    prediction_receipt_path,
    write_target_free_receipt,
)


def _producer_sources(contract: ExperimentContract, backbone: str) -> tuple[Path, ...]:
    relative = [
        "routers/bench_coe_query/ood_prediction.py",
        "routers/bench_coe_query/ood_common.py",
        "routers/bench_coe_query/ood_protocol.py",
        "routers/bench_coe_query/ood_gate.py",
        "routers/bench_coe_query/runtime_predict.py",
        "routers/bench_coe_query/predictor.py",
        "routers/bench_coe_query/target_free.py",
        "routers/bench_coe_query/checkpoints.py",
        "routers/bench_coe_query/reproducibility.py",
    ]
    if backbone == "bert":
        relative.append("routers/bench_coe_query/backbones/utils.py")
    else:
        relative.extend(
            [
                "routers/bench_coe_query/feature_cache.py",
                "routers/bench_coe_query/full_policy.py",
                "routers/bench_coe_query/heads.py",
            ]
        )
    return tuple(contract.dataset_root / value for value in relative)


def _verified_checkpoint(
    contract: ExperimentContract,
    backbone: str,
    seed: int,
    binding_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = ood_run_dir(contract, backbone, seed)
    completion = load_strict_json(run_dir / "OOD_TRAINING_COMPLETE.json")
    checkpoint = run_dir / "checkpoints/best/checkpoint.pt"
    digest = checkpoint_sha256(checkpoint)
    if (
        completion.get("passed") is not True
        or completion.get("backbone") != backbone
        or completion.get("seed") != seed
        or completion.get("training_binding_sha256") != binding_sha256
        or completion.get("best_checkpoint_sha256") != digest
        or completion.get("test_or_ood_used_for_selection") is not False
    ):
        raise RuntimeError(f"invalid OOD training completion for {backbone}/seed-{seed}")
    return checkpoint, completion


def _load_bert(
    contract: ExperimentContract,
    data: TargetFreeVLRouterData,
    checkpoint: Path,
    binding_sha256: str,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    verification = verify_pinned_input_content(
        contract, model_names=("bert",), include_visual_assets=False
    )
    model_path = effective_model_path(contract, "bert")
    require_model_identity(
        model_path,
        expected_model_type="bert",
        expected_architecture="BertForMaskedLM",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=True, local_files_only=True
    )
    id2label = {index: model_id for index, model_id in enumerate(data.model_order)}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=len(data.model_order),
        id2label=id2label,
        label2id={model_id: index for index, model_id in id2label.items()},
        local_files_only=True,
        ignore_mismatched_sizes=True,
    ).to(device)
    load_checkpoint(
        checkpoint,
        model=model,
        expected_config_sha256=binding_sha256,
        map_location=device,
    )
    return model, tokenizer, verification


def run_prediction(args: argparse.Namespace, contract: ExperimentContract) -> int:
    force_offline_environment()
    gate = assert_ood_execution_gate(contract, phase="predict", gate_path=args.gate)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("OOD prediction requires exactly one CUDA_VISIBLE_DEVICES GPU")
    seed_info = seed_everything(args.seed)
    device = torch.device("cuda")
    run_dir = ood_run_dir(contract, args.backbone, args.seed)

    with deny_target_matrix_access(contract) as guard:
        data = load_target_free_vlrb(args.dataset_dir, contract=contract)
        protocol, view = load_and_validate_protocol(
            contract,
            data,
            config_path=args.ood_config,
            protocol_path=args.protocol,
            validate_training_statistics=False,
        )
        binding_sha256, _binding = ood_training_binding(contract, protocol)
        checkpoint, completion = _verified_checkpoint(
            contract, args.backbone, args.seed, binding_sha256
        )
        checkpoint_id = checkpoint_sha256(checkpoint)
        logits_by_role: dict[str, np.ndarray] = {}
        annotations_by_role: dict[str, list[dict[str, Any]]] = {}
        common_binding: dict[str, Any]

        if args.backbone == "bert":
            model, tokenizer, verification = _load_bert(
                contract, data, checkpoint, binding_sha256, device
            )
            profile = contract.values["profiles"]["vlrb_adapted_v1"]["bert"]
            for role in ("id_test", "ood_test"):
                rows = view.rows[role]
                logits_by_role[role] = _predict_bert_logits(
                    model,
                    tokenizer,
                    [data.samples[row] for row in rows],
                    max_length=int(contract.values["models"]["bert"]["max_length"]),
                    batch_size=int(profile["batch_size_per_gpu"]),
                    device=device,
                )
                annotations_by_role[role] = [
                    {"protocol_role": role, "routing_input_status": "text_query"}
                    for _ in rows
                ]
            common_binding = {
                "kind": "verified_local_bert_payload",
                "verification": verification,
            }
        else:
            cache_root = full_feature_cache_root(contract, args.backbone)
            features, records, manifest, integrity = load_full_feature_cache(
                cache_root,
                dataset_root=contract.dataset_root,
                expected_contract_sha256=contract.sha256,
                expected_input_content=feature_input_content_fields(
                    contract, args.backbone
                ),
            )
            position_by_row = {
                int(record["row_index"]): position for position, record in enumerate(records)
            }
            if len(position_by_row) != len(records):
                raise RuntimeError("OOD feature cache contains duplicate rows")
            if any(
                data.samples[int(record["row_index"])].sample_id != str(record["sample_id"])
                for record in records
            ):
                raise RuntimeError("OOD feature cache sample IDs differ from target-free registry")
            profile = contract.values["profiles"]["vlrb_adapted_v1"]["frozen_head"]
            model = AuditedBenchCoEHead(
                hidden_dim=int(features.shape[1]),
                num_experts=len(data.model_order),
                dropout=float(profile["dropout"]),
            ).to(device)
            load_checkpoint(
                checkpoint,
                model=model,
                expected_config_sha256=binding_sha256,
                expected_feature_manifest_sha256=str(manifest["manifest_sha256"]),
                map_location=device,
            )
            strongest = int(protocol["training_statistics"]["strongest_expert_index"])
            for role in ("id_test", "ood_test"):
                rows = tuple(view.rows[role])
                feature_rows = tuple(row for row in rows if row in position_by_row)
                positions = np.asarray(
                    [position_by_row[row] for row in feature_rows], dtype=np.int64
                )
                feature_logits = _predict_head_logits(
                    model,
                    features[positions],
                    batch_size=int(profile["batch_size_per_gpu"]),
                    device=device,
                )
                logits, annotations, _fallback_rows = materialize_frozen_denominator_logits(
                    data=data,
                    official_rows=rows,
                    feature_rows=feature_rows,
                    feature_logits=feature_logits,
                    strongest_train_column=strongest,
                )
                logits_by_role[role] = logits
                annotations_by_role[role] = [
                    {**annotation, "protocol_role": role} for annotation in annotations
                ]
            common_binding = {
                "kind": "verified_frozen_feature_cache",
                "cache_root": integrity["cache_root"],
                "feature_signature": integrity["feature_signature"],
                "manifest_internal_sha256": integrity["manifest_internal_sha256"],
                "features_sha256": integrity["features_sha256"],
                "index_sha256": integrity["index_sha256"],
                "pointer_sha256": integrity["pointer_sha256"],
                "gate_sha256": integrity["gate_sha256"],
                "input_content": integrity["input_content"],
                "train_seen_only_fallback_policy": {
                    "strongest_expert_column": strongest,
                    "strongest_expert_id": data.model_order[strongest],
                    "statistics_source": "train_seen",
                },
                "passed": True,
            }

        predictions: dict[str, Any] = {}
        for role in ("id_test", "ood_test"):
            rows = tuple(view.rows[role])
            output = run_dir / f"predictions_{role}.jsonl"
            summary = write_logits_predictions(
                output,
                sample_ids=[data.samples[row].sample_id for row in rows],
                row_indices=rows,
                logits=logits_by_role[role],
                model_order=data.model_order,
                split="test",
                seed=args.seed,
                checkpoint_id=checkpoint_id,
                annotations=annotations_by_role[role],
            )
            input_binding = {
                **common_binding,
                "prompt15_protocol": {
                    "claim": "reconstructed_held_out_protocol",
                    "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
                    "protocol_file_sha256": sha256_file(
                        contract.dataset_root / OOD_PROTOCOL_PATH
                    ),
                    "protocol_role": role,
                    "role_manifest_sha256": protocol["views"][role]["sha256"],
                    "training_binding_sha256": binding_sha256,
                    "execution_gate_sha256": gate["gate_file_sha256"],
                    "official_split": "test",
                },
            }
            receipt_path = prediction_receipt_path(output)
            receipt = write_target_free_receipt(
                receipt_path,
                contract=contract,
                data=data,
                backbone=args.backbone,
                split="test",
                seed=args.seed,
                checkpoint_path=checkpoint,
                prediction_path=output,
                prediction_sha256=str(summary["sha256"]),
                prediction_records=int(summary["count"]),
                guard_audit=guard,
                producer_sources=_producer_sources(contract, args.backbone),
                input_binding=input_binding,
            )
            predictions[role] = {
                **summary,
                "receipt": receipt_path.relative_to(contract.dataset_root).as_posix(),
                "receipt_sha256": receipt["receipt_sha256"],
            }

    if guard.get("passed") is not True:
        raise RuntimeError("OOD prediction process target-matrix guard did not pass")
    freeze = {
        "schema_version": 1,
        "kind": "prompt15_target_free_prediction_freeze",
        "passed": True,
        "backbone": args.backbone,
        "seed": args.seed,
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "training_binding_sha256": binding_sha256,
        "checkpoint_sha256": checkpoint_id,
        "training_completion_sha256": sha256_file(
            run_dir / "OOD_TRAINING_COMPLETE.json"
        ),
        "target_free_input_manifest_sha256": data.input_manifest_sha256,
        "target_matrix_guard": guard,
        "seed_settings": seed_info,
        "device": str(device),
        "predictions": predictions,
        "evaluation_performed": False,
    }
    atomic_json(run_dir / "PREDICTION_FREEZE.json", freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("bert", "tinyllava", "qwen3vl"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--contract", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--ood-config", default=str(OOD_CONFIG_PATH))
    parser.add_argument("--protocol", default=str(OOD_PROTOCOL_PATH))
    parser.add_argument("--gate", default=str(OOD_GATE_PATH))
    args = parser.parse_args()
    return run_prediction(args, load_contract(args.contract))


if __name__ == "__main__":
    raise SystemExit(main())
