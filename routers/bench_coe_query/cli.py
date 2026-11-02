"""Unified, safety-gated CLI for Bench-CoE query routers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import assert_full_extraction_gates, assert_run_safety, load_contract


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/bench_coe_query/experiment_contract.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    audit = subparsers.add_parser("audit-data", parents=[common])
    audit.add_argument("--dataset-dir", default=".")
    extract = subparsers.add_parser("extract", parents=[common])
    extract.add_argument("--dataset-dir", default=".")
    extract.add_argument("--backbone", choices=("tinyllava", "qwen3vl"), required=True)
    extract.add_argument("--world-size", type=int)
    extract.add_argument("--split", choices=("train", "dev", "test", "all"), default="train")
    extract.add_argument("--batch-size", type=int)
    extract.add_argument("--output-dir")
    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--dataset-dir", default=".")
    train.add_argument("--backbone", choices=("bert",), required=True)
    train.add_argument("--epochs", type=int)
    head = subparsers.add_parser("train-head", parents=[common])
    head.add_argument("--dataset-dir", default=".")
    head.add_argument("--backbone", choices=("tinyllava", "qwen3vl"), required=True)
    head.add_argument("--feature-cache", required=True)
    head.add_argument("--epochs", type=int)
    predict = subparsers.add_parser("predict", parents=[common])
    predict.add_argument("--dataset-dir", default=".")
    predict.add_argument("--backbone", choices=("bert", "tinyllava", "qwen3vl"), required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--feature-cache")
    predict.add_argument("--split", choices=("train", "dev", "test"), required=True)
    predict.add_argument("--output", required=True)
    freeze = subparsers.add_parser("freeze-run", parents=[common])
    freeze.add_argument("--dataset-dir", default=".")
    freeze.add_argument("--backbone", choices=("bert", "tinyllava", "qwen3vl"), required=True)
    freeze.add_argument("--checkpoint", required=True)
    freeze.add_argument("--feature-cache")
    freeze.add_argument("--output-dir", required=True)
    evaluate = subparsers.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--dataset-dir", default=".")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--fallback-policy", choices=("none", "multi_image"), required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--prediction-sha256")
    evaluate.add_argument("--producer-receipt")
    evaluate.add_argument("--backbone", choices=("bert", "tinyllava", "qwen3vl"))
    evaluate.add_argument("--split", choices=("train", "dev", "test"))
    quality = subparsers.add_parser("quality-gate", parents=[common])
    quality.add_argument("--dataset-dir", default=".")
    quality.add_argument("--output-dir", default="outputs/bench_coe_query/smoke")
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--dataset-dir", default=".")
    aggregate.add_argument("--output-dir", default="outputs/bench_coe_query/final")
    return parser


def _dry_run_payload(args: argparse.Namespace, contract_sha256: str) -> dict[str, object]:
    return {
        "command": args.command,
        "dry_run": True,
        "contract_sha256": contract_sha256,
        "seed": args.seed,
        "run_id": args.run_id,
        "resume": args.resume,
        "max_samples": args.max_samples,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_contract(args.config)
    epochs = getattr(args, "epochs", None)
    full_operation = (
        args.command in {"extract", "train", "train-head", "predict", "freeze-run", "aggregate"}
        and args.max_samples is None
    )
    artifact_maintenance = bool(
        args.command in {"freeze-run", "aggregate"}
        and os.environ.get("BENCHCOE_ARTIFACT_MAINTENANCE", "0") == "1"
    )
    assert_run_safety(
        contract,
        requested_samples=args.max_samples,
        requested_epochs=epochs,
        full_operation=full_operation and not args.dry_run,
        artifact_maintenance=artifact_maintenance,
    )
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args, contract.sha256), indent=2, sort_keys=True))
        return 0
    if full_operation and args.command in {"train", "train-head", "predict", "freeze-run", "aggregate"}:
        assert_full_extraction_gates(contract)
    if args.command == "audit-data":
        from .data import data_audit, load_vlrb_strict

        print(json.dumps(data_audit(load_vlrb_strict(args.dataset_dir, contract=contract)), indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        from .evaluator import evaluate_prediction_file

        result = evaluate_prediction_file(
            args.predictions,
            args.dataset_dir,
            expected_fallback_policy=args.fallback_policy,
            contract_path=args.config,
            output_path=args.output,
            expected_split=args.split,
            expected_seed=args.seed,
            expected_checkpoint_path=args.checkpoint,
            expected_prediction_sha256=args.prediction_sha256,
            expected_producer_receipt_path=args.producer_receipt,
            expected_backbone=args.backbone,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "quality-gate":
        from .quality_gate import build_quality_gate

        gate = build_quality_gate(
            args.dataset_dir,
            contract_path=args.config,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "technical_smoke_passed": gate["technical_smoke_passed"],
                    "ready_for_artifact_maintenance": gate[
                        "ready_for_artifact_maintenance"
                    ],
                    "ready_for_full_run": gate["ready_for_full_run"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if gate["technical_smoke_passed"] else 1
    # Model-specific implementations register their dispatch in Prompts 06-09.
    if args.command == "extract":
        from .extraction import run_extraction

        return run_extraction(args, contract)
    if args.command == "train":
        from .bert_router import run_bert_training

        return run_bert_training(args, contract)
    if args.command == "train-head":
        from .head_trainer import run_head_training

        return run_head_training(args, contract)
    if args.command == "predict":
        from .runtime_predict import run_prediction

        return run_prediction(args, contract)
    if args.command == "freeze-run":
        from .runtime_predict import run_prediction_set

        return run_prediction_set(args, contract)
    if args.command == "aggregate":
        from .aggregate import run_aggregation

        return run_aggregation(args, contract)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
