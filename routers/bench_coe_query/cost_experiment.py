"""Offline workflow for the three Bench-CoE cost-reduction routes.

The workflow intentionally separates training (which may read fit/dev Y/C),
target-free prediction (which cannot load Y/C), and evaluation (which first
freezes prediction bytes and only then loads Y/C).  It never calls an expert,
remote model, or model ``generate`` method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .config import ExperimentContract, load_contract, sha256_file
from .cost_models import NonNegativeCostHead, identity_calibration
from .cost_policies import predicted_gap_policy, quality_only, utility_policy
from .cost_targets import (
    CostAwareTargets,
    build_targets,
    fit_cost_scale,
    masked_binary_cross_entropy,
    masked_soft_cross_entropy,
)
from .data import load_vlrb_strict
from .errors import DataIntegrityError
from .feature_cache import load_merged_features
from .heads import AuditedBenchCoEHead
from .labels import compute_train_expert_accuracy
from .registry import build_index, load_registry_list, require_exact_order
from .reproducibility import seed_everything
from .risk_bounds import clopper_pearson_upper, zero_harm_sample_requirement


DEFAULT_COST_CONFIG = "configs/bench_cost/protocol.yaml"
DEFAULT_OUTPUT_ROOT = "outputs/bench_cost"
FORBIDDEN_PREDICTION_KEYS = {
    "y",
    "Y",
    "c",
    "C",
    "gold",
    "answer",
    "correctness",
    "actual_cost",
    "actual_output_tokens",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_cost_config(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("cost protocol must be a YAML mapping")
    return value, sha256_file(resolved), resolved


def _load_protocol(path: str | Path) -> tuple[dict[str, Any], str, Path, ExperimentContract]:
    config, digest, resolved = _load_cost_config(path)
    contract = load_contract(config["base_contract"])
    if contract.sha256 != str(config["base_contract_sha256"]):
        raise DataIntegrityError(
            "base experiment contract changed: "
            f"expected={config['base_contract_sha256']}, observed={contract.sha256}"
        )
    roles = config.get("data_roles", {})
    expected_roles = {
        "fit": "SPLITS/train.jsonl",
        "dev": "SPLITS/dev.jsonl",
        "test": "SPLITS/test.jsonl",
    }
    if any(str(roles.get(name)) != value for name, value in expected_roles.items()):
        raise DataIntegrityError("cost protocol must preserve the exact frozen train/dev/test roles")
    if roles.get("qcal") is not None or roles.get("cert") is not None:
        raise DataIntegrityError("scoped protocol forbids constructing qcal/cert splits")
    return config, digest, resolved, contract


def _assert_environment(*, full: bool) -> None:
    if os.environ.get("RUN_API_REGEN", "0") != "0":
        raise RuntimeError("RUN_API_REGEN must equal 0")
    if full and os.environ.get("RUN_FULL", "0") != "1":
        raise RuntimeError("full cached-feature operation refused because RUN_FULL != 1")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name, "0") != "1":
            raise RuntimeError(f"{name} must equal 1")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise DataIntegrityError(f"blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DataIntegrityError(f"non-object JSONL record at {path}:{line_number}")
            records.append(value)
    return records


def _split_ids(contract: ExperimentContract, split: str) -> tuple[str, ...]:
    entry = contract.values["data"]["splits"][split]
    path = contract.resolve(str(entry["path"]))
    if sha256_file(path) != str(entry["sha256"]):
        raise DataIntegrityError(f"frozen {split} split hash mismatch")
    records = _read_jsonl(path)
    ids = tuple(str(record["sample_id"]) for record in records)
    if len(ids) != int(entry["count"]) or len(ids) != len(set(ids)):
        raise DataIntegrityError(f"invalid frozen {split} split")
    return ids


def _feature_cache(
    contract: ExperimentContract, backbone: str
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pointer_path = contract.dataset_root / str(contract.values["features"]["root"]) / backbone / "LATEST_FULL.json"
    with pointer_path.open("r", encoding="utf-8") as handle:
        pointer = json.load(handle)
    if pointer.get("contract_sha256") != contract.sha256:
        raise DataIntegrityError(f"{backbone} feature pointer contract mismatch")
    cache_root = contract.dataset_root / str(pointer["cache_root"])
    manifest_path = contract.dataset_root / str(pointer["manifest"])
    if cache_root / "merged/manifest.json" != manifest_path:
        raise DataIntegrityError(f"{backbone} pointer manifest/cache mismatch")
    if sha256_file(manifest_path) != str(pointer["manifest_sha256"]):
        raise DataIntegrityError(f"{backbone} frozen feature manifest hash mismatch")
    with manifest_path.open("r", encoding="utf-8") as handle:
        recorded_manifest = json.load(handle)
    features, records, manifest = load_merged_features(
        cache_root,
        expected_signature=str(recorded_manifest["signature"]),
        expected_manifest_sha256=str(recorded_manifest["manifest_sha256"]),
        expected_manifest_file_sha256=str(pointer["manifest_sha256"]),
    )
    evidence = {
        "backbone": backbone,
        "pointer": str(pointer_path.relative_to(contract.dataset_root)),
        "pointer_sha256": sha256_file(pointer_path),
        "cache_root": str(cache_root.relative_to(contract.dataset_root)),
        "manifest": str(manifest_path.relative_to(contract.dataset_root)),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "features_sha256": manifest["features_sha256"],
        "index_sha256": manifest["index_sha256"],
        "rows": len(records),
        "feature_dim": int(features.shape[1]),
        "dtype": str(features.dtype),
        "passed": True,
    }
    return features, records, manifest, evidence


def _positions_by_split(data: Any, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        split: np.flatnonzero(
            np.asarray([data.samples[int(row)].split == split for row in rows], dtype=bool)
        )
        for split in ("train", "dev", "test")
    }


def _method_config(config: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    matches = [dict(item) for item in config["route1"]["methods"] if item["id"] == method_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate Route-1 method: {method_id}")
    return matches[0]


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("cost_experiment.py", "cost_targets.py", "cost_policies.py", "cost_models.py", "risk_bounds.py")
    return {f"routers/bench_coe_query/{name}": sha256_file(root / name) for name in names}


def _command_result(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(command), "exit_code": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": list(command),
        "exit_code": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _write_step_artifact(
    step: str,
    *,
    status: str,
    artifacts: Sequence[str],
    checks: Mapping[str, Any],
    blockers: Sequence[str] = (),
) -> None:
    path = Path("artifacts/bench_cost/steps") / f"{step}.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "step": step,
            "status": status,
            "artifacts": list(artifacts),
            "checks": dict(checks),
            "blockers": list(blockers),
            "command": " ".join(sys.argv),
            "recorded_at_unix": time.time(),
        },
    )


def _baseline_metrics(data: Any) -> dict[str, Any]:
    train_rows = np.asarray(data.split_to_rows["train"], dtype=np.int64)
    dev_rows = np.asarray(data.split_to_rows["dev"], dtype=np.int64)
    accuracy = data.y[train_rows].astype(np.float64).mean(axis=0)
    mean_cost = data.c[train_rows].astype(np.float64).mean(axis=0)
    strongest = int(np.argmax(accuracy))
    cheapest = int(np.argmin(mean_cost))
    generator = np.random.default_rng(20260905)
    random_predictions = generator.integers(0, len(data.model_order), size=len(dev_rows))
    any_correct = data.y[dev_rows].max(axis=1)
    cheapest_correct = np.empty(len(dev_rows), dtype=np.int64)
    for position, row in enumerate(dev_rows):
        correct = np.flatnonzero(data.y[row] == 1)
        cheapest_correct[position] = (
            cheapest
            if not len(correct)
            else int(correct[np.lexsort((correct, -accuracy[correct], data.c[row, correct]))[0]])
        )

    def fixed(name: str, predictions: np.ndarray) -> dict[str, Any]:
        return {
            "name": name,
            "n": len(dev_rows),
            "accuracy": float(data.y[dev_rows, predictions].mean()),
            "expert_cost": float(data.c[dev_rows, predictions].astype(np.float64).mean()),
        }

    return {
        "train_global_accuracy": {data.model_order[index]: float(value) for index, value in enumerate(accuracy)},
        "train_mean_cost": {data.model_order[index]: float(value) for index, value in enumerate(mean_cost)},
        "policies": [
            fixed("source_fit_best_single", np.full(len(dev_rows), strongest, dtype=np.int64)),
            fixed("fixed_cheapest_fit_mean", np.full(len(dev_rows), cheapest, dtype=np.int64)),
            fixed("random_uniform_seed_20260905", random_predictions),
            {
                "name": "oracle_any_expert",
                "n": len(dev_rows),
                "accuracy": float(any_correct.mean()),
                "expert_cost": None,
                "status": "ORACLE_REFERENCE",
            },
            fixed("cheapest_correct_oracle", cheapest_correct),
        ],
        "strongest_index": strongest,
        "cheapest_index": cheapest,
    }


def run_prepare(args: argparse.Namespace) -> int:
    _assert_environment(full=False)
    config, protocol_sha, config_path, contract = _load_protocol(args.cost_config)
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    baseline = _baseline_metrics(data)
    feature_evidence: dict[str, Any] = {}
    for backbone in ("qwen3vl", "tinyllava"):
        features, _records, _manifest, evidence = _feature_cache(contract, backbone)
        feature_evidence[backbone] = evidence
        del features
    git = _command_result(["git", "status", "--short"])
    gpu = _command_result(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    train_rows = np.asarray(data.split_to_rows["train"], dtype=np.int64)
    kappa, kappa_status = fit_cost_scale(data.c, train_rows)
    input_paths = {
        "base_contract": config["base_contract"],
        "Y": contract.values["data"]["matrices"]["y"]["path"],
        "C": contract.values["data"]["matrices"]["c"]["path"],
        "train": contract.values["data"]["splits"]["train"]["path"],
        "dev": contract.values["data"]["splits"]["dev"]["path"],
        "test": contract.values["data"]["splits"]["test"]["path"],
        "models": contract.values["data"]["registry"]["models"]["path"],
        "samples": contract.values["data"]["registry"]["samples"]["path"],
        "meta": contract.values["data"]["registry"]["meta"]["path"],
    }
    inventory_files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(contract.resolve(str(path))),
            "bytes": contract.resolve(str(path)).stat().st_size,
        }
        for name, path in input_paths.items()
    }
    inventory = {
        "schema_version": 1,
        "status": "AUDITED_REAL_FROZEN_INPUTS",
        "protocol": str(config_path),
        "protocol_sha256": protocol_sha,
        "base_contract_sha256": contract.sha256,
        "samples": len(data.samples),
        "experts": len(data.model_order),
        "model_order": list(data.model_order),
        "split_counts": {name: len(data.split_to_rows[name]) for name in ("train", "dev", "test")},
        "files": inventory_files,
        "features": feature_evidence,
        "git": git,
        "source_hashes": _source_hashes(),
    }
    _atomic_json(Path("artifacts/bench_cost/inventory.json"), inventory)
    split_manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha,
        "external_splits_unchanged": True,
        "internal_resplit": "FORBIDDEN_NOT_CREATED",
        "roles": {
            role: {
                "path": str(contract.values["data"]["splits"][split]["path"]),
                "count": len(data.split_to_rows[split]),
                "sha256": str(contract.values["data"]["splits"][split]["sha256"]),
            }
            for role, split in (("fit", "train"), ("dev", "dev"), ("test", "test"))
        },
        "qcal": {"status": "UNAVAILABLE"},
        "cert": {"status": "UNAVAILABLE"},
    }
    _atomic_json(Path("configs/bench_cost/splits_manifest.json"), split_manifest)
    cost_snapshot = {
        "schema_version": 1,
        "unit": config["costs"]["expert_cost_unit"],
        "provenance": "frozen token-derived C.npy; see outputs/bench_coe_query/audit/DATA_INTEGRITY.json",
        "price_source": "config/pricing.yaml",
        "price_snapshot_sha256": sha256_file(Path("config/pricing.yaml")),
        "fit_only_kappa": kappa,
        "kappa_status": kappa_status,
        "minimum": float(data.c.min()),
        "maximum": float(data.c.max()),
        "zeros": int((data.c == 0).sum()),
        "nonfinite": int((~np.isfinite(data.c)).sum()),
        "fit_mean_by_expert": baseline["train_mean_cost"],
        "router_financial_cost": None,
        "router_cost_status": "NOT_MEASURED_IN_USD",
        "zero_harm_n_required_for_route3_numeric_threshold": zero_harm_sample_requirement(
            float(config["selection"]["risk_epsilon"]),
            float(config["selection"]["risk_delta"]),
        ),
    }
    _atomic_json(Path("artifacts/bench_cost/cost_snapshot.json"), cost_snapshot)
    _atomic_json(Path("artifacts/bench_cost/baseline_registry.json"), baseline)
    hardware = {
        "schema_version": 1,
        "gpu_probe": gpu,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "full_cached_feature_training_authorized": bool(
            yaml.safe_load(Path("configs/bench_cost/budgets.yaml").read_text(encoding="utf-8"))[
                "full_cached_feature_head_training"
            ]
        ),
    }
    _atomic_json(Path("artifacts/bench_cost/hardware.json"), hardware)

    docs = Path("docs/bench_cost")
    _atomic_text(
        docs / "REPO_AUDIT.md",
        "# Bench-CoE 三条降本路线仓库审计\n\n"
        f"- 冻结合同：`{config['base_contract']}`，SHA-256 `{contract.sha256}`。\n"
        f"- 数据：{len(data.samples):,} 条、{len(data.model_order)} 个专家；train/dev/test = "
        f"{len(data.split_to_rows['train']):,}/{len(data.split_to_rows['dev']):,}/{len(data.split_to_rows['test']):,}。\n"
        "- Y/C 完整、有限且严格按冻结 registry 对齐；C 是 token/价格快照导出的专家侧 USD 代理。\n"
        "- 可复用全量特征：Qwen3-VL 30,493×2,048，TinyLLaVA 30,493×5,120；47 个真实多图样本按既有合同结构化回退。\n"
        "- 当前工作区没有可用 Git metadata，无法生成可信 git diff/commit；改用逐文件 SHA-256 清单。\n"
        "- 原 Query 路由的标签是在正确集合中选 train 全局最强专家，成本不参与；全错行训练时 ignore、评价时保留。\n"
        "- Subject 字段不存在（只有 dataset/task），因此 R1-06 标记 NOT_APPLICABLE。\n",
    )
    _atomic_text(
        docs / "BASELINE_PROVENANCE.md",
        "# 参照系统来源\n\n"
        "主参照是已冻结的 `Query-Qwen3VL` 五种子运行，权重、预测和 target-free receipt 位于 "
        "`outputs/bench_coe_query/runs/qwen3vl/vlrb_adapted_v1/full-vlrb-adapted-v1/`。"
        "其 checkpoint 只由 dev hard-label loss 选择。该实现是 hash-pinned `adapted`，不是 faithful 原版声明。\n\n"
        "三条新路线的同骨干公平标签对照另以新训练的 `strongest_hard` 为局部参照；路线三部署参照固定到对应 seed 的既有 Query-Qwen3VL。\n",
    )
    _atomic_text(
        docs / "HARDWARE_AUDIT.md",
        "# 硬件与环境审计\n\n```text\n" + gpu.get("stdout", "") + "\n```\n\n"
        "仅复用冻结特征训练小型 LN/Linear 头；不重新提取特征、不下载模型、不调用专家。\n",
    )
    _atomic_text(
        docs / "BLOCKERS.md",
        "# 阻塞项\n\n"
        "- `BLOCKED_NO_QCAL`：目录级协议禁止新增切分，现有冻结协议没有独立 qcal。路线二真实运行采用明确命名的 identity calibration。\n"
        "- `BLOCKED_NO_CERT`：同一原因，没有与 fit/dev/test 分离的 cert；路线三不能标正式认证。\n"
        "- `NOT_MEASURED_ROUTER_USD`：路由器 GPU 秒与专家侧 USD 代理没有获准的换算，不能相加为完整财务成本。\n"
        "- `NOT_APPLICABLE_SUBJECT`：registry 无 subject，不能实施可部署 subject mapping。\n"
        "- `GIT_METADATA_UNAVAILABLE`：当前 `.git` 为空目录。\n",
    )
    _atomic_text(
        docs / "SPEC.md",
        "# 三路线冻结实验规格\n\n"
        f"协议 SHA-256：`{protocol_sha}`。外部分区保持原样；fit=train、dev=dev、test=test。"
        "主选择规则是在 dev 准确率不低于参照减 0.005 时取专家侧代理成本最低者；无可行项保留参照。"
        "所有 checkpoint 只以 `dev_hard_label_loss` 选取。test 在三条路线 freeze 后才评价。\n\n"
        "路线一比较 strongest/cheapest 硬标签、uniform soft 与五个 cost-soft λ。路线二训练 17 个独立 sigmoid，"
        "成本使用 fit 均值或固定 epoch 的非负预测头。路线三只在生成前决定是否降级，并在无资格/异常时回退参照。\n",
    )
    _atomic_text(
        docs / "DATA_ACCESS.md",
        "# 数据访问权限\n\n"
        "训练阶段只索引冻结 train/dev 行；预测阶段只加载冻结特征、split/registry、checkpoint 与既有参照预测，"
        "不加载 Y/C；评价阶段先读取并哈希预测及 receipt，随后才加载 Y/C。test 不参与训练、阈值或 checkpoint 选择。\n\n"
        "由于 Y/C 是单体矩阵文件，训练适配器会打开整个文件，但代码只索引 train/dev row registry；该限制与既有审计实现一致。\n",
    )
    _atomic_text(
        docs / "DATA_QUALITY.md",
        "# 数据与成本质量\n\n"
        f"Y/C shape 均为 `{tuple(data.y.shape)}`；Y 为二值，C 非负、非有限数 0、零值 {int((data.c == 0).sum())}。"
        f"fit-only κ={kappa:.17g}（{kappa_status}）。完整观测掩码为真；接口和测试仍显式保留 correctness/cost mask。\n",
    )
    _atomic_text(
        docs / "RUNBOOK.md",
        "# 运行手册\n\n"
        "所有真实全量命令必须设置 `RUN_FULL=1 RUN_API_REGEN=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。"
        "顺序为 prepare/opportunity → train-route1/train-route2 → target-free predict/score/materialize → dev evaluate/select/freeze → test evaluate。"
        "每个训练目录有 RUNNING/COMPLETE 状态；COMPLETE 且 identity/hash 一致时重跑会跳过，异常 identity 会 fail closed。\n",
    )
    _atomic_text(
        docs / "STATE.md",
        "# Bench-CoE 降本实验状态\n\n"
        "- C0–C4：COMPLETE（真实冻结输入审计、协议、适配器、基线、运行器）。\n"
        "- Route 1/2/3：IMPLEMENTED，真实训练与 dev/test 结果待本轮后续命令写入。\n"
        "- qcal：BLOCKED_NO_QCAL；formal certification：BLOCKED_NO_CERT。\n"
        "- paid API / expert regeneration / feature extraction：DISABLED。\n",
    )
    common_artifacts = [
        "artifacts/bench_cost/inventory.json",
        "configs/bench_cost/splits_manifest.json",
        "artifacts/bench_cost/cost_snapshot.json",
        "artifacts/bench_cost/baseline_registry.json",
    ]
    for step, checks in {
        "C0": {"frozen_inputs_verified": True, "large_task_run": False},
        "C1": {"external_splits_unchanged": True, "internal_resplit_created": False},
        "C2": {"y_c_alignment_verified": True, "inference_view_target_free": True},
        "C3": {"baseline_registry_created": True, "all_wrong_retained": True},
        "C4": {"runner_available": True, "paid_or_expert_calls": False},
    }.items():
        blockers = (
            ["BLOCKED_NO_QCAL", "BLOCKED_NO_CERT"] if step == "C1" else []
        )
        _write_step_artifact(step, status="COMPLETE_WITH_DISCLOSED_LIMITS", artifacts=common_artifacts, checks=checks, blockers=blockers)
    print(json.dumps({"status": "PREPARED", "inventory": inventory, "hardware": hardware}, indent=2, sort_keys=True))
    return 0


def run_opportunity(args: argparse.Namespace) -> int:
    _assert_environment(full=False)
    _config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    train_rows = np.asarray(data.split_to_rows["train"], dtype=np.int64)
    accuracy = data.y[train_rows].astype(np.float64).mean(axis=0)
    canonical = np.arange(len(data.model_order), dtype=np.int64)
    rows_output: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        for row in data.split_to_rows[split]:
            correct = np.flatnonzero(data.y[row] == 1)
            record: dict[str, Any] = {
                "split_role": "fit" if split == "train" else "dev",
                "row_index": int(row),
                "sample_id": data.samples[row].sample_id,
                "dataset": data.samples[row].dataset,
                "correct_expert_count": int(len(correct)),
                "stratum": "all_wrong" if not len(correct) else ("single_correct" if len(correct) == 1 else "multi_correct"),
                "strongest_correct_index": None,
                "cheapest_correct_index": None,
                "strongest_correct_cost": None,
                "cheapest_correct_cost": None,
                "ideal_absolute_saving": None,
            }
            if len(correct):
                strongest = int(correct[np.lexsort((canonical[correct], -accuracy[correct]))[0]])
                cheapest = int(
                    correct[np.lexsort((canonical[correct], -accuracy[correct], data.c[row, correct]))[0]]
                )
                strong_cost = float(data.c[row, strongest])
                cheap_cost = float(data.c[row, cheapest])
                record.update(
                    {
                        "strongest_correct_index": strongest,
                        "cheapest_correct_index": cheapest,
                        "strongest_correct_cost": strong_cost,
                        "cheapest_correct_cost": cheap_cost,
                        "ideal_absolute_saving": strong_cost - cheap_cost,
                    }
                )
            rows_output.append(record)
    report_root = Path("reports/bench_cost/route1")
    report_root.mkdir(parents=True, exist_ok=True)
    per_sample_path = report_root / "opportunity_per_sample.csv"
    with per_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_output[0]))
        writer.writeheader()
        writer.writerows(rows_output)
    summaries: list[dict[str, Any]] = []
    for role in ("fit", "dev"):
        selected = [item for item in rows_output if item["split_role"] == role]
        for dataset in ["__ALL__", *sorted({item["dataset"] for item in selected})]:
            group = selected if dataset == "__ALL__" else [item for item in selected if item["dataset"] == dataset]
            correct = [item for item in group if item["correct_expert_count"] > 0]
            multi = [item for item in group if item["correct_expert_count"] >= 2]
            saveable = [item for item in correct if float(item["ideal_absolute_saving"]) > 0]
            savings = np.asarray([float(item["ideal_absolute_saving"]) for item in correct], dtype=np.float64)
            summaries.append(
                {
                    "split_role": role,
                    "dataset": dataset,
                    "n": len(group),
                    "all_wrong": sum(item["correct_expert_count"] == 0 for item in group),
                    "single_correct": sum(item["correct_expert_count"] == 1 for item in group),
                    "multi_correct": len(multi),
                    "multi_correct_fraction": len(multi) / len(group),
                    "saveable_fraction_all": len(saveable) / len(group),
                    "mean_ideal_saving_correct_rows": None if not len(savings) else float(savings.mean()),
                    "median_ideal_saving_correct_rows": None if not len(savings) else float(np.median(savings)),
                    "p90_ideal_saving_correct_rows": None if not len(savings) else float(np.quantile(savings, 0.9)),
                }
            )
    summary_path = report_root / "opportunity_tables.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    correlation = np.corrcoef(data.y[train_rows].astype(np.float64), rowvar=False)
    correlation_path = report_root / "fit_expert_correctness_correlation.csv"
    with correlation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expert_id", *data.model_order])
        for expert, row_values in zip(data.model_order, correlation):
            writer.writerow([expert, *(format(float(value), ".17g") for value in row_values)])
    aggregate = {item["split_role"]: item for item in summaries if item["dataset"] == "__ALL__"}
    _atomic_text(
        report_root / "opportunity_dev.md",
        "# Route 1 标签机会诊断\n\n"
        f"协议 `{protocol_sha}`；仅使用 fit 计算全局专家强度，dev 沿用该顺序。\n\n"
        "| role | n | all-wrong | single | multi | 可省比例 | 正确行平均理想节省 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {role} | {item['n']} | {item['all_wrong']} | {item['single_correct']} | {item['multi_correct']} | "
            f"{item['saveable_fraction_all']:.4f} | {item['mean_ideal_saving_correct_rows']:.8g} |"
            for role, item in aggregate.items()
        )
        + "\n\n这里的节省是读取缓存真值后的标签层 oracle 机会，不是可部署路由器收益。全错行保留在分母和逐样本表中。\n",
    )
    _write_step_artifact(
        "R1-01",
        status="COMPLETE_REAL_FIT_DEV_DIAGNOSTIC",
        artifacts=[str(report_root / "opportunity_dev.md"), str(summary_path), str(per_sample_path), str(correlation_path)],
        checks={"test_opened_for_method_design": False, "all_wrong_retained": True, "fit_only_priority": True},
    )
    print(json.dumps({"status": "COMPLETE", "aggregate": aggregate}, indent=2, sort_keys=True))
    return 0


def _dev_results(root: Path, route: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("status") == "EVALUATED_FROM_FROZEN_TARGET_FREE_PREDICTIONS" and value.get("route") == route and value.get("split") == "dev":
            value["evaluation_path"] = str(path)
            results.append(value)
    if not results:
        raise RuntimeError(f"no dev evaluation results found for {route} under {root}")
    return results


def run_select(args: argparse.Namespace) -> int:
    _assert_environment(full=False)
    config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    results = _dev_results(Path(args.eval_root), args.route)
    epsilon = float(config["selection"]["accuracy_epsilon"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["method"]), []).append(result)
    summaries = []
    if args.route == "route1":
        if "strongest_hard" not in grouped:
            raise RuntimeError("Route-1 selection requires strongest_hard dev results")
        expected_seeds = set(int(seed) for seed in config["training"]["seeds"])
        for method, items in grouped.items():
            if {int(item["seed"]) for item in items} != expected_seeds:
                raise RuntimeError(f"Route-1 method {method} does not have the frozen five seeds")
        reference_accuracy = float(np.mean([item["accuracy"] for item in grouped["strongest_hard"]]))
    else:
        for method, items in grouped.items():
            seeds = [int(item["seed"]) for item in items]
            if args.route == "route2" and set(seeds) != set(int(seed) for seed in config["training"]["seeds"]):
                raise RuntimeError(f"Route-2 method {method} does not have the frozen five seeds")
            if args.route == "route3" and seeds != [int(config["route3"]["deployment_seed"])]:
                raise RuntimeError(f"Route-3 method {method} must use only the preregistered deployment seed")
        reference_values = [item["reference_accuracy"] for item in results]
        if any(value is None for value in reference_values):
            raise RuntimeError(f"{args.route} selection requires a frozen reference")
        reference_accuracy = float(np.mean(reference_values))
    threshold = reference_accuracy - epsilon
    for method, items in grouped.items():
        accuracy = float(np.mean([item["accuracy"] for item in items]))
        cost = float(np.mean([item["expert_cost"] for item in items]))
        summaries.append(
            {
                "method": method,
                "seeds": sorted(int(item["seed"]) for item in items),
                "accuracy_mean": accuracy,
                "expert_cost_mean": cost,
                "reference_accuracy_mean": reference_accuracy,
                "accuracy_threshold": threshold,
                "feasible": accuracy >= threshold,
                "evaluation_paths": [item["evaluation_path"] for item in sorted(items, key=lambda value: value["seed"])],
                "prediction_paths": [item["prediction_path"] for item in sorted(items, key=lambda value: value["seed"])],
                "prediction_sha256": [item["prediction_sha256"] for item in sorted(items, key=lambda value: value["seed"])],
            }
        )
    feasible = [item for item in summaries if item["feasible"]]
    if feasible:
        selected = min(feasible, key=lambda item: (item["expert_cost_mean"], -item["accuracy_mean"], item["method"]))
        selection_status = "SELECTED_ON_DEV"
    else:
        selected = None
        selection_status = "NO_FEASIBLE_POLICY_KEEP_REFERENCE"
    lock = {
        "schema_version": 1,
        "route": args.route,
        "status": selection_status,
        "protocol_sha256": protocol_sha,
        "base_contract_sha256": contract.sha256,
        "selection_split": "dev",
        "test_accessed": False,
        "accuracy_epsilon": epsilon,
        "reference_accuracy_mean": reference_accuracy,
        "accuracy_threshold": threshold,
        "selected": selected,
        "candidates": summaries,
        "created_at_unix": time.time(),
    }
    output = Path(args.output).resolve()
    _atomic_json(output, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


def _training_identity(
    *,
    route: str,
    method: str,
    seed: int,
    backbone: str,
    protocol_sha256: str,
    feature_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "route": route,
        "method": method,
        "seed": int(seed),
        "backbone": backbone,
        "protocol_sha256": protocol_sha256,
        "base_contract_sha256": feature_evidence.get("contract_sha256"),
        "feature_manifest_sha256": feature_evidence["manifest_file_sha256"],
        "source_hashes": _source_hashes(),
    }


def _batched_indices(count: int, batch_size: int, *, permutation: torch.Tensor | None = None):
    indices = torch.arange(count) if permutation is None else permutation
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size].numpy()


def _route1_eval(
    model: nn.Module,
    features: np.ndarray,
    positions: np.ndarray,
    rows: np.ndarray,
    target: CostAwareTargets,
    strongest_labels: np.ndarray,
    data: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for local in _batched_indices(len(positions), batch_size):
            values = torch.from_numpy(np.asarray(features[positions[local]])).to(device=device, dtype=torch.float32)
            logits_parts.append(model(values).float().cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    valid = target.selection_loss_mask
    log_probabilities = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    target_loss = float(-(target.probabilities[valid] * log_probabilities[valid]).sum() / valid.sum())
    hard_valid = strongest_labels != -100
    hard_loss = float(-log_probabilities[np.flatnonzero(hard_valid), strongest_labels[hard_valid]].mean())
    predictions = logits.argmax(axis=1)
    matrix_rows = rows[positions]
    return {
        "dev_target_loss": target_loss,
        "dev_hard_label_loss": hard_loss,
        "dev_hard_label_accuracy": float((predictions[hard_valid] == strongest_labels[hard_valid]).mean()),
        "dev_routed_accuracy": float(data.y[matrix_rows, predictions].mean()),
        "dev_expert_cost": float(data.c[matrix_rows, predictions].astype(np.float64).mean()),
        "dev_samples": int(len(matrix_rows)),
        "dev_trainable_samples": int(valid.sum()),
    }


def _checkpoint_path(output_root: Path, route: str, backbone: str, method: str, seed: int) -> Path:
    return output_root / "models" / route / backbone / method / f"seed-{seed}" / "checkpoint.pt"


def run_train_route1(args: argparse.Namespace) -> int:
    _assert_environment(full=True)
    config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    features, records, _manifest, feature_evidence = _feature_cache(contract, args.backbone)
    feature_evidence["contract_sha256"] = contract.sha256
    rows = np.asarray([int(record["row_index"]) for record in records], dtype=np.int64)
    if any(data.samples[int(row)].sample_id != str(record["sample_id"]) for row, record in zip(rows, records)):
        raise DataIntegrityError("feature rows/sample IDs do not match the frozen registry")
    positions = _positions_by_split(data, rows)
    train_positions, dev_positions = positions["train"], positions["dev"]
    train_rows, dev_rows = rows[train_positions], rows[dev_positions]
    train_global_accuracy = compute_train_expert_accuracy(
        data.y, data.split_to_rows["train"], data.model_order, expected_model_order=data.model_order
    )
    kappa, kappa_status = fit_cost_scale(data.c, data.split_to_rows["train"])
    method = _method_config(config, args.method)
    observed_train = np.ones((len(train_rows), len(data.model_order)), dtype=bool)
    observed_dev = np.ones((len(dev_rows), len(data.model_order)), dtype=bool)
    train_target = build_targets(
        data.y[train_rows], data.c[train_rows], observed_train, train_global_accuracy,
        str(method["mode"]), float(method["lambda_cost"]), kappa, model_order=data.model_order,
    )
    dev_target = build_targets(
        data.y[dev_rows], data.c[dev_rows], observed_dev, train_global_accuracy,
        str(method["mode"]), float(method["lambda_cost"]), kappa, model_order=data.model_order,
    )
    strongest_dev = build_targets(
        data.y[dev_rows], data.c[dev_rows], observed_dev, train_global_accuracy,
        "strongest_correct_hard", 0.0, kappa, model_order=data.model_order,
    ).hard_labels
    train_cfg = config["training"]
    output_root = Path(args.output_root).resolve()
    device = torch.device(args.device)
    seeds = [int(item) for item in args.seeds.split(",")]
    if any(seed not in tuple(int(item) for item in train_cfg["seeds"]) for seed in seeds):
        raise ValueError("requested seed is outside the frozen seed registry")

    for seed in seeds:
        path = _checkpoint_path(output_root, "route1", args.backbone, args.method, seed)
        status_path = path.parent / "status.json"
        identity = _training_identity(
            route="route1", method=args.method, seed=seed, backbone=args.backbone,
            protocol_sha256=protocol_sha, feature_evidence=feature_evidence,
        )
        if path.is_file() and status_path.is_file() and not args.force:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("identity") != identity:
                raise RuntimeError(f"existing Route-1 run identity mismatch: {path.parent}")
            if status.get("status") == "COMPLETE":
                print(json.dumps({"status": "SKIPPED_COMPLETE", "checkpoint": str(path)}))
                continue
        seed_everything(seed)
        model = AuditedBenchCoEHead(
            int(features.shape[1]), len(data.model_order), float(train_cfg["dropout"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        batch_size = int(train_cfg["batch_size"])
        generator = torch.Generator().manual_seed(seed)
        history: list[dict[str, Any]] = []
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        started = time.perf_counter()
        _atomic_json(status_path, {"schema_version": 1, "status": "RUNNING", "identity": identity})
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            model.train()
            permutation = torch.randperm(len(train_positions), generator=generator)
            total_loss = 0.0
            total_valid = 0
            for local in _batched_indices(len(train_positions), batch_size, permutation=permutation):
                source = train_positions[local]
                values = torch.from_numpy(np.asarray(features[source])).to(device=device, dtype=torch.float32)
                probabilities = torch.from_numpy(train_target.probabilities[local]).to(device=device, dtype=torch.float32)
                mask = torch.from_numpy(train_target.selection_loss_mask[local]).to(device=device)
                optimizer.zero_grad(set_to_none=True)
                loss, count = masked_soft_cross_entropy(model(values), probabilities, mask)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * count
                total_valid += count
            dev_metrics = _route1_eval(
                model, features, dev_positions, rows, dev_target, strongest_dev, data,
                batch_size=batch_size, device=device,
            )
            record = {
                "epoch": epoch,
                "train_target_loss": total_loss / max(1, total_valid),
                "trainable_samples": total_valid,
                **dev_metrics,
            }
            history.append(record)
            current = float(dev_metrics["dev_hard_label_loss"])
            if current < best_loss - 1e-6:
                best_loss = current
                best_epoch = epoch
                bad_epochs = 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                bad_epochs += 1
            _atomic_text(
                path.parent / "training_history.jsonl",
                "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in history),
            )
            print(json.dumps({"method": args.method, "seed": seed, **record}, sort_keys=True))
            if epoch >= int(train_cfg["min_epochs"]) and bad_epochs >= int(train_cfg["patience"]):
                break
        if best_state is None:
            raise RuntimeError("Route-1 training did not produce a finite best state")
        model.load_state_dict(best_state)
        final_metrics = _route1_eval(
            model, features, dev_positions, rows, dev_target, strongest_dev, data,
            batch_size=batch_size, device=device,
        )
        checkpoint = {
            "schema_version": 1,
            "identity": identity,
            "model_state": best_state,
            "architecture": "AuditedBenchCoEHead",
            "hidden_dim": int(features.shape[1]),
            "num_experts": len(data.model_order),
            "model_order": tuple(data.model_order),
            "fallback_expert_index": int(np.argmax(train_global_accuracy)),
            "train_global_accuracy": train_global_accuracy,
            "train_mean_cost": data.c[np.asarray(data.split_to_rows["train"], dtype=np.int64)].astype(np.float64).mean(axis=0),
            "kappa": kappa,
            "kappa_status": kappa_status,
            "target": method,
            "best_epoch": best_epoch,
            "best_dev_hard_label_loss": best_loss,
            "dev_metrics": final_metrics,
            "history": history,
            "feature_evidence": feature_evidence,
            "protocol_sha256": protocol_sha,
            "created_at_unix": time.time(),
        }
        _atomic_torch_save(path, checkpoint)
        status = {
            "schema_version": 1,
            "status": "COMPLETE",
            "identity": identity,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "dev_metrics": final_metrics,
        }
        _atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True))
    return 0


def _route2_eval_quality(
    model: nn.Module,
    features: np.ndarray,
    positions: np.ndarray,
    rows: np.ndarray,
    strongest_labels: np.ndarray,
    data: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for local in _batched_indices(len(positions), batch_size):
            values = torch.from_numpy(np.asarray(features[positions[local]])).to(device=device, dtype=torch.float32)
            pieces.append(model(values).float().cpu().numpy())
    logits = np.concatenate(pieces, axis=0)
    matrix_rows = rows[positions]
    labels = data.y[matrix_rows].astype(np.float64)
    bce = float(np.maximum(logits, 0).mean() - (labels * logits).mean() + np.log1p(np.exp(-np.abs(logits))).mean())
    # The expression above split means incorrectly if kept separate; compute the
    # exact element mean explicitly for clarity and numerical stability.
    bce = float((np.maximum(logits, 0) - labels * logits + np.log1p(np.exp(-np.abs(logits)))).mean())
    log_probabilities = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    valid = strongest_labels != -100
    hard_loss = float(-log_probabilities[np.flatnonzero(valid), strongest_labels[valid]].mean())
    predictions = logits.argmax(axis=1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    brier = float(np.square(probabilities - labels).mean())
    return {
        "dev_quality_bce": bce,
        "dev_quality_brier": brier,
        "dev_hard_label_loss": hard_loss,
        "dev_hard_label_accuracy": float((predictions[valid] == strongest_labels[valid]).mean()),
        "dev_quality_only_routed_accuracy": float(data.y[matrix_rows, predictions].mean()),
        "dev_quality_only_expert_cost": float(data.c[matrix_rows, predictions].astype(np.float64).mean()),
        "dev_samples": int(len(matrix_rows)),
        "dev_quality_cells": int(labels.size),
    }


def run_train_route2(args: argparse.Namespace) -> int:
    _assert_environment(full=True)
    config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    features, records, _manifest, feature_evidence = _feature_cache(contract, args.backbone)
    feature_evidence["contract_sha256"] = contract.sha256
    rows = np.asarray([int(record["row_index"]) for record in records], dtype=np.int64)
    if any(data.samples[int(row)].sample_id != str(record["sample_id"]) for row, record in zip(rows, records)):
        raise DataIntegrityError("feature rows/sample IDs do not match the frozen registry")
    positions = _positions_by_split(data, rows)
    train_positions, dev_positions = positions["train"], positions["dev"]
    train_rows, dev_rows = rows[train_positions], rows[dev_positions]
    accuracy = compute_train_expert_accuracy(
        data.y, data.split_to_rows["train"], data.model_order, expected_model_order=data.model_order
    )
    kappa, kappa_status = fit_cost_scale(data.c, data.split_to_rows["train"])
    strongest_dev = build_targets(
        data.y[dev_rows], data.c[dev_rows], np.ones_like(data.y[dev_rows], dtype=bool), accuracy,
        "strongest_correct_hard", 0.0, kappa, model_order=data.model_order,
    ).hard_labels
    train_cfg = config["training"]
    route_cfg = config["route2"]
    output_root = Path(args.output_root).resolve()
    device = torch.device(args.device)
    seeds = [int(item) for item in args.seeds.split(",")]
    for seed in seeds:
        path = _checkpoint_path(output_root, "route2", args.backbone, "quality_cost", seed)
        status_path = path.parent / "status.json"
        identity = _training_identity(
            route="route2", method="quality_cost", seed=seed, backbone=args.backbone,
            protocol_sha256=protocol_sha, feature_evidence=feature_evidence,
        )
        if path.is_file() and status_path.is_file() and not args.force:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("identity") != identity:
                raise RuntimeError(f"existing Route-2 run identity mismatch: {path.parent}")
            if status.get("status") == "COMPLETE":
                print(json.dumps({"status": "SKIPPED_COMPLETE", "checkpoint": str(path)}))
                continue
        seed_everything(seed)
        quality_model = AuditedBenchCoEHead(
            int(features.shape[1]), len(data.model_order), float(train_cfg["dropout"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            quality_model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        batch_size = int(train_cfg["batch_size"])
        generator = torch.Generator().manual_seed(seed)
        history: list[dict[str, Any]] = []
        best_loss, best_epoch, bad_epochs = math.inf, 0, 0
        best_state: dict[str, torch.Tensor] | None = None
        started = time.perf_counter()
        _atomic_json(status_path, {"schema_version": 1, "status": "RUNNING", "identity": identity})
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            quality_model.train()
            permutation = torch.randperm(len(train_positions), generator=generator)
            loss_sum, cell_count = 0.0, 0
            for local in _batched_indices(len(train_positions), batch_size, permutation=permutation):
                source = train_positions[local]
                values = torch.from_numpy(np.asarray(features[source])).to(device=device, dtype=torch.float32)
                labels = torch.from_numpy(data.y[train_rows[local]].astype(np.float32)).to(device)
                observed = torch.ones_like(labels, dtype=torch.bool)
                optimizer.zero_grad(set_to_none=True)
                loss, count = masked_binary_cross_entropy(quality_model(values), labels, observed)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * count
                cell_count += count
            metrics = _route2_eval_quality(
                quality_model, features, dev_positions, rows, strongest_dev, data,
                batch_size=batch_size, device=device,
            )
            record = {
                "epoch": epoch,
                "train_quality_bce": loss_sum / max(1, cell_count),
                "train_quality_cells": cell_count,
                **metrics,
            }
            history.append(record)
            current = float(metrics["dev_hard_label_loss"])
            if current < best_loss - 1e-6:
                best_loss, best_epoch, bad_epochs = current, epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in quality_model.state_dict().items()}
            else:
                bad_epochs += 1
            _atomic_text(
                path.parent / "quality_history.jsonl",
                "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in history),
            )
            print(json.dumps({"route": "route2", "seed": seed, **record}, sort_keys=True))
            if epoch >= int(train_cfg["min_epochs"]) and bad_epochs >= int(train_cfg["patience"]):
                break
        if best_state is None:
            raise RuntimeError("Route-2 quality training did not produce a finite best state")
        quality_model.load_state_dict(best_state)
        quality_metrics = _route2_eval_quality(
            quality_model, features, dev_positions, rows, strongest_dev, data,
            batch_size=batch_size, device=device,
        )

        # The cost model uses a fixed, preregistered epoch count: no checkpoint
        # is selected using dev cost or test data.
        seed_everything(seed + 10000)
        cost_model = NonNegativeCostHead(
            int(features.shape[1]), len(data.model_order), float(train_cfg["dropout"])
        ).to(device)
        cost_optimizer = torch.optim.AdamW(
            cost_model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        cost_generator = torch.Generator().manual_seed(seed + 10000)
        cost_history: list[dict[str, Any]] = []
        for epoch in range(1, int(route_cfg["cost_epochs"]) + 1):
            cost_model.train()
            permutation = torch.randperm(len(train_positions), generator=cost_generator)
            loss_sum, samples = 0.0, 0
            for local in _batched_indices(len(train_positions), batch_size, permutation=permutation):
                source = train_positions[local]
                values = torch.from_numpy(np.asarray(features[source])).to(device=device, dtype=torch.float32)
                target = torch.from_numpy((data.c[train_rows[local]].astype(np.float32) / kappa)).to(device)
                cost_optimizer.zero_grad(set_to_none=True)
                prediction = cost_model(values)
                loss = F.mse_loss(prediction, target)
                loss.backward()
                cost_optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(local)
                samples += len(local)
            cost_history.append({"epoch": epoch, "train_cost_scaled_mse": loss_sum / max(1, samples)})
        cost_model.eval()
        predicted_parts = []
        with torch.no_grad():
            for local in _batched_indices(len(dev_positions), batch_size):
                values = torch.from_numpy(np.asarray(features[dev_positions[local]])).to(device=device, dtype=torch.float32)
                predicted_parts.append(cost_model(values).float().cpu().numpy() * kappa)
        predicted_cost = np.concatenate(predicted_parts, axis=0)
        actual_cost = data.c[dev_rows].astype(np.float64)
        cost_metrics = {
            "dev_cost_mae": float(np.abs(predicted_cost - actual_cost).mean()),
            "dev_cost_rmse": float(np.sqrt(np.square(predicted_cost - actual_cost).mean())),
            "dev_cost_negative_predictions": int((predicted_cost < 0).sum()),
            "fixed_cost_epochs": int(route_cfg["cost_epochs"]),
            "checkpoint_selection": "fixed_epoch_no_dev_cost_selection",
        }
        calibration = identity_calibration(len(data.model_order))
        checkpoint = {
            "schema_version": 1,
            "identity": identity,
            "quality_state": best_state,
            "cost_state": {name: value.detach().cpu().clone() for name, value in cost_model.state_dict().items()},
            "quality_architecture": "AuditedBenchCoEHead_independent_sigmoid",
            "cost_architecture": "NonNegativeCostHead_direct_C_over_kappa",
            "hidden_dim": int(features.shape[1]),
            "num_experts": len(data.model_order),
            "model_order": tuple(data.model_order),
            "fallback_expert_index": int(np.argmax(accuracy)),
            "train_global_accuracy": accuracy,
            "train_mean_cost": data.c[np.asarray(data.split_to_rows["train"], dtype=np.int64)].astype(np.float64).mean(axis=0),
            "kappa": kappa,
            "kappa_status": kappa_status,
            "calibration_temperatures": calibration.temperatures,
            "calibration_status": calibration.source_role,
            "best_epoch": best_epoch,
            "best_dev_hard_label_loss": best_loss,
            "quality_metrics": quality_metrics,
            "cost_metrics": cost_metrics,
            "quality_history": history,
            "cost_history": cost_history,
            "feature_evidence": feature_evidence,
            "protocol_sha256": protocol_sha,
            "created_at_unix": time.time(),
        }
        _atomic_torch_save(path, checkpoint)
        status = {
            "schema_version": 1,
            "status": "COMPLETE",
            "identity": identity,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "quality_metrics": quality_metrics,
            "cost_metrics": cost_metrics,
            "calibration": "BLOCKED_NO_LEGAL_QCAL_IDENTITY_USED",
        }
        _atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True))
    return 0


def _load_checkpoint(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    resolved = Path(path).resolve()
    digest = sha256_file(resolved)
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(resolved, map_location="cpu")
    if not isinstance(value, dict) or int(value.get("schema_version", -1)) != 1:
        raise RuntimeError(f"invalid cost-route checkpoint: {resolved}")
    return value, digest, resolved


def _target_free_rows(
    contract: ExperimentContract,
    split: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...], dict[int, int], np.ndarray]:
    sample_registry = load_registry_list(
        contract.resolve(str(contract.values["data"]["registry"]["samples"]["path"])), name="sample"
    )
    sample_to_row = build_index(sample_registry, name="sample")
    split_sample_ids = _split_ids(contract, split)
    ordered_rows = np.asarray(sorted(sample_to_row[sample_id] for sample_id in split_sample_ids), dtype=np.int64)
    ordered_ids = tuple(sample_registry[int(row)] for row in ordered_rows)
    feature_position = {int(record["row_index"]): position for position, record in enumerate(records)}
    present = np.asarray([int(row) in feature_position for row in ordered_rows], dtype=bool)
    return ordered_rows, ordered_ids, feature_position, present


def _model_logits(
    model: nn.Module,
    features: np.ndarray,
    feature_positions: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for local in _batched_indices(len(feature_positions), batch_size):
            values = torch.from_numpy(np.asarray(features[feature_positions[local]])).to(device=device, dtype=torch.float32)
            parts.append(model(values).float().cpu().numpy())
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32)


def _write_prediction_records(
    output: Path,
    *,
    rows: np.ndarray,
    sample_ids: Sequence[str],
    predictions: np.ndarray,
    model_order: Sequence[str],
    split: str,
    seed: int,
    checkpoint_sha256: str,
    route: str,
    method: str,
    fallback: np.ndarray,
    policy: Mapping[str, Any],
    scores: np.ndarray | None = None,
    decision_costs: np.ndarray | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload_parts: list[str] = []
    for position, (row, sample_id, prediction) in enumerate(zip(rows, sample_ids, predictions)):
        record: dict[str, Any] = {
            "schema_version": 1,
            "row_index": int(row),
            "sample_id": str(sample_id),
            "split": split,
            "seed": int(seed),
            "route": route,
            "method": method,
            "checkpoint_sha256": checkpoint_sha256,
            "predicted_expert_index": int(prediction),
            "predicted_expert_id": str(model_order[int(prediction)]),
            "routing_input_status": "train_strongest_fallback" if fallback[position] else "frozen_feature",
            "policy": dict(policy),
        }
        if scores is not None:
            record["quality_scores"] = [float(value) for value in scores[position]]
        if decision_costs is not None:
            record["predicted_costs"] = [float(value) for value in decision_costs[position]]
        if FORBIDDEN_PREDICTION_KEYS.intersection(record):
            raise AssertionError("prediction schema unexpectedly contains a target field")
        payload_parts.append(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    payload = "".join(payload_parts).encode("utf-8")
    _atomic_bytes(output, payload)
    receipt = {
        "schema_version": 1,
        "status": "TARGET_FREE_PREDICTION_FROZEN",
        "prediction_path": str(output),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "records": len(rows),
        "split": split,
        "seed": int(seed),
        "route": route,
        "method": method,
        "checkpoint_sha256": checkpoint_sha256,
        "producer_loaded_y_or_c": False,
        "forbidden_matrix_paths": ["data/matrices/Y.npz", "data/matrices/C.npy"],
        "source_hashes": _source_hashes(),
        "policy": dict(policy),
        "extra_provenance": dict(extra_provenance or {}),
    }
    _atomic_json(output.with_suffix(output.suffix + ".target_free_receipt.json"), receipt)
    return receipt


def run_predict_route1(args: argparse.Namespace) -> int:
    _assert_environment(full=True)
    _config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    checkpoint, checkpoint_sha, _checkpoint_path_value = _load_checkpoint(args.checkpoint)
    if checkpoint.get("protocol_sha256") != protocol_sha or checkpoint["identity"]["route"] != "route1":
        raise RuntimeError("Route-1 checkpoint/protocol mismatch")
    require_exact_order(checkpoint["model_order"], contract.model_order, name="checkpoint model order")
    features, records, _manifest, feature_evidence = _feature_cache(contract, args.backbone)
    if feature_evidence["manifest_file_sha256"] != checkpoint["feature_evidence"]["manifest_file_sha256"]:
        raise RuntimeError("Route-1 checkpoint feature cache mismatch")
    rows, sample_ids, feature_position, present = _target_free_rows(contract, args.split, records)
    selected_positions = np.asarray([feature_position[int(row)] for row in rows[present]], dtype=np.int64)
    device = torch.device(args.device)
    model = AuditedBenchCoEHead(
        int(checkpoint["hidden_dim"]), int(checkpoint["num_experts"]),
        float(_load_cost_config(args.cost_config)[0]["training"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logits = np.full((len(rows), len(contract.model_order)), -1e9, dtype=np.float32)
    logits[present] = _model_logits(
        model, features, selected_positions, batch_size=int(args.batch_size), device=device
    )
    fallback_index = int(checkpoint["fallback_expert_index"])
    logits[~present, fallback_index] = 0.0
    predictions = logits.argmax(axis=1).astype(np.int64)
    receipt = _write_prediction_records(
        Path(args.output).resolve(), rows=rows, sample_ids=sample_ids, predictions=predictions,
        model_order=contract.model_order, split=args.split, seed=int(checkpoint["identity"]["seed"]),
        checkpoint_sha256=checkpoint_sha, route="route1", method=str(checkpoint["identity"]["method"]),
        fallback=~present, policy={"name": "classification_argmax"}, scores=logits,
        extra_provenance={"feature_evidence": feature_evidence},
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def run_score_route2(args: argparse.Namespace) -> int:
    _assert_environment(full=True)
    _config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    checkpoint, checkpoint_sha, _checkpoint_path_value = _load_checkpoint(args.checkpoint)
    if checkpoint.get("protocol_sha256") != protocol_sha or checkpoint["identity"]["route"] != "route2":
        raise RuntimeError("Route-2 checkpoint/protocol mismatch")
    require_exact_order(checkpoint["model_order"], contract.model_order, name="checkpoint model order")
    features, records, _manifest, feature_evidence = _feature_cache(contract, args.backbone)
    if feature_evidence["manifest_file_sha256"] != checkpoint["feature_evidence"]["manifest_file_sha256"]:
        raise RuntimeError("Route-2 checkpoint feature cache mismatch")
    rows, sample_ids, feature_position, present = _target_free_rows(contract, args.split, records)
    selected_positions = np.asarray([feature_position[int(row)] for row in rows[present]], dtype=np.int64)
    device = torch.device(args.device)
    protocol = _load_cost_config(args.cost_config)[0]
    quality_model = AuditedBenchCoEHead(
        int(checkpoint["hidden_dim"]), int(checkpoint["num_experts"]), float(protocol["training"]["dropout"])
    ).to(device)
    quality_model.load_state_dict(checkpoint["quality_state"])
    quality_logits = np.full((len(rows), len(contract.model_order)), -50.0, dtype=np.float32)
    quality_logits[present] = _model_logits(
        quality_model, features, selected_positions, batch_size=int(args.batch_size), device=device
    )
    temperatures = np.asarray(checkpoint["calibration_temperatures"], dtype=np.float64)
    quality = 1.0 / (1.0 + np.exp(-np.clip(quality_logits / temperatures[None, :], -50, 50)))
    cost_model = NonNegativeCostHead(
        int(checkpoint["hidden_dim"]), int(checkpoint["num_experts"]), float(protocol["training"]["dropout"])
    ).to(device)
    cost_model.load_state_dict(checkpoint["cost_state"])
    learned = np.tile(np.asarray(checkpoint["train_mean_cost"], dtype=np.float64), (len(rows), 1))
    if present.any():
        learned[present] = _model_logits(
            cost_model, features, selected_positions, batch_size=int(args.batch_size), device=device
        ) * float(checkpoint["kappa"])
    mean_cost = np.tile(np.asarray(checkpoint["train_mean_cost"], dtype=np.float64), (len(rows), 1))
    fallback_index = int(checkpoint["fallback_expert_index"])
    quality[~present] = 0.0
    quality[~present, fallback_index] = 1.0
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            rows=rows,
            quality=quality.astype(np.float32),
            fit_mean_cost=mean_cost.astype(np.float32),
            learned_cost=learned.astype(np.float32),
            fallback=(~present),
        )
        produced = Path(temporary + ".npz") if Path(temporary + ".npz").is_file() else Path(temporary)
        os.replace(produced, output)
    finally:
        for candidate in (Path(temporary), Path(temporary + ".npz")):
            if candidate.exists():
                candidate.unlink()
    metadata = {
        "schema_version": 1,
        "status": "TARGET_FREE_SCORE_BUNDLE_FROZEN",
        "bundle": str(output),
        "bundle_sha256": sha256_file(output),
        "split": args.split,
        "rows": len(rows),
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "seed": int(checkpoint["identity"]["seed"]),
        "protocol_sha256": protocol_sha,
        "base_contract_sha256": contract.sha256,
        "model_order": list(contract.model_order),
        "kappa": float(checkpoint["kappa"]),
        "calibration_status": checkpoint["calibration_status"],
        "producer_loaded_y_or_c": False,
        "feature_evidence": feature_evidence,
        "source_hashes": _source_hashes(),
    }
    _atomic_json(output.with_suffix(output.suffix + ".json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _load_reference_predictions(path: str | Path, rows: np.ndarray, split: str) -> tuple[np.ndarray, str]:
    resolved = Path(path).resolve()
    digest = sha256_file(resolved)
    records = _read_jsonl(resolved)
    by_row: dict[int, int] = {}
    for record in records:
        if str(record.get("split")) != split:
            raise DataIntegrityError("reference prediction split mismatch")
        row = int(record["row_index"])
        if row in by_row:
            raise DataIntegrityError("duplicate reference prediction row")
        by_row[row] = int(record["predicted_expert_index"])
    if set(by_row) != set(int(row) for row in rows):
        raise DataIntegrityError("reference prediction coverage mismatch")
    return np.asarray([by_row[int(row)] for row in rows], dtype=np.int64), digest


def run_materialize_route2(args: argparse.Namespace) -> int:
    _assert_environment(full=True)
    _config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    bundle_path = Path(args.bundle).resolve()
    metadata_path = bundle_path.with_suffix(bundle_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("bundle_sha256") != sha256_file(bundle_path):
        raise RuntimeError("Route-2 score bundle hash mismatch")
    if metadata.get("protocol_sha256") != protocol_sha or metadata.get("base_contract_sha256") != contract.sha256:
        raise RuntimeError("Route-2 score bundle protocol mismatch")
    if metadata.get("split") != args.split:
        raise RuntimeError("Route-2 score bundle split mismatch")
    with np.load(bundle_path, allow_pickle=False) as archive:
        rows = np.asarray(archive["rows"], dtype=np.int64)
        quality = np.asarray(archive["quality"], dtype=np.float64)
        costs = np.asarray(archive[f"{args.cost_model}_cost"], dtype=np.float64)
        fallback = np.asarray(archive["fallback"], dtype=bool)
    split_ids = _split_ids(contract, args.split)
    sample_registry = load_registry_list(
        contract.resolve(str(contract.values["data"]["registry"]["samples"]["path"])), name="sample"
    )
    expected_rows = np.asarray(sorted(build_index(sample_registry, name="sample")[sample_id] for sample_id in split_ids), dtype=np.int64)
    if not np.array_equal(rows, expected_rows):
        raise DataIntegrityError("score bundle rows differ from the frozen split")
    sample_ids = tuple(sample_registry[int(row)] for row in rows)
    reference_digest = None
    if args.policy == "quality_only":
        predictions = quality_only(quality, costs)
        policy = {"name": "quality_only", "cost_model": args.cost_model}
    elif args.policy == "utility":
        predictions = utility_policy(
            quality, costs, lambda_cost=float(args.lambda_cost), kappa=float(metadata["kappa"])
        )
        policy = {"name": "utility", "lambda_cost": float(args.lambda_cost), "cost_model": args.cost_model}
    else:
        if not args.reference_predictions:
            raise ValueError("predicted_gap requires --reference-predictions")
        reference, reference_digest = _load_reference_predictions(args.reference_predictions, rows, args.split)
        predictions = predicted_gap_policy(
            quality,
            costs,
            reference,
            max_quality_gap=float(args.max_quality_gap),
            minimum_saving=float(args.minimum_saving),
        )
        policy = {
            "name": "predicted_gap",
            "max_quality_gap": float(args.max_quality_gap),
            "minimum_saving": float(args.minimum_saving),
            "cost_model": args.cost_model,
            "reference_prediction_sha256": reference_digest,
        }
    if fallback.any():
        predictions[fallback] = np.argmax(quality[fallback], axis=1)
    if args.method_id:
        method = args.method_id
    elif args.policy == "quality_only":
        method = f"quality_only_{args.cost_model}"
    elif args.policy == "utility":
        method = f"utility_{args.cost_model}_l{args.lambda_cost}"
    else:
        method = f"predicted_gap_{args.cost_model}_g{args.max_quality_gap}_s{args.minimum_saving}"
    receipt = _write_prediction_records(
        Path(args.output).resolve(), rows=rows, sample_ids=sample_ids, predictions=predictions,
        model_order=contract.model_order, split=args.split, seed=int(metadata["seed"]),
        checkpoint_sha256=str(metadata["checkpoint_sha256"]), route=args.route_name,
        method=method, fallback=fallback, policy=policy, scores=quality, decision_costs=costs,
        extra_provenance={"score_bundle_sha256": metadata["bundle_sha256"], "reference_sha256": reference_digest},
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _load_frozen_predictions(path: str | Path) -> tuple[list[dict[str, Any]], str, bytes]:
    resolved = Path(path).resolve()
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    records = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        value = json.loads(line)
        if FORBIDDEN_PREDICTION_KEYS.intersection(value):
            raise DataIntegrityError(f"prediction contains target field at line {line_number}")
        records.append(value)
    receipt_candidates = (
        resolved.with_suffix(resolved.suffix + ".target_free_receipt.json"),
        resolved.with_suffix(".target_free_receipt.json"),
    )
    receipt_path = next((candidate for candidate in receipt_candidates if candidate.is_file()), None)
    if receipt_path is None:
        raise DataIntegrityError(f"target-free receipt missing for {resolved}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_prediction_sha = receipt.get("prediction_sha256")
    if receipt_prediction_sha is None and isinstance(receipt.get("prediction"), dict):
        receipt_prediction_sha = receipt["prediction"].get("sha256")
    loaded_targets = receipt.get("producer_loaded_y_or_c")
    if loaded_targets is None and isinstance(receipt.get("process_boundary"), dict):
        loaded_targets = receipt["process_boundary"].get("prediction_process_loaded_y_or_c")
    if receipt_prediction_sha != digest or loaded_targets is not False:
        raise DataIntegrityError("target-free prediction receipt mismatch")
    return records, digest, payload


def _bootstrap_interval(values: np.ndarray, *, seed: int = 20260905, draws: int = 2000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return math.nan, math.nan
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = generator.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_evaluate(args: argparse.Namespace) -> int:
    _assert_environment(full=True)
    config, protocol_sha, _config_path, contract = _load_protocol(args.cost_config)
    # Freeze and validate prediction/reference bytes before the evaluator opens Y/C.
    records, prediction_sha, _prediction_payload = _load_frozen_predictions(args.predictions)
    reference_records = reference_sha = None
    if args.reference_predictions:
        reference_records, reference_sha, _reference_payload = _load_frozen_predictions(args.reference_predictions)
    data = load_vlrb_strict(contract.dataset_root, contract=contract)
    expected_rows = np.asarray(data.split_to_rows[args.split], dtype=np.int64)
    by_row: dict[int, dict[str, Any]] = {}
    for record in records:
        row = int(record["row_index"])
        if row in by_row or record.get("split") != args.split:
            raise DataIntegrityError("prediction has duplicate rows or wrong split")
        by_row[row] = record
    if set(by_row) != set(int(row) for row in expected_rows):
        raise DataIntegrityError("prediction coverage differs from the frozen split")
    ordered = [by_row[int(row)] for row in expected_rows]
    predictions = np.asarray([int(record["predicted_expert_index"]) for record in ordered], dtype=np.int64)
    if (predictions < 0).any() or (predictions >= len(data.model_order)).any():
        raise DataIntegrityError("prediction expert index out of range")
    outcomes = data.y[expected_rows, predictions].astype(np.int8)
    costs = data.c[expected_rows, predictions].astype(np.float64)
    reference_predictions = reference_outcomes = reference_costs = harm = gain = changed = None
    delta_accuracy = cost_saving = more_expensive_rate = None
    interval = (None, None)
    if reference_records is not None:
        reference_by_row = {int(record["row_index"]): record for record in reference_records}
        if set(reference_by_row) != set(int(row) for row in expected_rows):
            raise DataIntegrityError("reference prediction coverage mismatch")
        reference_predictions = np.asarray(
            [int(reference_by_row[int(row)]["predicted_expert_index"]) for row in expected_rows], dtype=np.int64
        )
        reference_outcomes = data.y[expected_rows, reference_predictions].astype(np.int8)
        reference_costs = data.c[expected_rows, reference_predictions].astype(np.float64)
        harm = ((reference_outcomes == 1) & (outcomes == 0)).astype(np.int8)
        gain = ((reference_outcomes == 0) & (outcomes == 1)).astype(np.int8)
        paired = outcomes.astype(np.float64) - reference_outcomes.astype(np.float64)
        delta_accuracy = float(paired.mean())
        low, high = _bootstrap_interval(paired)
        interval = (low, high)
        changed = predictions != reference_predictions
        more_expensive_rate = float((costs[changed] > reference_costs[changed]).mean()) if changed.any() else None
        reference_mean_cost = float(reference_costs.mean())
        cost_saving = None if reference_mean_cost == 0 else float((reference_mean_cost - costs.mean()) / reference_mean_cost)
    usage_counts = np.bincount(predictions, minlength=len(data.model_order))
    per_dataset: dict[str, Any] = {}
    for dataset in sorted({data.samples[int(row)].dataset for row in expected_rows}):
        mask = np.asarray([data.samples[int(row)].dataset == dataset for row in expected_rows], dtype=bool)
        item: dict[str, Any] = {
            "n": int(mask.sum()),
            "accuracy": float(outcomes[mask].mean()),
            "expert_cost": float(costs[mask].mean()),
        }
        if reference_outcomes is not None:
            item.update(
                {
                    "reference_accuracy": float(reference_outcomes[mask].mean()),
                    "delta_accuracy": float((outcomes[mask] - reference_outcomes[mask]).mean()),
                    "harm": int(harm[mask].sum()),
                    "gain": int(gain[mask].sum()),
                    "downgrade_rate": float(changed[mask].mean()),
                }
            )
        per_dataset[dataset] = item
    first = ordered[0]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "EVALUATED_FROM_FROZEN_TARGET_FREE_PREDICTIONS",
        "route": first["route"],
        "method": first["method"],
        "seed": int(first["seed"]),
        "split": args.split,
        "n": int(len(expected_rows)),
        "accuracy": float(outcomes.mean()),
        "expert_cost": float(costs.mean()),
        "expert_cost_unit": config["costs"]["expert_cost_unit"],
        "router_cost": None,
        "total_cost": None,
        "complete_net_saving_status": "NOT_MEASURED_IN_COMMON_FINANCIAL_UNIT",
        "prediction_path": str(Path(args.predictions).resolve()),
        "prediction_sha256": prediction_sha,
        "reference_prediction_path": None if args.reference_predictions is None else str(Path(args.reference_predictions).resolve()),
        "reference_prediction_sha256": reference_sha,
        "reference_accuracy": None if reference_outcomes is None else float(reference_outcomes.mean()),
        "reference_expert_cost": None if reference_costs is None else float(reference_costs.mean()),
        "delta_accuracy": delta_accuracy,
        "paired_bootstrap_95_low": interval[0],
        "paired_bootstrap_95_high": interval[1],
        "paired_interval_status": "DESCRIPTIVE_ROW_BOOTSTRAP_GROUP_IDS_UNAVAILABLE" if reference_outcomes is not None else None,
        "harm_count": None if harm is None else int(harm.sum()),
        "gain_count": None if gain is None else int(gain.sum()),
        "joint_harm_rate": None if harm is None else float(harm.mean()),
        "downgrade_rate": None if changed is None else float(changed.mean()),
        "conditional_harm_rate": None if changed is None or not changed.any() else float(harm[changed].mean()),
        "actual_more_expensive_replacement_rate": more_expensive_rate,
        "expert_cost_saving_fraction": cost_saving,
        "certification_status": "UNCERTIFIED_NO_INDEPENDENT_CERT_SPLIT",
        "post_freeze_test_risk_diagnostic": None,
        "fallback_rows": int(sum(record["routing_input_status"] != "frozen_feature" for record in ordered)),
        "expert_usage": {data.model_order[index]: int(count) for index, count in enumerate(usage_counts)},
        "per_dataset": per_dataset,
        "protocol_sha256": protocol_sha,
        "base_contract_sha256": contract.sha256,
    }
    if args.split == "test" and harm is not None:
        delta = float(config["selection"]["risk_delta"])
        upper = clopper_pearson_upper(int(harm.sum()), len(harm), delta)
        result["post_freeze_test_risk_diagnostic"] = {
            "status": "DIAGNOSTIC_NOT_FORMAL_CERTIFICATE_NO_SEPARATE_CERT_ROLE",
            "epsilon": float(config["selection"]["risk_epsilon"]),
            "delta": delta,
            "harms": int(harm.sum()),
            "samples": len(harm),
            "one_sided_exact_upper": upper,
            "would_meet_numeric_threshold": bool(upper is not None and upper <= float(config["selection"]["risk_epsilon"])),
        }
    output = Path(args.output).resolve()
    _atomic_json(output, result)
    trace_path = output.with_suffix(".per_sample.csv")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=trace_path.parent, prefix=f".{trace_path.name}.") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index", "sample_id", "dataset", "prediction", "selected_correct", "selected_cost",
                "reference_prediction", "reference_correct", "reference_cost", "harm", "gain", "changed",
            ],
        )
        writer.writeheader()
        for position, row in enumerate(expected_rows):
            writer.writerow(
                {
                    "row_index": int(row),
                    "sample_id": data.samples[int(row)].sample_id,
                    "dataset": data.samples[int(row)].dataset,
                    "prediction": int(predictions[position]),
                    "selected_correct": int(outcomes[position]),
                    "selected_cost": format(float(costs[position]), ".17g"),
                    "reference_prediction": "" if reference_predictions is None else int(reference_predictions[position]),
                    "reference_correct": "" if reference_outcomes is None else int(reference_outcomes[position]),
                    "reference_cost": "" if reference_costs is None else format(float(reference_costs[position]), ".17g"),
                    "harm": "" if harm is None else int(harm[position]),
                    "gain": "" if gain is None else int(gain[position]),
                    "changed": "" if changed is None else int(changed[position]),
                }
            )
        temporary = handle.name
    os.replace(temporary, trace_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-config", default=DEFAULT_COST_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare")
    subparsers.add_parser("opportunity")

    train1 = subparsers.add_parser("train-route1")
    train1.add_argument("--backbone", default="qwen3vl", choices=("qwen3vl", "tinyllava"))
    train1.add_argument("--method", required=True)
    train1.add_argument("--seeds", default="0,1,2,3,4")
    train1.add_argument("--device", default="cuda")
    train1.add_argument("--force", action="store_true")

    train2 = subparsers.add_parser("train-route2")
    train2.add_argument("--backbone", default="qwen3vl", choices=("qwen3vl", "tinyllava"))
    train2.add_argument("--seeds", default="0,1,2,3,4")
    train2.add_argument("--device", default="cuda")
    train2.add_argument("--force", action="store_true")

    predict1 = subparsers.add_parser("predict-route1")
    predict1.add_argument("--backbone", default="qwen3vl", choices=("qwen3vl", "tinyllava"))
    predict1.add_argument("--checkpoint", required=True)
    predict1.add_argument("--split", required=True, choices=("train", "dev", "test"))
    predict1.add_argument("--output", required=True)
    predict1.add_argument("--batch-size", type=int, default=512)
    predict1.add_argument("--device", default="cuda")

    score2 = subparsers.add_parser("score-route2")
    score2.add_argument("--backbone", default="qwen3vl", choices=("qwen3vl", "tinyllava"))
    score2.add_argument("--checkpoint", required=True)
    score2.add_argument("--split", required=True, choices=("train", "dev", "test"))
    score2.add_argument("--output", required=True)
    score2.add_argument("--batch-size", type=int, default=512)
    score2.add_argument("--device", default="cuda")

    materialize = subparsers.add_parser("materialize-route2")
    materialize.add_argument("--bundle", required=True)
    materialize.add_argument("--split", required=True, choices=("train", "dev", "test"))
    materialize.add_argument("--policy", required=True, choices=("quality_only", "utility", "predicted_gap"))
    materialize.add_argument("--cost-model", default="fit_mean", choices=("fit_mean", "learned"))
    materialize.add_argument("--lambda-cost", type=float, default=0.0)
    materialize.add_argument("--max-quality-gap", type=float, default=0.0)
    materialize.add_argument("--minimum-saving", type=float, default=0.0)
    materialize.add_argument("--reference-predictions")
    materialize.add_argument("--route-name", default="route2", choices=("route2", "route3"))
    materialize.add_argument("--method-id")
    materialize.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--reference-predictions")
    evaluate.add_argument("--split", required=True, choices=("train", "dev", "test"))
    evaluate.add_argument("--output", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--route", required=True, choices=("route1", "route2", "route3"))
    select.add_argument("--eval-root", required=True)
    select.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "prepare": run_prepare,
        "opportunity": run_opportunity,
        "train-route1": run_train_route1,
        "train-route2": run_train_route2,
        "predict-route1": run_predict_route1,
        "score-route2": run_score_route2,
        "materialize-route2": run_materialize_route2,
        "evaluate": run_evaluate,
        "select": run_select,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
