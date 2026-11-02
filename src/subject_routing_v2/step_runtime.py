"""Receipt and rolling-state helpers for the 40-step V2 study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ._io import atomic_write_text, load_json


PACKAGE_INDEX = Path(
    "codex_prompts/bench_coe_subject_v2_research_codex/STEP_INDEX.json"
)
DEFAULT_RESOURCES: dict[str, Any] = {
    "training_runs": 0,
    "gpu_ids": [],
    "gpu_seconds": 0,
    "paid_api_calls": 0,
    "paid_api_usd": 0,
    "expert_generation_calls": 0,
    "test_outcome_rows_opened": 0,
}


def load_step_index(repository_root: Path) -> list[dict[str, Any]]:
    """Load the package's authoritative ordered step index."""

    return load_json(repository_root / PACKAGE_INDEX)


def dependency_statuses(
    repository_root: Path, dependencies: Iterable[str]
) -> list[dict[str, Any]]:
    """Fail closed on missing or nonterminal dependency receipts."""

    from ._io import sha256_file

    statuses: list[dict[str, Any]] = []
    for step_id in dependencies:
        path = repository_root / f"artifacts/subject_routing_v2/steps/{step_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing dependency receipt: {path}")
        receipt = load_json(path)
        status = str(receipt.get("status", "UNKNOWN"))
        if status in {"FAIL", "NOT_RUN", "UNKNOWN"}:
            raise RuntimeError(f"dependency {step_id} is not usable: {status}")
        statuses.append(
            {
                "step_id": step_id,
                "status": status,
                "receipt_sha256": sha256_file(path),
            }
        )
    return statuses


def render_state(
    repository_root: Path,
    *,
    current_receipt: dict[str, Any],
    next_step: str | None,
    highlights: Iterable[str],
) -> None:
    """Render STATE.md using receipts plus the pending current receipt."""

    index = load_step_index(repository_root)
    receipts: dict[str, dict[str, Any]] = {}
    receipt_root = repository_root / "artifacts/subject_routing_v2/steps"
    for entry in index:
        path = receipt_root / f"{entry['id']}.json"
        if path.is_file():
            receipts[entry["id"]] = load_json(path)
    receipts[str(current_receipt["step_id"])] = current_receipt

    completed = sum(
        str(value.get("status")) not in {"NOT_RUN", "FAIL", "UNKNOWN"}
        for value in receipts.values()
    )
    timestamp = str(current_receipt.get("completed_at") or current_receipt.get("audited_at") or "UNKNOWN")
    lines = [
        "# Subject Routing V2 状态",
        "",
        f"更新时间：{timestamp}",
        "",
        "## 当前进度",
        "",
        f"- 已产生终态 receipt：{completed}/40。",
        f"- 当前步骤：`{current_receipt['step_id']}` → `{current_receipt['status']}`。",
        f"- 下一步：`{next_step}`。" if next_step else "- 下一步：无（全流程已终结）。",
        "- 历史 `artifacts/subject_routing` 只读；V2 仅写 `subject_routing_v2` 命名空间。",
        "- 默认资源边界：CPU/cache-only；GPU、付费 API、专家重生成均默认禁用。",
        "",
        "## 累计关键结论",
        "",
        "- 历史 OOD q100 主结论保持 `SUPERIORITY_NOT_ESTABLISHED`；ID14 是已曝光的 post-hoc 扩展。",
        "- 现有 query 预测与 subject 的 UID/专家池匹配，但特征、监督、预算不全匹配；严格差距假说仍待 B00 source-only 检验。",
    ]
    for item in highlights:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 步骤状态",
            "",
            "| 步骤 | 状态 | 摘要 |",
            "|---|---|---|",
        ]
    )
    for entry in index:
        receipt = receipts.get(entry["id"])
        if receipt is None:
            status = "NOT_RUN"
            summary = "尚未执行"
        else:
            status = str(receipt.get("status", "UNKNOWN"))
            summary = str(
                receipt.get("summary")
                or receipt.get("reason")
                or entry["title"]
            ).replace("|", "/")
        lines.append(f"| {entry['id']} | {status} | {summary} |")
    lines.extend(
        [
            "",
            "## 固定科学边界",
            "",
            "- 所有既有 seen11、heldout3 和 ID14 test 均为 `OLD_EXPOSED`；旧 test 新方法只可称 `LEGACY_EXPLORATORY_REPLAY`。",
            "- 未到 E04 不读取 test 逐题 Y/C；source LODO 的 held-out source 不得参与 teacher/profile/PCA/taxonomy 或调参。",
            "- InputView 禁止答案、专家回答、实际输出长度、事后成本和 source_id 路由特征。",
            "- full-Y 与 sparse-label 轨隔离；后者所有标签只能经 RevealService 获取并计数。",
            "- micro、dataset-macro、每数据集、专家成本、router 开销分别保存；不把 seed SD 当置信区间。",
        ]
    )
    atomic_write_text(
        repository_root / "docs/subject_routing_v2/STATE.md",
        "\n".join(lines) + "\n",
    )
