"""两次运行的对比与回归门禁。

总准确率不变，也可能是“修好 3 条、改坏 3 条”。所以对比以逐条翻转为核心：
  修好：上次错 → 这次对        改坏（回归）：上次对 → 这次错
再用 McNemar 检验判断翻转的不对称是否显著。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .datasets import EvalHubError
from .db import Run
from .stats import mcnemar_exact

KEY_METRICS = {
    "policy": ["accuracy", "false_approve", "false_reject", "extra_review", "human_review_share",
               "llm_calls", "tokens"],
    "extraction": ["doc_accuracy", "silent_errors", "flagged", "model_calls", "tokens", "avg_latency_ms"],
}
# 越小越好的指标
LOWER_IS_BETTER = {"false_approve", "false_reject", "extra_review", "human_review_share", "llm_calls",
                   "tokens", "silent_errors", "flagged", "model_calls", "avg_latency_ms"}


@dataclass
class Comparison:
    base: Run
    new: Run
    fixed: list[dict] = field(default_factory=list)
    broken: list[dict] = field(default_factory=list)
    changed: list[dict] = field(default_factory=list)     # 结论变了但对错没变（如两次都错但错法不同）
    p_value: float = 1.0
    deltas: dict = field(default_factory=dict)


def compare(base: Run, new: Run) -> Comparison:
    if base.version_id != new.version_id:
        raise EvalHubError(f"两次运行不在同一个评测集版本上（{base.version.ref} vs {new.version.ref}），"
                           "逐条对比没有意义")
    cmp = Comparison(base, new)
    old = {r.case_id: r for r in base.results}
    for r in new.results:
        o = old.get(r.case_id)
        if o is None:
            continue
        row = {"key": r.case.key, "tags": r.case.tags, "gold": r.gold, "before": o.predicted,
               "after": r.predicted}
        if o.correct is True and r.correct is False:
            cmp.broken.append(row)
        elif o.correct is False and r.correct is True:
            cmp.fixed.append(row)
        elif o.predicted != r.predicted:
            cmp.changed.append(row)
    cmp.p_value = mcnemar_exact(len(cmp.broken), len(cmp.fixed))
    for k in KEY_METRICS.get(new.task, []):
        a, b = base.metrics.get(k), new.metrics.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            cmp.deltas[k] = (a, b)
    return cmp


def format_comparison(cmp: Comparison, limit: int = 20) -> str:
    b, n = cmp.base, cmp.new
    lines = [f"对比 #{b.id}（{b.note or '无备注'}） → #{n.id}（{n.note or '无备注'}），评测集 {n.version.ref}"]
    same = [k for k, (x, y) in {"代码": (b.code_hash, n.code_hash), "提示词": (b.prompt_hash, n.prompt_hash),
                                "制度": (b.policy_hash, n.policy_hash), "模型": (b.models, n.models)}.items() if x == y]
    diff = [k for k in ("代码", "提示词", "制度", "模型") if k not in same]
    lines.append(f"变化项：{'、'.join(diff) or '无（两次运行配置相同，可用来看模型输出是否稳定）'}")
    lines.append("\n| 指标 | 之前 | 之后 | 变化 |\n|---|---|---|---|")
    for k, (x, y) in cmp.deltas.items():
        d = y - x
        good = (d < 0) if k in LOWER_IS_BETTER else (d > 0)
        mark = "" if d == 0 else (" ✓" if good else " ✗")
        fmt = (lambda v: f"{v:.1%}") if k in ("accuracy", "doc_accuracy", "human_review_share") else str
        lines.append(f"| {k} | {fmt(x)} | {fmt(y)} | {d:+.4g}{mark} |")
    lines.append(f"\n修好 {len(cmp.fixed)} 条，改坏 {len(cmp.broken)} 条，"
                 f"结论变化但对错不变 {len(cmp.changed)} 条；McNemar p = {cmp.p_value:.3f}"
                 + ("（差异显著）" if cmp.p_value < 0.05 else "（差异不显著，可能是偶然波动）"))
    for title, rows in (("改坏的（回归）", cmp.broken), ("修好的", cmp.fixed), ("结论变化", cmp.changed)):
        if rows:
            lines.append(f"\n{title}：")
            for r in rows[:limit]:
                tag = r["tags"].get("family") or r["tags"].get("scenario") or ""
                lines.append(f"  {r['key']} [{tag}] 标准答案 {r['gold']}：{r['before']} → {r['after']}")
            if len(rows) > limit:
                lines.append(f"  …… 另有 {len(rows) - limit} 条")
    return "\n".join(lines)


def gate(cmp: Comparison, max_broken: int = 0, waive: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """回归门禁：返回不通过的原因，空列表表示通过。

    看的是代价高的错误，不是总准确率：误通过、误驳回、静默错误只要比基线多就不通过；
    改坏的用例超过 max_broken 条也不通过（默认一条都不允许）。
    waive：人工确认过、允许改坏的用例（例如标注人自己也拿不准的灰色用例），必须写明理由。
    """
    reasons = []
    for k in ("false_approve", "false_reject", "silent_errors"):
        if k in cmp.deltas and cmp.deltas[k][1] > cmp.deltas[k][0]:
            reasons.append(f"{k} 从 {cmp.deltas[k][0]} 增加到 {cmp.deltas[k][1]}")
    broken = [r for r in cmp.broken if r["key"] not in waive]
    if len(broken) > max_broken:
        reasons.append(f"有 {len(broken)} 条用例从对变错（允许 {max_broken} 条）："
                       + "、".join(r["key"] for r in broken))
    return reasons
