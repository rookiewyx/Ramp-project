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


# ---------------- 多次运行的稳定性 ----------------
def stability_from(rows_per_run: list[dict[str, tuple]]) -> dict:
    """rows_per_run：每次运行一个 {用例: (是否正确, 系统结论, 标准答案)}。

    Agent 带随机性（即使 temperature=0，接口侧也不保证完全确定），一次运行的准确率可能是运气。这里给出：
      - 每次的准确率和均值 ± 标准差；
      - pass^k：k 次都对的用例占比（τ-bench 的口径），衡量“上线后每次都对”；
      - pass@k：至少对一次的占比，和 pass^k 的差就是不稳定的部分；
      - 结论不稳定的用例：同一张单子不同次给出不同结论，逐条列出。
    """
    import statistics

    k = len(rows_per_run)
    keys = sorted(set.intersection(*(set(r) for r in rows_per_run))) if rows_per_run else []
    scored = [key for key in keys if all(r[key][0] is not None for r in rows_per_run)]
    accs = [sum(bool(r[key][0]) for key in scored) / len(scored) for r in rows_per_run] if scored else []
    false_approve = [sum(1 for key in scored if r[key][1] == "approve" and r[key][2] not in (None, "approve"))
                     for r in rows_per_run]
    unstable = []
    for key in keys:
        preds = [r[key][1] for r in rows_per_run]
        if len(set(preds)) > 1:
            unstable.append({"key": key, "gold": rows_per_run[0][key][2], "predicted": preds,
                             "correct": sum(bool(r[key][0]) for r in rows_per_run)})
    return {
        "runs": k, "cases": len(keys), "scored": len(scored),
        "accuracy": [round(a, 4) for a in accs],
        "mean": round(statistics.mean(accs), 4) if accs else None,
        "std": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
        "pass_all": round(sum(all(r[key][0] for r in rows_per_run) for key in scored) / len(scored), 4)
        if scored else None,
        "pass_any": round(sum(any(r[key][0] for r in rows_per_run) for key in scored) / len(scored), 4)
        if scored else None,
        "false_approve": false_approve,
        "unstable": unstable,
    }


def stability(runs: list[Run]) -> dict:
    if not runs:
        raise EvalHubError("至少要一次运行")
    if len({r.version_id for r in runs}) > 1 or len({r.task for r in runs}) > 1:
        raise EvalHubError("这些运行不在同一个评测集版本上，不能算稳定性")
    out = stability_from([{x.case.key: (x.correct, x.predicted, x.gold) for x in r.results} for r in runs])
    out["run_ids"] = [r.id for r in runs]
    out["ref"] = runs[0].version.ref
    out["tokens"] = [r.metrics.get("tokens") for r in runs]
    configs = {(r.code_hash, r.prompt_hash, r.policy_hash, str(r.models), str(r.params)) for r in runs}
    out["same_config"] = len(configs) == 1
    return out


def format_stability(st: dict, limit: int = 20) -> str:
    k = st["runs"]
    lines = [f"稳定性：{k} 次运行 {st.get('run_ids', '')}，评测集 {st.get('ref', '')}，{st['scored']} 条参与打分"]
    if not st.get("same_config", True):
        lines.append("注意：这几次运行的代码、提示词、模型或参数不完全一样，差异不全是随机性")
    if st["mean"] is None:
        return "\n".join(lines + ["没有可打分的用例"])
    lines.append("每次准确率：" + "、".join(f"{a:.1%}" for a in st["accuracy"])
                 + f"；均值 {st['mean']:.1%} ± {st['std']:.1%}")
    lines.append(f"pass^{k}（{k} 次都对）{st['pass_all']:.1%}；pass@{k}（至少对一次）{st['pass_any']:.1%}")
    lines.append("每次误通过：" + "、".join(str(x) for x in st["false_approve"]))
    lines.append(f"结论不稳定的用例 {len(st['unstable'])} 条" + ("：" if st["unstable"] else ""))
    for u in st["unstable"][:limit]:
        lines.append(f"  {u['key']} 标准答案 {u['gold']}：{' / '.join(str(p) for p in u['predicted'])}"
                     f"（{u['correct']}/{k} 次对）")
    if len(st["unstable"]) > limit:
        lines.append(f"  …… 另有 {len(st['unstable']) - limit} 条")
    return "\n".join(lines)
