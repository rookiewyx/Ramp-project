"""汇总链路追踪数据：一笔报销的时间花在哪里。

输入是 OTEL_EXPORTER=file 写出的 JSONL（默认 reports/traces.jsonl）。输出：
  1. 端到端：提交接口返回 → 进入队列 → worker 开始处理 → 审核完成，各段的 P50 / P95
  2. worker 内部各环节：领取、取文件、识别（视觉模型）、语义判断（文本模型）、规则、落库
  3. 所有 span 按名称汇总

用法：.venv\\Scripts\\python scripts\\trace_report.py [reports\\traces.jsonl] [--since 纳秒时间戳]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = "POST /v1/expenses"
PRODUCE = "apply_async/ramp.review_expense"
RUN = "run/ramp.review_expense"
STAGES = ["pipeline.acquire", "pipeline.fetch_file", "extraction", "policy.semantic", "policy.rules",
          "pipeline.persist"]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def row(name: str, xs: list[float]) -> str:
    return (f"| {name} | {len(xs)} | {pct(xs, .5):,.0f} | {pct(xs, .95):,.0f} | {pct(xs, .99):,.0f} |"
            f" {sum(xs) / len(xs) if xs else 0:,.0f} |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="reports/traces.jsonl")
    p.add_argument("--since", type=int, default=0, help="只统计这个时间（纳秒）之后开始的 span")
    a = p.parse_args()
    path = Path(a.path) if Path(a.path).is_absolute() else ROOT / a.path

    spans = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    spans = [s for s in spans if s["start"] >= a.since]
    by_trace: dict[str, list[dict]] = defaultdict(list)
    by_name: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
        by_name[s["name"]].append(s["duration_ms"])

    api, queue, work, e2e = [], [], [], []
    e2e_by_tier: dict[str, list[float]] = defaultdict(list)
    stage: dict[str, list[float]] = defaultdict(list)
    model: dict[str, list[float]] = defaultdict(list)
    for ss in by_trace.values():
        sub = next((s for s in ss if s["name"] == SUBMIT and s["kind"] == "SERVER"), None)
        prod = next((s for s in ss if s["name"] == PRODUCE), None)
        runs = [s for s in ss if s["name"] == RUN]
        if not (sub and prod and runs):
            continue
        # 有重试时取真正完成审核的那一次（outcome=done）；重复投递被跳过的不算
        done = [s for s in runs if s["attributes"].get("pipeline.outcome") == "done"]
        run = max(done or runs, key=lambda s: s["end"])
        api.append(sub["duration_ms"])
        queue.append((run["start"] - prod["end"]) / 1e6)
        work.append(run["duration_ms"])
        e2e.append((run["end"] - sub["start"]) / 1e6)
        ext = next((s for s in ss if s["name"] == "extraction" and s["parent_id"] == run["span_id"]), None)
        tier = ext["attributes"].get("extraction.tier", "?") if ext else "none"
        e2e_by_tier[tier].append(e2e[-1])
        ids = {run["span_id"]}
        for s in sorted(ss, key=lambda s: s["start"]):        # 找出 run 下面的全部子孙 span
            if s["parent_id"] in ids:
                ids.add(s["span_id"])
        for name in STAGES:
            d = sum(s["duration_ms"] for s in ss if s["name"] == name and s["span_id"] in ids)
            stage[name].append(d)
        for s in ss:
            if s["span_id"] in ids and s["name"].startswith("chat "):
                model[s["name"]].append(s["duration_ms"])

    print(f"# 链路分析：{path.name}\n\n共 {len(spans)} 个 span，{len(e2e)} 笔完整链路（提交 → 审核完成）\n")
    head = "| 环节 | 次数 | P50 ms | P95 ms | P99 ms | 平均 ms |\n|---|---|---|---|---|---|"
    print("## 端到端\n\n" + head)
    print(row("提交接口（员工等待）", api))
    print(row("排队（投递 → worker 开始）", queue))
    print(row("worker 处理", work))
    print(row("**端到端（提交 → 审核完成）**", e2e))
    names = {"pdf_text": "文字版 PDF", "vlm": "拍照件（视觉模型）", "failed": "识别失败", "none": "没附发票"}
    for tier, xs in sorted(e2e_by_tier.items(), key=lambda kv: -len(kv[1])):
        print(row(f"　端到端 · {names.get(tier, tier)}", xs))

    if work:
        total = sum(work)
        print("\n## worker 内部各环节（每笔报销）\n\n" + head.replace("|---|---|---|---|---|---|",
                                                                 "|---|---|---|---|---|---|") )
        for name in STAGES:
            print(row(name, stage[name]))
        print("\n占 worker 总耗时：" + "，".join(
            f"{n} {sum(stage[n]) / total:.0%}" for n in STAGES if sum(stage[n]) / total >= 0.005))
    sub_children: dict[str, list[float]] = defaultdict(list)
    for ss in by_trace.values():
        for sub in (s for s in ss if s["name"] == SUBMIT and s["kind"] == "SERVER"):
            ids = {sub["span_id"]}
            for s in sorted(ss, key=lambda s: s["start"]):
                if s["parent_id"] in ids and s["service"] == sub["service"]:   # 只看接口进程里的
                    ids.add(s["span_id"])
                    key = "SQL" if s["attributes"].get("db.system") else s["name"]
                    sub_children[key].append(s["duration_ms"])
    if sub_children:
        print("\n## 提交接口内部\n\n" + head)
        for n, xs in sorted(sub_children.items(), key=lambda kv: -sum(kv[1])):
            print(row(n, xs))
    if model:
        print("\n## 模型调用\n\n" + head)
        for n, xs in sorted(model.items()):
            print(row(n, xs))

    print("\n## 全部 span\n\n" + head)
    for n, xs in sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[:25]:
        print(row(n, xs))


if __name__ == "__main__":
    main()
