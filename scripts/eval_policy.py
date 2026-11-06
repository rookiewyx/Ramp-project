"""制度审核评测：结论准确率、混淆矩阵、误通过 / 误驳回、条款级召回、大模型调用量。

这是不经过评测平台的“快速版”：直接读 jsonl，结果写到 reports/。
需要版本管理、人工标注、运行对比和门禁时，用评测平台（python -m ramp_agent.evalhub）。
两边共用 src/ramp_agent/evalhub/adapters/policy.py 里的审核和打分逻辑，算出来的数字一致。

两种输入：
  标准答案模式（默认）：发票字段直接用数据里的标准答案，只评审核逻辑本身。
  端到端模式（--extraction）：发票字段用模块一的识别结果（复用识别评测保存的逐张结果，不再调用视觉模型）。

两类错误代价不同，分开统计：
  误通过：本该转人工或驳回，却建议通过——会让违规报销直接放行，最危险。
  误驳回：本该通过或转人工，却建议驳回——员工被冤枉，体验和信任受损。
  多转人工：本该通过却转了人工——只是多花一点人工，代价最小。

用法：
  .venv\\Scripts\\python scripts\\eval_policy.py --no-llm                    # 只用规则，不需要密钥
  .venv\\Scripts\\python scripts\\eval_policy.py                             # 规则 + 大模型
  .venv\\Scripts\\python scripts\\eval_policy.py --data evalsets/policy_challenge.jsonl   # 语义难例集
  .venv\\Scripts\\python scripts\\eval_policy.py --data data/synthetic_test/expenses.jsonl ^
        --extraction reports\\extraction_synthetic-synthetic-test_tiered_xxx.jsonl     # 端到端
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ramp_agent.evalhub.adapters.policy import DECISIONS, policy_metrics, run_policy
from ramp_agent.evalhub.cli import connect_llm

REPORTS = ROOT / "reports"


def main() -> None:
    ap = argparse.ArgumentParser(description="制度审核评测")
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/expenses.jsonl"))
    ap.add_argument("--extraction", type=Path, help="识别评测的逐张结果 .jsonl（端到端模式）")
    ap.add_argument("--no-llm", action="store_true", help="不调用大模型，只用规则")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    data = args.data if args.data.is_absolute() else ROOT / args.data
    records = [json.loads(x) for x in data.read_text(encoding="utf-8").splitlines() if x.strip()]
    lookup = None
    if args.extraction:
        ext = args.extraction if args.extraction.is_absolute() else ROOT / args.extraction
        table = {}
        for line in ext.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            table[row["id"]] = row
        lookup = table.get
    judge = None if args.no_llm else connect_llm()
    mode = ("e2e" if lookup else "oracle") + ("_rules" if judge is None else "_llm")
    print(f"[eval] {data.relative_to(ROOT)}：{len(records)} 笔；发票字段："
          f"{'识别结果' if lookup else '标准答案'}；大模型：{'关闭' if judge is None else judge.model}")

    t0 = time.perf_counter()
    try:
        outputs = run_policy(records, judge=judge, extraction_lookup=lookup, workers=args.workers)
    except KeyError as exc:
        sys.exit(f"{exc}，请确认 --data 和 --extraction 是同一批数据")
    wall = time.perf_counter() - t0

    items, rows = [], []
    for rec, out in zip(records, outputs):
        exp = rec["expected"]
        exp_rules = sorted({v["rule"] for v in exp["violations"]})
        items.append({"gold": exp["decision"], "gold_rules": exp_rules, "predicted": out["decision"],
                      "violated": out["violated"], "tags": {"scenario": rec["scenario"]},
                      "llm_calls": out["llm_calls"], "tokens": out["tokens"]})
        rows.append({"expense_id": rec["expense_id"], "scenario": rec["scenario"], "expected": exp["decision"],
                     "predicted": out["decision"], "expected_rules": exp_rules, **out})
    m = policy_metrics(items)
    summary = {"data": str(data.relative_to(ROOT)), "mode": mode,
               "llm_model": judge.model if judge else None, "claims": len(records),
               "wall_seconds": round(wall, 1), **m,
               "llm_decided_rules": sum(1 for o in outputs for r in o["results"] if r["by"] == "llm")}

    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPORTS / f"policy_{data.parent.name if data.name == 'expenses.jsonl' else data.stem}_{mode}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    base.with_suffix(".jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print_report(summary, rows)
    print(f"\n[done] 汇总：{base.with_suffix('.json')}\n       逐笔结论（含解释）：{base.with_suffix('.jsonl')}")


def print_report(s: dict, rows: list[dict]) -> None:
    print(f"\n用时 {s['wall_seconds']}s，大模型调用 {s['llm_calls']} 次，tokens {s['tokens']}，"
          f"由大模型判定的条款 {s['llm_decided_rules']} 条")
    lo, hi = s["accuracy_ci95"]
    print(f"结论准确率 {s['accuracy']:.1%}（95% 区间 {lo:.1%}–{hi:.1%}）；误通过 {s['false_approve']}；"
          f"误驳回 {s['false_reject']}；多转人工 {s['extra_review']}")
    if s["auto_approve_rate"] is not None:
        print(f"合规报销自动通过率 {s['auto_approve_rate']:.1%}；转人工占比 {s['human_review_share']:.1%}")
    print("\n混淆矩阵（行=标准答案，列=系统结论）\n| | approve | review | reject |\n|---|---|---|---|")
    for e in DECISIONS:
        c = s["confusion"][e]
        print(f"| {e} | {c['approve']} | {c['review']} | {c['reject']} |")
    print("\n| 条款 | 召回 | 精确 | TP | FN | FP |\n|---|---|---|---|---|---|")
    for r, v in s["rules"].items():
        def fmt(x):
            return "-" if x is None else f"{x:.0%}"
        print(f"| {r} | {fmt(v['recall'])} | {fmt(v['precision'])} | {v['tp']} | {v['fn']} | {v['fp']} |")
    bad = [r for r in rows if r["expected"] != r["predicted"]]
    if bad:
        print(f"\n结论不一致的 {len(bad)} 笔（最多显示 15 笔）：")
        for r in bad[:15]:
            print(f"  {r['expense_id']} [{r['scenario']}] 期望 {r['expected']} 实际 {r['predicted']}；"
                  f"违规 {r['violated']} 不确定 {r['uncertain']}")


if __name__ == "__main__":
    main()
