"""审核任务适配器：报销记录 → Claim → 审核结论，以及审核任务的指标。

scripts/eval_policy.py 和评测平台共用这里的逻辑，保证两边算出来的数字一致。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ...extraction import InvoiceExtractor, InvoiceFields, LineItem
from ...policy import Claim, PolicyReviewer, Review, SemanticJudge
from ..stats import wilson

DECISIONS = ("approve", "review", "reject")
CLAIM_KEYS = ["expense_id", "report_id", "employee_id", "category", "amount", "submit_date", "note",
              "city", "trip_start", "trip_end", "nights", "headcount", "preapproval_id"]


def invoice_from(d: dict) -> InvoiceFields:
    return InvoiceFields(**{**{k: d.get(k) for k in InvoiceFields.model_fields if k != "items"},
                            "items": [LineItem(**i) for i in d.get("items") or []]})


def claim_from_record(record: dict, extraction: dict | None = None,
                      validator: InvoiceExtractor | None = None) -> Claim:
    """extraction=None：发票字段用记录里的标准答案；否则用这份识别结果（含 pred / tier）。"""
    inv = record["invoice"]
    c = Claim(**{k: record[k] for k in CLAIM_KEYS if record.get(k) is not None},
              has_attachment=inv.get("file") is not None)
    if not c.has_attachment:
        return c
    if extraction is None:
        c.invoice = invoice_from(inv)
        return c
    c.invoice = invoice_from(extraction["pred"])
    c.extraction_failed = extraction["tier"] == "failed"
    if not c.extraction_failed:
        _, c.uncertain_fields = (validator or InvoiceExtractor(None)).revalidate(c.invoice)
    return c


def review_output(rv: Review) -> dict:
    return {"decision": rv.decision.value, "violated": rv.violated_rules,
            "uncertain": rv.uncertain_rules, "explanation": rv.explanation,
            "llm_calls": rv.llm_calls, "tokens": rv.prompt_tokens + rv.completion_tokens,
            "results": [r.model_dump(exclude={"clause"}) for r in rv.results]}


def run_policy(records: list[dict], *, judge: SemanticJudge | None,
               extraction_lookup: Callable[[str], dict | None] | None = None,
               workers: int = 4) -> list[dict]:
    """按顺序审核一批报销（查重依赖顺序）。大模型调用先并发做完，规则部分再逐笔执行。"""
    validator = InvoiceExtractor(None)
    claims = []
    for r in records:
        ext = None
        if extraction_lookup is not None and r["invoice"].get("file") is not None:
            ext = extraction_lookup(r["invoice"]["invoice_no"])
            if ext is None:
                raise KeyError(f"识别结果里没有发票 {r['invoice']['invoice_no']}")
        claims.append(claim_from_record(r, ext, validator))
    reviewer = PolicyReviewer(judge=judge)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sems = list(pool.map(reviewer.semantic, claims))
    return [review_output(reviewer.review(c, s)) for c, s in zip(claims, sems)]


def policy_metrics(items: list[dict]) -> dict:
    """items: 每条 {gold, gold_rules, predicted, violated, tags, llm_calls, tokens}；gold 为 None 的不计分。"""
    scored = [it for it in items if it["gold"] is not None]
    confusion: dict[str, Counter] = defaultdict(Counter)
    rule_stats: dict[str, Counter] = defaultdict(Counter)
    scen: dict[str, Counter] = defaultdict(Counter)
    for it in scored:
        g, p = it["gold"], it["predicted"]
        confusion[g][p] += 1
        key = it.get("tags", {}).get("family") or it.get("tags", {}).get("scenario") or "-"
        scen[key]["n"] += 1
        scen[key]["ok"] += g == p
        exp_rules = set(it.get("gold_rules") or [])
        got = set(it.get("violated") or []) - {"AMT"}            # AMT 是系统校验项，标签里没有
        for rule in exp_rules | got:
            rule_stats[rule]["tp" if rule in exp_rules and rule in got else
                             "fn" if rule in exp_rules else "fp"] += 1
    n = len(scored)
    correct = sum(confusion[d][d] for d in DECISIONS)
    exp_approve = sum(confusion["approve"].values())
    lo, hi = wilson(correct, n)
    return {
        "scored": n, "unscored": len(items) - n,
        "accuracy": round(correct / n, 4) if n else None,
        "accuracy_ci95": [round(lo, 4), round(hi, 4)] if n else None,
        "false_approve": sum(confusion[e]["approve"] for e in ("review", "reject")),
        "false_reject": sum(confusion[e]["reject"] for e in ("approve", "review")),
        "extra_review": confusion["approve"]["review"],
        "auto_approve_rate": round(confusion["approve"]["approve"] / exp_approve, 4) if exp_approve else None,
        "human_review_share": round(sum(confusion[e]["review"] for e in DECISIONS) / n, 4) if n else None,
        "confusion": {e: {p: confusion[e][p] for p in DECISIONS} for e in DECISIONS},
        "rules": {r: {"recall": round(s["tp"] / (s["tp"] + s["fn"]), 3) if s["tp"] + s["fn"] else None,
                      "precision": round(s["tp"] / (s["tp"] + s["fp"]), 3) if s["tp"] + s["fp"] else None,
                      "tp": s["tp"], "fn": s["fn"], "fp": s["fp"]} for r, s in sorted(rule_stats.items())},
        "groups": {k: {"n": v["n"], "accuracy": round(v["ok"] / v["n"], 3)} for k, v in sorted(scen.items())},
        "llm_calls": sum(it.get("llm_calls", 0) for it in items),
        "tokens": sum(it.get("tokens", 0) for it in items),
    }
