"""识别任务适配器：样本构造、逐字段比较、识别指标。scripts/eval_extraction.py 也使用这里的函数。"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...config import ROOT
from ...extraction import InvoiceExtractor
from ..stats import wilson

FIELDS = ["invoice_no", "issue_date", "seller", "buyer_name", "buyer_tax_id", "items", "total",
          "tax_rate"]


def samples_from_records(records: list[dict]) -> list[dict]:
    """合成报销记录 → 识别样本。缺附件的不算识别样本；同一个发票文件只评一次。"""
    seen, samples = set(), []
    for r in records:
        inv = r["invoice"]
        if not inv["file"] or inv["file"] in seen:
            continue
        seen.add(inv["file"])
        gt = {k: inv[k] for k in FIELDS}
        gt["items"] = sorted([i["name"], round(i["amount"], 2)] for i in inv["items"])
        samples.append({"id": inv["invoice_no"], "path": inv["file"], "kind": inv["file_type"], "gt": gt})
    return samples


def digits(value) -> str:
    """CORD 是印尼小票，'60.000' 表示六万；统一去掉分隔符和小数部分的 0 再比较。"""
    if value is None:
        return ""
    if isinstance(value, float):
        value = f"{value:.2f}"
    s = re.sub(r"[^\d.,]", "", str(value))
    s = re.sub(r"[.,]00$", "", s)
    return re.sub(r"\D", "", s)


def _items_key(items) -> list:
    out = []
    for it in items:
        name, amount = (it.name, it.amount) if hasattr(it, "name") else (it[0], it[1])
        out.append((re.sub(r"\s+", "", name), round(float(amount), 2)))   # 项目名称里的空格不算错
    return sorted(out)


def field_equal(name: str, pred, gt, dataset: str = "synthetic") -> bool:
    if name == "total" and dataset == "cord":
        return digits(pred) == digits(gt) != ""
    if name in ("total", "tax_rate"):
        return pred is not None and gt is not None and abs(pred - gt) < 0.011
    if name == "items":
        return _items_key(pred) == _items_key(gt)
    return pred == gt


def run_extraction(samples: list[dict], extractor: InvoiceExtractor, workers: int = 4) -> list[dict]:
    def one(smp: dict) -> dict:
        path = Path(smp["path"])
        r = extractor.extract(path if path.is_absolute() else ROOT / path)
        wrong = [k for k, g in smp["gt"].items() if not field_equal(k, getattr(r.fields, k), g)]
        return {"tier": r.tier.value, "pred": r.fields.model_dump(), "uncertain": r.uncertain_fields,
                "checks_failed": r.checks_failed, "error": r.error, "flagged": r.needs_review,
                "wrong_fields": wrong, "latency_ms": r.latency_ms, "model_calls": r.model_calls,
                "tokens": r.prompt_tokens + r.completion_tokens}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, samples))


def extraction_metrics(outputs: list[dict]) -> dict:
    evaluated = [o for o in outputs if o["tier"] != "failed"]
    n = len(evaluated)
    correct = sum(1 for o in evaluated if not o["wrong_fields"])
    silent = sum(1 for o in evaluated if o["wrong_fields"] and not o["flagged"])
    lo, hi = wilson(correct, n)
    tiers: dict[str, int] = {}
    for o in outputs:
        tiers[o["tier"]] = tiers.get(o["tier"], 0) + 1
    return {
        "samples": len(outputs), "tiers": tiers,
        "doc_accuracy": round(correct / n, 4) if n else None,
        "doc_accuracy_ci95": [round(lo, 4), round(hi, 4)] if n else None,
        "silent_errors": silent, "silent_error_rate": round(silent / n, 4) if n else None,
        "flagged": sum(1 for o in evaluated if o["flagged"]),
        "model_calls": sum(o["model_calls"] for o in outputs),
        "tokens": sum(o["tokens"] for o in outputs),
        "avg_latency_ms": round(sum(o["latency_ms"] for o in outputs) / len(outputs), 1) if outputs else 0,
    }
