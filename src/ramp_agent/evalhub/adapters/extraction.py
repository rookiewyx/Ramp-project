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


# ---------------- 交通票据（真实扫描件，SCID） ----------------
def ticket_field_equal(name: str, pred, gt) -> bool:
    """站名类字段放宽：标注有时带航站楼、机场名（“T2拉萨-贡嘎”），模型可能只读出“拉萨”，互相包含就算对。"""
    from ...extraction.tickets import station

    if pred is None:
        return False
    if name in ("amount", "insurance"):
        return abs(float(pred) - float(gt)) < 0.011
    if name in ("origin", "destination", "entry", "exit"):
        a, b = station(pred), station(gt)
        return bool(a and b) and (a == b or (len(a) >= 2 and a in b) or (len(b) >= 2 and b in a))
    return pred == gt


def run_ticket(smp: dict, vlm) -> dict:
    from ...extraction.tickets import TicketFields, TicketResult, extract_ticket

    path = Path(smp["path"])
    path = path if path.is_absolute() else ROOT / path
    if vlm is None or not path.exists():
        r = TicketResult(fields=TicketFields(), error="no_vlm" if vlm is None else "no_file")
    else:
        r = extract_ticket(vlm, path.read_bytes())
    failed = bool(r.error)
    wrong = [] if failed else [k for k, g in smp["gt"].items()
                               if not ticket_field_equal(k, getattr(r.fields, k), g)]
    # 没读出来（留空）的字段下游会当作缺失，转人工；读出了一个错的值才是静默错误
    silent = [k for k in wrong if getattr(r.fields, k) is not None and k not in r.uncertain_fields]
    return {"tier": "failed" if failed else "vlm", "pred": r.fields.model_dump(), "uncertain": r.uncertain_fields,
            "checks_failed": [], "error": r.error, "flagged": r.needs_review, "wrong_fields": wrong,
            "silent_fields": silent, "type_ok": r.fields.doc_type == smp["kind"], "kind": smp["kind"],
            "scored_fields": list(smp["gt"]), "latency_ms": r.latency_ms, "model_calls": r.model_calls,
            "tokens": r.prompt_tokens + r.completion_tokens}


def ticket_metrics(outputs: list[dict]) -> dict:
    """按票种、按字段的准确率。字段准确率的分母是“标注里有、且标注本身可读”的字段。"""
    from collections import defaultdict

    kinds: dict[str, dict] = defaultdict(lambda: {"n": 0, "ok": 0, "silent": 0, "type_ok": 0})
    fields: dict[str, dict] = defaultdict(lambda: {"n": 0, "ok": 0, "silent": 0})
    for o in outputs:
        if o["tier"] == "failed":
            continue
        k = kinds[o["kind"]]
        k["n"] += 1
        k["ok"] += not o["wrong_fields"]
        k["silent"] += bool(o["silent_fields"])
        k["type_ok"] += o["type_ok"]
        for f in o["scored_fields"]:
            fields[f]["n"] += 1
            fields[f]["ok"] += f not in o["wrong_fields"]
            fields[f]["silent"] += f in o["silent_fields"]
    n_fields = sum(v["n"] for v in fields.values())
    return {
        "field_accuracy": round(sum(v["ok"] for v in fields.values()) / n_fields, 4) if n_fields else None,
        "silent_field_rate": round(sum(v["silent"] for v in fields.values()) / n_fields, 4) if n_fields else None,
        "type_accuracy": round(sum(v["type_ok"] for v in kinds.values()) / max(1, sum(v["n"] for v in kinds.values())), 4),
        "groups": {k: {"n": v["n"], "accuracy": round(v["ok"] / v["n"], 4), "silent": v["silent"],
                       "type_accuracy": round(v["type_ok"] / v["n"], 4)} for k, v in sorted(kinds.items())},
        "fields": {f: {"n": v["n"], "accuracy": round(v["ok"] / v["n"], 4), "silent": v["silent"]}
                   for f, v in sorted(fields.items())},
    }


def run_extraction(samples: list[dict], extractor: InvoiceExtractor, workers: int = 4) -> list[dict]:
    def one(smp: dict) -> dict:
        if smp.get("schema") == "ticket":
            return run_ticket(smp, extractor.vlm)
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
    # 静默错误：读错了又没有标记。交通票据按字段算（读错的字段没被标为可疑），增值税发票按整张算
    silent = sum(1 for o in evaluated if (o["silent_fields"] if "silent_fields" in o
                                          else (o["wrong_fields"] and not o["flagged"])))
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
        **(ticket_metrics(outputs) if any("kind" in o for o in outputs) else {}),
    }
