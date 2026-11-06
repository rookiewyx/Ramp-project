"""票据识别评测：字段级准确率、分级占比、模型调用次数、耗时，以及最关键的“静默错误率”。

静默错误 = 识别结果有字段读错，但系统没有把它标成 uncertain。
这类错误会直接流进制度审核，导致误判，是识别模块最需要压低的指标；
读错但被标记出来的，只会多一次人工复核，代价小得多。

数据集：
  synthetic  合成报销数据里的发票（文字版 PDF + 拍照 JPG），全部字段有标准答案
  cn         Invoice Checkmark 中文子集（200 张图片，标注是部分字段）
  cord       CORD v2 测试集小票（只比较总金额）

用法：
  .venv\\Scripts\\python scripts\\eval_extraction.py                    # 合成数据，有视觉模型配置就一起评
  .venv\\Scripts\\python scripts\\eval_extraction.py --no-vlm           # 只评文字版 PDF（不需要任何密钥）
  .venv\\Scripts\\python scripts\\eval_extraction.py --force-vlm        # 全部走视觉模型，作为成本对照组
  .venv\\Scripts\\python scripts\\eval_extraction.py --dataset cn --limit 50
  .venv\\Scripts\\python scripts\\eval_extraction.py --rescore reports\\xxx.jsonl
      用最新的校验和打分逻辑重新评一份旧结果，不再调用模型（改了校验规则后用，省钱）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import openai
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ramp_agent.extraction import (
    ExtractionResult,
    InvoiceExtractor,
    InvoiceFields,
    OpenAICompatibleVLM,
    Tier,
)
from ramp_agent.extraction.normalize import clean_invoice_no, clean_text, parse_amount, parse_date

REPORTS = ROOT / "reports"
from ramp_agent.evalhub.adapters.extraction import FIELDS, field_equal

# ---------------- 载入数据：统一成 {id, path, gt: {字段: 值}} ----------------
SYNTHETIC_DIR = ROOT / "data/synthetic"


def load_synthetic(limit: int | None) -> list[dict]:
    seen, samples = set(), []
    lines = (SYNTHETIC_DIR / "expenses.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        inv = json.loads(line)["invoice"]
        if not inv["file"] or inv["file"] in seen:      # 缺附件的不算识别样本；重复发票只评一次
            continue
        seen.add(inv["file"])
        gt = {k: inv[k] for k in FIELDS}
        gt["items"] = sorted((i["name"], round(i["amount"], 2)) for i in inv["items"])
        samples.append({"id": inv["invoice_no"], "path": str(ROOT / inv["file"]),
                        "kind": inv["file_type"], "gt": gt})
    return samples[:limit] if limit else samples


CN_FIELD_MAP = {
    "Unique Invoice Identifier": ("invoice_no", clean_invoice_no),
    "Invoice Issuance Date": ("issue_date", parse_date),
    "Customer/Buyer Name": ("buyer_name", clean_text),
    "Final Grand Total Payable": ("total", parse_amount),
}


def load_cn(limit: int | None) -> list[dict]:
    raw = ROOT / "data/raw/invoice-checkmark/Chinese.zip"
    out = ROOT / "data/processed/invoice-cn"
    if not (out / "label.txt").exists():
        if not raw.exists():
            sys.exit("没找到 Chinese.zip，请先运行 scripts/download_datasets.py")
        with zipfile.ZipFile(raw) as z:
            z.extractall(out)
    samples = []
    for line in (out / "label.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gt = {}
        for item in rec["data"]:
            if item["field"] in CN_FIELD_MAP:
                key, fn = CN_FIELD_MAP[item["field"]]
                gt[key] = fn(item["text"])
        if gt:     # 这个数据集每张只标了部分字段，只评有标注的
            samples.append({"id": rec["file_name"], "path": str(out / rec["file_name"]),
                            "kind": "jpg", "gt": gt})
    return samples[:limit] if limit else samples


def load_cord(limit: int | None) -> list[dict]:
    samples = []
    for line in (ROOT / "data/processed/cord-v2/test.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        total = (rec["fields"].get("total") or {}).get("total_price")
        if isinstance(total, str):
            samples.append({"id": rec["id"], "path": str(ROOT / rec["image"]), "kind": "jpg",
                            "gt": {"total": total}})
    return samples[:limit] if limit else samples


def connect_vlm() -> OpenAICompatibleVLM | None:
    """先探测一下视觉模型服务，连不上就退回只评文字版，避免每张图都等超时重试。"""
    vlm = OpenAICompatibleVLM()
    try:
        vlm.client.with_options(timeout=10, max_retries=0).models.list()
    except openai.AuthenticationError:
        sys.exit("[eval] 视觉模型的 API Key 无效，请检查 .env 里的 VLM_API_KEY")
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        print(f"[eval] 视觉模型 {vlm.client.base_url} 连不上（{type(exc).__name__}），"
              "本次只评文字版 PDF；扫描件记为 failed")
        return None
    except openai.APIStatusError:
        pass      # 有的云服务不提供 /models 接口（返回 404 等），能连上就算可用
    return vlm


def load_previous(path: Path, extractor: InvoiceExtractor) -> tuple[list[dict], list[ExtractionResult]]:
    """读取旧的逐张结果，用当前的校验逻辑重新判定是否需要复核（模型输出保持不变）。"""
    samples, results = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gt = row["gt"]
        if "items" in gt:
            gt["items"] = sorted(tuple(x) for x in gt["items"])
        tier = Tier(row["tier"])
        res = ExtractionResult(fields=InvoiceFields(**row["pred"]), tier=tier, error=row.get("error"),
                               latency_ms=row["latency_ms"],
                               model_calls=row.get("model_calls", int(tier == Tier.VLM)),
                               prompt_tokens=row.get("tokens", 0))
        if tier != Tier.FAILED:
            res.checks_failed, res.uncertain_fields = extractor.revalidate(res.fields)
        samples.append({"id": row["id"], "kind": row["kind"], "path": None, "gt": gt})
        results.append(res)
    return samples, results


# ---------------- 主流程 ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="票据识别评测")
    ap.add_argument("--dataset", choices=["synthetic", "cn", "cord"], default="synthetic")
    ap.add_argument("--limit", type=int, help="只评前 N 张")
    ap.add_argument("--no-vlm", action="store_true", help="不调用视觉模型")
    ap.add_argument("--force-vlm", action="store_true", help="全部走视觉模型（成本对照组）")
    ap.add_argument("--workers", type=int, default=4, help="并发数（调视觉模型时生效）")
    ap.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"),
                    help="合成数据目录，例如独立测试集 data/synthetic_test")
    ap.add_argument("--rescore", type=Path, help="重新评分一份旧的逐张结果 .jsonl，不调用模型")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    global SYNTHETIC_DIR
    SYNTHETIC_DIR = args.synthetic_dir if args.synthetic_dir.is_absolute() else ROOT / args.synthetic_dir
    tag = "" if SYNTHETIC_DIR == ROOT / "data/synthetic" else "-" + SYNTHETIC_DIR.name.replace("_", "-")

    if args.rescore:
        # 文件名形如 extraction_{dataset}_{mode}_{时间}.jsonl
        parts = args.rescore.stem.split("_")
        args.dataset, mode = parts[1].split("-")[0], "_".join(parts[2:-2]) + "_rescored"
        tag = parts[1][len(args.dataset):]
        vlm = None
        extractor = InvoiceExtractor(None, strict_cn=(args.dataset == "synthetic"))
        samples, results = load_previous(args.rescore, extractor)
        print(f"[eval] 重新评分 {args.rescore.name}：{len(samples)} 张，不调用模型")
        wall = 0.0
    else:
        samples = {"synthetic": load_synthetic, "cn": load_cn, "cord": load_cord}[args.dataset](args.limit)
        vlm = None if args.no_vlm else connect_vlm()
        if args.force_vlm and vlm is None:
            sys.exit("--force-vlm 需要可用的视觉模型，请检查 .env 里的 VLM_* 配置")
        extractor = InvoiceExtractor(vlm, strict_cn=(args.dataset == "synthetic"),
                                     force_vlm=args.force_vlm)
        mode = "force_vlm" if args.force_vlm else "tiered"
        print(f"[eval] {args.dataset}: {len(samples)} 张；视觉模型："
              f"{'关闭' if vlm is None else vlm.model}；模式：{'全部走视觉模型' if args.force_vlm else '分级'}")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda s: extractor.extract(s["path"]), samples))
        wall = time.perf_counter() - t0

    tiers = Counter()
    by_tier = defaultdict(lambda: {"n": 0, "correct_docs": 0, "flagged": 0, "silent_errors": 0,
                                   "latency_ms": 0.0, "model_calls": 0, "tokens": 0})
    field_hits, field_total = Counter(), Counter()
    rows = []
    for s, r in zip(samples, results):
        tier = r.tier.value
        tiers[tier] += 1
        wrong = []
        for name, gt in s["gt"].items():
            ok = field_equal(name, getattr(r.fields, name), gt, args.dataset)
            if tier != "failed":        # 没识别的样本单独统计，不拉低字段准确率
                field_total[name] += 1
                field_hits[name] += ok
            if not ok:
                wrong.append(name)
        flagged = r.needs_review
        b = by_tier[tier]
        b["n"] += 1
        b["correct_docs"] += not wrong
        b["flagged"] += flagged
        b["silent_errors"] += bool(wrong) and not flagged
        b["latency_ms"] += r.latency_ms
        b["model_calls"] += r.model_calls
        b["tokens"] += r.prompt_tokens + r.completion_tokens
        rows.append({"id": s["id"], "kind": s["kind"], "tier": tier, "wrong_fields": wrong,
                     "flagged": flagged, "uncertain": r.uncertain_fields,
                     "checks_failed": r.checks_failed, "error": r.error,
                     "latency_ms": r.latency_ms, "model_calls": r.model_calls,
                     "tokens": r.prompt_tokens + r.completion_tokens, "pred": r.fields.model_dump(),
                     "gt": {k: (v if k != "items" else [list(x) for x in v]) for k, v in s["gt"].items()}})

    # 分母只算真正评到的样本：没有视觉模型时，扫描件会进 failed 档单独列出
    evaluated = [t for t in by_tier if t != "failed"]
    n_eval = sum(by_tier[t]["n"] for t in evaluated)
    summary = {
        "dataset": args.dataset, "mode": mode,
        "vlm_model": None if vlm is None else vlm.model, "samples": len(samples),
        "wall_seconds": round(wall, 1), "tiers": dict(tiers),
        "doc_accuracy": round(sum(by_tier[t]["correct_docs"] for t in evaluated) / n_eval, 4) if n_eval else None,
        "silent_error_rate": round(sum(by_tier[t]["silent_errors"] for t in evaluated) / n_eval, 4) if n_eval else None,
        "field_accuracy": {k: round(field_hits[k] / field_total[k], 4) for k in field_total},
        "by_tier": {t: {**v, "avg_latency_ms": round(v["latency_ms"] / v["n"], 1)}
                    for t, v in by_tier.items()},
        "model_calls": sum(r.model_calls for r in results),
        "model_calls_saved_vs_all_vlm": tiers.get("pdf_text", 0),
    }

    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = args.dataset + (tag if args.dataset == "synthetic" else "")
    base = REPORTS / f"extraction_{dataset_tag}_{summary['mode']}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    base.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    print_report(summary)
    print(f"\n[done] 汇总：{base.with_suffix('.json')}\n       逐张结果（错例分析用）：{base.with_suffix('.jsonl')}")


def print_report(s: dict) -> None:
    print(f"\n用时 {s['wall_seconds']}s，模型调用 {s['model_calls']} 次，"
          f"相比全部走视觉模型少调用 {s['model_calls_saved_vs_all_vlm']} 次")
    if s["doc_accuracy"] is not None:
        print(f"整张全对率 {s['doc_accuracy']:.1%}，静默错误率 {s['silent_error_rate']:.1%}")
    print("\n| 路径 | 张数 | 整张全对 | 被标记复核 | 静默错误 | 平均耗时 ms | 模型调用 | tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for t, v in s["by_tier"].items():
        print(f"| {t} | {v['n']} | {v['correct_docs']} | {v['flagged']} | {v['silent_errors']} | "
              f"{v['avg_latency_ms']} | {v['model_calls']} | {v['tokens']} |")
    if "failed" in s["by_tier"]:
        print("（failed 多为未配置视觉模型的扫描件，不计入准确率）")
    print("\n| 字段 | 准确率 |\n|---|---|")
    for k, v in s["field_accuracy"].items():
        print(f"| {k} | {v:.1%} |")


if __name__ == "__main__":
    main()
