"""从 SCID（真实扫描的交通 / 定额票据，海康威视研究院）里抽一批做识别评测集。

SCID 不随仓库分发（标注许可 CC BY-NC-SA 4.0，仅限非商业用途）。下载方式见
https://davar-lab.github.io/dataset/scid.html ，放到项目根目录的 SCID/ 下：
    SCID/testA/gt.json  ocr.json  testA_images.zip
    SCID/train/...      SCID/testB/...

这个脚本做三件事：
  1. 判断每张票的票种。SCID 的标注里没有票种，按“标了哪些字段 + 票面文字里的关键词”推断；
  2. 把标注归一化成和模型输出同一种写法（金额转数字、日期转 YYYY-MM-DD）。标注里的 “#” 表示票面本身看不清，
     这样的字段不计分；金额写成时间（18:00）之类明显标错的也不计分；
  3. 按票种分层抽样（固定随机种子），把抽中的图片解压出来，写成评测平台能导入的样本文件。

用法：
    .venv\\Scripts\\python scripts\\build_scid_evalset.py                 # 默认 testA，600 张
    .venv\\Scripts\\python -m ramp_agent.evalhub import SCID/eval/testA_600/samples.jsonl --dataset scid-testA --task samples --freeze
    .venv\\Scripts\\python -m ramp_agent.evalhub run extraction --dataset scid-testA --note "真实票据基线"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ramp_agent.extraction.tickets import ticket_amount, ticket_code, ticket_date  # noqa: E402

# 默认每类抽多少张：差旅报销常见的四类为主，定额和过路费各少量
QUOTA = {"taxi": 150, "train": 150, "coach": 120, "flight": 100, "toll": 40, "quota": 40}
FIELD = {"发票代码": "invoice_code", "发票号码": "invoice_no", "日期": "date", "金额": "amount",
         "始发站": "origin", "到达站": "destination", "座位类型": "seat_class", "入口": "entry", "出口": "exit",
         "保险费": "insurance"}
CN_NUM = re.compile(r"[壹贰叁肆伍陆柒捌玖拾佰仟]")


def kind_of(gt: dict, text: str) -> str:
    if "保险费" in gt or "行程单" in text or "航空运输" in text:
        return "flight"
    if "座位类型" in gt or "12306" in text or "铁路" in text:
        return "train"
    if "入口" in gt or "出口" in gt or "通行费" in text:
        return "toll"
    if "始发站" in gt or "到达站" in gt:
        return "coach"
    if "定额" in text or CN_NUM.search(str(gt.get("金额", ""))):
        return "quota"
    if "日期" in gt or any(w in text for w in ("上车", "车号", "出租")):
        return "taxi"
    return "other"


def normalize(gt: dict) -> tuple[dict, list[str]]:
    """返回（归一化后的标准答案，不计分的字段及原因）。"""
    out, skipped = {}, []
    for zh, raw in gt.items():
        name = FIELD.get(zh)
        raw = str(raw).strip()
        if name is None:
            continue
        if not raw or "#" in raw or raw.upper() == "XXX":
            skipped.append(f"{name}：票面看不清（{raw or '空'}）")
            continue
        if name in ("amount", "insurance"):
            v = ticket_amount(raw)
        elif name == "date":
            v = ticket_date(raw)
        elif name in ("invoice_code", "invoice_no"):
            v = ticket_code(raw)
        else:
            v = re.sub(r"\s+", "", raw) or None
        if v is None:
            skipped.append(f"{name}：标注无法解析（{raw}）")
        else:
            out[name] = v
    return out, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA", choices=["train", "testA", "testB"])
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--scale", type=float, default=1.0, help="每类张数的倍数，如 0.1 只抽 60 张试跑")
    a = ap.parse_args()

    src = ROOT / "SCID" / a.split
    gts = json.loads((src / "gt.json").read_text(encoding="utf-8"))
    ocr = json.loads((src / "ocr.json").read_text(encoding="utf-8"))
    archive = next((p for p in src.glob("*.zip")), None)
    if archive is None:
        sys.exit(f"{src} 里没有 .zip 图片包。testB 是 .rar，请先用 7-Zip 解压后重新打成 zip，或改用 testA")

    by_kind: dict[str, list[str]] = defaultdict(list)
    for key in sorted(gts):
        text = "".join(ocr.get(key, {}).get("content_ann", {}).get("texts", []))
        by_kind[kind_of(gts[key], text)].append(key)
    print("全量票种分布（推断）：", dict(Counter({k: len(v) for k, v in by_kind.items()})))

    rng = random.Random(a.seed)
    quota = {k: max(1, round(n * a.scale)) for k, n in QUOTA.items()}
    total = sum(quota.values())
    out_dir = ROOT / "SCID" / "eval" / f"{a.split}_{total}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    samples, skipped_n = [], Counter()
    with zipfile.ZipFile(archive) as z:
        members = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
        for kind, n in quota.items():
            picked = 0
            for key in rng.sample(by_kind[kind], len(by_kind[kind])):
                if picked >= n:
                    break
                gt, skipped = normalize(gts[key])
                if not gt or key not in members:          # 没有任何可计分字段的票不要
                    continue
                (out_dir / "images" / key).write_bytes(z.read(members[key]))
                samples.append({"id": Path(key).stem, "path": f"SCID/eval/{out_dir.name}/images/{key}",
                                "kind": kind, "schema": "ticket", "gt": gt, "gt_raw": gts[key],
                                "skipped": skipped})
                skipped_n.update(s.split("：")[1].split("（")[0] for s in skipped)
                picked += 1
    path = out_dir / "samples.jsonl"
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    print(f"抽样 {len(samples)} 张 → {path.relative_to(ROOT)}")
    print("各类张数：", dict(Counter(s["kind"] for s in samples)))
    print("可计分字段数：", dict(Counter(f for s in samples for f in s["gt"])))
    print("不计分的字段：", dict(skipped_n))


if __name__ == "__main__":
    main()
