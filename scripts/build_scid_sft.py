"""把 SCID 训练集转成微调数据（ms-swift 的 messages + images 格式）。

训练目标是一个 JSON：doc_type + 这类票据的全部字段（extraction/tickets.py 的 TYPE_FIELDS）。
和评测用同一套归一化，所以模型学到的输出格式就是评测比较的格式。

几条规则，都是为了不把错的东西教给模型：
  - 标注里 “#” 表示票面看不清 → 目标填 null。这是这次微调最想教会的事：看不清就不填，不要编；
  - 机票保险费印 XXX → 0（行程单上 XXX 就是没买保险；testA 也这样标。上一轮验证集的教训）；
  - 一类票据该有的字段，标注里缺了任何一个，整张不要：分不清是“票面没有”还是“标注者漏标”，
    当成 null 教给模型，会让它学会把读得出的字段也留空；
  - 标注本身无法解析的（金额写成 18:00、日期年份不合理），整张不要；
  - 调试集用过的 300 张（SCID/eval/train_300）不进训练集；
  - 定额发票有 8000 多张、和差旅关系不大，最多取 1500 张，免得模型偏向它。

图片按推理时同样的方式缩小（长边 1600、JPEG 85），训练和推理看到的是同样大小的图。

用法：
    .venv\\Scripts\\python scripts\\build_scid_sft.py
输出 SCID/sft/：train.jsonl、val.jsonl、images/、stats.json，以及打包好用来上传的 SCID/sft.zip
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from PIL import Image  # noqa: E402

from build_scid_evalset import FIELD, kind_of  # noqa: E402
from ramp_agent.extraction.tickets import (  # noqa: E402
    PROMPT_FT,
    TYPE_FIELDS,
    ticket_amount,
    ticket_code,
    ticket_date,
    toll_station,
)
from ramp_agent.extraction.vlm import MAX_SIDE  # noqa: E402

ZH = {v: k for k, v in FIELD.items()}            # amount → 金额
CAP = {"quota": 1500}


def target(kind: str, gt: dict) -> tuple[dict | None, str]:
    """标注 → 训练目标。返回 (目标, 不要这张票的原因)。"""
    out: dict = {"doc_type": kind}
    for name in TYPE_FIELDS[kind]:
        if ZH[name] not in gt:
            return None, f"缺字段 {name}"
        raw = str(gt[ZH[name]]).strip()
        if name == "insurance" and raw.upper() == "XXX":
            out[name] = 0.0
            continue
        if not raw or "#" in raw or raw.upper() == "XXX":
            out[name] = None                     # 票面看不清：教模型不填
            continue
        if name in ("amount", "insurance"):
            v = ticket_amount(raw)
        elif name == "date":
            v = ticket_date(raw)
            if v and not ("2010-01-01" <= v <= "2026-12-31"):
                v = None
        elif name in ("invoice_code", "invoice_no"):
            v = ticket_code(raw)
        elif name in ("entry", "exit"):
            v = toll_station(raw)
        else:
            v = "".join(raw.split()) or None
        if v is None:
            return None, f"标注无法解析 {name}"
        out[name] = v
    return out, ""


def shrink(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 张试跑（检查格式用）")
    ap.add_argument("--no-zip", action="store_true", help="不打包 SCID/sft.zip")
    a = ap.parse_args()

    src = ROOT / "SCID" / "train"
    gts = json.loads((src / "gt.json").read_text(encoding="utf-8"))
    ocr = json.loads((src / "ocr.json").read_text(encoding="utf-8"))
    dev = ROOT / "SCID" / "eval" / "train_300" / "samples.jsonl"
    used = {json.loads(x)["id"] for x in dev.read_text(encoding="utf-8").splitlines()} if dev.exists() else set()

    pool: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    dropped: Counter = Counter()
    for key in sorted(gts):
        if Path(key).stem in used:
            dropped["调试集用过"] += 1
            continue
        kind = kind_of(gts[key], "".join(ocr.get(key, {}).get("content_ann", {}).get("texts", [])))
        if kind not in TYPE_FIELDS:
            dropped["票种不明"] += 1
            continue
        t, why = target(kind, gts[key])
        if t is None:
            dropped[why.split(" ")[0]] += 1
            continue
        pool[kind].append((key, t))

    rng = random.Random(a.seed)
    rows: list[tuple[str, dict]] = []
    for kind, items in sorted(pool.items()):
        rng.shuffle(items)
        if kind in CAP and len(items) > CAP[kind]:
            dropped["定额发票超出上限"] += len(items) - CAP[kind]
            items = items[:CAP[kind]]
        rows += items
    rng.shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    val_n = min(a.val, len(rows) // 10)
    splits = {"val": rows[:val_n], "train": rows[val_n:]}

    out = ROOT / "SCID" / "sft"
    (out / "images").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src / "Train_image.zip") as z:
        members = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
        for split, items in splits.items():
            lines = []
            for key, t in items:
                img = out / "images" / key
                if not img.exists():
                    img.write_bytes(shrink(z.read(members[key])))
                lines.append(json.dumps({
                    "messages": [{"role": "user", "content": "<image>" + PROMPT_FT},
                                 {"role": "assistant", "content": json.dumps(t, ensure_ascii=False)}],
                    "images": [f"images/{key}"]}, ensure_ascii=False))
            (out / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    nulls = Counter(f for _, t in rows for f, v in t.items() if v is None)
    stats = {"train": len(splits["train"]), "val": len(splits["val"]),
             "by_kind": dict(Counter(t["doc_type"] for _, t in rows)),
             "null_targets": dict(nulls), "dropped": dict(dropped), "seed": a.seed}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))

    if not a.no_zip:
        pack = ROOT / "SCID" / "sft.zip"
        with zipfile.ZipFile(pack, "w", zipfile.ZIP_STORED) as z:     # JPEG 已经压缩过，不再压
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, Path("sft") / p.relative_to(out))
        print(f"已打包 {pack.relative_to(ROOT)}（{pack.stat().st_size / 2**20:.0f} MB），上传到训练机器")


if __name__ == "__main__":
    main()
