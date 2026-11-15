"""交通票据识别：出租车发票、火车票、汽车客票、机票行程单、过路费发票、定额发票。

增值税发票有统一的版式和字段（schema.InvoiceFields）；这几类票据各有各的版式，字段也少：
没有购买方抬头和税号，没有明细，只有号码、日期、金额、起讫站这些。所以单独一套字段和提示词。

用真实扫描件评测（海康威视 SCID 数据集，见 scripts/build_scid_evalset.py）。真实票据的难点和合成数据不同：
热敏纸褪色、盖章压字、金额写成大写（壹拾元）、日期只印两位年份。读不清的字段不猜，留空并标记，
和增值税发票一样的原则：宁可转人工，不要悄悄读错。
"""

from __future__ import annotations

import json
import re
import time
from datetime import date

from pydantic import BaseModel, Field

from ..llm import parse_json_object
from .normalize import parse_amount

TYPES = {"taxi": "出租车发票", "train": "火车票", "coach": "汽车客票", "flight": "机票行程单",
         "toll": "过路费发票", "quota": "定额发票"}

# 每类票据应该读出来的字段；读不出来就标记为可疑（转人工）
REQUIRED = {
    "taxi": ["invoice_code", "invoice_no", "date", "amount"],
    "train": ["origin", "destination", "date", "amount", "seat_class"],
    "coach": ["origin", "destination", "date", "amount"],
    "flight": ["origin", "destination", "date", "amount"],
    "toll": ["invoice_code", "invoice_no", "amount"],
    "quota": ["invoice_code", "invoice_no", "amount"],
}

PROMPT = """你是票据信息抽取器。图片是一张中国的交通或定额票据的扫描件，只输出一个 JSON 对象，不要输出任何解释。
字段：
- doc_type: 票据类型，取值 taxi（出租车发票）/ train（火车票）/ coach（汽车客票）/ flight（航空运输电子客票行程单）/ toll（过路费、通行费发票）/ quota（定额发票）/ other
- invoice_code: 发票代码（票面“发票代码”后的那串，通常 12 位）
- invoice_no: 发票号码（票面“发票号码”后的那串，通常 8 位）
- date: 票面上的乘车 / 乘机 / 开票日期，格式 YYYY-MM-DD
- amount: 金额，数字，单位元。定额发票的金额是大写（如“壹拾元”），请换算成数字；机票行程单取合计金额
- origin: 始发站 / 出发地（火车票、汽车票、机票）
- destination: 到达站 / 目的地（火车票、汽车票、机票）
- seat_class: 座位类型（火车票，如 二等座、新空调硬座）
- entry: 入口收费站（过路费发票）
- exit: 出口收费站（过路费发票）
- insurance: 保险费，数字（机票行程单）
票面上没有的字段填 null。字迹褪色、被印章盖住、看不清的字段也填 null，不要猜测，不要根据常识补全。"""


class TicketFields(BaseModel):
    doc_type: str | None = None
    invoice_code: str | None = None
    invoice_no: str | None = None
    date: str | None = Field(default=None, description="YYYY-MM-DD")
    amount: float | None = None
    origin: str | None = None
    destination: str | None = None
    seat_class: str | None = None
    entry: str | None = None
    exit: str | None = None
    insurance: float | None = None


class TicketResult(BaseModel):
    fields: TicketFields
    uncertain_fields: list[str] = Field(default_factory=list)
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def needs_review(self) -> bool:
        return bool(self.error) or bool(self.uncertain_fields)


# ---------------- 归一化（模型输出和数据集标注都用这一套，保证比较的是同一种写法） ----------------
_CN_DIGIT = {"零": 0, "〇": 0, "壹": 1, "一": 1, "贰": 2, "二": 2, "两": 2, "叁": 3, "三": 3, "肆": 4, "四": 4,
             "伍": 5, "五": 5, "陆": 6, "六": 6, "柒": 7, "七": 7, "捌": 8, "八": 8, "玖": 9, "九": 9}
_CN_UNIT = {"拾": 10, "十": 10, "佰": 100, "百": 100, "仟": 1000, "千": 1000}


def parse_cn_amount(text: str) -> float | None:
    """'壹拾元' → 10.0，'伍元整' → 5.0，'壹佰元整' → 100.0，'贰拾伍元伍角' → 25.5。不是大写金额返回 None。"""
    s = re.sub(r"[\s整正圆]", "", text).replace("圓", "元")
    if not s or not all(ch in _CN_DIGIT or ch in _CN_UNIT or ch in "元角分万萬" for ch in s):
        return None
    total, section, num = 0.0, 0, 0
    seen = False
    for ch in s:
        if ch in _CN_DIGIT:
            num, seen = _CN_DIGIT[ch], True
        elif ch in _CN_UNIT:
            section += (num or 1) * _CN_UNIT[ch]
            num, seen = 0, True
        elif ch in "万萬":
            total += (section + num) * 10000
            section = num = 0
        elif ch == "元":
            total += section + num
            section = num = 0
        elif ch == "角":
            total += num * 0.1
            num = 0
        elif ch == "分":
            total += num * 0.01
            num = 0
    total += section + num                      # 没写“元”的，如“拾”
    return round(total, 2) if seen else None


def ticket_amount(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).strip()
    cn = parse_cn_amount(text.lstrip("￥¥"))
    if cn is not None:
        return cn
    if ":" in text or "：" in text:            # 时间（18:00）不是金额
        return None
    return parse_amount(text)


def ticket_date(value: object) -> str | None:
    """2021年03月05日 / 2021-3-5 / 2021.3.5 / 2021·3·5 / 20210305 / 21-03-05 → 2021-03-05。缺年份等读不全的返回 None。"""
    if not value:
        return None
    s = str(value).strip()
    m = (re.search(r"(?<!\d)(\d{4})\s*[-/.·年]\s*(\d{1,2})\s*[-/.·月]\s*(\d{1,2})", s)
         or re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s))
    y2 = None if m else re.fullmatch(r"(\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        if y2:
            return date(2000 + int(y2[1]), int(y2[2]), int(y2[3])).isoformat()
    except ValueError:
        return None
    return None


def ticket_code(value: object) -> str | None:
    if value is None:
        return None
    s = re.sub(r"[^0-9A-Za-z]", "", str(value)).upper()
    return s or None


def station(value: object) -> str | None:
    """站名：去空格、连字符和括号里的备注，去掉结尾的“站”和航站楼前缀（T2）。“遵义-茅台”和“遵义茅台”是同一个。"""
    if value is None:
        return None
    s = re.sub(r"[\s\-—－·]+", "", str(value))
    s = re.sub(r"[（(].*?[)）]", "", s).replace("(", "").replace(")", "")
    s = re.sub(r"^T\d", "", s)
    s = re.sub(r"站$", "", s)
    return s or None


def text(value: object) -> str | None:
    if value is None:
        return None
    s = re.sub(r"\s+", "", str(value))
    return s or None


def to_ticket(raw: dict) -> TicketFields:
    t = text(raw.get("doc_type"))
    return TicketFields(
        doc_type=t if t in TYPES else ("other" if t else None),
        invoice_code=ticket_code(raw.get("invoice_code")), invoice_no=ticket_code(raw.get("invoice_no")),
        date=ticket_date(raw.get("date")), amount=ticket_amount(raw.get("amount")),
        origin=text(raw.get("origin")), destination=text(raw.get("destination")),
        seat_class=text(raw.get("seat_class")), entry=text(raw.get("entry")), exit=text(raw.get("exit")),
        insurance=ticket_amount(raw.get("insurance")),
    )


def check(f: TicketFields) -> list[str]:
    """该有却没读出来的字段，以及明显不合理的值。标记后转人工。"""
    bad = [k for k in REQUIRED.get(f.doc_type or "", ["amount"]) if getattr(f, k) in (None, "")]
    if f.doc_type not in TYPES:
        bad.append("doc_type")
    if f.amount is not None and not (0 < f.amount < 100000):
        bad.append("amount")
    if f.date and not ("2000-01-01" <= f.date <= date.today().isoformat()):
        bad.append("date")
    return sorted(set(bad))


def extract_ticket(vlm, image: bytes) -> TicketResult:
    """vlm：有 ask(image, prompt) -> (文本, 输入 tokens, 输出 tokens) 的视觉模型（OpenAICompatibleVLM）。"""
    t0 = time.perf_counter()
    calls = pt = ct = 0
    try:
        out, a, b = vlm.ask(image, PROMPT)
        calls, pt, ct = 1, a, b
        try:
            raw = parse_json_object(out)
        except (ValueError, json.JSONDecodeError):
            out, a, b = vlm.ask(image, PROMPT + "\n上一次输出不是合法 JSON，请只输出 JSON 对象。")
            calls, pt, ct = 2, pt + a, ct + b
            raw = parse_json_object(out)
        fields = to_ticket(raw)
        res = TicketResult(fields=fields, uncertain_fields=check(fields))
    except Exception as exc:  # noqa: BLE001  模型或解析失败：整张标记为失败，转人工
        res = TicketResult(fields=TicketFields(), error=f"{type(exc).__name__}: {exc}"[:300])
    res.model_calls, res.prompt_tokens, res.completion_tokens = calls, pt, ct
    res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res
