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


# 第二版提示词（2026-11-15）。针对真实票据基线（#36）里的两类静默错误：
#   - 机票行程单有两个日期，第一版没说要哪个，模型常取右下角的填开日期；
#   - 出租车票热敏纸褪色，“金额”一行看不清时，模型会把里程、单价、等候时间当成金额。
# 做法：让模型把容易混的几个数分开填（航班日期 / 填开日期，金额 / 单价 / 里程），
# 再由代码核对金额和“单价 × 里程”是否对得上（check）。在 SCID train 抽样上调，不看 testA。
PROMPT_V2 = """你是票据信息抽取器。图片是一张中国的交通或定额票据的扫描件，只输出一个 JSON 对象，不要输出任何解释。

先判断票据类型，再按类型读取字段：
- doc_type: taxi（出租车发票）/ train（火车票）/ coach（汽车客票）/ flight（航空运输电子客票行程单）/ toll（过路费、通行费发票）/ quota（定额发票）/ other

通用字段：
- invoice_code: 发票代码（“发票代码”后的那串，通常 12 位）
- invoice_no: 发票号码（“发票号码”后的那串，通常 8 位）
- date: 日期，格式 YYYY-MM-DD。出租车票取“日期”一行；火车票、汽车票取乘车日期；机票行程单取航班日期（见下）
- amount: 金额，数字，单位元

出租车发票（热敏小票，字迹常常褪色）：
- amount 只取“金额”这一行的数。不要用单价、里程、等候时间、电调费、燃油附加费来代替；“金额”一行看不清就填 null
- unit_price: “单价”一行的数（元/公里）
- distance_km: “里程”一行的数（公里）

机票行程单（票面上有两个日期，容易混）：
- date: 航班日期，在中间表格“日期 DATE”一栏、和航班号同一行
- issue_date: 填开日期，在右下角“填开日期 DATE OF ISSUE”处。它不是航班日期，不要填到 date 里
- amount: “合计 TOTAL”的数，不是票价 FARE
- insurance: “保险费 INSURANCE”的数；印的是 XXX 或空白就填 null
- origin / destination: “自 FROM”和第一个“至 TO”的城市、机场，照票面写（如 T3杭州-萧山）

火车票、汽车客票：
- origin: 始发站；destination: 到达站。注意方向，不要写反
- seat_class: 座位类型（火车票，如 二等座、新空调硬座）
- 汽车客票的 amount 取票价，不要取座位号、车次、检票口

过路费发票：
- entry: 入口收费站；exit: 出口收费站

定额发票：
- amount: 票面大写金额（如“壹拾元”）换算成的数字

票面上没有的字段填 null。字迹褪色、被印章盖住、看不清的字段也填 null：宁可留空，不要猜，不要用别的行的数字代替，不要根据常识补全（例如年份看不清时不要按今年填）。"""

PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2}


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
    # 下面三个只用来核对，不参与评测计分
    unit_price: float | None = None       # 出租车：单价
    distance_km: float | None = None      # 出租车：里程
    issue_date: str | None = None         # 机票：填开日期


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


def toll_station(value: object) -> str | None:
    """过路费的入口 / 出口：模型常把车型（客1、货2）或车道号（1036）填进来，提示词里说了也没用，在这里去掉。"""
    s = text(value)
    return None if s is None or re.fullmatch(r"[客货]车?\d|\d+", s) else s


def to_ticket(raw: dict) -> TicketFields:
    t = text(raw.get("doc_type"))
    return TicketFields(
        doc_type=t if t in TYPES else ("other" if t else None),
        invoice_code=ticket_code(raw.get("invoice_code")), invoice_no=ticket_code(raw.get("invoice_no")),
        date=ticket_date(raw.get("date")), amount=ticket_amount(raw.get("amount")),
        origin=text(raw.get("origin")), destination=text(raw.get("destination")),
        seat_class=text(raw.get("seat_class")), entry=toll_station(raw.get("entry")),
        exit=toll_station(raw.get("exit")),
        insurance=ticket_amount(raw.get("insurance")),
        unit_price=ticket_amount(raw.get("unit_price")), distance_km=ticket_amount(raw.get("distance_km")),
        issue_date=ticket_date(raw.get("issue_date")),
    )


def taxi_amount_suspicious(f: TicketFields) -> str:
    """出租车金额和单价、里程对不对得上。返回可疑的原因，没问题返回空串。

    车费 ≈ 起步价 + 单价 × 超出起步里程的部分，再加等候费，所以大致在“单价 × 里程”附近：
    短途因为起步价会偏高，长途可能有返空费。这里只抓明显对不上的——金额和里程或单价是同一个数
    （多半是读串了行），或者金额不到“单价 × 里程”的一半。
    """
    a, u, d = f.amount, f.unit_price, f.distance_km
    if a is None:
        return ""
    if d is not None and abs(a - d) < 0.011:
        return "金额和里程是同一个数"
    if u is not None and abs(a - u) < 0.011:
        return "金额和单价是同一个数"
    if u and d and u < 20 and d < 500:                 # 单价、里程本身读得离谱就不拿来核对
        est = u * d
        if est >= 15 and a < 0.5 * est:
            return f"金额 {a} 不到 单价×里程（{est:.1f}）的一半"
        if a > 3 * est + 30:
            return f"金额 {a} 远高于 单价×里程（{est:.1f}）"
    return ""


def check(f: TicketFields) -> list[str]:
    """该有却没读出来的字段，以及明显不合理的值。标记后转人工。"""
    bad = [k for k in REQUIRED.get(f.doc_type or "", ["amount"]) if getattr(f, k) in (None, "")]
    if f.doc_type not in TYPES:
        bad.append("doc_type")
    if f.amount is not None and not (0 < f.amount < 100000):
        bad.append("amount")
    if f.date and not ("2000-01-01" <= f.date <= date.today().isoformat()):
        bad.append("date")
    if f.doc_type == "taxi" and taxi_amount_suspicious(f):
        bad.append("amount")
    return sorted(set(bad))


def enhance(image: bytes) -> bytes:
    """把褪色的热敏小票变清楚一些：转灰度、拉开对比度、把浅灰的字压深。只用于第二遍复读。"""
    import io

    from PIL import Image, ImageOps

    img = ImageOps.autocontrast(Image.open(io.BytesIO(image)).convert("L"), cutoff=1)
    img = img.point(lambda p: int(255 * (p / 255) ** 2.2))          # 浅灰 → 深灰，白底基本不变
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# 第三版（v3）= v2 的提示词 + 出租车票读两遍。
# v2 在调试集上把机票的日期问题解决了，但出租车票没有改善：褪色的金额、日期模型照样会“读”出一个数，
# 连用来核对的单价、里程也是编的，提示词管不住。所以换一个思路——不指望读对，而是要知道哪些没读准：
# 原图读一遍、增强对比度后再读一遍，两遍对不上的字段标为可疑、转人工。只对出租车票多调一次模型。
DOUBLE_READ = {"taxi": ["amount", "date", "invoice_code", "invoice_no"]}

PROMPT_V3 = PROMPT_V2.replace(
    "- entry: 入口收费站；exit: 出口收费站",
    "- entry: 入口收费站的站名；exit: 出口收费站的站名。“客1”“货2”这类是车型，不是站名；票面上没有印站名就填 null")
PROMPTS["v3"] = PROMPT_V3


def _ask(vlm, image: bytes, prompt: str) -> tuple[dict, int, int, int]:
    out, pt, ct = vlm.ask(image, prompt)
    calls = 1
    try:
        raw = parse_json_object(out)
    except (ValueError, json.JSONDecodeError):
        out, a, b = vlm.ask(image, prompt + "\n上一次输出不是合法 JSON，请只输出 JSON 对象。")
        calls, pt, ct = 2, pt + a, ct + b
        raw = parse_json_object(out)
    return raw, calls, pt, ct


def extract_ticket(vlm, image: bytes, version: str | None = None) -> TicketResult:
    """vlm：有 ask(image, prompt) -> (文本, 输入 tokens, 输出 tokens) 的视觉模型（OpenAICompatibleVLM）。
    version：提示词版本，默认取配置 TICKET_PROMPT（v1 最初的基线，v2 分开两个日期 + 金额核对，v3 再加出租车票复读）。"""
    from ..config import get_settings

    version = version or get_settings().ticket_prompt
    prompt = PROMPTS[version]
    t0 = time.perf_counter()
    calls = pt = ct = 0
    try:
        raw, calls, pt, ct = _ask(vlm, image, prompt)
        fields = to_ticket(raw)
        uncertain = check(fields)
        if version == "v3" and fields.doc_type in DOUBLE_READ:
            raw2, c2, a, b = _ask(vlm, enhance(image), prompt)
            calls, pt, ct = calls + c2, pt + a, ct + b
            second = to_ticket(raw2)
            for k in DOUBLE_READ[fields.doc_type]:
                x, y = getattr(fields, k), getattr(second, k)
                same = abs(x - y) < 0.011 if isinstance(x, float) and isinstance(y, float) else x == y
                if not same:
                    uncertain.append(k)                  # 两遍读得不一样：至少有一遍是错的
            uncertain = sorted(set(uncertain))
        res = TicketResult(fields=fields, uncertain_fields=uncertain)
    except Exception as exc:  # noqa: BLE001  模型或解析失败：整张标记为失败，转人工
        res = TicketResult(fields=TicketFields(), error=f"{type(exc).__name__}: {exc}"[:300])
    res.model_calls, res.prompt_tokens, res.completion_tokens = calls, pt, ct
    res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res
