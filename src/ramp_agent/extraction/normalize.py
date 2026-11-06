"""字段归一化：不同来源的写法（¥1,739.00 / 1739 CNY / 2026年9月19日）统一成可比较的形式。"""

from __future__ import annotations

import re
from datetime import date

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_amount(value: object) -> float | None:
    """'¥1,739.00' / '5,046.28 CNY' / 1739 -> 1739.0。读不出数字返回 None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).replace(",", "").replace("，", "").replace(" ", "")
    m = _NUM.search(text)
    return round(float(m.group()), 2) if m else None


def parse_rate(value: object) -> float | None:
    """'6%' / 6 / 0.06 -> 0.06"""
    if value is None:
        return None
    if isinstance(value, str):
        num = parse_amount(value)
        if num is None:
            return None
        return round(num / 100, 4) if ("%" in value or num > 1) else num
    num = float(value)
    return round(num / 100, 4) if num > 1 else round(num, 4)


_DATE_PATTERNS = [
    re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})"),
]


def parse_date(value: object) -> str | None:
    """'2026年9月19日' / '2026/09/19' / '2026-09-19' -> '2026-09-19'；不合法返回 None。"""
    if not value:
        return None
    for pat in _DATE_PATTERNS:
        m = pat.search(str(value))
        if m:
            try:
                return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
            except ValueError:
                return None
    return None


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value))
    return text or None


def clean_invoice_no(value: object) -> str | None:
    """只保留字母数字，去掉 '#'、空格等。"""
    if value is None:
        return None
    text = re.sub(r"[^0-9A-Za-z]", "", str(value))
    return text or None
