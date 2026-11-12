"""个人信息脱敏：发给外部模型的文字在这里统一处理。

报销说明和明细是员工填写的，里面可能出现手机号、身份证号、银行卡号、邮箱，或者员工自己的名字。
审核用不到这些信息，发给云端大模型前替换成占位符：判断照样能做，个人信息不出公司。

只处理文字。发票图片要交给视觉模型才能识别，图片里的信息无法在这一步脱敏；
对隐私要求高的部署，把 VLM_BASE_URL 指向自建的视觉模型（如 AutoDL 上的 vLLM）即可。
"""

from __future__ import annotations

import re

# 顺序有讲究：先匹配长的号码，免得身份证号被当成手机号截掉一段
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("身份证号", re.compile(r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9A-Za-z])")),
    ("银行卡号", re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")),
    ("手机号", re.compile(r"(?<!\d)(?:\+?86[ -]?)?1[3-9]\d(?:[ -]?\d{4}){2}(?!\d)")),
    ("邮箱", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


def mask(text: str, names: list[str] | tuple[str, ...] = ()) -> str:
    """把文字里的个人信息换成占位符，如“[手机号]”。names：需要隐去的人名（如员工本人）。"""
    if not text:
        return text
    for label, pat in _PATTERNS:
        text = pat.sub(f"[{label}]", text)
    for n in sorted({n.strip() for n in names if n and len(n.strip()) >= 2}, key=len, reverse=True):
        text = text.replace(n, "[姓名]")
    return text
