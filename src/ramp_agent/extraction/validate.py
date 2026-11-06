"""字段校验：用票据自身的约束交叉检查识别结果，发现读错的字段。

这一步是分级识别能放心走“零成本路径”的前提：文字解析出来的结果只要通过校验就直接采用，
不通过才升级到视觉模型；视觉模型的结果不通过，就把可疑字段标成 uncertain 交给人工。
"""

from __future__ import annotations

import re

from .normalize import parse_date
from .schema import InvoiceFields

# 统一社会信用代码：18 位数字/大写字母。
# 国标字符集其实不含 I O Z S V，但项目里的虚构税号用到了 S、Z，这里只做长度和字符类型的宽松校验；
# 接入真实发票时可以换成严格版：^[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}$
TAX_ID_RE = re.compile(r"^[0-9A-Z]{18}$")
# 全电发票号码 20 位；传统发票号码 8 位（另有 10/12 位发票代码）
INVOICE_NO_RE = re.compile(r"^(\d{20}|\d{8}|\d{10}|\d{12})$")

REQUIRED = ("invoice_no", "issue_date", "buyer_name", "total")

# OCR / 视觉模型最常混淆的字符对。评测里 5 张发票把税号中的 Z 读成 2、S 读成 5
_CONFUSABLE = str.maketrans({"Z": "2", "S": "5", "O": "0", "D": "0", "Q": "0", "I": "1",
                             "L": "1", "B": "8", "G": "6", "T": "7"})


def confusable_key(text: str) -> str:
    """把易混字符统一成数字，用来判断两个号码是否“只差在易混字符上”。"""
    return text.upper().translate(_CONFUSABLE)


def validate(fields: InvoiceFields, *, strict_cn: bool = True,
             known_tax_ids: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """返回 (未通过的校验项, 可疑字段)。

    strict_cn:     是否按中国发票格式校验号码和税号
    known_tax_ids: 已知的公司税号。读出来的税号和它“只差在易混字符上”时，
                   分不清是读错了还是发票真的开错了，标为可疑交人工，而不是自动改成正确值——
                   自动纠正会让故意开成相近税号的发票蒙混过关。
    """
    failed: list[str] = []
    suspicious: set[str] = set()

    for name in REQUIRED:
        if getattr(fields, name) in (None, ""):
            failed.append(f"missing:{name}")
            suspicious.add(name)

    if fields.issue_date and parse_date(fields.issue_date) != fields.issue_date:
        failed.append("format:issue_date")
        suspicious.add("issue_date")

    if strict_cn:
        if fields.invoice_no and not INVOICE_NO_RE.match(fields.invoice_no):
            failed.append("format:invoice_no")
            suspicious.add("invoice_no")
        if fields.buyer_tax_id and not TAX_ID_RE.match(fields.buyer_tax_id):
            failed.append("format:buyer_tax_id")
            suspicious.add("buyer_tax_id")

    tid = fields.buyer_tax_id
    for known in known_tax_ids:
        if tid and tid != known and confusable_key(tid) == confusable_key(known):
            failed.append(f"confusable:buyer_tax_id~{known}")
            suspicious.add("buyer_tax_id")

    if fields.total is not None and fields.total <= 0:
        failed.append("range:total")
        suspicious.add("total")

    # 明细合计必须等于价税合计：同时校验金额和明细两类字段，是最有用的一条
    if fields.items and fields.total is not None:
        items_sum = round(sum(i.amount for i in fields.items), 2)
        if abs(items_sum - fields.total) > 0.02:
            failed.append(f"sum:items={items_sum}!=total={fields.total}")
            suspicious.update({"items", "total"})

    return failed, sorted(suspicious)
