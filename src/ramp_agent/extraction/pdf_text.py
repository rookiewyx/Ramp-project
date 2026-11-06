"""第一级：文字版 PDF 直接解析（零模型调用）。

电子发票是带文字层的 PDF，文字可以原样提取，用规则解析就能拿到全部字段，毫秒级、零成本。
规则是按本项目的电子发票版式写的；遇到没见过的版式，解析结果过不了 validate()，
路由会自动升级到视觉模型，所以规则写得不全不会导致读错，只会多花一次模型调用。
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from .normalize import clean_invoice_no, clean_text, parse_amount, parse_date, parse_rate
from .schema import InvoiceFields, LineItem

MIN_TEXT_CHARS = 20          # 少于这个字数视为扫描件（没有文字层）
WATERMARK_HINTS = ("样例数据", "SYNTHETIC")


def read_pdf_text(path: Path) -> str:
    """提取第一页文字。目前一张发票对应一页，多页发票后续再扩展。"""
    with pymupdf.open(path) as doc:
        return doc[0].get_text("text") if doc.page_count else ""


def has_text_layer(text: str) -> bool:
    return len(re.sub(r"\s+", "", text)) >= MIN_TEXT_CHARS


def render_first_page(path: Path, dpi: int = 150) -> bytes:
    """扫描版 PDF 没有文字层，渲染成图片交给视觉模型。"""
    with pymupdf.open(path) as doc:
        return doc[0].get_pixmap(dpi=dpi).tobytes("png")


def _after(label: str, text: str) -> str | None:
    m = re.search(rf"{label}\s*[:：]\s*(.+)", text)
    return m.group(1).strip() if m else None


def parse_einvoice_text(text: str) -> InvoiceFields:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not any(h in ln for h in WATERMARK_HINTS)]
    body = "\n".join(lines)

    fields = InvoiceFields(
        invoice_no=clean_invoice_no(_after("发票号码", body)),
        issue_date=parse_date(_after("开票日期", body)),
        buyer_name=clean_text(_after("购买方名称", body)),
        buyer_tax_id=clean_invoice_no(_after("纳税人识别号", body)),   # 第一个税号是购买方
        seller=clean_text(_after("销售方名称", body)),
    )
    m = re.search(r"价税合计[^¥￥\n]*[¥￥]\s*([\d,]+(?:\.\d+)?)", body)
    fields.total = parse_amount(m.group(1)) if m else None

    # 明细区：表头“税额”之后、“价税合计”之前，每 4 行一组：名称 / 金额 / 税率 / 税额
    try:
        start = lines.index("税额") + 1
        end = next(i for i, ln in enumerate(lines) if ln.startswith("价税合计"))
    except (ValueError, StopIteration):
        return fields
    rows = lines[start:end]
    if len(rows) % 4 != 0:
        return fields          # 版式不认识，不硬解析，交给校验和降级
    rates = []
    for i in range(0, len(rows), 4):
        name, net, rate, tax = rows[i:i + 4]
        net_v, tax_v = parse_amount(net), parse_amount(tax)
        if net_v is None or tax_v is None:
            return fields.model_copy(update={"items": []})
        fields.items.append(LineItem(name=name, amount=round(net_v + tax_v, 2)))
        rates.append(parse_rate(rate))
    fields.tax_rate = rates[0] if rates and len(set(rates)) == 1 else None
    return fields
