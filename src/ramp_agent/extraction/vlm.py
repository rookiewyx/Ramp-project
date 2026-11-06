"""第二级：视觉模型识别扫描件和照片。

走 OpenAI 兼容接口，所以既可以接 AutoDL 上 vLLM 部署的 Qwen3-VL（包括自己微调过的），
也可以接阿里云百炼等云端视觉模型，只需要改 .env 里的 VLM_* 配置。
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from PIL import Image

from ..config import Settings, get_settings
from ..llm import parse_json_object
from .normalize import clean_invoice_no, clean_text, parse_amount, parse_date, parse_rate
from .schema import InvoiceFields, LineItem

MAX_SIDE = 1600   # 长边超过就缩小，控制图片 token 数

PROMPT = """你是发票信息抽取器。读取图片中的发票，只输出一个 JSON 对象，不要输出任何解释。
字段：
- invoice_no: 发票号码（只保留数字和字母）
- issue_date: 开票日期，格式 YYYY-MM-DD
- seller: 销售方名称
- buyer_name: 购买方名称
- buyer_tax_id: 购买方纳税人识别号
- items: 明细数组，每项 {"name": 项目名称, "amount": 该项含税金额（金额+税额）}
- total: 价税合计（数字）
- tax_rate: 税率（小数，如 0.06）
看不清或图片上没有的字段填 null，不要猜测。"""


@dataclass
class VLMOutput:
    fields: InvoiceFields
    prompt_tokens: int = 0
    completion_tokens: int = 0


class VLMBackend(Protocol):
    def extract(self, image: bytes) -> VLMOutput: ...


def prepare_image(image: bytes) -> str:
    """缩放 + 转 JPEG + base64 data URL。"""
    img = Image.open(io.BytesIO(image)).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def to_fields(raw: dict) -> InvoiceFields:
    items = []
    for it in raw.get("items") or []:
        if isinstance(it, dict) and it.get("name"):
            amt = parse_amount(it.get("amount"))
            if amt is not None:
                items.append(LineItem(name=str(it["name"]).strip(), amount=amt))
    return InvoiceFields(
        invoice_no=clean_invoice_no(raw.get("invoice_no")),
        issue_date=parse_date(raw.get("issue_date")),
        seller=clean_text(raw.get("seller")),
        buyer_name=clean_text(raw.get("buyer_name")),
        buyer_tax_id=clean_invoice_no(raw.get("buyer_tax_id")),
        items=items,
        total=parse_amount(raw.get("total")),
        tax_rate=parse_rate(raw.get("tax_rate")),
    )


class OpenAICompatibleVLM:
    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        self.model = s.vlm_model
        # openai 客户端自带指数退避重试（连接错误、429、5xx）
        self.client = OpenAI(base_url=s.vlm_base_url, api_key=s.vlm_api_key or "EMPTY",
                             timeout=s.vlm_timeout_s, max_retries=s.vlm_max_retries)

    def _call(self, image_url: str, extra_hint: str = "") -> tuple[str, int, int]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": PROMPT + extra_hint},
            ]}],
        )
        usage = resp.usage
        return (resp.choices[0].message.content or "",
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0)

    def extract(self, image: bytes) -> VLMOutput:
        url = prepare_image(image)
        text, pt, ct = self._call(url)
        try:
            raw = parse_json_object(text)
        except (ValueError, json.JSONDecodeError):
            # 输出格式坏了重试一次，并明确提醒只输出 JSON
            text, pt2, ct2 = self._call(url, "\n上一次输出不是合法 JSON，请只输出 JSON 对象。")
            pt, ct = pt + pt2, ct + ct2
            raw = parse_json_object(text)
        return VLMOutput(fields=to_fields(raw), prompt_tokens=pt, completion_tokens=ct)
