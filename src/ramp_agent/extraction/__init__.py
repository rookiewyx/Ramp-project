"""票据识别：文字版 PDF 规则解析 → 视觉模型 → 校验兜底。"""

from .router import InvoiceExtractor
from .schema import ExtractionResult, InvoiceFields, LineItem, Tier
from .validate import validate
from .vlm import OpenAICompatibleVLM, VLMOutput

__all__ = ["ExtractionResult", "InvoiceExtractor", "InvoiceFields", "LineItem",
           "OpenAICompatibleVLM", "Tier", "VLMOutput", "validate"]
