"""票据识别的输入输出结构。

识别模块只负责“读出票面上写了什么”，不判断能不能报销——那是制度审核模块的事。
读不准的字段不猜，留空并写进 uncertain_fields，交给后续环节（转人工或补充材料）。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    name: str
    amount: float = Field(description="含税金额")


class InvoiceFields(BaseModel):
    invoice_no: str | None = None
    issue_date: str | None = Field(default=None, description="YYYY-MM-DD")
    seller: str | None = None
    buyer_name: str | None = None
    buyer_tax_id: str | None = None
    items: list[LineItem] = Field(default_factory=list)
    total: float | None = Field(default=None, description="价税合计")
    tax_rate: float | None = Field(default=None, description="小数，如 0.06")


class Tier(str, Enum):
    """分级识别的路径，按成本从低到高。"""

    PDF_TEXT = "pdf_text"   # 文字版 PDF：直接解析文字层，零模型调用
    VLM = "vlm"             # 扫描件/照片：调用视觉模型
    FAILED = "failed"       # 没有文件或所有路径都失败


class ExtractionResult(BaseModel):
    fields: InvoiceFields
    tier: Tier
    uncertain_fields: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list, description="未通过的校验项")
    fallback_from: Tier | None = Field(default=None, description="从哪一级降级而来")
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.tier == Tier.FAILED or bool(self.uncertain_fields)
