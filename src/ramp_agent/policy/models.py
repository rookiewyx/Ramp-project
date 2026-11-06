"""制度审核的输入输出结构。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ..extraction.schema import InvoiceFields


class Claim(BaseModel):
    """一笔报销：员工填的报销单信息 + 识别模块读出的发票。"""

    expense_id: str
    report_id: str = ""
    employee_id: str = ""
    category: str                       # hotel / meal / entertainment / taxi / train / flight / conference / other
    amount: float                       # 员工填写的报销金额
    submit_date: str
    note: str = ""
    city: str | None = None
    trip_start: str | None = None
    trip_end: str | None = None
    nights: int | None = None
    headcount: int | None = None
    preapproval_id: str | None = None

    has_attachment: bool = True
    invoice: InvoiceFields = Field(default_factory=InvoiceFields)
    uncertain_fields: list[str] = Field(default_factory=list, description="识别模块标记的可疑字段")
    extraction_failed: bool = False


class Status(str, Enum):
    PASS = "pass"
    VIOLATED = "violated"
    UNCERTAIN = "uncertain"     # 证据不足（字段可疑、信息缺失、需要语义判断但没有模型）
    NA = "na"                   # 不适用（如非差旅类不查出差日期）


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    REJECT = "reject"


class RuleResult(BaseModel):
    rule: str
    status: Status
    reason: str = ""
    clause: str = ""
    by: str = "code"            # code / llm：这条结论是代码算出来的还是大模型判断的


class Review(BaseModel):
    expense_id: str
    decision: Decision
    results: list[RuleResult]
    explanation: str
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def violated_rules(self) -> list[str]:
        return [r.rule for r in self.results if r.status == Status.VIOLATED]

    @property
    def uncertain_rules(self) -> list[str]:
        return [r.rule for r in self.results if r.status == Status.UNCERTAIN]
