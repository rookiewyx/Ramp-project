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
    employee_name: str = ""             # 只用于脱敏：发给大模型前从说明里隐去
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

    # 调查 Agent 补充的信息（见 agent/）。初审时都为空
    supplements: list[str] = Field(default_factory=list, description="员工对追问的答复，等同于补充的报销说明")
    merchant: dict | None = Field(default=None, description="销售方的工商登记信息：category、scope")

    @property
    def full_note(self) -> str:
        """报销说明 + 员工的补充答复。审核时两者一起看。"""
        return "\n".join([self.note, *self.supplements]).strip() if self.supplements else self.note


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
    # 没通过的原因属于哪一类，调查 Agent 据此判断能不能补证据（见 agent/plan.py）：
    # missing_info 报销单缺信息、no_preapproval 未附预审批、outside_trip 不在出差期间、
    # field_uncertain 发票字段识别可疑、unclear_item 看不出是什么商品、llm_item 只凭名称判定的禁报明细；
    # 空表示不能靠补证据解决
    kind: str = ""


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
