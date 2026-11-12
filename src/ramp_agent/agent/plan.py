"""初审结果 → 要不要调查、调查什么、给 Agent 哪些工具。

只在“补齐证据就可能改变结论”时才调查：初审里每一项没通过的，都必须属于能靠查证据解决的那几类。
有一项属于制度明确要人工审核的（住宿超标、餐饮超标、逾期提交……），财务反正要看，调查省不了人工，
不花这笔模型调用。

工具按需授予：只给和这次待查问题相关的工具，模型看不到、也调用不了其他工具。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..policy.models import Claim, Decision, Review, Status

# 每类问题 → 能用的工具
TOOLS_FOR_KIND = {
    "no_preapproval": ["search_preapprovals"],
    "outside_trip": ["search_travel_requests"],
    "field_uncertain": ["verify_invoice"],
    "unclear_item": ["lookup_merchant"],
    "llm_item": ["lookup_merchant"],      # 大模型只凭明细名称判了禁报（如“雅间服务费”），查销售方登记再判一次
    "missing_info": ["ask_employee"],
}


@dataclass
class OpenItem:
    rule: str
    status: str
    kind: str
    reason: str
    clause: str


@dataclass
class Plan:
    items: list[OpenItem] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    skip_reason: str = ""

    @property
    def should_run(self) -> bool:
        return not self.skip_reason


def make_plan(claim: Claim, review: Review) -> Plan:
    if review.decision == Decision.APPROVE:
        return Plan(skip_reason="初审已通过")
    open_ = [r for r in review.results if r.status in (Status.VIOLATED, Status.UNCERTAIN)]
    blocked = [r for r in open_ if r.kind not in TOOLS_FOR_KIND]
    if blocked:
        return Plan(skip_reason="有补证据解决不了的问题："
                    + "、".join(f"{r.rule}（{r.reason or r.status.value}）" for r in blocked))
    if claim.extraction_failed:
        return Plan(skip_reason="发票未能识别")
    items = [OpenItem(r.rule, r.status.value, r.kind, r.reason, r.clause) for r in open_]
    tools: list[str] = []
    for it in items:
        for t in TOOLS_FOR_KIND[it.kind]:
            if t not in tools:
                tools.append(t)
    return Plan(items=items, tools=tools)
