"""Agent 提交的证据 → 校验 → 补进报销单。

模型负责“找”：在一堆记录里挑出和这笔报销对得上的那一条、把识别错的发票号改对再查、问员工缺的事实。
代码负责“验”：每条证据都要满足硬条件才采纳，不满足的拒绝并写明原因，供财务和评测查看。

  预审批    本人、已批准、费用类别一致（或不限类别）、报销金额不超过批准上限、开票日期在有效期内
  出差申请  本人、已批准、城市一致、开票日期在申请的起止日期内（含变更后的日期）
  发票查验  税务登记里发票状态为正常；采纳后用税务登记的票面信息替换识别出的可疑字段
  商户登记  登记名称与发票上的销售方一致；采纳后把经营类别交给 R8 的语义判断
  员工答复  原样作为补充说明，交给 R2 / R3 的语义判断（模型判断理由是否成立、提取人数）

采纳之后不由 Agent 下结论：补好证据的报销单交回规则引擎，从头再审一遍。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..policy.config import PolicyConfig, same_name
from ..policy.models import Claim
from .tools import ToolBox

FIELDS_FROM_TAX = ("invoice_no", "issue_date", "total", "buyer_name", "buyer_tax_id", "seller")


@dataclass
class Finding:
    kind: str                 # preapproval / travel_request / invoice / merchant / employee_reply
    ref: str                  # 记录编号或查验流水号
    accepted: bool
    reason: str               # 采纳了什么 / 为什么拒绝

    def to_dict(self) -> dict:
        return asdict(self)


def _same_city(a: str | None, b: str | None) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    return bool(a and b) and (a.startswith(b) or b.startswith(a))


def _check_preapproval(c: Claim, r: dict) -> str:
    date = c.invoice.issue_date or c.submit_date
    if r.get("employee_id") != c.employee_id:
        return "不是提交人本人的预审批"
    if r.get("status") != "approved":
        return f"审批状态为 {r.get('status')}，不是已批准"
    if r.get("category") not in (c.category, "any"):
        return f"预审批的费用类别是 {r.get('category')}，这笔报销是 {c.category}"
    if c.amount > float(r.get("amount_cap") or 0):
        return f"报销 {c.amount} 元超过批准上限 {r.get('amount_cap')} 元"
    if not (str(r.get("valid_from", "")) <= date <= str(r.get("valid_to", "9999"))):
        return f"开票日期 {date} 不在有效期 {r.get('valid_from')}~{r.get('valid_to')} 内"
    return ""


def _check_trip(c: Claim, r: dict) -> str:
    if r.get("employee_id") != c.employee_id:
        return "不是提交人本人的出差申请"
    if r.get("status") != "approved":
        return f"出差申请状态为 {r.get('status')}，不是已批准"
    if c.city and not _same_city(c.city, r.get("city")):
        return f"出差城市是 {r.get('city')}，报销单写的是 {c.city}"
    d = c.invoice.issue_date or ""
    if not (str(r.get("start", "")) <= d <= str(r.get("end", ""))):
        return f"开票日期 {d} 不在出差申请的 {r.get('start')}~{r.get('end')} 内"
    return ""


def apply_findings(claim: Claim, submitted: dict, box: ToolBox, policy: PolicyConfig) -> tuple[Claim, list[Finding]]:
    """返回补充了证据的报销单副本和每条证据的处理结果。原报销单不改。"""
    c = claim.model_copy(deep=True)
    out: list[Finding] = []

    def lookup(field: str, kind: str) -> tuple[str, dict | None]:
        ref = str(submitted.get(field) or "").strip()
        if not ref:
            return "", None
        got = box.ledger.get(ref)
        if got is None or got[0] != kind:
            out.append(Finding(kind, ref, False, "调查中没有查到过这个编号，不采纳"))
            return ref, None
        return ref, got[1]

    # 发票查验放在最前面：查验可能改正开票日期，后面的预审批、出差申请都按改正后的日期核对
    ref, rec = lookup("verification_id", "invoice")
    if rec is not None:
        if rec.get("status") != "正常":
            out.append(Finding("invoice", ref, False, f"税务登记的发票状态为 {rec.get('status')}"))
        else:
            changed = []
            for f in FIELDS_FROM_TAX:
                if rec.get(f) is not None and getattr(c.invoice, f) != rec[f]:
                    changed.append(f"{f}：{getattr(c.invoice, f)} → {rec[f]}")
                    setattr(c.invoice, f, rec[f])
            c.uncertain_fields = [f for f in c.uncertain_fields if f not in FIELDS_FROM_TAX]
            out.append(Finding("invoice", ref, True, "按税务登记的票面信息核对"
                               + ("，更正 " + "；".join(changed) if changed else "，与识别结果一致")))

    ref, rec = lookup("preapproval_id", "preapproval")
    if rec is not None:
        why = _check_preapproval(c, rec)
        if why:
            out.append(Finding("preapproval", ref, False, why))
        else:
            c.preapproval_id = ref
            out.append(Finding("preapproval", ref, True,
                               f"已批准，上限 {rec.get('amount_cap')} 元，有效期 {rec.get('valid_from')}~{rec.get('valid_to')}"))

    ref, rec = lookup("travel_request_id", "travel_request")
    if rec is not None:
        why = _check_trip(c, rec)
        if why:
            out.append(Finding("travel_request", ref, False, why))
        else:
            c.trip_start, c.trip_end = rec["start"], rec["end"]
            c.city = c.city or rec.get("city")
            out.append(Finding("travel_request", ref, True, f"出差 {rec.get('city')} {rec['start']}~{rec['end']}，已批准"))

    ref, rec = lookup("merchant_id", "merchant")
    if rec is not None:
        if not same_name(rec.get("name"), c.invoice.seller):
            out.append(Finding("merchant", ref, False, f"登记名称“{rec.get('name')}”与发票销售方“{c.invoice.seller}”不一致"))
        else:
            c.merchant = {"category": rec.get("category", ""), "scope": rec.get("scope", "")}
            out.append(Finding("merchant", ref, True, f"经营类别：{rec.get('category')}"))

    if box.reply and box.reply.strip():
        c.supplements.append(box.reply.strip()[:1000])
        out.append(Finding("employee_reply", "", True, "员工答复作为补充说明，交给审核规则判断"))
    elif box.asked:
        out.append(Finding("employee_reply", "", False, "员工没有答复"))
    return c, out
