"""数据闭环：把线上的审核结果变成评测数据。

评测集是人出的题，总有没想到的情况。线上每天都在产生真实的“考题”，其中最有价值的是三类：

  overturned  系统给了定论（通过 / 驳回），财务复核时推翻了。这是系统确定答错了的地方。
  resolved    系统转人工，财务给出了最终结论。可以看出系统在哪些问题上拿不准，人最后是怎么判的。
  audit       系统自动通过或驳回、没人复核过的单子，随机抽一部分。
              只看财务推翻的单子有盲区：自动通过的单子没人看，错了也不会被发现。抽检能估计这部分的错误率。

流程：
    evalhub harvest                    从业务库收集，追加到评测集 policy-prod（草稿）
    evalhub serve                      第二个人盲标（看不到系统和财务的结论）
    evalhub agreement policy-prod      财务和第二个人不一致的，讨论后定答案
    evalhub freeze policy-prod
    evalhub run policy --dataset policy-prod

财务的复核结论作为标注人 finance 导入；抽检的单子没有财务结论，只能靠盲标。
写进评测集之前统一脱敏：员工姓名去掉、工号换成哈希、说明和明细里的手机号等个人信息替换掉，
评测集导出后可以放进代码仓库。

不收集的单子（会记在报告里）：
  - 发票识别失败：发票字段本身不可靠，属于识别问题，应该进识别评测；
  - 触发了重复报销（R6）：取决于当时的发票登记状态，离线评测复现不了。
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..pii import mask
from ..service.db import Expense, ExpenseStatus, session as service_session
from .datasets import existing_keys, open_for_append
from .db import Case, Label

FINANCE = "finance"
REASONS = {"overturned": "系统定论被推翻", "resolved": "转人工后财务定论", "audit": "自动结论抽检"}


@dataclass
class HarvestReport:
    dataset: str = ""
    version: str = ""
    added: Counter = field(default_factory=Counter)          # 理由 → 新增条数
    skipped: Counter = field(default_factory=Counter)        # 原因 → 跳过条数
    transitions: Counter = field(default_factory=Counter)    # (系统, 财务) → 条数
    rules_overturned: Counter = field(default_factory=Counter)  # 被推翻的单子里，系统依据的条款

    def format(self) -> str:
        lines = [f"[harvest] {self.dataset} {self.version}：新增 {sum(self.added.values())} 条"]
        for k, n in self.added.items():
            lines.append(f"  {REASONS[k]}：{n}")
        if self.skipped:
            lines.append("  跳过：" + "，".join(f"{k} {n}" for k, n in self.skipped.items()))
        if self.transitions:
            lines.append("系统 → 财务：" + "，".join(f"{a}→{b} {n}" for (a, b), n in self.transitions.most_common()))
        if self.rules_overturned:
            lines.append("被推翻的单子涉及的条款：" + "，".join(
                f"{r} {n}" for r, n in self.rules_overturned.most_common()))
        return "\n".join(lines)


def _key(expense_id: str) -> str:
    """评测集里的用例编号：由线上编号哈希得到，同一笔报销永远对应同一个编号，用来去重。"""
    return "P" + hashlib.sha256(expense_id.encode()).hexdigest()[:10]


def _pseudonym(employee_id: str) -> str:
    return "E" + hashlib.sha256(("emp:" + employee_id).encode()).hexdigest()[:8] if employee_id else ""


def to_record(e: Expense) -> dict:
    """线上报销单 → 评测集用例（和合成数据同一格式），脱敏。"""
    rv = e.reviews[-1]
    names = [e.employee_name] if e.employee_name else []
    fields = dict(rv.extraction.get("fields") or {})
    fields["items"] = [{"name": mask(i.get("name") or "", names), "amount": i.get("amount")}
                       for i in fields.get("items") or []]
    return {
        "expense_id": _key(e.id), "report_id": "", "employee_id": _pseudonym(e.employee_id),
        "category": e.category, "amount": float(e.amount), "submit_date": e.submit_date,
        "note": mask(e.note or "", names), "city": e.city, "trip_start": e.trip_start,
        "trip_end": e.trip_end, "nights": e.nights, "headcount": e.headcount,
        "preapproval_id": e.preapproval_id,
        "invoice": {**{k: fields.get(k) for k in ("invoice_no", "issue_date", "seller", "buyer_name",
                                                  "buyer_tax_id", "items", "total", "tax_rate")},
                    "file": "production" if e.file_key else None},
    }


def _flagged_rules(rv) -> list[str]:
    return sorted({r["rule"] for r in rv.results if r["status"] in ("violated", "uncertain")})


def select_expenses(service_url: str, audit: int, seed: int) -> list[tuple[str, Expense]]:
    """按三类理由挑出要收集的报销单。"""
    with service_session(service_url) as s:
        rows = s.scalars(select(Expense).where(Expense.status == ExpenseStatus.DONE)
                         .options(selectinload(Expense.reviews), selectinload(Expense.overrides))
                         .order_by(Expense.created_at)).all()
    picked: list[tuple[str, Expense]] = []
    pool: list[Expense] = []
    for e in rows:
        if not e.reviews:
            continue
        if e.final_decision is None:
            if e.decision in ("approve", "reject"):
                pool.append(e)
        elif e.decision == "review":
            picked.append(("resolved", e))
        elif e.final_decision != e.decision:
            picked.append(("overturned", e))
    rng = random.Random(seed)
    picked += [("audit", e) for e in rng.sample(pool, min(audit, len(pool)))]
    return picked


def harvest(service_url: str, s: Session, dataset: str = "policy-prod", *, audit: int = 20,
            seed: int = 0, dry_run: bool = False) -> HarvestReport:
    report = HarvestReport(dataset=dataset)
    seen = existing_keys(s, dataset)
    todo: list[tuple[str, Expense]] = []
    for reason, e in select_expenses(service_url, audit, seed):
        rv = e.reviews[-1]
        if _key(e.id) in seen:
            report.skipped["已收集过"] += 1
            continue
        if rv.extraction.get("tier") == "failed":
            report.skipped["发票识别失败"] += 1
            continue
        if any(r["rule"] == "R6" and r["status"] == "violated" for r in rv.results):
            report.skipped["重复报销（离线无法复现）"] += 1
            continue
        todo.append((reason, e))
        if reason != "audit":
            report.transitions[(e.decision, e.final_decision)] += 1
        if reason == "overturned":
            report.rules_overturned.update(_flagged_rules(rv) or ["（无违规条款）"])
        report.added[reason] += 1

    if dry_run or not todo:
        report.version = "（未写入）" if dry_run else "（无新增）"
        return report

    v = open_for_append(s, dataset, "policy",
                        description="线上数据：财务推翻、转人工后的定论、自动结论抽检（已脱敏）")
    report.version = v.ref
    for reason, e in todo:
        rec = to_record(e)
        case = Case(version=v, key=rec["expense_id"], payload=rec,
                    tags={"family": reason, "origin": "production",
                          "system_decision": e.decision, "system_flags": _flagged_rules(e.reviews[-1])})
        s.add(case)
        if reason != "audit":
            o = e.overrides[-1]
            # 财务复核只给结论，不勾条款；条款级的答案由盲标的人补上
            s.add(Label(case=case, labeler=FINANCE, decision=e.final_decision, rules=[],
                        note=mask(o.note or "", [e.employee_name] if e.employee_name else [])))
    s.flush()
    return report
