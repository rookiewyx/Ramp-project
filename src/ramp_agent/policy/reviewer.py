"""制度审核：规则引擎 + 大模型语义判断 + 不确定性传递。

流程：
  1. 代码逐条检查 R1–R9 和金额一致性。能算清楚的直接得出结论（通过 / 违规）。
  2. 需要理解语义的问题（变相烟酒娱乐、超标理由、说明里的人数）合并成一次大模型调用。
  3. 汇总：有“驳回类”违规 → 驳回；否则有“复核类”违规或任何不确定 → 转人工；全部通过 → 建议通过。

两个关键设计：
  - 不确定性传递：识别模块标为可疑的字段，依赖它的规则只能得出“不确定”，不能据此驳回。
    例如税号被读成易混字符，R4 给出“不确定 → 转人工”，而不是错误驳回一张合规发票。
  - 保守：大模型不可用或答不上来时，对应规则记为“不确定”，转人工，不会自动通过。
"""

from __future__ import annotations

from datetime import date

from loguru import logger

from .config import PolicyConfig, load_policy, same_name
from .models import Claim, Decision, Review, RuleResult, Status
from .registry import InMemoryRegistry, InvoiceRegistry
from .semantic import SemanticJudge, SemanticRequest, SemanticResult

class PolicyReviewer:
    def __init__(self, policy: PolicyConfig | None = None, judge: SemanticJudge | None = None,
                 registry: InvoiceRegistry | None = None):
        self.policy = policy or load_policy()
        self.judge = judge
        self.registry = registry or InMemoryRegistry()

    # ---------------- 对外入口 ----------------
    def semantic(self, claim: Claim) -> tuple[SemanticResult, int]:
        """需要的话调用一次大模型，返回 (结果, 调用次数)。可以提前并发执行，再按顺序审核。"""
        sem_req = self._semantic_request(claim)
        if sem_req.empty or self.judge is None:
            return SemanticResult(), 0
        try:
            return self.judge.judge(sem_req), 1
        except Exception as exc:  # noqa: BLE001  模型出错时相关规则记为不确定，转人工
            logger.warning("语义判断失败 {}: {}", claim.expense_id, exc)
            return SemanticResult(), 1

    def review(self, claim: Claim, semantic: tuple[SemanticResult, int] | None = None) -> Review:
        """审核一笔报销。查重依赖登记顺序，所以同一批报销要按提交顺序调用。"""
        sem, llm_calls = semantic if semantic is not None else self.semantic(claim)

        results = [r for r in self._run_rules(claim, sem) if r.status != Status.NA]
        for r in results:
            r.clause = self.policy.rules[r.rule].clause
        decision = self._decide(results)
        return Review(expense_id=claim.expense_id, decision=decision, results=results,
                      explanation=self._explain(decision, results), llm_calls=llm_calls,
                      prompt_tokens=sem.prompt_tokens, completion_tokens=sem.completion_tokens)

    # ---------------- 语义问题收集 ----------------
    def _semantic_request(self, c: Claim) -> SemanticRequest:
        req = SemanticRequest(category=c.category, amount=c.amount, note=c.note)
        if c.has_attachment:
            for i in c.invoice.items:
                hit = self.policy.prohibited_hit(i.name)
                if hit:                                   # 命中关键词：请模型确认商品本身
                    req.keyword_hits[i.name] = hit[1]
                if (hit or not self.policy.is_safe_item(i.name)) and i.name not in req.items_to_check:
                    req.items_to_check.append(i.name)
        lim = self.policy.limits
        if c.category == "taxi" and c.amount > lim.taxi_per_ride and c.note.strip():
            req.need_taxi_reason = True
        if c.category == "entertainment" and not c.headcount and c.note.strip():
            req.need_headcount = True
        return req

    # ---------------- 规则 ----------------
    def _certain(self, c: Claim, *names: str) -> bool:
        """这些发票字段是否可信：识别没失败、没被标为可疑、有值。"""
        if c.extraction_failed:
            return False
        return all(n not in c.uncertain_fields and getattr(c.invoice, n) not in (None, "", [])
                   for n in names)

    def _run_rules(self, c: Claim, sem: SemanticResult) -> list[RuleResult]:
        P, lim = self.policy, self.policy.limits
        out: list[RuleResult] = []

        def add(rule: str, status: Status, reason: str = "", by: str = "code") -> None:
            out.append(RuleResult(rule=rule, status=status, reason=reason, by=by))

        # R9 附件
        if not c.has_attachment:
            add("R9", Status.VIOLATED, "未附发票")
            return out          # 没有发票，其他发票相关检查无从谈起
        if c.extraction_failed:
            add("R9", Status.UNCERTAIN, "发票未能识别，需人工查看原件")
        else:
            add("R9", Status.PASS)

        # 金额一致：报销金额 vs 发票价税合计（能发现“读错但自洽”的识别错误）
        if not self._certain(c, "total"):
            add("AMT", Status.UNCERTAIN, "发票金额未能可靠识别")
        elif round(c.amount * 100) != round(c.invoice.total * 100):   # 按分比较，差 1 分也算不一致
            add("AMT", Status.VIOLATED, f"报销 {c.amount} 元与发票价税合计 {c.invoice.total} 元不一致")
        else:
            add("AMT", Status.PASS)

        # R1 住宿
        if c.category == "hotel":
            cap = P.hotel_cap(c.city)
            if not c.nights:
                add("R1", Status.UNCERTAIN, "未填写住宿晚数")
            else:
                per = round(c.amount / c.nights, 2)
                add("R1", Status.VIOLATED if per > cap else Status.PASS,
                    f"{c.city or '未知城市'}住宿 {per} 元/晚，标准 {cap:.0f} 元")

        # R2 餐饮 / 招待
        if c.category == "meal":
            add("R2", Status.VIOLATED if c.amount > lim.meal_per_day else Status.PASS,
                f"出差个人餐饮 {c.amount} 元，标准每天 {lim.meal_per_day:.0f} 元")
        if c.category == "entertainment":
            heads, by = c.headcount, "code"
            if not heads and sem.headcount:
                heads, by = sem.headcount, "llm"
            if not heads and (not c.note.strip() or sem.answered):
                add("R2", Status.VIOLATED, "报销说明未注明招待人数，无法计算人均",
                    "llm" if sem.answered else "code")
            elif not heads:
                add("R2", Status.UNCERTAIN, f"需要从说明“{c.note}”中确认招待人数")
            else:
                per = round(c.amount / heads, 2)
                add("R2", Status.VIOLATED if per > lim.entertainment_per_head else Status.PASS,
                    f"招待 {heads} 人，人均 {per} 元，标准 {lim.entertainment_per_head:.0f} 元", by)

        # R3 交通
        names = [i.name for i in c.invoice.items]
        over_class = [w for w in P.travel_class_over if any(w in n for n in names)]
        if over_class:
            add("R3", Status.VIOLATED, f"舱位/座席超标：{'、'.join(over_class)}")
        elif c.category == "taxi" and c.amount > lim.taxi_per_ride:
            if not c.note.strip():
                add("R3", Status.VIOLATED, f"打车 {c.amount} 元超过 {lim.taxi_per_ride:.0f} 元，且未说明原因")
            elif sem.taxi_reason_valid is True:
                add("R3", Status.PASS, f"超标理由成立：{sem.taxi_reason_explanation}", "llm")
            elif sem.taxi_reason_valid is False:
                add("R3", Status.VIOLATED, f"超标理由不成立：{sem.taxi_reason_explanation}", "llm")
            else:
                add("R3", Status.UNCERTAIN, f"打车超标，需判断说明“{c.note}”是否成立")
        elif c.category in ("taxi", "train", "flight"):
            add("R3", Status.PASS)

        # R4 抬头与税号：两项分开判断。抬头可信且明确不符（如“个人”），不必等税号；
        # 税号为空但识别环节没有标为可疑，说明发票上确实没有税号，也算不符。
        name_ok = self._certain(c, "buyer_name")
        tid_flagged = c.extraction_failed or "buyer_tax_id" in c.uncertain_fields
        tid = (c.invoice.buyer_tax_id or "").strip()
        bad = []
        if name_ok and not same_name(c.invoice.buyer_name, P.company.name):
            bad.append(f"抬头“{c.invoice.buyer_name}”与公司全称不一致")
        if not tid_flagged and tid != P.company.tax_id:
            bad.append(f"税号 {tid} 与公司不一致" if tid else "发票未填写购买方税号")
        if bad:
            add("R4", Status.VIOLATED, "；".join(bad))
        elif not name_ok or tid_flagged:
            add("R4", Status.UNCERTAIN, "发票抬头或税号未能可靠识别，需人工核对")
        else:
            add("R4", Status.PASS)

        # R5 日期
        if not self._certain(c, "issue_date"):
            add("R5_late", Status.UNCERTAIN, "开票日期未能可靠识别")
        else:
            issued = date.fromisoformat(c.invoice.issue_date)
            age = (date.fromisoformat(c.submit_date) - issued).days
            add("R5_late", Status.VIOLATED if age > lim.submit_within_days else Status.PASS,
                f"开票后 {age} 天提交，期限 {lim.submit_within_days} 天")
            if c.category in P.travel_categories and c.trip_start and c.trip_end:
                inside = c.trip_start <= c.invoice.issue_date <= c.trip_end
                add("R5_outside_trip", Status.PASS if inside else Status.VIOLATED,
                    f"开票日 {c.invoice.issue_date}，出差 {c.trip_start}~{c.trip_end}")

        # R6 重复报销
        if not self._certain(c, "invoice_no"):
            add("R6", Status.UNCERTAIN, "发票号码未能可靠识别，无法查重")
        else:
            owner = self.registry.claim(c.invoice.invoice_no, c.expense_id)
            add("R6", Status.VIOLATED if owner else Status.PASS,
                f"发票号 {c.invoice.invoice_no} 已在报销 {owner} 中使用" if owner else "")

        # R7 大额预审批
        if c.amount > lim.preapproval_threshold:
            add("R7", Status.PASS if c.preapproval_id else Status.VIOLATED,
                f"单笔 {c.amount} 元，预审批：{c.preapproval_id or '无'}")

        # R8 不予报销项目
        #   - 命中禁报关键词：有大模型时由它确认商品本身是否就是这类物品（“红酒杯”不是酒）；
        #     没有大模型时按命中处理。
        #   - 不在正常名单里的：交给大模型；它答不上来或看不出是什么，转人工。
        hits, unknown, llm_hits, cleared = [], [], [], False
        for n in names:
            hit = P.prohibited_hit(n)
            if not hit and P.is_safe_item(n):
                continue
            v = sem.items.get(n)
            if v is None and hit:
                hits.append(f"{n}（{hit[0]}）")
            elif v is None or v.prohibited is None:
                unknown.append(n)
            elif v.prohibited:
                llm_hits.append(f"{n}（{v.category or (hit[0] if hit else '')}：{v.reason}）")
            else:
                cleared = True
        if hits:
            add("R8", Status.VIOLATED, "明细含不予报销项目：" + "；".join(hits))
        elif llm_hits:
            add("R8", Status.VIOLATED, "明细含不予报销项目：" + "；".join(llm_hits), "llm")
        elif unknown:
            add("R8", Status.UNCERTAIN, "以下明细需要人工判断：" + "、".join(unknown))
        elif not names:
            add("R8", Status.UNCERTAIN, "未识别出发票明细")
        else:
            add("R8", Status.PASS, by="llm" if cleared else "code")
        return out

    # ---------------- 汇总 ----------------
    def _decide(self, results: list[RuleResult]) -> Decision:
        violated = [r for r in results if r.status == Status.VIOLATED]
        if any(self.policy.rules[r.rule].outcome == "reject" for r in violated):
            return Decision.REJECT
        if violated or any(r.status == Status.UNCERTAIN for r in results):
            return Decision.REVIEW
        return Decision.APPROVE

    def _explain(self, decision: Decision, results: list[RuleResult]) -> str:
        head = {Decision.APPROVE: "建议通过：所有条款检查通过。",
                Decision.REVIEW: "需人工审核：",
                Decision.REJECT: "建议驳回："}[decision]
        lines = [head]
        for r in results:
            if r.status in (Status.VIOLATED, Status.UNCERTAIN):
                spec = self.policy.rules[r.rule]
                tag = "违反" if r.status == Status.VIOLATED else "无法确认"
                lines.append(f"- [{r.rule} {spec.title}·{tag}] {r.reason}。依据：“{spec.clause}”")
        return "\n".join(lines)
