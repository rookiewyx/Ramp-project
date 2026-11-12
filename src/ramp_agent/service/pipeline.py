"""审核流水线：一笔报销从“已受理”到“有结论”。

    领取任务（原子地把 queued 改成 processing）
      └► 取发票文件 ─► 分级识别 ─► 组装 Claim ─► 制度审核（规则 + 至多一次大模型）
           └► 写审核记录、更新状态、记审计日志（同一个事务）

设计要点：
  - 可重入：消息可能被投递两次（worker 崩溃后重投、客户端重试），领取用条件更新，
    同一时间只有一个 worker 能拿到；已完成的直接跳过。
  - 崩溃恢复：处理中超过 stale_after_s 没完成，视为 worker 已崩溃，允许别的 worker 接手。
  - 模型临时故障先重试，再降级：识别和审核模块遇到模型错误会降级成“转人工”。
    在服务里，网络抖动不该直接把单子推给财务，所以前几次尝试遇到模型故障抛 ModelUnavailable 交给队列重试，
    最后一次才接受降级结果。
  - 其他异常（数据库、存储故障）原样抛出，交给任务队列重试。
  - 调查 Agent：初审转人工、而且问题都能靠补证据解决的，先调查再定结论（见 agent/）。
    调查要追问员工时，结论先按转人工保存，agent_status 记为等待答复；员工答复（或等待超时）后
    重新排队，这次不再识别和初审，直接从调查的断点继续（resume）。
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from loguru import logger
from opentelemetry.trace import get_current_span as trace_current
from sqlalchemy import or_, update

from ..agent import AsyncEmployee, FixtureSystems, InvestigationResult, Investigator
from ..agent.systems import EnterpriseSystems
from ..config import Settings
from ..extraction import ExtractionResult, InvoiceExtractor, InvoiceFields, Tier
from ..policy import Claim, InvoiceRegistry, PolicyReviewer, Review, SemanticJudge, load_policy
from ..telemetry import tracer
from .db import (
    AgentStatus,
    Expense,
    ExpenseStatus,
    Investigation,
    ReviewRecord,
    audit,
    session,
    utcnow,
)
from .storage import ObjectStore


@dataclass
class Deps:
    """流水线依赖。worker 启动时组装一次；测试里换成假的模型和内存存储。"""

    settings: Settings
    store: ObjectStore
    registry: InvoiceRegistry
    extractor: InvoiceExtractor
    judge: SemanticJudge | None
    versions: dict
    investigator: Investigator | None = None      # 不配置就不调查，转人工的直接交给财务
    systems: EnterpriseSystems | None = None


class ModelUnavailable(RuntimeError):
    """模型调用失败，稍后重试可能恢复。"""


def acquire(url: str, expense_id: str, stale_after_s: int) -> bool:
    """把报销从 queued（或卡死的 processing）改成 processing。返回是否领取成功。"""
    stale = utcnow() - timedelta(seconds=stale_after_s)
    with session(url) as s:
        res = s.execute(
            update(Expense)
            .where(Expense.id == expense_id,
                   or_(Expense.status == ExpenseStatus.QUEUED,
                       (Expense.status == ExpenseStatus.PROCESSING) & (Expense.updated_at < stale)))
            .values(status=ExpenseStatus.PROCESSING, attempts=Expense.attempts + 1, updated_at=utcnow()))
        return res.rowcount == 1


def build_claim(e: Expense, ext: ExtractionResult | None) -> Claim:
    c = Claim(expense_id=e.id, report_id=e.report_id, employee_id=e.employee_id,
              employee_name=e.employee_name, category=e.category, amount=float(e.amount),
              submit_date=e.submit_date, note=e.note, city=e.city, trip_start=e.trip_start,
              trip_end=e.trip_end, nights=e.nights, headcount=e.headcount,
              preapproval_id=e.preapproval_id, has_attachment=e.file_key is not None)
    if ext is not None:
        c.invoice = ext.fields
        c.uncertain_fields = list(ext.uncertain_fields)
        c.extraction_failed = ext.tier == Tier.FAILED
    return c


def extract(deps: Deps, e: Expense) -> ExtractionResult | None:
    if e.file_key is None:
        return None
    with tracer.start_as_current_span("pipeline.fetch_file") as span:
        data = deps.store.get(e.file_key)
        span.set_attribute("file.size", len(data))
    # 识别模块按文件路径读取；写到临时文件，用完即删
    with tracer.start_as_current_span("extraction") as span, \
            tempfile.TemporaryDirectory(prefix="ramp-") as tmp:
        path = Path(tmp) / ("invoice" + Path(e.file_key).suffix)
        path.write_bytes(data)
        ext = deps.extractor.extract(path)
        span.set_attribute("extraction.tier", ext.tier.value)
        span.set_attribute("extraction.model_calls", ext.model_calls)
        span.set_attribute("extraction.uncertain_fields", len(ext.uncertain_fields))
        return ext


def process(deps: Deps, expense_id: str, *, final_attempt: bool = True) -> str:
    """审核一笔报销。返回 done / skipped。异常向上抛，由任务队列决定是否重试。

    final_attempt=False 时，模型故障抛 ModelUnavailable 而不是降级成转人工。
    """
    url = deps.settings.database_url
    span = trace_current()
    span.set_attribute("expense.id", expense_id)
    with tracer.start_as_current_span("pipeline.acquire"):
        acquired = acquire(url, expense_id, deps.settings.stale_after_s)
    if not acquired:
        logger.info("跳过 {}：已完成或正在被其他 worker 处理", expense_id)
        span.set_attribute("pipeline.outcome", "skipped")
        return "skipped"

    t0 = time.perf_counter()
    with session(url) as s:
        e = s.get(Expense, expense_id)
        _ = e.reviews, e.investigations
        audit(s, expense_id, "review_started", attempt=e.attempts)
    if e.agent_status in (AgentStatus.REPLIED, AgentStatus.EXPIRED) and e.investigations:
        return _resume(deps, e, final_attempt, t0)

    ext = extract(deps, e)
    if not final_attempt and ext is not None and (ext.error or "").startswith("vlm_error"):
        raise ModelUnavailable(f"视觉模型调用失败：{ext.error}")
    claim = build_claim(e, ext)
    reviewer = PolicyReviewer(load_policy(), deps.judge, deps.registry)
    with tracer.start_as_current_span("policy.semantic") as sp:
        sem = reviewer.semantic(claim)             # 先做语义判断，失败时还没登记发票，重试是干净的
        sp.set_attribute("policy.llm_calls", sem[1])
    if not final_attempt and sem[1] and not sem[0].answered:
        raise ModelUnavailable("大模型调用失败")
    with tracer.start_as_current_span("policy.rules"):
        rv = reviewer.review(claim, sem)

    inv = None
    if deps.investigator is not None and deps.settings.agent_enabled:
        inv = deps.investigator.run(claim, rv, reviewer, deps.systems or FixtureSystems(), AsyncEmployee(), sem)
        if inv.status == "error" and not final_attempt:
            raise ModelUnavailable(f"调查时模型调用失败：{inv.error}")
        if inv.status == "skipped":
            inv = None
    final = inv.review if inv is not None and inv.review is not None else rv
    latency = int((time.perf_counter() - t0) * 1000)

    ext_info = {"tier": ext.tier.value, "fields": ext.fields.model_dump(),
                "uncertain_fields": ext.uncertain_fields, "checks_failed": ext.checks_failed,
                "error": ext.error, "latency_ms": ext.latency_ms} if ext else {"tier": "none"}
    with tracer.start_as_current_span("pipeline.persist"), session(url) as s:
        s.add(_record(deps, expense_id, final, ext_info, latency,
                      llm_calls=rv.llm_calls + (inv.llm_calls if inv else 0),
                      tokens=rv.prompt_tokens + rv.completion_tokens + (inv.tokens if inv else 0)
                      + ((ext.prompt_tokens + ext.completion_tokens) if ext else 0),
                      vlm_calls=ext.model_calls if ext else 0))
        if inv is not None:
            s.add(Investigation(expense_id=expense_id, status=inv.status, decision_before=inv.decision_before,
                                decision_after=inv.decision_after if inv.review else None,
                                question=inv.question, reply=inv.reply,
                                asked_at=utcnow() if inv.question else None,
                                result={**inv.to_dict(), "initial_results": _results(rv)}, state=inv.state,
                                llm_calls=inv.llm_calls, tool_calls=inv.tool_calls, tokens=inv.tokens,
                                latency_ms=inv.latency_ms))
            _audit_investigation(s, expense_id, inv)
        s.execute(update(Expense).where(Expense.id == expense_id).values(
            status=ExpenseStatus.DONE, decision=final.decision.value, error=None,
            agent_status=_agent_status(inv), updated_at=utcnow()))
        audit(s, expense_id, "reviewed", decision=final.decision.value,
              violated=final.violated_rules, uncertain=final.uncertain_rules, latency_ms=latency)
    # 日志里只记 ID 和结论，不记报销说明等可能含个人信息的内容
    logger.info("审核完成 {} → {}（{} ms，大模型 {} 次{}）", expense_id, final.decision.value, latency,
                rv.llm_calls, f"，调查 {inv.status}" if inv else "")
    span.set_attribute("pipeline.outcome", "done")
    span.set_attribute("review.decision", final.decision.value)
    return "done"


def _resume(deps: Deps, e: Expense, final_attempt: bool, t0: float) -> str:
    """员工答复了（或等待超时），从调查的断点继续。不重新识别发票，用上次保存的识别结果。"""
    url = deps.settings.database_url
    row = e.investigations[-1]
    last = e.reviews[-1]
    reply = row.reply if e.agent_status == AgentStatus.REPLIED else None
    if deps.investigator is None or not row.state:
        # 调查功能被关掉了，或断点丢失：保持转人工，交给财务
        with session(url) as s:
            s.execute(update(Investigation).where(Investigation.id == row.id).values(status="error", state=None))
            s.execute(update(Expense).where(Expense.id == e.id).values(
                status=ExpenseStatus.DONE, agent_status=AgentStatus.ERROR, updated_at=utcnow()))
            audit(s, e.id, "investigation_abandoned", reason="调查未启用或断点丢失")
        return "done"
    claim = claim_from_record(e, last.extraction)
    reviewer = PolicyReviewer(load_policy(), deps.judge, deps.registry)
    inv = deps.investigator.resume(claim, None, reviewer, deps.systems or FixtureSystems(), row.state,
                                   row.result, reply)
    if inv.status == "error" and not final_attempt:
        raise ModelUnavailable(f"调查时模型调用失败：{inv.error}")
    final = inv.review
    latency = int((time.perf_counter() - t0) * 1000)
    with session(url) as s:
        if final is not None:
            s.add(_record(deps, e.id, final, last.extraction, latency, llm_calls=inv.llm_calls - row.llm_calls,
                          tokens=inv.tokens - row.tokens, vlm_calls=0))
        s.execute(update(Investigation).where(Investigation.id == row.id).values(
            status=inv.status, decision_after=final.decision.value if final else None,
            result={**inv.to_dict(), "initial_results": row.result.get("initial_results", [])}, state=None,
            llm_calls=inv.llm_calls, tool_calls=inv.tool_calls, tokens=inv.tokens, latency_ms=inv.latency_ms,
            updated_at=utcnow()))
        _audit_investigation(s, e.id, inv)
        decision = final.decision.value if final else e.decision
        s.execute(update(Expense).where(Expense.id == e.id).values(
            status=ExpenseStatus.DONE, decision=decision, error=None, agent_status=_agent_status(inv),
            updated_at=utcnow()))
        audit(s, e.id, "reviewed", decision=decision, after_investigation=True)
    logger.info("调查继续 {} → {}（{}）", e.id, decision, inv.status)
    return "done"


def _record(deps: Deps, expense_id: str, rv: Review, ext_info: dict, latency: int, *, llm_calls: int,
            tokens: int, vlm_calls: int) -> ReviewRecord:
    return ReviewRecord(expense_id=expense_id, decision=rv.decision.value, results=_results(rv),
                        explanation=rv.explanation, extraction=ext_info, llm_calls=llm_calls, vlm_calls=vlm_calls,
                        tokens=tokens, latency_ms=latency, versions=deps.versions)


def _results(rv: Review) -> list[dict]:
    return [r.model_dump(mode="json") for r in rv.results]


def _agent_status(inv: InvestigationResult | None) -> str | None:
    if inv is None:
        return None
    return {"awaiting_employee": AgentStatus.AWAITING, "resolved": AgentStatus.RESOLVED,
            "unresolved": AgentStatus.UNRESOLVED}.get(inv.status, AgentStatus.ERROR)


def _audit_investigation(s, expense_id: str, inv: InvestigationResult) -> None:
    if inv.status == "awaiting_employee":
        audit(s, expense_id, "employee_asked", actor="agent", question=inv.question)
    else:
        audit(s, expense_id, "investigated", actor="agent", status=inv.status, before=inv.decision_before,
              after=inv.decision_after, tool_calls=inv.tool_calls,
              accepted=[f["kind"] for f in inv.findings if f["accepted"]])


def claim_from_record(e: Expense, extraction: dict) -> Claim:
    """用保存的识别结果重建报销单（恢复调查时不再重新识别发票）。"""
    c = build_claim(e, None)
    tier = extraction.get("tier", "none")
    if tier != "none" and extraction.get("fields"):
        c.invoice = InvoiceFields.model_validate(extraction["fields"])
        c.uncertain_fields = list(extraction.get("uncertain_fields") or [])
        c.extraction_failed = tier == Tier.FAILED.value
    return c


def release(url: str, expense_id: str, error: str, final: bool) -> None:
    """处理失败：还有重试机会就放回 queued，用尽了标记 failed。"""
    with session(url) as s:
        s.execute(update(Expense).where(Expense.id == expense_id).values(
            status=ExpenseStatus.FAILED if final else ExpenseStatus.QUEUED,
            error=error[:2000], updated_at=utcnow()))
        audit(s, expense_id, "failed" if final else "retry", error=error[:500])
