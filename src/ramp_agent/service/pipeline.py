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

from ..config import Settings
from ..extraction import ExtractionResult, InvoiceExtractor, Tier
from ..policy import Claim, InvoiceRegistry, PolicyReviewer, SemanticJudge, load_policy
from ..telemetry import tracer
from .db import Expense, ExpenseStatus, ReviewRecord, audit, session, utcnow
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
        audit(s, expense_id, "review_started", attempt=e.attempts)
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
    latency = int((time.perf_counter() - t0) * 1000)

    ext_info = {"tier": ext.tier.value, "fields": ext.fields.model_dump(),
                "uncertain_fields": ext.uncertain_fields, "checks_failed": ext.checks_failed,
                "error": ext.error, "latency_ms": ext.latency_ms} if ext else {"tier": "none"}
    with tracer.start_as_current_span("pipeline.persist"), session(url) as s:
        s.add(ReviewRecord(
            expense_id=expense_id, decision=rv.decision.value,
            results=[r.model_dump(mode="json") for r in rv.results], explanation=rv.explanation,
            extraction=ext_info, llm_calls=rv.llm_calls, vlm_calls=ext.model_calls if ext else 0,
            tokens=rv.prompt_tokens + rv.completion_tokens
            + ((ext.prompt_tokens + ext.completion_tokens) if ext else 0),
            latency_ms=latency, versions=deps.versions))
        s.execute(update(Expense).where(Expense.id == expense_id).values(
            status=ExpenseStatus.DONE, decision=rv.decision.value, error=None, updated_at=utcnow()))
        audit(s, expense_id, "reviewed", decision=rv.decision.value,
              violated=rv.violated_rules, uncertain=rv.uncertain_rules, latency_ms=latency)
    # 日志里只记 ID 和结论，不记报销说明等可能含个人信息的内容
    logger.info("审核完成 {} → {}（{} ms，大模型 {} 次）", expense_id, rv.decision.value, latency, rv.llm_calls)
    span.set_attribute("pipeline.outcome", "done")
    span.set_attribute("review.decision", rv.decision.value)
    return "done"


def release(url: str, expense_id: str, error: str, final: bool) -> None:
    """处理失败：还有重试机会就放回 queued，用尽了标记 failed。"""
    with session(url) as s:
        s.execute(update(Expense).where(Expense.id == expense_id).values(
            status=ExpenseStatus.FAILED if final else ExpenseStatus.QUEUED,
            error=error[:2000], updated_at=utcnow()))
        audit(s, expense_id, "failed" if final else "retry", error=error[:500])
