"""任务队列（Celery + Redis）。

可靠性配置：
  - acks_late + reject_on_worker_lost：任务做完才确认；worker 中途崩溃，消息会重新投递，
    配合流水线里的条件领取，不会丢单也不会重复审核。
  - prefetch=1：每个 worker 一次只拿一条，审核慢的单子不会把后面的堵在同一个 worker 里。
  - 指数退避重试：模型故障、数据库抖动都按 5s、10s、20s… 重试，用尽后标记 failed。
  - 两个队列：拍照件进 vlm 队列，文字版 PDF 和无附件的进 default 队列，分开的 worker 处理，
    视觉模型排队时不会拖慢秒级就能审完的单子（压测发现的问题，见 docs/perf_log.md）。
  - 保护模型接口靠并发上限（VLM/LLM_MAX_CONCURRENCY），不靠任务限速：
    任务限速会把不调模型的单子一起卡住，压测时 60/m 的默认值让端到端 P50 变成 18 秒。

本地开发不想起 Redis：TASK_EAGER=true，任务在 API 进程里同步执行。
"""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from loguru import logger
from sqlalchemy import or_, select, update

from ..config import get_settings
from .db import AgentStatus, Expense, ExpenseStatus, Investigation, session, utcnow
from .pipeline import Deps, process, release

settings = get_settings()

app = Celery("ramp", broker=settings.redis_url)
app.conf.update(
    task_always_eager=settings.task_eager,
    task_eager_propagates=False,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": settings.stale_after_s * 2},
    task_default_queue="default",
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
)


@worker_process_init.connect
def _init_process(**_) -> None:
    """prefork 池：每个子进程 fork 出来后各自初始化链路追踪（导出线程不能跨 fork 继承）。"""
    from ..telemetry import setup_tracing

    setup_tracing("worker")


@worker_process_shutdown.connect
def _shutdown_process(**_) -> None:
    from ..telemetry import shutdown_tracing

    shutdown_tracing()


@lru_cache
def deps() -> Deps:
    """每个 worker 进程组装一次依赖（模型客户端、存储、登记表）。"""
    from ..agent import FixtureSystems, Investigator
    from ..config import ROOT
    from ..evalhub.runner import code_hash, policy_hash, prompt_hash
    from ..extraction import InvoiceExtractor, OpenAICompatibleVLM
    from ..policy import LLMJudge
    from .registry import RedisRegistry, SqlRegistry
    from .storage import make_store

    s = get_settings()
    registry = RedisRegistry(s.redis_url) if s.registry_backend == "redis" else SqlRegistry(s.database_url)
    vlm = OpenAICompatibleVLM() if s.vlm_base_url else None
    judge = LLMJudge() if s.llm_api_key else None
    if judge is None:
        logger.warning("未配置 LLM_API_KEY：需要语义判断的条款会转人工")
    investigator = Investigator.from_settings(s) if (s.llm_api_key and s.agent_enabled) else None
    fixture = s.enterprise_fixture
    if fixture and not Path(fixture).is_absolute():
        fixture = str(ROOT / fixture)
    systems = FixtureSystems.from_file(fixture)
    versions = {"code": code_hash()[:12], "prompt": prompt_hash()[:12], "policy": policy_hash()[:12],
                "llm": judge.model if judge else None, "vlm": vlm.model if vlm else None,
                "agent": investigator.model if investigator else None}
    return Deps(settings=s, store=make_store(s), registry=registry,
                extractor=InvoiceExtractor(vlm), judge=judge, versions=versions,
                investigator=investigator, systems=systems)


@app.task(bind=True, name="ramp.review_expense", max_retries=settings.task_max_retries,
          rate_limit=settings.task_rate_limit or None)
def review_expense(self, expense_id: str) -> str:
    final = self.request.retries >= (self.max_retries or 0)
    d = deps()
    try:
        return process(d, expense_id, final_attempt=final)
    except Exception as exc:
        logger.exception("审核失败 {}（第 {} 次）", expense_id, self.request.retries + 1)
        release(d.settings.database_url, expense_id, f"{type(exc).__name__}: {exc}", final=final)
        if final:
            return "failed"
        raise self.retry(exc=exc, countdown=5 * 2 ** self.request.retries) from exc


QUEUE_FAST, QUEUE_VLM = "default", "vlm"


def queue_for(file_key: str | None) -> str:
    """拍照件要走视觉模型（几秒、受配额限制），放进单独的队列，由单独的 worker 处理；
    文字版 PDF 和没附发票的毫秒级就能审完，走默认队列，不会被排在拍照件后面（舱壁隔离）。
    没有文字层的扫描版 PDF 提交时分辨不出来，仍走默认队列，在默认 worker 里调视觉模型。"""
    return QUEUE_VLM if file_key and file_key.lower().endswith((".jpg", ".png")) else QUEUE_FAST


def enqueue(expense_id: str, file_key: str | None = None) -> bool:
    """投递审核任务。投递失败（Redis 不可用）返回 False，单子留在 queued，由 requeue 补投。"""
    try:
        review_expense.apply_async(args=[expense_id], queue=queue_for(file_key))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("投递任务失败 {}：{}", expense_id, exc)
        return False


def expire_waiting(timeout_h: int | None = None) -> int:
    """追问员工后超过 AGENT_REPLY_TIMEOUT_H 没有答复的：不再等，按没有答复收尾（仍转人工）。"""
    s = get_settings()
    cutoff = utcnow() - timedelta(hours=s.agent_reply_timeout_h if timeout_h is None else timeout_h)
    with session(s.database_url) as db:
        ids = db.scalars(select(Investigation.expense_id).where(
            Investigation.status == "awaiting_employee", Investigation.asked_at < cutoff)).all()
        moved = []
        for i in ids:
            res = db.execute(update(Expense).where(
                Expense.id == i, Expense.agent_status == AgentStatus.AWAITING, Expense.final_decision.is_(None))
                .values(status=ExpenseStatus.QUEUED, agent_status=AgentStatus.EXPIRED, updated_at=utcnow()))
            if res.rowcount:
                moved.append(i)
    for i in moved:
        enqueue(i)
    return len(moved)


def requeue(older_than_s: int = 60) -> int:
    """补投：受理了但一直没被处理的单子（投递失败、消息丢失）、处理中卡死的单子。"""
    s = get_settings()
    now = utcnow()
    with session(s.database_url) as db:
        rows = db.execute(select(Expense.id, Expense.file_key).where(or_(
            (Expense.status == ExpenseStatus.QUEUED)
            & (Expense.updated_at < now - timedelta(seconds=older_than_s)),
            (Expense.status == ExpenseStatus.PROCESSING)
            & (Expense.updated_at < now - timedelta(seconds=s.stale_after_s))))).all()
    return sum(enqueue(i, k) for i, k in rows)
