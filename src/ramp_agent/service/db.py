"""业务库：报销单、审核记录、财务复核、审计日志、发票登记。

和评测平台（evalhub）分开建库：评测数据是实验用的，业务数据是要留存审计的，生命周期不同。
默认 SQLite，换成 MySQL 只需改 DATABASE_URL。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from ..config import ROOT


# MySQL 的 DATETIME 默认精确到秒，同一秒内的事件分不出先后；指定到微秒
Timestamp = DateTime().with_variant(mysql.DATETIME(fsp=6), "mysql")


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)     # 统一存 UTC，不带时区，MySQL / SQLite 行为一致


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    # 约束名固定下来：迁移脚本在 MySQL / PostgreSQL / SQLite 上生成的名字一致，以后改约束才找得到
    metadata = MetaData(naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })


class ExpenseStatus:
    QUEUED = "queued"           # 已受理，等待审核
    PROCESSING = "processing"   # worker 正在审核
    DONE = "done"               # 审核完成（结论可能是通过 / 转人工 / 驳回）
    FAILED = "failed"           # 重试用尽仍失败，需要运维介入


class AgentStatus:
    """调查 Agent 的进度（Expense.agent_status）。没调查过为空。"""
    AWAITING = "awaiting_employee"   # 已向员工追问，等答复；这期间结论仍是转人工，财务也可以直接复核
    REPLIED = "replied"              # 员工已答复，排队继续调查
    EXPIRED = "expired"              # 等答复超时，排队收尾（按没有答复处理）
    RESOLVED = "resolved"            # 调查后有了定论（通过 / 驳回）
    UNRESOLVED = "unresolved"        # 调查后仍需人工
    ERROR = "error"                  # 调查出错，保持初审结论


class Expense(Base):
    __tablename__ = "expenses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # 幂等键：客户端重试同一次提交时带同一个键，只会受理一次
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    request_hash: Mapped[str] = mapped_column(String(64), default="")   # 同一个幂等键必须对应同样的内容
    report_id: Mapped[str] = mapped_column(String(64), default="")
    employee_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    employee_name: Mapped[str] = mapped_column(String(64), default="")
    category: Mapped[str] = mapped_column(String(32))
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))          # 金额用定点数，不用浮点
    submit_date: Mapped[str] = mapped_column(String(10))
    note: Mapped[str] = mapped_column(Text, default="")
    city: Mapped[str | None] = mapped_column(String(64))
    trip_start: Mapped[str | None] = mapped_column(String(10))
    trip_end: Mapped[str | None] = mapped_column(String(10))
    nights: Mapped[int | None] = mapped_column(Integer)
    headcount: Mapped[int | None] = mapped_column(Integer)
    preapproval_id: Mapped[str | None] = mapped_column(String(64))

    file_key: Mapped[str | None] = mapped_column(String(255))       # 对象存储里的位置；为空表示没附发票
    file_name: Mapped[str | None] = mapped_column(String(255))
    file_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    file_size: Mapped[int | None] = mapped_column(Integer)

    status: Mapped[str] = mapped_column(String(16), default=ExpenseStatus.QUEUED)
    decision: Mapped[str | None] = mapped_column(String(16))         # 系统结论
    final_decision: Mapped[str | None] = mapped_column(String(16))   # 财务复核后的最终结论
    agent_status: Mapped[str | None] = mapped_column(String(24))     # 调查 Agent 的进度，见 AgentStatus
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow, onupdate=utcnow)

    reviews: Mapped[list[ReviewRecord]] = relationship(
        back_populates="expense", order_by="ReviewRecord.created_at", cascade="all, delete-orphan")
    overrides: Mapped[list[Override]] = relationship(
        back_populates="expense", order_by="Override.created_at", cascade="all, delete-orphan")
    investigations: Mapped[list[Investigation]] = relationship(
        back_populates="expense", order_by="Investigation.id", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_expenses_status_decision", "status", "decision"),)


class ReviewRecord(Base):
    """一次审核的完整记录：结论、逐条依据、识别结果、成本，以及当时的代码 / 提示词 / 制度版本。"""

    __tablename__ = "review_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expense_id: Mapped[str] = mapped_column(ForeignKey("expenses.id"), index=True)
    decision: Mapped[str] = mapped_column(String(16))
    results: Mapped[list] = mapped_column(JSON)
    explanation: Mapped[str] = mapped_column(Text)
    extraction: Mapped[dict] = mapped_column(JSON)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    vlm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    versions: Mapped[dict] = mapped_column(JSON)       # code / prompt / policy 哈希，用于追溯
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)

    expense: Mapped[Expense] = relationship(back_populates="reviews")


class Override(Base):
    """财务复核：人对系统结论的最终裁决。也是之后改进系统的数据来源。"""

    __tablename__ = "overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expense_id: Mapped[str] = mapped_column(ForeignKey("expenses.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)

    expense: Mapped[Expense] = relationship(back_populates="overrides")


class Investigation(Base):
    """调查 Agent 的一次调查：查了什么、找到什么、问了员工什么，以及调查前后的结论。

    追问员工时调查会暂停：state 里存着对话和查到的记录，员工答复后从这里继续，不重复已做过的查询。
    """

    __tablename__ = "investigations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expense_id: Mapped[str] = mapped_column(ForeignKey("expenses.id"), index=True)
    status: Mapped[str] = mapped_column(String(24))
    decision_before: Mapped[str] = mapped_column(String(16))
    decision_after: Mapped[str | None] = mapped_column(String(16))
    question: Mapped[str | None] = mapped_column(Text)
    reply: Mapped[str | None] = mapped_column(Text)
    asked_at: Mapped[datetime | None] = mapped_column(Timestamp)
    replied_at: Mapped[datetime | None] = mapped_column(Timestamp)
    result: Mapped[dict] = mapped_column(JSON)          # 调查过程和证据（InvestigationResult.to_dict）
    state: Mapped[dict | None] = mapped_column(JSON)    # 暂停时的断点；调查结束后清空
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow, onupdate=utcnow)

    expense: Mapped[Expense] = relationship(back_populates="investigations")


class AuditEvent(Base):
    """审计日志：只追加，不修改。谁在什么时候对这笔报销做了什么。"""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expense_id: Mapped[str] = mapped_column(String(32), index=True)
    event: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(64), default="system")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class InvoiceClaim(Base):
    """发票登记（R6 查重）：发票号是主键，靠数据库唯一约束保证并发提交也不会重复登记。"""

    __tablename__ = "invoice_claims"

    invoice_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    expense_id: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


def audit(s: Session, expense_id: str, event_name: str, actor: str = "system", **detail) -> None:
    s.add(AuditEvent(expense_id=expense_id, event=event_name, actor=actor, detail=detail))


# ---------------- 连接 ----------------
def resolve_url(url: str) -> str:
    """SQLite 的相对路径按项目根目录解析，不受启动目录影响。"""
    prefix = "sqlite:///"
    if url.startswith(prefix) and not url.startswith(prefix + "/") and ":memory:" not in url:
        path = Path(url[len(prefix):])
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return prefix + path.as_posix()
    return url


@lru_cache
def get_engine(url: str) -> Engine:
    url = resolve_url(url)
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _pragma(conn, _):                      # WAL：API 和 worker 同时读写 SQLite 不互相锁死
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    return engine


@lru_cache
def _factory(url: str) -> sessionmaker[Session]:
    return sessionmaker(get_engine(url), expire_on_commit=False)


@contextmanager
def session(url: str) -> Iterator[Session]:
    s = _factory(url)()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
