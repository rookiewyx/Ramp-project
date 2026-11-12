"""审核服务 HTTP 接口。

  POST /v1/expenses                    提交报销（表单字段 + 发票文件），立即返回受理编号，审核在后台进行
  GET  /v1/expenses/{id}               查询状态、结论、逐条依据
  GET  /v1/expenses                    列表；财务待办：?decision=review&pending=true
  POST /v1/expenses/{id}/override      财务复核：给出最终结论（推翻系统结论时必须写理由）
  POST /v1/expenses/{id}/rereview      重新审核（制度更新后、失败的单子）
  GET  /v1/expenses/{id}/audit         审计日志
  GET  /v1/expenses/{id}/file          发票原件（工作台预览用）
  GET  /v1/stats                       自动通过率、转人工率、人机一致率
  GET  /workbench                      财务审核工作台（网页）
  GET  /healthz  /readyz               存活 / 就绪检查

提交接口支持 Idempotency-Key：客户端超时重试时带同一个键，只受理一次；同一个键换了内容返回 409。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Security,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError

from ..config import Settings, get_settings
from .db import AuditEvent, Expense, ExpenseStatus, Override, audit, session, utcnow
from .migrate import ensure_schema
from .storage import CONTENT_TYPES, ObjectStore, make_store, object_key, sniff

STATIC = Path(__file__).parent / "static"
CATEGORIES = ("hotel", "meal", "entertainment", "taxi", "train", "flight", "conference", "other")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DecisionT = Literal["approve", "review", "reject"]
CategoryT = Literal["hotel", "meal", "entertainment", "taxi", "train", "flight", "conference", "other"]
# 用安全方案声明 API Key，接口文档页右上角会出现 Authorize 按钮
API_KEY = APIKeyHeader(name="X-API-Key", auto_error=False, description="在 .env 的 API_KEYS 里配置")


# ---------------- 输出结构 ----------------
class RuleOut(BaseModel):
    rule: str
    status: str
    reason: str
    clause: str
    by: str


class ReviewOut(BaseModel):
    decision: str
    explanation: str
    results: list[RuleOut]
    extraction_tier: str
    uncertain_fields: list[str]
    invoice_fields: dict = Field(default_factory=dict)     # 从发票上识别出的字段
    llm_calls: int
    vlm_calls: int
    tokens: int
    latency_ms: int
    versions: dict
    created_at: datetime


class OverrideOut(BaseModel):
    reviewer: str
    decision: str
    note: str
    created_at: datetime


class ExpenseOut(BaseModel):
    id: str
    status: str
    decision: str | None
    final_decision: str | None
    category: str
    amount: str
    employee_id: str
    employee_name: str
    submit_date: str
    note: str
    has_attachment: bool
    file_name: str | None = None
    file_type: str | None = None
    claim: dict = Field(default_factory=dict)              # 报销单上的其他字段（城市、出差日期、人数……）
    attempts: int
    error: str | None
    created_at: datetime
    updated_at: datetime
    review: ReviewOut | None = None
    overrides: list[OverrideOut] = Field(default_factory=list)


class OverrideIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    decision: DecisionT
    note: str = Field(default="", max_length=2000)


def expense_out(e: Expense, detail: bool = True) -> ExpenseOut:
    out = ExpenseOut(id=e.id, status=e.status, decision=e.decision, final_decision=e.final_decision,
                     category=e.category, amount=f"{e.amount:.2f}", employee_id=e.employee_id,
                     employee_name=e.employee_name, submit_date=e.submit_date, note=e.note,
                     has_attachment=e.file_key is not None, file_name=e.file_name,
                     file_type=(e.file_key.rsplit(".", 1)[-1] if e.file_key else None),
                     attempts=e.attempts, error=e.error, created_at=e.created_at, updated_at=e.updated_at)
    if detail:
        out.claim = {"report_id": e.report_id, "city": e.city, "trip_start": e.trip_start,
                     "trip_end": e.trip_end, "nights": e.nights, "headcount": e.headcount,
                     "preapproval_id": e.preapproval_id}
    if detail and e.reviews:
        r = e.reviews[-1]
        out.review = ReviewOut(
            decision=r.decision, explanation=r.explanation, results=[RuleOut(**x) for x in r.results],
            extraction_tier=r.extraction.get("tier", "none"),
            uncertain_fields=r.extraction.get("uncertain_fields", []),
            invoice_fields=r.extraction.get("fields") or {}, llm_calls=r.llm_calls,
            vlm_calls=r.vlm_calls, tokens=r.tokens, latency_ms=r.latency_ms, versions=r.versions,
            created_at=r.created_at)
    if detail:
        out.overrides = [OverrideOut(reviewer=o.reviewer, decision=o.decision, note=o.note,
                                     created_at=o.created_at) for o in e.overrides]
    return out


# ---------------- 应用 ----------------
def create_app(settings: Settings | None = None, store: ObjectStore | None = None,
               enqueue=None) -> FastAPI:
    """settings / store / enqueue 可以注入，测试时换成临时目录和同步执行。"""
    s = settings or get_settings()
    ensure_schema(s.database_url, s.auto_migrate)
    store = store or make_store(s)
    if enqueue is None:
        from .tasks import enqueue
    api_keys = {k.strip() for k in s.api_keys.split(",") if k.strip()}
    db = s.database_url

    app = FastAPI(
        title="RAMP 报销审核服务", version="0.1.0",
        description="员工提交报销单和发票，系统自动识别发票、对照公司制度审核，给出"
                    "**建议通过 / 需人工审核 / 建议驳回**，并引用制度条款说明理由；财务对转人工的单子做最终复核。\n\n"
                    "试用方法：点开一个接口 → 右上角 **Try it out** → 填参数 → **Execute**，下方会显示返回结果。"
                    "开启了鉴权时，先点页面右上角 **Authorize** 填入 API Key。",
        openapi_tags=[
            {"name": "1. 员工：提交与查询", "description": "提交报销、查看审核进度和结论"},
            {"name": "2. 财务：复核", "description": "待办列表、给出最终结论、查看审计日志、重新审核"},
            {"name": "3. 运营指标", "description": "自动通过率、转人工率、人机一致率"},
            {"name": "4. 运维", "description": "健康检查，给负载均衡和容器编排用"},
        ])
    EMP, FIN, OPS, SYS = "1. 员工：提交与查询", "2. 财务：复核", "3. 运营指标", "4. 运维"

    app.add_middleware(RequestIdMiddleware)

    def auth(x_api_key: Annotated[str | None, Security(API_KEY)] = None) -> None:
        if api_keys and x_api_key not in api_keys:
            raise HTTPException(401, "缺少或错误的 X-API-Key")

    def load(expense_id: str) -> Expense:
        with session(db) as sess:
            e = sess.get(Expense, expense_id)
            if e is None:
                raise HTTPException(404, "报销单不存在")
            _ = e.reviews, e.overrides            # 在会话内加载关联数据
            return e

    # ---------- 提交 ----------
    # 用普通 def 而不是 async def：里面全是同步的数据库、文件、Redis 调用，写成 async def 会堵住事件循环，
    # 压测时所有查询请求都要排在它后面。普通 def 会被放进线程池执行。
    @app.post("/v1/expenses", status_code=202, dependencies=[Depends(auth)], tags=[EMP],
              summary="提交报销",
              description="填写报销单并上传发票（PDF / JPG / PNG）。接口立即返回受理编号 `id`，"
                          "审核在后台进行；用“查询报销单”看结果。不传发票会被按 R9 驳回。")
    def submit(
        category: Annotated[CategoryT, Form(description="费用类别：hotel 住宿 / meal 餐饮 / entertainment 业务招待 / taxi 市内打车 / train 火车 / flight 飞机 / conference 会议 / other 其他")],
        amount: Annotated[str, Form(description="报销金额，元，最多两位小数，如 88.00")],
        submit_date: Annotated[str, Form(description="提交日期，YYYY-MM-DD")],
        employee_id: Annotated[str, Form(max_length=64, description="员工工号")] = "",
        employee_name: Annotated[str, Form(max_length=64, description="员工姓名（只用于脱敏，不会发给大模型）")] = "",
        report_id: Annotated[str, Form(max_length=64, description="报销单号（一张报销单可以有多笔费用）")] = "",
        note: Annotated[str, Form(max_length=2000, description="报销说明，如打车原因、招待人数")] = "",
        city: Annotated[str | None, Form(max_length=64, description="出差城市（差旅类必填）")] = None,
        trip_start: Annotated[str | None, Form(description="出差开始日期，YYYY-MM-DD")] = None,
        trip_end: Annotated[str | None, Form(description="出差结束日期，YYYY-MM-DD")] = None,
        nights: Annotated[int | None, Form(ge=0, le=365, description="住宿晚数")] = None,
        headcount: Annotated[int | None, Form(ge=0, le=1000, description="招待人数（含我方）")] = None,
        preapproval_id: Annotated[str | None, Form(max_length=64, description="预审批编号（超过 5000 元需要）")] = None,
        file: Annotated[UploadFile | None, File(description="发票文件")] = None,
        idempotency_key: Annotated[str | None, Header(max_length=128,
                                                      description="幂等键：重试同一次提交时带同一个值，只受理一次")] = None,
    ):
        if category not in CATEGORIES:
            raise HTTPException(422, f"category 必须是 {', '.join(CATEGORIES)} 之一")
        try:
            amt = Decimal(amount)
        except InvalidOperation:
            raise HTTPException(422, "amount 不是合法金额") from None
        if amt <= 0 or amt != amt.quantize(Decimal("0.01")) or amt >= Decimal("1e10"):
            raise HTTPException(422, "amount 必须大于 0，最多两位小数")
        for name, v in (("submit_date", submit_date), ("trip_start", trip_start), ("trip_end", trip_end)):
            if v and not DATE_RE.match(v):
                raise HTTPException(422, f"{name} 格式应为 YYYY-MM-DD")

        data = b""
        suffix = None
        if file is not None and file.filename:
            limit = s.max_upload_mb * 1024 * 1024
            data = file.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"发票文件超过 {s.max_upload_mb} MB")
            suffix = sniff(data)
            if suffix is None:
                raise HTTPException(415, "只支持 PDF、JPG、PNG 格式的发票")

        fields = {"category": category, "amount": str(amt), "submit_date": submit_date,
                  "employee_id": employee_id, "employee_name": employee_name, "report_id": report_id,
                  "note": note, "city": city or None, "trip_start": trip_start or None,
                  "trip_end": trip_end or None, "nights": nights, "headcount": headcount,
                  "preapproval_id": preapproval_id or None}
        key = digest = None
        if data:
            key, digest = object_key(data, suffix)
        req_hash = hashlib.sha256(json.dumps({**fields, "file": digest}, sort_keys=True,
                                             ensure_ascii=False).encode()).hexdigest()

        if idempotency_key:
            existing = _by_idempotency(db, idempotency_key)
            if existing is not None:
                return _replay(existing, req_hash)

        if data:
            store.put(key, data, CONTENT_TYPES[suffix])      # 先存文件再落库：落库失败只会多一个孤立文件
        e = Expense(**{k: v for k, v in fields.items() if k != "amount"}, amount=amt,
                    idempotency_key=idempotency_key, request_hash=req_hash, file_key=key,
                    file_name=(file.filename[:255] if data else None), file_sha256=digest,
                    file_size=len(data) or None)
        try:
            with session(db) as sess:
                sess.add(e)
                sess.flush()
                audit(sess, e.id, "submitted", actor=employee_id or "anonymous",
                      has_attachment=bool(data), file_sha256=digest)
        except IntegrityError:                    # 并发的两次重试撞上同一个幂等键
            existing = _by_idempotency(db, idempotency_key)
            if existing is None:
                raise
            return _replay(existing, req_hash)

        queued = enqueue(e.id, e.file_key)                    # 投递失败也已受理：单子留在 queued，由 requeue 补投
        out = expense_out(load(e.id))
        return JSONResponse(status_code=202, content={**json.loads(out.model_dump_json()),
                                                      "enqueued": queued})

    # ---------- 查询 ----------
    @app.get("/v1/expenses/{expense_id}", response_model=ExpenseOut, dependencies=[Depends(auth)],
             tags=[EMP], summary="查询报销单",
             description="填提交时返回的 `id`。`status` 是处理进度（queued 排队 / processing 审核中 / done 完成 / "
                         "failed 失败），`decision` 是系统结论，`review.results` 逐条列出每个条款的判断和依据。")
    def get_expense(expense_id: str):
        return expense_out(load(expense_id))

    @app.get("/v1/expenses", response_model=list[ExpenseOut], dependencies=[Depends(auth)],
             tags=[FIN], summary="报销单列表 / 财务待办",
             description="财务待办：`decision=review` 且 `pending=true`，即系统转人工、财务还没复核的单子。")
    def list_expenses(status: str | None = None, decision: str | None = None,
                      pending: Annotated[bool, Query(description="只看还没有财务复核的")] = False,
                      reviewed: Annotated[bool, Query(description="只看财务已经复核过的")] = False,
                      employee_id: str | None = None,
                      limit: Annotated[int, Query(ge=1, le=200)] = 50,
                      offset: Annotated[int, Query(ge=0)] = 0):
        q = select(Expense).order_by(Expense.created_at.desc()).limit(limit).offset(offset)
        if status:
            q = q.where(Expense.status == status)
        if decision:
            q = q.where(Expense.decision == decision)
        if pending:
            q = q.where(Expense.final_decision.is_(None), Expense.status == ExpenseStatus.DONE)
        if reviewed:
            q = q.where(Expense.final_decision.is_not(None))
        if employee_id:
            q = q.where(Expense.employee_id == employee_id)
        with session(db) as sess:
            return [expense_out(e, detail=False) for e in sess.scalars(q)]

    @app.get("/v1/policy", dependencies=[Depends(auth)], tags=[FIN], summary="当前制度",
             description="公司主体信息和每个条款的标题、原文、违规结论。工作台用来展示依据。")
    def get_policy():
        from ..policy import load_policy

        p = load_policy()
        return {"company": p.company.model_dump(),
                "rules": {k: {"title": r.title, "clause": r.clause, "outcome": r.outcome}
                          for k, r in p.rules.items()}}

    @app.get("/v1/expenses/{expense_id}/file", dependencies=[Depends(auth)], tags=[FIN],
             summary="发票原件", description="返回员工上传的发票文件，工作台用来预览。")
    def get_file(expense_id: str):
        e = load(expense_id)
        if e.file_key is None:
            raise HTTPException(404, "这笔报销没有附发票")
        suffix = "." + e.file_key.rsplit(".", 1)[-1]
        return Response(content=store.get(e.file_key), media_type=CONTENT_TYPES.get(suffix, "application/octet-stream"),
                        headers={"Content-Disposition": "inline", "Cache-Control": "private, max-age=3600",
                                 "X-Content-Type-Options": "nosniff"})

    @app.get("/v1/expenses/{expense_id}/audit", dependencies=[Depends(auth)], tags=[FIN],
             summary="审计日志", description="这笔报销的全部操作记录：谁、什么时候、做了什么。")
    def get_audit(expense_id: str):
        load(expense_id)
        with session(db) as sess:
            rows = sess.scalars(select(AuditEvent).where(AuditEvent.expense_id == expense_id)
                                .order_by(AuditEvent.id))
            return [{"event": r.event, "actor": r.actor, "detail": r.detail, "at": r.at} for r in rows]

    # ---------- 财务复核 ----------
    @app.post("/v1/expenses/{expense_id}/override", response_model=ExpenseOut,
              dependencies=[Depends(auth)], tags=[FIN], summary="财务复核（给出最终结论）",
              description="decision 填 approve / review / reject。和系统结论不同时，note 必须写明理由。")
    def override(expense_id: str, body: OverrideIn):
        e = load(expense_id)
        if e.status != ExpenseStatus.DONE:
            raise HTTPException(409, f"报销单状态为 {e.status}，审核完成后才能复核")
        if body.decision != e.decision and not body.note.strip():
            raise HTTPException(422, "推翻系统结论时必须写明理由")
        with session(db) as sess:
            sess.add(Override(expense_id=expense_id, reviewer=body.reviewer, decision=body.decision,
                              note=body.note))
            sess.execute(update(Expense).where(Expense.id == expense_id)
                         .values(final_decision=body.decision, updated_at=utcnow()))
            audit(sess, expense_id, "overridden", actor=body.reviewer, system=e.decision,
                  final=body.decision, agree=body.decision == e.decision)
        return expense_out(load(expense_id))

    @app.post("/v1/expenses/{expense_id}/rereview", status_code=202, response_model=ExpenseOut,
              dependencies=[Depends(auth)], tags=[FIN], summary="重新审核",
              description="制度更新后，或审核失败的单子，重新跑一遍审核。")
    def rereview(expense_id: str, actor: str = "system"):
        with session(db) as sess:
            res = sess.execute(update(Expense).where(
                Expense.id == expense_id,
                Expense.status.in_([ExpenseStatus.DONE, ExpenseStatus.FAILED]))
                .values(status=ExpenseStatus.QUEUED, attempts=0, error=None, updated_at=utcnow()))
            if res.rowcount == 0:
                load(expense_id)                          # 不存在时返回 404
                raise HTTPException(409, "只有审核完成或失败的单子可以重新审核")
            audit(sess, expense_id, "rereview_requested", actor=actor)
        e = load(expense_id)
        enqueue(expense_id, e.file_key)
        return expense_out(load(expense_id))

    # ---------- 运营指标 ----------
    @app.get("/v1/stats", dependencies=[Depends(auth)], tags=[OPS], summary="运营指标",
             description="各结论的数量、自动通过率、转人工率；人机一致率（系统建议通过或驳回、财务复核后结论相同的比例）；转人工的单子财务最后怎么定的。")
    def stats():
        with session(db) as sess:
            by_status = dict(sess.execute(select(Expense.status, func.count()).group_by(Expense.status)).all())
            by_decision = dict(sess.execute(select(Expense.decision, func.count())
                                            .where(Expense.status == ExpenseStatus.DONE)
                                            .group_by(Expense.decision)).all())
            reviewed = sess.execute(select(Expense.decision, Expense.final_decision)
                                    .where(Expense.final_decision.is_not(None))).all()
        done = sum(by_decision.values())
        # 人机一致率只看系统下了定论（通过 / 驳回）的单子：转人工本来就是让人来定，不存在“一致”与否
        decided = [(d, f) for d, f in reviewed if d in ("approve", "reject")]
        agree = sum(1 for d, f in decided if d == f)
        resolved = {}
        for d, f in reviewed:
            if d == "review":
                resolved[f] = resolved.get(f, 0) + 1
        return {
            "by_status": by_status, "by_decision": by_decision,
            "auto_approve_rate": round(by_decision.get("approve", 0) / done, 4) if done else None,
            "human_review_rate": round(by_decision.get("review", 0) / done, 4) if done else None,
            "overridden": len(reviewed),
            # 人机一致率：系统建议通过 / 驳回、财务也复核过的单子里，最终结论与系统相同的比例
            "human_agreement_rate": round(agree / len(decided), 4) if decided else None,
            "agreement_base": len(decided),
            # 转人工的单子，财务最后怎么定的
            "review_resolved": resolved,
        }

    # ---------- 健康检查 ----------
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/workbench")

    @app.get("/workbench", include_in_schema=False)
    def workbench():
        # 页面本身不需要鉴权，页面里的接口调用会带上 X-API-Key
        return FileResponse(STATIC / "workbench.html", headers={"Cache-Control": "no-cache"})

    @app.get("/healthz", tags=[SYS], summary="存活检查")
    def healthz():
        return {"ok": True}

    @app.get("/readyz", tags=[SYS], summary="就绪检查（数据库、存储、队列是否可用）")
    def readyz():
        checks = {}
        try:
            with session(db) as sess:
                sess.execute(text("SELECT 1"))
            checks["database"] = True
        except Exception:  # noqa: BLE001
            checks["database"] = False
        try:
            checks["storage"] = bool(store.ping())
        except Exception:  # noqa: BLE001
            checks["storage"] = False
        if not s.task_eager:
            try:
                import redis

                checks["broker"] = bool(redis.Redis.from_url(s.redis_url, socket_timeout=2).ping())
            except Exception:  # noqa: BLE001
                checks["broker"] = False
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"ok": ok, **checks})

    from ..telemetry import setup_tracing

    setup_tracing("api", s, app)
    return app


class RequestIdMiddleware:
    """给每个请求一个 X-Request-ID，写进日志上下文和响应头。

    用纯 ASGI 中间件，不用 @app.middleware("http")：后者（BaseHTTPMiddleware）在另一个任务里读请求体，
    压测时发现会让链路上下文串到同一个长连接的下一个请求上（几笔报销被记成同一条链路）。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        hdrs = dict(scope.get("headers") or [])
        rid = (hdrs.get(b"x-request-id") or b"").decode()[:64] or uuid.uuid4().hex[:16]

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        with logger.contextualize(request_id=rid):
            await self.app(scope, receive, send_with_id)


def _by_idempotency(db: str, key: str | None) -> Expense | None:
    if not key:
        return None
    with session(db) as sess:
        e = sess.scalar(select(Expense).where(Expense.idempotency_key == key))
        if e is not None:
            _ = e.reviews, e.overrides
        return e


def _replay(e: Expense, req_hash: str):
    if e.request_hash != req_hash:
        raise HTTPException(409, "同一个 Idempotency-Key 已用于内容不同的提交")
    out = expense_out(e)
    return JSONResponse(status_code=200, content={**json.loads(out.model_dump_json()), "replayed": True})
