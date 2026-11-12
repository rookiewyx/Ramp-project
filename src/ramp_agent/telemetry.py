"""链路追踪（OpenTelemetry）。

一笔报销的完整链路：
    POST /v1/expenses（API）
      └─ apply_async ramp.review_expense ── 消息队列 ──► run ramp.review_expense（worker）
                                                          ├─ pipeline.acquire / pipeline.fetch_file
                                                          ├─ extraction（chat <视觉模型>）
                                                          ├─ policy.semantic（chat <文本模型>）
                                                          ├─ policy.rules
                                                          └─ pipeline.persist（SQL 语句）
API 和 worker 是两个进程，Celery 埋点会把链路上下文放进消息头，所以在 Jaeger 里是同一条链路。

模型调用的 span 按 OpenTelemetry GenAI 语义约定命名（chat <模型名>）和打属性
（gen_ai.request.model、gen_ai.usage.input_tokens……）。**不记录提示词和回答内容**：里面有员工填写的信息。

没开 OTEL_ENABLED 时，get_tracer 返回的是空实现，埋点几乎没有开销。
导出方式（OTEL_EXPORTER）：
    otlp  发给 Jaeger 等（docker compose 里的 Jaeger，界面在 http://127.0.0.1:16686）
    file  每个 span 一行 JSON，写到 OTEL_FILE，用 scripts/trace_report.py 汇总各环节耗时
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from opentelemetry import trace

from .config import ROOT, Settings, get_settings

tracer = trace.get_tracer("ramp_agent")
_lock = threading.Lock()
_configured = False


@contextmanager
def genai_span(model: str, operation: str = "chat", system: str = "openai") -> Iterator[trace.Span]:
    """一次模型调用。调用方拿到 span 后用 record_usage 记录 tokens。"""
    with tracer.start_as_current_span(f"{operation} {model}", kind=trace.SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.system", system)
        span.set_attribute("gen_ai.request.model", model)
        yield span


def record_usage(span: trace.Span, input_tokens: int, output_tokens: int) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens or 0))
    span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens or 0))


class FileSpanExporter:
    """每个 span 写一行 JSON。没装 Jaeger 时也能分析各环节耗时，压测时用。"""

    def __init__(self, path: str | Path):
        p = Path(path)
        self.path = p if p.is_absolute() else ROOT / p
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence) -> object:
        from opentelemetry.sdk.trace.export import SpanExportResult

        lines = []
        for s in spans:
            ctx = s.get_span_context()
            lines.append(json.dumps({
                "name": s.name, "service": s.resource.attributes.get("service.name"),
                "trace_id": f"{ctx.trace_id:032x}", "span_id": f"{ctx.span_id:016x}",
                "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                "kind": s.kind.name, "start": s.start_time, "end": s.end_time,
                "duration_ms": round((s.end_time - s.start_time) / 1e6, 3),
                "status": s.status.status_code.name,
                "attributes": {k: v for k, v in (s.attributes or {}).items()
                               if isinstance(v, (str, int, float, bool))},
            }, ensure_ascii=False))
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def setup_tracing(role: str, settings: Settings | None = None, app=None) -> bool:
    """初始化链路追踪。role：api / worker，会拼进服务名。重复调用无副作用。返回是否已启用。"""
    global _configured
    s = settings or get_settings()
    if not s.otel_enabled:
        return False
    with _lock:
        if not _configured:
            from opentelemetry.instrumentation.celery import CeleryInstrumentor
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=Resource.create(
                {"service.name": f"{s.otel_service_name}-{role}"}))
            if s.otel_exporter == "file":
                exporter = FileSpanExporter(s.otel_file)
            elif s.otel_exporter == "otlp":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=s.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces")
            else:
                raise ValueError(f"未知的 OTEL_EXPORTER：{s.otel_exporter}")
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            CeleryInstrumentor().instrument()          # 链路上下文随消息从 API 传到 worker

            from .service.db import get_engine

            instrument_engine(get_engine(s.database_url))
            _configured = True
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # 不记录每次读写请求体 / 响应体的小 span（http receive / send），它们只占数量不提供信息
        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz",
                                           exclude_spans=["receive", "send"])
    return True


class FreshTraceContext:
    """最外层的 ASGI 包装：每个请求从空的链路上下文开始。

    压测时发现：同一个长连接上，客户端发得快时，uvicorn 会在上一个请求发送响应的过程中
    就为下一个请求创建任务，新任务复制了当时的上下文，于是下一个请求被挂到了上一个请求的链路下面
    （约 8% 的请求“串号”）。请求头里带了 traceparent 的，埋点中间件仍会按请求头接上上游链路。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        from opentelemetry import context

        token = context.attach(context.Context())
        try:
            await self.app(scope, receive, send)
        finally:
            context.detach(token)


def instrument_engine(engine) -> None:
    """每条 SQL 一个 span。

    没用官方的 opentelemetry-instrumentation-sqlalchemy：它目前只支持 SQLAlchemy < 2.1，
    装了 2.1 时会静默跳过，链路里一条 SQL 都看不到（压测时发现的）。这里直接挂 SQLAlchemy 的事件。
    只记录带占位符的语句，不记录参数值（参数里可能有员工填写的内容）。
    """
    from sqlalchemy import event

    if getattr(engine, "_ramp_traced", False):
        return
    engine._ramp_traced = True
    system = engine.dialect.name

    @event.listens_for(engine, "before_cursor_execute")
    def _start(conn, cursor, statement, params, context, executemany):
        op = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else "SQL"
        span = tracer.start_span(f"{op} {system}", kind=trace.SpanKind.CLIENT)
        span.set_attribute("db.system", system)
        span.set_attribute("db.operation", op)
        span.set_attribute("db.statement", " ".join(statement.split())[:300])
        conn.info.setdefault("_ramp_spans", []).append(span)

    def _end(conn, error=None):
        spans = conn.info.get("_ramp_spans")
        if spans:
            span = spans.pop()
            if error is not None:
                span.record_exception(error)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
            span.end()

    @event.listens_for(engine, "after_cursor_execute")
    def _after(conn, cursor, statement, params, context, executemany):
        _end(conn)

    @event.listens_for(engine, "handle_error")
    def _error(ctx):
        if ctx.connection is not None:
            _end(ctx.connection, ctx.original_exception)


def shutdown_tracing() -> None:
    """进程退出前把缓冲里的 span 发出去。"""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
