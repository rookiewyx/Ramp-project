"""审核服务：接口、流水线、幂等、查重、重试、脱敏。

全部在本地跑：SQLite 临时库 + 本地目录存储 + 同步执行任务 + 假的大模型，不需要 Redis 和密钥。
"""

from __future__ import annotations

import json
import threading
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from sqlalchemy import update

from ramp_agent.config import Settings
from ramp_agent.extraction import InvoiceExtractor
from ramp_agent.pii import mask
from ramp_agent.policy import LLMJudge, SemanticRequest, SemanticResult
from ramp_agent.service.api import create_app
from ramp_agent.service.db import Expense, ExpenseStatus, InvoiceClaim, session, utcnow
from ramp_agent.service.pipeline import Deps, ModelUnavailable, acquire, process
from ramp_agent.service.registry import SqlRegistry
from ramp_agent.service.storage import LocalStore, sniff

KEY = {"X-API-Key": "k1"}


class FakeJudge:
    def __init__(self, result: SemanticResult | None = None, fail: bool = False):
        self.result, self.fail, self.calls = result, fail, 0

    def judge(self, req):
        self.calls += 1
        if self.fail:
            raise ConnectionError("模型接口超时")
        return self.result or SemanticResult(answered=True)


@pytest.fixture
def svc(tmp_path, db_url):
    s = Settings(_env_file=None, database_url=db_url,
                 storage_dir=str(tmp_path / "objects"), task_eager=True, api_keys="k1",
                 max_upload_mb=1, llm_api_key="", vlm_base_url="")
    store = LocalStore(s.storage_dir)
    deps = Deps(settings=s, store=store, registry=SqlRegistry(s.database_url),
                extractor=InvoiceExtractor(None), judge=FakeJudge(), versions={"code": "test"})
    state = {"enqueue_ok": True}

    def enqueue(expense_id: str, file_key: str | None = None) -> bool:
        if not state["enqueue_ok"]:
            return False
        process(deps, expense_id)
        return True

    client = TestClient(create_app(s, store, enqueue))
    return client, deps, state


@pytest.fixture
def make_pdf(tmp_path):
    from gen_synthetic import COMPANY, COMPANY_TAX_ID, Invoice, render_pdf

    def _make(invoice_no="89000000000000000001", items=(("餐饮服务", 88.0),), issue="2026-10-28"):
        inv = Invoice(invoice_no=invoice_no, issue_date=issue, seller="样例商户", buyer_name=COMPANY,
                      buyer_tax_id=COMPANY_TAX_ID, items=[{"name": n, "amount": a} for n, a in items],
                      total=round(sum(a for _, a in items), 2), tax_rate=0.06)
        path = tmp_path / f"{invoice_no}.pdf"
        render_pdf(inv, path)
        return path.read_bytes()
    return _make


def form(**kw):
    base = {"category": "meal", "amount": "88.00", "submit_date": "2026-11-02", "employee_id": "E001",
            "employee_name": "张三", "city": "武汉", "trip_start": "2026-10-26", "trip_end": "2026-10-30",
            "note": "出差午餐"}
    return {**base, **kw}


def submit(client, data=None, pdf=None, headers=None):
    files = {"file": ("发票.pdf", pdf, "application/pdf")} if pdf is not None else None
    return client.post("/v1/expenses", data=data or form(), files=files, headers={**KEY, **(headers or {})})


# ---------------- 主流程 ----------------
def test_submit_text_pdf_is_reviewed_without_model(svc, make_pdf):
    client, _deps, _ = svc
    r = submit(client, pdf=make_pdf())
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "done" and body["decision"] == "approve" and body["enqueued"]
    rv = body["review"]
    assert rv["extraction_tier"] == "pdf_text" and rv["llm_calls"] == 0 and rv["vlm_calls"] == 0
    assert all(x["clause"] for x in rv["results"]) and rv["versions"] == {"code": "test"}

    got = client.get(f"/v1/expenses/{body['id']}", headers=KEY).json()
    assert got["review"]["decision"] == "approve"
    events = [e["event"] for e in client.get(f"/v1/expenses/{body['id']}/audit", headers=KEY).json()]
    assert events == ["submitted", "review_started", "reviewed"]


def test_missing_attachment_rejected(svc):
    client, _, _ = svc
    body = submit(client).json()
    assert body["decision"] == "reject" and not body["has_attachment"]
    assert any(x["rule"] == "R9" and x["status"] == "violated" for x in body["review"]["results"])


def test_same_file_stored_once(svc, make_pdf, tmp_path):
    client, _deps, _ = svc
    pdf = make_pdf()
    submit(client, pdf=pdf)
    submit(client, data=form(note="再交一次"), pdf=pdf)
    assert len(list((tmp_path / "objects").rglob("*.pdf"))) == 1


# ---------------- 幂等 ----------------
def test_idempotency_key_replays_and_rejects_changed_content(svc, make_pdf):
    client, deps, _ = svc
    pdf = make_pdf()
    first = submit(client, pdf=pdf, headers={"Idempotency-Key": "abc"})
    again = submit(client, pdf=pdf, headers={"Idempotency-Key": "abc"})
    assert first.status_code == 202 and again.status_code == 200
    assert again.json()["id"] == first.json()["id"] and again.json()["replayed"]
    changed = submit(client, data=form(amount="99.00"), pdf=pdf, headers={"Idempotency-Key": "abc"})
    assert changed.status_code == 409
    with session(deps.settings.database_url) as s:
        assert s.query(Expense).count() == 1


# ---------------- 查重 ----------------
def test_duplicate_invoice_rejected_and_rereview_is_not_duplicate(svc, make_pdf):
    client, _, _ = svc
    pdf = make_pdf()
    a = submit(client, pdf=pdf).json()
    b = submit(client, data=form(note="同一张发票又报一次"), pdf=pdf).json()
    assert a["decision"] == "approve"
    assert b["decision"] == "reject" and any(
        x["rule"] == "R6" and x["status"] == "violated" for x in b["review"]["results"])
    # 第一笔重新审核：发票登记在它自己名下，不算重复
    again = client.post(f"/v1/expenses/{a['id']}/rereview", headers=KEY)
    assert again.status_code == 202 and again.json()["decision"] == "approve"


def test_sql_registry_is_atomic_under_concurrency(svc):
    _, deps, _ = svc
    reg, results = deps.registry, []
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        results.append(reg.claim("89000000000000000999", f"exp{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(None) == 1                       # 只有一笔登记成功
    with session(deps.settings.database_url) as s:
        assert s.query(InvoiceClaim).count() == 1


# ---------------- 输入校验与鉴权 ----------------
@pytest.mark.parametrize("data,files,code", [
    (form(category="bribe"), None, 422),
    (form(amount="-1"), None, 422),
    (form(amount="12.345"), None, 422),
    (form(trip_start="2026/10/26"), None, 422),
    (form(), {"file": ("a.pdf", b"MZ\x90\x00not a pdf", "application/pdf")}, 415),
    (form(), {"file": ("a.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 10), "application/pdf")}, 413),
])
def test_input_validation(svc, data, files, code):
    client, _, _ = svc
    assert client.post("/v1/expenses", data=data, files=files, headers=KEY).status_code == code


def test_api_key_required(svc):
    client, _, _ = svc
    assert client.post("/v1/expenses", data=form()).status_code == 401
    assert client.get("/v1/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/healthz").status_code == 200


def test_sniff_ignores_declared_type():
    assert sniff(b"%PDF-1.7 ...") == ".pdf" and sniff(b"\xff\xd8\xff\xe0") == ".jpg"
    assert sniff(b"<html>") is None


# ---------------- 财务复核与指标 ----------------
def test_override_and_stats(svc, make_pdf):
    client, _, _ = svc
    a = submit(client, pdf=make_pdf("89000000000000000011")).json()
    b = submit(client).json()                                         # 缺发票 → 驳回
    r = client.post(f"/v1/expenses/{b['id']}/override", json={"reviewer": "财务李", "decision": "approve"},
                    headers=KEY)
    assert r.status_code == 422                                       # 推翻系统结论必须写理由
    r = client.post(f"/v1/expenses/{b['id']}/override",
                    json={"reviewer": "财务李", "decision": "approve", "note": "纸质发票已线下核对"}, headers=KEY)
    assert r.status_code == 200 and r.json()["final_decision"] == "approve"
    client.post(f"/v1/expenses/{a['id']}/override", json={"reviewer": "财务李", "decision": "approve"},
                headers=KEY)

    pending = client.get("/v1/expenses?pending=true", headers=KEY).json()
    assert pending == []
    st = client.get("/v1/stats", headers=KEY).json()
    assert st["overridden"] == 2 and st["human_agreement_rate"] == 0.5
    assert st["by_decision"] == {"approve": 1, "reject": 1}


def test_override_only_after_review(svc):
    client, _, state = svc
    state["enqueue_ok"] = False
    b = submit(client).json()
    assert b["status"] == "queued" and b["enqueued"] is False        # 投递失败也已受理，等 requeue
    r = client.post(f"/v1/expenses/{b['id']}/override", json={"reviewer": "x", "decision": "approve"},
                    headers=KEY)
    assert r.status_code == 409


# ---------------- 可靠性 ----------------
def test_acquire_is_exclusive_and_recovers_stale(svc):
    client, deps, state = svc
    state["enqueue_ok"] = False
    eid = submit(client).json()["id"]
    url = deps.settings.database_url
    assert acquire(url, eid, 600) is True
    assert acquire(url, eid, 600) is False                           # 第二个 worker 拿不到
    with session(url) as s:                                          # 模拟 worker 崩溃：处理中 20 分钟没动
        s.execute(update(Expense).where(Expense.id == eid)
                  .values(updated_at=utcnow() - timedelta(minutes=20)))
    assert acquire(url, eid, 600) is True                            # 超时后可以被接手
    assert process(deps, eid) == "skipped"                           # 已被领取（处理中），这次跳过


def test_done_expense_not_processed_twice(svc):
    client, deps, _ = svc
    eid = submit(client).json()["id"]
    assert process(deps, eid) == "skipped"
    with session(deps.settings.database_url) as s:
        assert len(s.get(Expense, eid).reviews) == 1


def test_model_failure_retries_before_degrading(svc, make_pdf):
    client, deps, state = svc
    state["enqueue_ok"] = False
    deps.judge = FakeJudge(fail=True)
    # 打车超标，需要问大模型理由是否成立
    eid = submit(client, data=form(category="taxi", amount="150.00", note="凌晨无地铁"),
                 pdf=make_pdf(items=(("客运服务费", 150.0),))).json()["id"]
    url = deps.settings.database_url

    with pytest.raises(ModelUnavailable):
        process(deps, eid, final_attempt=False)
    with session(url) as s:                                          # 失败时还没登记发票，也没有审核记录
        assert s.query(InvoiceClaim).count() == 0 and not s.get(Expense, eid).reviews
        s.execute(update(Expense).where(Expense.id == eid).values(status=ExpenseStatus.QUEUED))

    assert process(deps, eid, final_attempt=True) == "done"          # 最后一次：降级成转人工
    got = client.get(f"/v1/expenses/{eid}", headers=KEY).json()
    assert got["decision"] == "review" and "R3" in [x["rule"] for x in got["review"]["results"]
                                                     if x["status"] == "uncertain"]


# ---------------- 脱敏 ----------------
def test_mask_patterns():
    t = mask("张三 手机 13812345678，身份证 110101199003077777，卡号 6222021234567890123，a@b.com",
             ["张三"])
    assert "138" not in t and "110101" not in t and "6222" not in t and "@" not in t and "张三" not in t
    keep = "发票号 89000000000000000001 税号 91310000MA1XHSZ00X 共 12 人 1500 元"
    assert mask(keep) == keep                                        # 发票号、税号、金额不受影响


def test_llm_receives_masked_text_and_results_map_back():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        content = json.dumps({"items": [{"index": 0, "name": "x", "prohibited": False, "category": "无",
                                         "reason": "正常"}], "taxi_reason": {"valid": True, "reason": "深夜"}})
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]})

    judge = LLMJudge.__new__(LLMJudge)
    judge.model = "fake"
    judge.client = OpenAI(base_url="http://fake/v1", api_key="x", max_retries=0,
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    item = "代订服务（联系人 13812345678）"
    req = SemanticRequest("taxi", 150, "张三凌晨落地，电话 13900001111", items_to_check=[item],
                          need_taxi_reason=True, pii_names=["张三"])
    res = judge.judge(req)
    payload = json.dumps(sent[0], ensure_ascii=False)
    assert "13812345678" not in payload and "13900001111" not in payload and "张三" not in payload
    assert item in res.items and res.items[item].prohibited is False   # 结果按原明细名对回去


# ---------------- Celery 任务 ----------------
def test_celery_task_eager_and_failure_marks_failed(svc, make_pdf, monkeypatch):
    from ramp_agent.service import tasks

    client, deps, state = svc
    monkeypatch.setattr(tasks, "deps", lambda: deps)
    state["enqueue_ok"] = False
    ok_id = submit(client, pdf=make_pdf("89000000000000000021")).json()["id"]
    assert tasks.review_expense.apply(args=[ok_id]).get() == "done"

    bad_id = submit(client, pdf=make_pdf("89000000000000000022")).json()["id"]

    class BrokenStore:
        def get(self, key):
            raise OSError("对象存储不可用")

    monkeypatch.setattr(deps, "store", BrokenStore())
    monkeypatch.setattr(tasks.review_expense, "max_retries", 0)      # 不重试，直接看最终失败的处理
    assert tasks.review_expense.apply(args=[bad_id]).get() == "failed"
    got = client.get(f"/v1/expenses/{bad_id}", headers=KEY).json()
    assert got["status"] == "failed" and "对象存储不可用" in got["error"]
    events = [e["event"] for e in client.get(f"/v1/expenses/{bad_id}/audit", headers=KEY).json()]
    assert events[-1] == "failed"
    # 失败的单子可以重新审核
    monkeypatch.setattr(deps, "store", LocalStore(deps.settings.storage_dir))
    state["enqueue_ok"] = True
    assert client.post(f"/v1/expenses/{bad_id}/rereview", headers=KEY).json()["status"] == "done"


# ---------------- 模型并发上限 ----------------
def test_model_slot_caps_concurrency():
    import time

    from ramp_agent.limits import model_slot

    inflight, peak, lock = [0], [0], threading.Lock()

    def call():
        with model_slot("test-cap", 3):
            with lock:
                inflight[0] += 1
                peak[0] = max(peak[0], inflight[0])
            time.sleep(0.05)
            with lock:
                inflight[0] -= 1

    threads = [threading.Thread(target=call) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 3


def test_request_id_header(svc):
    client, _, _ = svc
    r = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    assert len(client.get("/healthz").headers["x-request-id"]) == 16


def test_sql_spans_without_official_instrumentation():
    """SQLAlchemy 2.1 下官方埋点会静默失效，自己挂事件的版本要能产出 span，且不带参数值。"""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from sqlalchemy import create_engine, text

    from ramp_agent.telemetry import instrument_engine

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    engine = create_engine("sqlite://")
    instrument_engine(engine)
    instrument_engine(engine)                     # 重复调用不会重复挂事件
    import ramp_agent.telemetry as tel

    old = tel.tracer
    tel.tracer = provider.get_tracer("t")
    try:
        with engine.connect() as c:
            c.execute(text("select :secret"), {"secret": "张三的手机号"})
    finally:
        tel.tracer = old
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["SELECT sqlite"]
    assert "张三" not in str(dict(spans[0].attributes))


def test_scans_go_to_their_own_queue():
    from ramp_agent.service.tasks import queue_for

    assert queue_for("invoices/ab/abc.jpg") == "vlm" and queue_for("invoices/ab/abc.png") == "vlm"
    assert queue_for("invoices/ab/abc.pdf") == "default" and queue_for(None) == "default"


# ---------------- 审核工作台 ----------------
def test_workbench_endpoints(svc, make_pdf):
    client, _, _ = svc
    page = client.get("/workbench")
    assert page.status_code == 200 and "财务审核工作台" in page.text
    assert client.get("/", follow_redirects=False).headers["location"] == "/workbench"

    pol = client.get("/v1/policy", headers=KEY).json()
    assert pol["company"]["tax_id"] and pol["rules"]["R8"]["title"] == "不予报销项目"

    pdf = make_pdf("89000000000000000031")
    a = submit(client, data=form(headcount="4"), pdf=pdf).json()
    assert a["claim"]["city"] == "武汉" and a["claim"]["headcount"] == 4 and a["employee_name"] == "张三"
    assert a["file_type"] == "pdf" and a["review"]["invoice_fields"]["total"] == 88.0

    f = client.get(f"/v1/expenses/{a['id']}/file", headers=KEY)
    assert f.status_code == 200 and f.headers["content-type"] == "application/pdf" and f.content == pdf
    assert client.get(f"/v1/expenses/{a['id']}/file").status_code == 401         # 文件也要鉴权
    b = submit(client).json()
    assert client.get(f"/v1/expenses/{b['id']}/file", headers=KEY).status_code == 404

    client.post(f"/v1/expenses/{a['id']}/override", json={"reviewer": "李", "decision": "approve"}, headers=KEY)
    done = client.get("/v1/expenses?reviewed=true", headers=KEY).json()
    assert [x["id"] for x in done] == [a["id"]]
