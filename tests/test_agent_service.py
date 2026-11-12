"""调查 Agent 接入审核服务：转人工的单子先调查；追问员工时暂停，员工答复后从断点继续。"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from test_agent import FakeLLM, Judge
from test_service import KEY, form, submit

from ramp_agent.agent import FixtureSystems, Investigator
from ramp_agent.config import Settings
from ramp_agent.extraction import InvoiceExtractor
from ramp_agent.service import tasks
from ramp_agent.service.api import create_app
from ramp_agent.service.db import Investigation, session, utcnow
from ramp_agent.service.pipeline import Deps, process
from ramp_agent.service.registry import SqlRegistry
from ramp_agent.service.storage import LocalStore

WORLD = {"preapprovals": [{"id": "PA-1", "employee_id": "E001", "category": "conference", "amount_cap": 8000,
                           "status": "approved", "valid_from": "2026-10-01", "valid_to": "2026-12-31"}]}


@pytest.fixture
def agent_svc(tmp_path, db_url, monkeypatch):
    s = Settings(_env_file=None, database_url=db_url, storage_dir=str(tmp_path / "objects"), task_eager=True,
                 api_keys="k1", llm_api_key="", vlm_base_url="")
    store = LocalStore(s.storage_dir)
    llm = FakeLLM()
    deps = Deps(settings=s, store=store, registry=SqlRegistry(db_url), extractor=InvoiceExtractor(None),
                judge=Judge(), versions={}, investigator=Investigator(llm, "fake"), systems=FixtureSystems(WORLD))

    def enqueue(expense_id, file_key=None):
        process(deps, expense_id)
        return True

    monkeypatch.setattr(tasks, "enqueue", enqueue)
    monkeypatch.setattr(tasks, "get_settings", lambda: s)
    return TestClient(create_app(s, store, enqueue)), llm


def taxi_form(**kw):
    return form(category="taxi", amount="168.00", note="", city="北京", trip_start="", trip_end="", **kw)


def test_conference_without_preapproval_is_resolved_by_agent(agent_svc, make_pdf):
    client, llm = agent_svc
    llm.turns = [[("search_preapprovals", {})], [("submit_findings", {"preapproval_id": "PA-1", "summary": "找到预审批"})]]
    pdf = make_pdf(items=(("会议费", 6800.0),))
    d = submit(client, form(category="conference", amount="6800.00", note="峰会", trip_start="", trip_end=""), pdf).json()
    d = client.get(f"/v1/expenses/{d['id']}", headers=KEY).json()
    assert d["decision"] == "approve" and d["agent_status"] == "resolved"
    iv = d["investigation"]
    assert iv["decision_before"] == "review" and iv["decision_after"] == "approve" and iv["summary"] == "找到预审批"
    assert any(r["rule"] == "R7" and r["status"] == "violated" for r in iv["initial_results"])
    assert any(r["rule"] == "R7" and r["status"] == "pass" for r in d["review"]["results"])
    events = [a["event"] for a in client.get(f"/v1/expenses/{d['id']}/audit", headers=KEY).json()]
    assert "investigated" in events
    st = client.get("/v1/stats", headers=KEY).json()["agent"]
    assert st["investigated"] == 1 and st["resolved_rate"] == 1.0


def test_question_reply_and_resume(agent_svc, make_pdf):
    client, llm = agent_svc
    llm.turns = [[("ask_employee", {"question": "请说明打车原因"})], [("submit_findings", {"summary": "员工已说明"})]]
    pdf = make_pdf(items=(("客运服务费", 168.0),))
    eid = submit(client, taxi_form(), pdf).json()["id"]
    d = client.get(f"/v1/expenses/{eid}", headers=KEY).json()
    assert d["decision"] == "review" and d["agent_status"] == "awaiting_employee"
    assert d["investigation"]["question"] == "请说明打车原因"
    todo = client.get("/v1/expenses?agent_status=awaiting_employee", headers=KEY).json()
    assert [x["id"] for x in todo] == [eid]

    r = client.post(f"/v1/expenses/{eid}/reply", json={"reply": "凌晨没地铁", "employee_id": "E002"}, headers=KEY)
    assert r.status_code == 403                                    # 只有提交人能答复
    r = client.post(f"/v1/expenses/{eid}/reply", json={"reply": "凌晨一点下飞机，地铁停运", "employee_id": "E001"},
                    headers=KEY)
    assert r.status_code == 202
    d = client.get(f"/v1/expenses/{eid}", headers=KEY).json()
    assert d["decision"] == "approve" and d["agent_status"] == "resolved"
    assert d["investigation"]["reply"] == "凌晨一点下飞机，地铁停运" and len(d["investigation"]["steps"]) == 1
    assert len(llm.seen) == 2                                     # 恢复后没有重新开始，只多调了一轮
    r = client.post(f"/v1/expenses/{eid}/reply", json={"reply": "再答一次", "employee_id": "E001"}, headers=KEY)
    assert r.status_code == 409


def test_finance_can_decide_while_waiting_and_late_reply_is_refused(agent_svc, make_pdf):
    client, llm = agent_svc
    llm.turns = [[("ask_employee", {"question": "原因？"})]]
    eid = submit(client, taxi_form(), make_pdf(items=(("客运服务费", 168.0),))).json()["id"]
    r = client.post(f"/v1/expenses/{eid}/override", json={"reviewer": "财务李", "decision": "reject",
                                                             "note": "无正当理由"}, headers=KEY)
    assert r.status_code == 200
    r = client.post(f"/v1/expenses/{eid}/reply", json={"reply": "凌晨", "employee_id": "E001"}, headers=KEY)
    assert r.status_code == 409


def test_waiting_too_long_is_closed_without_reply(agent_svc, make_pdf, db_url):
    client, llm = agent_svc
    llm.turns = [[("ask_employee", {"question": "原因？"})], [("submit_findings", {"summary": "员工未答复"})]]
    eid = submit(client, taxi_form(), make_pdf(items=(("客运服务费", 168.0),))).json()["id"]
    assert tasks.expire_waiting() == 0                             # 还没到时间
    with session(db_url) as s:
        s.execute(update(Investigation).values(asked_at=utcnow() - timedelta(hours=100)))
    assert tasks.expire_waiting() == 1
    d = client.get(f"/v1/expenses/{eid}", headers=KEY).json()
    assert d["decision"] == "review" and d["agent_status"] == "unresolved"
    assert d["investigation"]["findings"][-1]["reason"] == "员工没有答复"


def test_unfixable_review_is_not_investigated(agent_svc, make_pdf):
    client, llm = agent_svc
    pdf = make_pdf(items=(("餐饮服务", 210.0),))
    d = submit(client, form(amount="210.00"), pdf).json()
    d = client.get(f"/v1/expenses/{d['id']}", headers=KEY).json()
    assert d["decision"] == "review" and d["agent_status"] is None and d["investigation"] is None
    assert llm.seen == []
