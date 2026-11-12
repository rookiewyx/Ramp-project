"""数据闭环：线上财务复核的结果收进评测集。

用真实的审核服务（同步执行 + 假的大模型）造出几种典型情况，再收集，检查挑选、脱敏、去重、版本和盲标。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ramp_agent.config import Settings
from ramp_agent.evalhub import datasets as dsm
from ramp_agent.evalhub.adapters.policy import claim_from_record
from ramp_agent.evalhub.db import session as eh_session
from ramp_agent.evalhub.harvest import harvest
from ramp_agent.evalhub.web import _view
from ramp_agent.extraction import InvoiceExtractor
from ramp_agent.policy import SemanticResult
from ramp_agent.service.api import create_app
from ramp_agent.service.pipeline import Deps, process
from ramp_agent.service.registry import SqlRegistry
from ramp_agent.service.storage import LocalStore


class Judge:
    """打车理由一律判不成立，让打车超标的单子转人工。"""

    def judge(self, req):
        return SemanticResult(answered=True, taxi_reason_valid=False if req.need_taxi_reason else None,
                              taxi_reason_explanation="理由不充分")


@pytest.fixture
def world(tmp_path, db_url):
    from gen_synthetic import COMPANY, COMPANY_TAX_ID, Invoice, render_pdf

    s = Settings(_env_file=None, database_url=db_url, storage_dir=str(tmp_path / "obj"),
                 task_eager=True, api_keys="", llm_api_key="", vlm_base_url="")
    store = LocalStore(s.storage_dir)
    deps = Deps(settings=s, store=store, registry=SqlRegistry(db_url),
                extractor=InvoiceExtractor(None), judge=Judge(), versions={})
    client = TestClient(create_app(s, store, lambda i, k=None: process(deps, i) == "done"))

    def pdf(no: str, items=(("餐饮服务", 88.0),)) -> bytes:
        inv = Invoice(invoice_no=no, issue_date="2026-10-28", seller="样例商户", buyer_name=COMPANY,
                      buyer_tax_id=COMPANY_TAX_ID, items=[{"name": n, "amount": a} for n, a in items],
                      total=round(sum(a for _, a in items), 2), tax_rate=0.06)
        path = tmp_path / f"{no}.pdf"
        render_pdf(inv, path)
        return path.read_bytes()

    def submit(file: bytes | None = None, **kw) -> str:
        data = {"category": "meal", "amount": "88.00", "submit_date": "2026-11-02", "employee_id": "E001",
                "employee_name": "张三", "city": "武汉", "trip_start": "2026-10-26", "trip_end": "2026-10-30",
                "note": "出差午餐", **kw}
        files = {"file": ("f.pdf", file, "application/pdf")} if file else None
        return client.post("/v1/expenses", data=data, files=files).json()["id"]

    def override(eid: str, decision: str, note: str = "复核") -> None:
        r = client.post(f"/v1/expenses/{eid}/override", json={"reviewer": "财务李", "decision": decision,
                                                               "note": note})
        assert r.status_code == 200, r.text

    return client, submit, override, pdf, db_url


def test_harvest_end_to_end(world, tmp_path):
    client, submit, override, pdf, src = world
    eh = f"sqlite:///{(tmp_path / 'eh.db').as_posix()}"
    first_pdf = pdf("89000000000000000101")

    e1 = submit(first_pdf, note="张三请客户吃饭，客户电话 13812345678")     # 系统通过 → 财务驳回
    e2 = submit(None)                                                        # 缺发票驳回 → 财务通过
    e3 = submit(pdf("89000000000000000103", (("客运服务费", 150.0),)), category="taxi",
                amount="150.00", note="打车方便")                            # 转人工 → 财务驳回
    e4 = submit(pdf("89000000000000000104"))                                 # 自动通过，没人复核
    e5 = submit(first_pdf, note="同一张发票再报一次")                         # 重复报销
    assert [client.get(f"/v1/expenses/{e}").json()["decision"] for e in (e1, e2, e3, e4, e5)] == \
        ["approve", "reject", "review", "approve", "reject"]
    r3 = next(x for x in client.get(f"/v1/expenses/{e3}").json()["review"]["results"] if x["rule"] == "R3")
    assert r3["status"] == "violated" and r3["by"] == "llm"           # 大模型判理由不成立
    override(e1, "reject", "张三说是私人聚餐，手机 13812345678")
    override(e2, "approve", "纸质发票已线下核对")
    override(e3, "reject", "个人便利不构成理由")
    override(e5, "approve", "误判")

    with eh_session(eh) as s:
        rep = harvest(src, s, audit=5, seed=1)
        s.commit()
    assert rep.added == {"overturned": 2, "resolved": 1, "audit": 1}
    assert rep.skipped == {"重复报销（离线无法复现）": 1}
    assert rep.transitions[("approve", "reject")] == 1 and rep.transitions[("review", "reject")] == 1
    assert rep.version == "policy-prod@v1" and "系统定论被推翻：2" in rep.format()

    with eh_session(eh) as s:
        v = dsm.get_version(s, "policy-prod")
        cases = {c.tags["family"]: c for c in v.cases if c.tags["family"] != "overturned"}
        over = [c for c in v.cases if c.tags["family"] == "overturned"]
        assert len(v.cases) == 4 and len(over) == 2
        for c in v.cases:
            p = c.payload
            text = str(p) + "".join(lb.note for lb in c.labels)
            # 脱敏：姓名、手机号、原始工号、原始编号都不进评测集
            assert "张三" not in text and "13812345678" not in text and "E001" not in text
            assert p["expense_id"].startswith("P") and p["employee_id"].startswith("E")
            claim_from_record(p)                                    # 能直接拿来跑评测
            view = str(_view(p))                                    # 盲标页面看不到系统和财务的结论
            assert "finance" not in view and "system" not in view and "approve" not in view
        # 财务的结论作为 finance 的标注；抽检的单子没有标注，必须盲标
        assert {lb.labeler for lb in cases["resolved"].labels} == {"finance"}
        assert dsm.resolve_gold(cases["resolved"]).decision == "reject"
        assert cases["audit"].labels == []
        assert dsm.resolve_gold(cases["audit"]).status == "unlabeled"
        # 第二个人盲标，和财务意见不一致时记为有争议，不参与打分
        dsm.save_label(s, cases["resolved"], "kin", "review", ["R3"])
        assert dsm.resolve_gold(cases["resolved"]).status == "disputed"
        dsm.freeze(s, v)
        s.commit()

    # 再收一次：已收集的不重复；评测集已冻结，新数据进派生出的 v2
    e6 = submit(pdf("89000000000000000106"))
    override(e6, "reject", "发票是个人消费")
    with eh_session(eh) as s:
        rep2 = harvest(src, s, audit=0)
        s.commit()
        assert rep2.added == {"overturned": 1} and rep2.skipped["已收集过"] == 3
        v2 = dsm.get_version(s, "policy-prod@v2")
        assert v2.status == "draft" and len(v2.cases) == 5


def test_dry_run_writes_nothing(world, tmp_path):
    client, submit, override, pdf, src = world
    e = submit(pdf("89000000000000000201"))
    override(e, "reject", "私人消费")
    eh = f"sqlite:///{(tmp_path / 'eh2.db').as_posix()}"
    with eh_session(eh) as s:
        rep = harvest(src, s, dry_run=True)
        assert rep.added == {"overturned": 1}
        assert dsm.existing_keys(s, "policy-prod") == set()

