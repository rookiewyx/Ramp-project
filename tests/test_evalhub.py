from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ramp_agent.evalhub import datasets as dsm
from ramp_agent.evalhub.compare import compare, format_comparison, gate
from ramp_agent.evalhub.db import session
from ramp_agent.evalhub.runner import run_extraction_eval, run_policy_eval
from ramp_agent.evalhub.stats import cohen_kappa, mcnemar_exact, wilson
from ramp_agent.evalhub.web import create_app
from ramp_agent.policy import SemanticResult


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{(tmp_path / 'eh.db').as_posix()}"


def record(eid: str, note: str, amount: float, expected: str, rules=(), designer: str | None = None) -> dict:
    """一笔打车报销：超过 100 元时结论取决于说明里的理由。"""
    r = {"expense_id": eid, "category": "taxi", "amount": amount, "submit_date": "2026-10-20", "note": note,
         "city": "杭州", "trip_start": "2026-10-10", "trip_end": "2026-10-15", "nights": None,
         "headcount": None, "preapproval_id": None, "scenario": "taxi",
         "invoice": {"invoice_no": f"1{eid[-3:]:0>19}", "issue_date": "2026-10-12", "seller": "s",
                     "buyer_name": "星河数智科技有限公司", "buyer_tax_id": "91310000MA1XHSZ00X",
                     "items": [{"name": "客运服务费", "amount": amount}], "total": amount, "tax_rate": 0.03,
                     "file": "x", "file_type": "none"},
         "expected": {"decision": expected, "violations": [{"rule": x} for x in rules]}}
    if designer:
        r["designer_label"] = {"decision": designer, "rules": list(rules)}
    return r


@pytest.fixture
def jsonl(tmp_path):
    rows = [record("T001", "", 60, "approve"),
            record("T002", "凌晨航班落地", 160, "approve", designer="approve"),
            record("T003", "习惯打车", 150, "review", ("R3",), designer="review"),
            record("T004", "", 150, "review", ("R3",))]
    p = tmp_path / "set.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


class Judge:
    """假的大模型：说明里有“凌晨”就认为理由成立。"""

    def judge(self, req):
        return SemanticResult(taxi_reason_valid="凌晨" in req.note, answered=True)


# ---------------- 统计 ----------------
def test_stats():
    lo, hi = wilson(27, 27)
    assert hi == 1.0 and 0.87 < lo < 0.88          # 27/27 全对，下限仍只有约 87%
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(3, 3) == 1.0
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("aabb")) == 0.0


# ---------------- 评测集 ----------------
def test_import_freeze_fork(db_url, jsonl):
    with session(db_url) as s:
        v = dsm.import_policy(s, jsonl, "taxi")
        assert v.ref == "taxi@v1" and v.status == "draft" and len(v.cases) == 4
        assert {lb.labeler for c in v.cases for lb in c.labels} == {"generator", "designer"}
        dsm.freeze(s, v)
        with pytest.raises(dsm.EvalHubError):
            dsm.save_label(s, v.cases[0], "kin", "approve")        # 冻结后不能改
        assert dsm.import_policy(s, jsonl, "taxi").ref == "taxi@v2"   # 冻结后可以导入新版本
        with pytest.raises(dsm.EvalHubError):
            dsm.import_policy(s, jsonl, "taxi")                        # 已有草稿时不能再建
        with pytest.raises(dsm.EvalHubError):
            dsm.import_extraction(s, jsonl, "taxi")                    # 任务类型不能混
        dsm.freeze(s, dsm.get_version(s, "taxi@v2"))
        s.commit()
    with session(db_url) as s:
        v3 = dsm.fork(s, "taxi@v1", "追加用例")
        assert v3.ref == "taxi@v3" and v3.status == "draft" and v3.source == "fork:taxi@v1"
        assert len(v3.cases[1].labels) == len(dsm.get_version(s, "taxi@v1").cases[1].labels)
        assert dsm.get_version(s, "taxi").id == v3.id


def test_gold_resolution(db_url, jsonl):
    with session(db_url) as s:
        v = dsm.import_policy(s, jsonl, "g", with_labels=False)
        c2, c3 = v.cases[1], v.cases[2]
        assert dsm.resolve_gold(c2).decision == "approve"          # 只有 designer 时用 designer
        dsm.save_label(s, c2, "kin", "review", ["R3"])
        assert dsm.resolve_gold(c2).decision == "review"           # 有人工标注时排除 designer
        assert dsm.resolve_gold(c2, "designer").decision == "approve"
        dsm.save_label(s, c2, "amy", "approve")
        assert dsm.resolve_gold(c2).status == "disputed"
        dsm.save_label(s, c3, "kin", "review", ["R3"], unsure=True)
        assert dsm.resolve_gold(c3).status == "unsure"
        assert dsm.resolve_gold(v.cases[0]).status == "unlabeled"
        ag = dsm.agreement(v)
        pair = next(p for p in ag["pairs"] if p["pair"] == ["designer", "kin"])
        assert pair["n"] == 2 and pair["agree"] == 1


# ---------------- 运行、对比、门禁 ----------------
def test_run_compare_gate(db_url, jsonl):
    with session(db_url) as s:
        v = dsm.import_policy(s, jsonl, "r")
        dsm.freeze(s, v)
        base = run_policy_eval(s, v, judge=None, note="规则")
        new = run_policy_eval(s, v, judge=Judge(), note="规则+模型")
        s.commit()
        assert base.metrics["accuracy"] == 0.75 and base.metrics["extra_review"] == 1
        assert new.metrics["accuracy"] == 1.0 and new.metrics["llm_calls"] == 2
        assert base.code_hash and base.prompt_hash and base.policy_hash
        cmp = compare(base, new)
        assert [r["key"] for r in cmp.fixed] == ["T002"] and not cmp.broken
        assert gate(cmp) == []
        back = compare(new, base)                       # 反过来就是一次回归
        assert [r["key"] for r in back.broken] == ["T002"]
        assert any("从对变错" in x for x in gate(back))
        assert gate(back, waive={"T002"}) == []         # 人工确认后可以豁免
        assert "修好 0 条，改坏 1 条" in format_comparison(back)


def test_extraction_run(db_url, tmp_path, sample_pdf, sample_invoice):
    from dataclasses import asdict

    inv = {**asdict(sample_invoice), "file": str(sample_pdf), "file_type": "pdf"}
    p = tmp_path / "ext.jsonl"
    p.write_text(json.dumps({"invoice": inv}, ensure_ascii=False), encoding="utf-8")
    with session(db_url) as s:
        v = dsm.import_extraction(s, p, "ext")
        dsm.freeze(s, v)
        run = run_extraction_eval(s, v, vlm=None)
        assert run.metrics["doc_accuracy"] == 1.0 and run.metrics["tiers"] == {"pdf_text": 1}


# ---------------- 标注页面接口 ----------------
def test_labeling_api_is_blind(db_url, jsonl):
    with session(db_url) as s:
        v = dsm.import_policy(s, jsonl, "web")       # 带 generator 和 designer 的标注
        vid = v.id
        s.commit()
    client = TestClient(create_app(db_url))
    assert "RAMP 报销标注" in client.get("/").text
    assert client.get("/api/datasets").json()[0]["ref"] == "web@v1"
    assert len(client.get("/api/policy").json()["rules"]) >= 10

    listing = client.get(f"/api/versions/{vid}/cases", params={"labeler": "kin"}).json()
    assert listing["done"] == 0 and all(c["decision"] is None for c in listing["cases"])
    cid = listing["cases"][1]["case_id"]
    detail = client.get(f"/api/cases/{cid}", params={"labeler": "kin"}).json()
    assert detail["label"] is None                    # 看不到 designer / generator 的答案
    assert "scenario" not in json.dumps(detail) and "expected" not in json.dumps(detail)

    r = client.put(f"/api/cases/{cid}/label", json={"labeler": "kin", "decision": "approve"})
    assert r.status_code == 200 and r.json()["done"] == 1
    assert client.get(f"/api/cases/{cid}", params={"labeler": "kin"}).json()["label"]["decision"] == "approve"
    assert client.put(f"/api/cases/{cid}/label", json={"labeler": "designer", "decision": "approve"}).status_code == 400
    assert client.put(f"/api/cases/{cid}/label", json={"labeler": "kin", "decision": "maybe"}).status_code == 409

    with session(db_url) as s:
        dsm.freeze(s, dsm.get_version(s, "web"))
        s.commit()
    assert client.put(f"/api/cases/{cid}/label", json={"labeler": "kin", "decision": "review"}).status_code == 409
