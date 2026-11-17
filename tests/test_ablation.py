"""消融开关：每个开关拿掉 harness 的一个部件，确认拿掉之后行为确实变了（变成会出错的样子）；以及稳定性统计。"""

from __future__ import annotations

import pytest
from test_agent import CO, TID, FakeLLM, claim, investigate, last_tool, pa

from ramp_agent.agent import FixtureSystems, Investigator, ScriptedEmployee
from ramp_agent.agent.investigator import ABLATIONS, SYSTEM, SYSTEM_DECIDES
from ramp_agent.agent.tools import ToolBox, tool_specs
from ramp_agent.evalhub.compare import format_stability, stability_from
from ramp_agent.policy import PolicyReviewer, load_policy


def run_with(ablations, c, llm, world=None):
    reviewer = PolicyReviewer(load_policy())
    sem = reviewer.semantic(c)
    rv = reviewer.review(c, sem)
    agent = Investigator(llm, "fake", ablations=ablations)
    return agent.run(c, rv, reviewer, FixtureSystems(world), ScriptedEmployee(None), sem)


def test_unknown_ablation_is_rejected():
    with pytest.raises(ValueError, match="未知的消融开关"):
        Investigator(None, "fake", ablations=("no_such_thing",))
    assert set(ABLATIONS) == {"trust_model", "no_auto_adopt", "all_tools", "no_limits", "model_decides"}


def test_trust_model_accepts_an_inapplicable_record():
    """证据校验的作用：待审批的预审批，正常会被拒绝；不校验就被采纳，报销被误通过。"""
    def llm():
        return FakeLLM([("search_preapprovals", {})], [("submit_findings", {"preapproval_id": "PA-9", "summary": ""})])

    world = {"preapprovals": [pa("PA-9", status="pending")]}
    normal, _ = investigate(claim(), llm(), world)
    ablated = run_with(("trust_model",), claim(), llm(), world)
    assert normal.decision_after == "review" and not normal.findings[0]["accepted"]
    assert ablated.decision_after == "approve" and ablated.findings[0]["accepted"]


def test_no_auto_adopt_drops_a_verification_the_model_did_not_cite():
    """唯一事实自动采纳的作用：模型查到了却没引用（B014 那种），正常照样采纳；关掉就丢了。"""
    real = "89000000000000000107"
    c = dict(category="meal", amount=96, city="杭州", trip_start="2026-10-20", trip_end="2026-10-23", note="午餐",
             invoice={"invoice_no": "8900000000000000O107", "total": 96, "items": [{"name": "餐饮服务", "amount": 96}]},
             uncertain_fields=["invoice_no"])
    world = {"invoices": [{"invoice_no": real, "issue_date": "2026-10-22", "total": 96, "buyer_name": CO,
                           "buyer_tax_id": TID, "seller": "样例商户", "status": "正常"}]}

    def llm():
        return FakeLLM([("verify_invoice", {"invoice_no": real, "issue_date": "2026-10-22", "total": 96})],
                       [("submit_findings", {"summary": "查到了"})])      # 没填 verification_id

    normal, _ = investigate(claim(**c), llm(), world)
    ablated = run_with(("no_auto_adopt",), claim(**c), llm(), world)
    assert normal.decision_after == "approve" and "自动采纳" in normal.findings[0]["reason"]
    assert ablated.decision_after == "review" and ablated.findings == []


def test_all_tools_offers_every_tool():
    llm = FakeLLM([("submit_findings", {"summary": ""})])
    res = run_with(("all_tools",), claim(), llm, {"preapprovals": [pa("PA-1")]})
    assert sorted(res.tools_offered) == sorted(["search_preapprovals", "search_travel_requests", "verify_invoice",
                                                "lookup_merchant", "ask_employee"])
    assert len(llm.seen[0]["tools"]) == 6


def test_no_limits_lifts_caps_and_duplicate_check_but_still_asks_once():
    box = ToolBox("E1", FixtureSystems(), ScriptedEmployee("答复"), ["verify_invoice", "ask_employee"], limits=False)
    args = {"invoice_no": "1", "issue_date": "2026-10-01", "total": 1}
    for _ in range(5):
        assert box.call("verify_invoice", args)["found"] is False       # 不拦重复、不限次数
    assert box.call("ask_employee", {"question": "?"})["replied"] is True
    assert "上限" in box.call("ask_employee", {"question": "再问一次"})["error"]


def test_model_decides_skips_the_rule_engine():
    """规则引擎重审的作用：模型说 approve 就 approve，即使什么证据都没交。"""
    llm = FakeLLM([("submit_findings", {"decision": "approve", "summary": "看起来没问题"})])
    res = run_with(("model_decides",), claim(), llm, {"preapprovals": [pa("PA-1")]})
    assert res.decision_after == "approve" and res.review.decision.value == "review"   # 规则引擎仍是转人工
    assert llm.seen[0]["messages"][0]["content"] == SYSTEM_DECIDES != SYSTEM
    spec = tool_specs(["search_preapprovals"], decide=True)[-1]["function"]["parameters"]
    assert "decision" in spec["required"]
    assert "decision" not in tool_specs(["search_preapprovals"])[-1]["function"]["parameters"]["properties"]


def test_model_decides_with_a_bad_decision_falls_back_to_review():
    llm = FakeLLM([("submit_findings", {"decision": "通过", "summary": ""})])
    assert run_with(("model_decides",), claim(), llm).decision_after == "review"


def test_default_behaviour_unchanged_by_the_switches():
    world = {"preapprovals": [pa("PA-2")]}
    llm = FakeLLM([("search_preapprovals", {})],
                  [("submit_findings", lambda m: {"preapproval_id": last_tool(m)["records"][0]["id"], "summary": ""})])
    res = run_with((), claim(), llm, world)
    assert res.decision_after == "approve" and llm.seen[0]["tools"] == ["search_preapprovals", "submit_findings"]


def test_stability_statistics():
    runs = [
        {"A": (True, "approve", "approve"), "B": (True, "review", "review"), "C": (False, "approve", "reject")},
        {"A": (True, "approve", "approve"), "B": (False, "approve", "review"), "C": (False, "review", "reject")},
        {"A": (True, "approve", "approve"), "B": (True, "review", "review"), "C": (True, "reject", "reject")},
    ]
    st = stability_from(runs)
    assert st["accuracy"] == [0.6667, 0.3333, 1.0] and st["mean"] == 0.6667
    assert st["pass_all"] == 0.3333 and st["pass_any"] == 1.0          # 只有 A 三次都对；每条都至少对过一次
    assert st["false_approve"] == [1, 1, 0]
    assert [u["key"] for u in st["unstable"]] == ["B", "C"]
    text = format_stability(st)
    assert "pass^3" in text and "B 标准答案 review：review / approve / review（2/3 次对）" in text
