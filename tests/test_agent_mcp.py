"""公司系统经 MCP 接入：MCP server 的工具、客户端的契约检查与出错处理，以及“换成 MCP 结论不变”。

进程内（MCPSystems(build_server(...))）测协议往返和返回结构；stdio 子进程测真实部署的那条路。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from mcp.server.mcpserver import MCPServer

from ramp_agent.agent import FixtureSystems, Investigator, ScriptedEmployee, ScriptedPlanner, SystemsError
from ramp_agent.agent.mcp_server import TOOLS, build_server
from ramp_agent.agent.mcp_systems import MCPSystems
from ramp_agent.agent.tools import SPECS, ToolBox
from ramp_agent.evalhub.adapters.policy import run_policy

ROOT = Path(__file__).resolve().parents[1]
TID = "91310000MA1XHSZ00X"

WORLD = {
    "preapprovals": [
        {"id": "PA-1", "employee_id": "E1", "category": "conference", "amount_cap": 8000, "status": "approved",
         "valid_from": "2026-10-01", "valid_to": "2026-12-31"},
        {"id": "PA-X", "employee_id": "E9", "category": "conference", "amount_cap": 99999, "status": "approved"}],
    "travel_requests": [{"id": "TR-1", "employee_id": "E1", "city": "杭州", "start": "2026-10-20",
                         "end": "2026-10-23", "status": "approved"}],
    "invoices": [{"invoice_no": "89000000000000000107", "issue_date": "2026-10-22", "total": 96,
                  "buyer_name": "星河数智科技有限公司", "buyer_tax_id": TID, "seller": "样例商户", "status": "正常"}],
    "merchants": [{"id": "M-1", "name": "样例商户", "category": "餐饮", "scope": "中餐；本商户发票均可报销"}],
}


@pytest.fixture
def inproc():
    with MCPSystems(build_server(FixtureSystems(WORLD))) as s:
        yield s


def test_mcp_answers_exactly_like_the_fixture(inproc):
    fx = FixtureSystems(WORLD)
    for emp in ("E1", "E9", "nobody", ""):
        assert inproc.preapprovals(emp) == fx.preapprovals(emp)
        assert inproc.travel_requests(emp) == fx.travel_requests(emp)
    for args in [("89000000000000000107", "2026-10-22", 96), ("8900000000000000O107", "2026-10-22", 96),
                 ("89000000000000000107", "2026-10-22", 96.01), ("89000000000000000107", "2026-10-22", "九十六")]:
        assert inproc.verify_invoice(*args) == fx.verify_invoice(*args)
    assert inproc.verify_invoice("89000000000000000107", "2026-10-22", 96)["verification_id"].startswith("VF")
    assert inproc.merchant("样例商户") == fx.merchant("样例商户")
    assert inproc.merchant("不存在的公司") is None


def test_server_exposes_read_only_tools_and_model_never_sees_them():
    import asyncio

    from mcp import Client

    async def listed():
        async with Client(build_server(FixtureSystems())) as c:
            return (await c.list_tools()).tools

    tools = asyncio.run(listed())
    assert sorted(t.name for t in tools) == sorted(TOOLS)
    assert all(t.annotations.read_only_hint for t in tools)
    # 系统接口要员工编号；给模型的工具没有这个参数——模型只看 harness 定义的那一套
    assert "employee_id" in next(t for t in tools if t.name == "list_preapprovals").input_schema["properties"]
    assert all("employee_id" not in json.dumps(s["parameters"]) for s in SPECS.values())


def test_least_privilege_holds_over_mcp(inproc):
    box = ToolBox("E1", inproc, ScriptedEmployee(None), ["search_preapprovals"])
    got = box.call("search_preapprovals", {})
    assert [r["id"] for r in got["records"]] == ["PA-1"]          # E9 的那条看不到
    assert "employee_id" not in got["records"][0]                 # 也不把员工编号回给模型


def test_server_missing_a_tool_fails_the_contract_check():
    bad = MCPServer("incomplete")

    @bad.tool()
    def list_preapprovals(employee_id: str) -> dict:
        return {"records": []}

    with MCPSystems(bad) as s, pytest.raises(SystemsError, match="缺少工具"):
        s.preapprovals("E1")


class Down:
    """公司系统挂了：每次都报不可用。"""

    def __getattr__(self, name):
        def fail(*_a, **_k):
            raise SystemsError("连接超时")
        return fail


def test_unavailable_system_is_reported_to_the_model_and_can_be_retried():
    box = ToolBox("E1", Down(), ScriptedEmployee(None), ["verify_invoice", "search_preapprovals"])
    args = {"invoice_no": "1", "issue_date": "2026-10-01", "total": 1}
    first = box.call("verify_invoice", args)
    assert "暂时不可用" in first["error"]
    assert "暂时不可用" in box.call("verify_invoice", args)["error"]      # 失败的查询允许原样重试
    assert box.counts["verify_invoice"] == 2                              # 但仍占次数，不会无限重试
    assert box.ledger == {}


def test_unreachable_server_becomes_a_tool_error_not_a_crash():
    gone = MCPSystems(StdioServerParameters(command=sys.executable, args=["-c", "pass"]), connect_timeout=20)
    box = ToolBox("E1", gone, ScriptedEmployee(None), ["search_preapprovals"])
    assert "暂时不可用" in box.call("search_preapprovals", {})["error"]
    gone.close()


def test_stdio_server_end_to_end(tmp_path):
    fixture = tmp_path / "world.json"
    fixture.write_text(json.dumps(WORLD, ensure_ascii=False), encoding="utf-8")
    with MCPSystems.for_fixture(fixture) as s:
        assert [r["id"] for r in s.preapprovals("E1")] == ["PA-1"]
        assert s.verify_invoice("89000000000000000107", "2026-10-22", 96)["status"] == "正常"


def test_spec_parsing():
    assert MCPSystems.from_spec("http://127.0.0.1:8765/mcp").server == "http://127.0.0.1:8765/mcp"
    s = MCPSystems.from_spec("python -m ramp_agent.agent.mcp_server --fixture x.json").server
    assert s.command == sys.executable and s.args[-2:] == ["--fixture", "x.json"]
    assert "LLM_API_KEY" not in s.env                     # 不把模型密钥传给子进程


def test_same_verdicts_through_mcp_on_the_dev_set():
    """换成 MCP 接入，结论、采纳的证据要和直接读数据逐条一致（写死流程，不调模型）。"""
    recs = [json.loads(x) for x in
            (ROOT / "evalsets" / "investigate_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    agent = Investigator(ScriptedPlanner(), "scripted")
    direct = run_policy(recs, judge=None, agent=agent)
    via_mcp = run_policy(recs, judge=None, agent=agent, systems="mcp", workers=8)

    def key(o):
        return o["decision"], [(f["kind"], f["ref"], f["accepted"]) for f in o["agent"]["findings"]]

    assert [key(o) for o in via_mcp] == [key(o) for o in direct]
