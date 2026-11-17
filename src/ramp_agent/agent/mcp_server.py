"""公司系统的 MCP server：把出差申请、预审批（OA）、发票查验（税务）、工商登记做成 MCP 工具。

真实企业里这几个系统归不同团队维护，各自暴露一个 MCP server，审核 Agent、财务助手、桌面客户端都能复用，
不用每接一个 Agent 就写一遍对接代码。这里一个 server 包了四个系统，数据用 FixtureSystems（JSON）模拟，
换成真实接口只需要改这个文件里的后端。

注意分工：这里的工具是“系统接口”，参数里有员工编号，调用方是审核服务（harness），不是模型。
模型看到的工具仍然由 agent/tools.py 定义：没有员工编号参数、有次数上限、结果记台账，
最小权限和证据校验都留在 harness 这一侧。MCP 只是连接和发现工具的协议，不替我们做安全。

用法：
    python -m ramp_agent.agent.mcp_server --fixture evalsets/enterprise_demo.json                 # stdio
    python -m ramp_agent.agent.mcp_server --fixture evalsets/enterprise_demo.json \\
        --transport streamable-http --host 127.0.0.1 --port 8765                                  # HTTP，地址 /mcp
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .systems import EnterpriseSystems, FixtureSystems

# 工具名是 harness 和 server 之间的契约：客户端连上后先核对这四个工具都在，缺了直接报错
TOOLS = ("list_travel_requests", "list_preapprovals", "verify_invoice", "lookup_merchant")

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def build_server(backend: EnterpriseSystems, name: str = "ramp-enterprise") -> MCPServer:
    """用任意 EnterpriseSystems 实现搭一个 MCP server。四个工具都是只读查询。"""
    server = MCPServer(name, instructions="公司内部系统查询：出差申请、费用预审批、税务发票查验、工商登记。只读。")

    @server.tool(annotations=_READ_ONLY)
    def list_travel_requests(employee_id: str) -> dict[str, Any]:
        """OA：员工的出差申请（编号、城市、起止日期、审批状态、变更记录）。"""
        return {"records": backend.travel_requests(employee_id)}

    @server.tool(annotations=_READ_ONLY)
    def list_preapprovals(employee_id: str) -> dict[str, Any]:
        """OA：员工的费用预审批（编号、费用类别、批准上限、审批状态、有效期、审批人）。"""
        return {"records": backend.preapprovals(employee_id)}

    @server.tool(annotations=_READ_ONLY)
    def verify_invoice(invoice_no: str, issue_date: str, total: float) -> dict[str, Any]:
        """税务发票查验：号码、开票日期（YYYY-MM-DD）、价税合计三项与税务登记完全一致才查得到。"""
        rec = backend.verify_invoice(invoice_no, issue_date, total)
        return {"found": rec is not None, "record": rec}

    @server.tool(annotations=_READ_ONLY)
    def lookup_merchant(name: str) -> dict[str, Any]:
        """工商登记：按名称查销售方的经营类别和经营范围。"""
        rec = backend.merchant(name)
        return {"found": rec is not None, "record": rec}

    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="公司系统 MCP server（数据来自 JSON 文件）")
    ap.add_argument("--fixture", default="", help="公司系统数据（JSON），格式见 FixtureSystems")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    server = build_server(FixtureSystems.from_file(a.fixture))
    if a.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
