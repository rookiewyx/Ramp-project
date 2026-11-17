"""通过 MCP 访问公司系统：EnterpriseSystems 的 MCP 客户端实现。

调查 Agent（harness）是 MCP client，公司系统是 MCP server（mcp_server.py，或别的团队按同样契约提供的 server）。
几条设计：

  - 模型看不到 MCP server 的工具。给模型的工具定义仍然是 agent/tools.py 里那一套（没有员工编号参数、
    有次数上限、结果记台账）。server 的工具描述不进提示词：第三方 server 的描述里写了什么（工具投毒），
    都影响不到模型；server 换了实现，模型这边也不变。
  - 契约检查：连上后先列工具，四个工具缺一个就报错，不等到调查进行到一半才发现。
  - 出错不中断调查：超时、断开、server 报错都转成 SystemsError，工具把它作为错误返回给模型，
    模型可以用已有的证据提交；查不到的证据就不采纳，结论最多是转人工，不会因此误通过。
  - 同步接口：调查循环是同步代码（跑在 Celery worker 和评测线程池里），MCP SDK 是异步的。
    所有连接共用一个后台事件循环线程，连接在第一次调用时才建立（Celery 预派生的子进程里才起线程）。

连接方式：
    MCPSystems("http://127.0.0.1:8765/mcp")                 Streamable HTTP（服务部署）
    MCPSystems.from_spec("python -m ... --fixture x.json")  stdio，启动子进程
    MCPSystems.for_fixture("x.json")                       stdio，起本项目自带的 server（评测用）
    MCPSystems(build_server(...))                          进程内（测试用）
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shlex
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment

from .mcp_server import TOOLS
from .systems import SystemsError

_SRC = str(Path(__file__).resolve().parents[2])           # src/，子进程靠它找到 ramp_agent

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """所有 MCP 连接共用的事件循环，跑在一个守护线程里。"""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="mcp-client", daemon=True).start()
            _loop = loop
        return _loop


class MCPSystems:
    def __init__(self, server: Any, *, timeout: float = 15.0, connect_timeout: float = 30.0):
        self.server = server
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._client: Client | None = None
        self._stop: asyncio.Event | None = None
        self._holder: concurrent.futures.Future | None = None
        self._lock = threading.Lock()

    # ---------------- 连接方式 ----------------
    @classmethod
    def from_spec(cls, spec: str, **kw) -> MCPSystems:
        """ENTERPRISE_MCP 的写法：http(s):// 开头是 Streamable HTTP 地址，否则是启动 stdio server 的命令行。"""
        spec = spec.strip()
        if spec.startswith(("http://", "https://")):
            return cls(spec, **kw)
        parts = shlex.split(spec, posix=os.name != "nt")
        if not parts:
            raise ValueError("ENTERPRISE_MCP 为空")
        command = sys.executable if parts[0] in ("python", "python3") else parts[0]
        return cls(StdioServerParameters(command=command, args=parts[1:], env=_child_env()), **kw)

    @classmethod
    def for_fixture(cls, path: str | Path, **kw) -> MCPSystems:
        """用本项目的 mcp_server 起一个 stdio 子进程，数据来自 path。"""
        args = ["-m", "ramp_agent.agent.mcp_server", "--fixture", str(path)]
        return cls(StdioServerParameters(command=sys.executable, args=args, env=_child_env()), **kw)

    # ---------------- EnterpriseSystems ----------------
    def travel_requests(self, employee_id: str) -> list[dict]:
        return list(self._call("list_travel_requests", {"employee_id": employee_id}).get("records") or [])

    def preapprovals(self, employee_id: str) -> list[dict]:
        return list(self._call("list_preapprovals", {"employee_id": employee_id}).get("records") or [])

    def verify_invoice(self, invoice_no: str, issue_date: str, total: float) -> dict | None:
        try:
            total = float(total)
        except (TypeError, ValueError):
            return None                       # 和 FixtureSystems 一致：金额都不是数，肯定查不到
        out = self._call("verify_invoice", {"invoice_no": str(invoice_no), "issue_date": str(issue_date),
                                            "total": total})
        return out.get("record") if out.get("found") else None

    def merchant(self, name: str) -> dict | None:
        out = self._call("lookup_merchant", {"name": str(name)})
        return out.get("record") if out.get("found") else None

    # ---------------- 连接管理 ----------------
    def _ensure(self) -> Client:
        with self._lock:
            if self._client is not None and self._holder is not None and not self._holder.done():
                return self._client
            self._client = None                       # 第一次调用，或者连接断了：重新连
            ready: concurrent.futures.Future = concurrent.futures.Future()
            self._holder = asyncio.run_coroutine_threadsafe(self._hold(ready), _background_loop())
            try:
                self._client = ready.result(timeout=self.connect_timeout)
            except concurrent.futures.TimeoutError as exc:
                self._holder.cancel()
                raise SystemsError(f"连接公司系统 MCP server 超时（{self.connect_timeout:.0f} 秒）") from exc
            except SystemsError:
                raise
            except Exception as exc:  # noqa: BLE001  子进程起不来、地址不通，都按系统不可用处理
                raise SystemsError(f"连接公司系统 MCP server 失败：{_leaf(exc)!r}") from exc
            return self._client

    async def _hold(self, ready: concurrent.futures.Future) -> None:
        """在后台事件循环里打开连接并一直持有，直到 close()。连接的进出必须在同一个任务里。"""
        self._stop = asyncio.Event()
        try:
            async with Client(self.server, read_timeout_seconds=self.timeout) as client:
                listed = await client.list_tools()
                missing = sorted(set(TOOLS) - {t.name for t in listed.tools})
                if missing:
                    ready.set_exception(SystemsError(f"MCP server 缺少工具 {missing}，和审核服务的契约不一致"))
                    return
                ready.set_result(client)
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001  连接失败要交给等待中的调用方
            if not ready.done():
                ready.set_exception(_leaf(exc))
            if not isinstance(exc, Exception):
                raise

    def _call(self, name: str, args: dict) -> dict:
        client = self._ensure()
        fut = asyncio.run_coroutine_threadsafe(client.call_tool(name, args), _background_loop())
        try:
            res = fut.result(timeout=self.timeout + 5)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise SystemsError(f"{name} 超时") from exc
        except Exception as exc:  # noqa: BLE001  连接断开、协议错误
            raise SystemsError(f"{name} 调用失败：{_leaf(exc)!r}") from exc
        if res.is_error:
            text = " ".join(getattr(c, "text", "") for c in res.content)[:300]
            raise SystemsError(f"{name} 返回错误：{text}")
        if isinstance(res.structured_content, dict):
            return res.structured_content
        for c in res.content:                     # 只返回文本的 server：内容是 JSON
            try:
                data = json.loads(getattr(c, "text", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict):
                return data
        raise SystemsError(f"{name} 的返回不是 JSON 对象")

    def close(self) -> None:
        with self._lock:
            holder, stop = self._holder, self._stop
            self._client = self._holder = None
        if holder is None or stop is None or holder.done():
            return
        _background_loop().call_soon_threadsafe(stop.set)
        try:
            holder.result(timeout=10)
        except Exception:  # noqa: BLE001  关闭时子进程的退出错误不重要
            pass

    def __enter__(self) -> MCPSystems:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _leaf(exc: BaseException) -> BaseException:
    """anyio 的任务组会把异常包成 ExceptionGroup，取出里面真正的那个，错误信息才看得懂。"""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _child_env() -> dict[str, str]:
    """子进程的环境：SDK 默认的最小环境（不把模型密钥之类传给 server）+ 找到 ramp_agent 的路径 + UTF-8。"""
    env = get_default_environment()
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_SRC, os.environ.get("PYTHONPATH", "")) if p)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env
