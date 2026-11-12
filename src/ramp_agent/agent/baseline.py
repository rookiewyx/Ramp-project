"""写死流程的“调查”：不用大模型，作为调查 Agent 的对照组，也给测试和离线演示用。

它和大模型走同一套协议（假装是一个 chat.completions 客户端），所以工具、校验、重审完全一样，
差别只在“决策”：
  - 第一轮把提供的工具全部调一遍：查询类直接查；发票按识别结果原样查验（不会修正易混字符）；
    追问员工用固定模板；
  - 第二轮把每类查询返回的第一条记录当作证据提交（不会在多条记录里挑选）。
评测时和大模型 Agent 对比，能看出模型的判断到底贡献了多少（见 docs/eval_log.md）。
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

QUESTIONS = {
    "R3": "打车超过 100 元的原因",
    "R2": "招待的人数（含我方）和对象",
    "R1": "住宿的晚数",
}


def _call(i: int, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(id=f"call_{i}_{name}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)))


def _reply(calls: list | None = None, content: str = "") -> SimpleNamespace:
    msg = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))


class ScriptedPlanner:
    model = "scripted"

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages: list[dict], tools: list[dict], **_) -> SimpleNamespace:
        offered = [t["function"]["name"] for t in tools]
        user = next(m["content"] for m in messages if m["role"] == "user")
        data = json.loads(re.search(r"```json\n(.*)\n```", user, re.S).group(1))
        results = [m for m in messages if m["role"] == "tool"]
        if not any(m["role"] == "assistant" for m in messages):
            return _reply(self._first_round(offered, data))
        found: dict[str, str] = {}
        for m in results:
            r = json.loads(m["content"])
            if r.get("records"):
                rec = r["records"][0]
                key = "preapproval_id" if str(rec.get("id", "")).startswith("PA") else "travel_request_id"
                found.setdefault(key, rec["id"])
            if r.get("found"):
                rec = r["record"]
                if "verification_id" in rec:
                    found.setdefault("verification_id", rec["verification_id"])
                else:
                    found.setdefault("merchant_id", rec["id"])
        return _reply([_call(99, "submit_findings", {**found, "summary": "按固定流程查询，提交每类查询的第一条结果"})])

    @staticmethod
    def _first_round(offered: list[str], data: dict) -> list:
        inv = data["发票识别结果"]
        calls = []
        for i, name in enumerate(offered):
            if name in ("search_preapprovals", "search_travel_requests"):
                calls.append(_call(i, name, {}))
            elif name == "verify_invoice":
                calls.append(_call(i, name, {"invoice_no": inv["发票号码"] or "", "issue_date": inv["开票日期"] or "",
                                             "total": inv["价税合计"] or 0}))
            elif name == "lookup_merchant":
                calls.append(_call(i, name, {"name": inv["销售方"] or ""}))
            elif name == "ask_employee":
                need = [QUESTIONS[x["条款"]] for x in data["待查问题"] if x["条款"] in QUESTIONS]
                calls.append(_call(i, name, {"question": "请补充说明" + "、".join(need or ["相关情况"]) + "。"}))
        return calls
