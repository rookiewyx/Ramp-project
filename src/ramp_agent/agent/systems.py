"""调查 Agent 能查的公司系统，以及向员工追问的通道。

真实部署里这些是对接 OA（出差申请、预审批）、税务发票查验平台、工商登记查询的 HTTP 接口；
这里先定接口，用一份 JSON 数据实现（评测时每条用例自带一份，服务里从 ENTERPRISE_FIXTURE 文件读）。

最小权限：查询接口按员工过滤，Agent 只能看到提交这笔报销的员工本人的记录，
工具也不接受员工编号参数（见 tools.py），模型没办法去查别人的数据。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Protocol

from ..policy.config import same_name


class EnterpriseSystems(Protocol):
    def travel_requests(self, employee_id: str) -> list[dict]:
        """员工的出差申请：id、city、start、end、status（approved / pending / cancelled）、purpose、changes。"""
        ...

    def preapprovals(self, employee_id: str) -> list[dict]:
        """员工的费用预审批：id、category、amount_cap、status、valid_from、valid_to、approver、purpose。"""
        ...

    def verify_invoice(self, invoice_no: str, issue_date: str, total: float) -> dict | None:
        """发票查验：三项与税务登记完全一致才返回登记的票面信息，否则 None。"""
        ...

    def merchant(self, name: str) -> dict | None:
        """销售方工商登记：id、name、category、scope。"""
        ...


class FixtureSystems:
    """用一份 JSON 数据模拟公司系统：{"travel_requests": [...], "preapprovals": [...],
    "invoices": [...], "merchants": [...]}。缺哪一类就当作那个系统里没有记录。"""

    def __init__(self, data: dict | None = None):
        self.data = data or {}

    @classmethod
    def from_file(cls, path: str | Path | None) -> FixtureSystems:
        if not path or not Path(path).exists():
            return cls()
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _mine(self, kind: str, employee_id: str) -> list[dict]:
        return [copy.deepcopy(r) for r in self.data.get(kind, [])
                if employee_id and r.get("employee_id") == employee_id]

    def travel_requests(self, employee_id: str) -> list[dict]:
        return self._mine("travel_requests", employee_id)

    def preapprovals(self, employee_id: str) -> list[dict]:
        return self._mine("preapprovals", employee_id)

    def verify_invoice(self, invoice_no: str, issue_date: str, total: float) -> dict | None:
        try:
            cents = round(float(total) * 100)
        except (TypeError, ValueError):
            return None
        for r in self.data.get("invoices", []):
            if (r.get("invoice_no") == str(invoice_no).strip() and r.get("issue_date") == str(issue_date).strip()
                    and round(float(r.get("total", 0)) * 100) == cents):
                rec = copy.deepcopy(r)
                # 查验流水号：同一张发票每次查到的都一样，Agent 提交证据时引用它
                rec["verification_id"] = "VF" + hashlib.sha256(rec["invoice_no"].encode()).hexdigest()[:8].upper()
                return rec
        return None

    def merchant(self, name: str) -> dict | None:
        for r in self.data.get("merchants", []):
            if same_name(r.get("name"), name):
                return copy.deepcopy(r)
        return None


# ---------------- 向员工追问 ----------------
class AwaitingEmployee(Exception):
    """问题已经发给员工，答复要过一段时间才来：调查暂停，保存进度，等答复到了再接着做。"""

    def __init__(self, question: str):
        super().__init__(question)
        self.question = question


class EmployeeChannel(Protocol):
    def ask(self, question: str) -> str | None:
        """返回员工的答复；None 表示员工没有答复（超时）。答复要等待时抛 AwaitingEmployee。"""
        ...


class ScriptedEmployee:
    """评测用：员工的答复事先写在用例里，不管问什么都这样回答；没写表示员工不回复。"""

    def __init__(self, reply: str | None):
        self.reply = reply
        self.questions: list[str] = []

    def ask(self, question: str) -> str | None:
        self.questions.append(question)
        return self.reply


class AsyncEmployee:
    """服务用：第一次追问时把问题发出去并暂停；员工答复后（reply 有值）恢复调查时直接返回答复。"""

    def __init__(self, reply: str | None = None, timed_out: bool = False):
        self.reply = reply
        self.timed_out = timed_out

    def ask(self, question: str) -> str | None:
        if self.reply is not None:
            return self.reply
        if self.timed_out:
            return None
        raise AwaitingEmployee(question)
