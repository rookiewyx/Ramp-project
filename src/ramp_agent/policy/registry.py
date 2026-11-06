"""发票登记：发现重复报销（R6）。

接口先定下来，现在用内存实现跑评测；接入服务时换成 Redis（SET NX，原子操作，多实例并发提交也不会漏判）
或 MySQL 唯一索引，审核代码不用改。
"""

from __future__ import annotations

import threading
from typing import Protocol


class InvoiceRegistry(Protocol):
    def claim(self, invoice_no: str, expense_id: str) -> str | None:
        """登记发票号。已被别的报销登记过时，返回那笔报销的 ID；否则登记并返回 None。"""
        ...


class InMemoryRegistry:
    def __init__(self) -> None:
        self._owner: dict[str, str] = {}
        self._lock = threading.Lock()     # 检查和登记必须是一个原子操作

    def claim(self, invoice_no: str, expense_id: str) -> str | None:
        with self._lock:
            owner = self._owner.get(invoice_no)
            if owner is not None and owner != expense_id:
                return owner
            self._owner[invoice_no] = expense_id
            return None
