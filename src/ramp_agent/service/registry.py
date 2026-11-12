"""服务用的发票登记（R6 查重），实现 policy.registry.InvoiceRegistry 接口。

SqlRegistry（默认）：发票号做主键，靠唯一约束判重。多个 worker 同时审核同一张发票的两笔报销，
  只有一个 INSERT 能成功，另一个一定会看到对方——检查和登记是一个原子操作。数据落在业务库里，持久。
RedisRegistry：SET NX，速度快，适合把查重放在数据库之前挡一层；Redis 没开持久化时重启会丢登记。

同一笔报销重新审核（重试、人工触发复审）时，发票已经登记在它自己名下，不算重复。
"""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError

from .db import InvoiceClaim, session


class SqlRegistry:
    def __init__(self, url: str):
        self.url = url

    def claim(self, invoice_no: str, expense_id: str) -> str | None:
        try:
            with session(self.url) as s:
                s.add(InvoiceClaim(invoice_no=invoice_no, expense_id=expense_id))
            return None
        except IntegrityError:
            with session(self.url) as s:
                owner = s.get(InvoiceClaim, invoice_no)
            if owner is None:            # 极少见：刚好被删除，按未登记处理再试一次
                return self.claim(invoice_no, expense_id)
            return None if owner.expense_id == expense_id else owner.expense_id


class RedisRegistry:
    def __init__(self, url: str, prefix: str = "ramp:invoice:"):
        import redis

        self.r = redis.Redis.from_url(url)
        self.prefix = prefix

    def claim(self, invoice_no: str, expense_id: str) -> str | None:
        key = self.prefix + invoice_no
        if self.r.set(key, expense_id, nx=True):
            return None
        owner = self.r.get(key)
        owner = owner.decode() if isinstance(owner, bytes) else owner
        return None if owner == expense_id else owner
