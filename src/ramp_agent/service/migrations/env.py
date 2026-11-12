"""Alembic 迁移环境。由 ramp_agent.service.migrate 以编程方式调用，不需要 alembic.ini。"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from ramp_agent.service.db import Base

config = context.config
target_metadata = Base.metadata


def _configure(**kw) -> None:
    context.configure(
        target_metadata=target_metadata,
        compare_type=True,                 # 字段类型变了也要生成迁移
        render_as_batch=True,              # SQLite 不支持大部分 ALTER，用“建新表、拷数据”的方式改表
        **kw,
    )


def run_offline() -> None:
    """只输出 SQL，不连数据库（给 DBA 审核用：python -m ramp_agent.service migrate --sql）。"""
    _configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    conn = config.attributes.get("connection")
    if conn is not None:                   # 调用方已经给了连接（和外层共用一个事务）
        _configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.connect() as c:
        _configure(connection=c)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
