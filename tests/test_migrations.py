"""数据库迁移：迁移脚本建出来的结构必须和模型一致；旧库能平滑接上；版本不对时拒绝启动。

默认在临时 SQLite 上跑；CI 里设置 RAMP_TEST_DATABASE_URL 后，同样的测试在 MySQL 和 PostgreSQL 上再跑一遍。
"""

from __future__ import annotations

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, text

from ramp_agent.service import migrate
from ramp_agent.service.db import Base, Expense, get_engine, session


def diff(url: str) -> list:
    with get_engine(url).connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        return compare_metadata(ctx, Base.metadata)


def test_migrations_build_exactly_the_models(db_url):
    migrate.upgrade(db_url)
    assert migrate.current(db_url) == migrate.head()
    assert diff(db_url) == []          # 改了 db.py 却忘了写迁移脚本，这里会失败


def test_downgrade_and_upgrade_round_trip(db_url):
    from alembic import command

    migrate.upgrade(db_url)
    command.downgrade(migrate.config(db_url), "base")
    assert "expenses" not in inspect(get_engine(db_url)).get_table_names()
    migrate.upgrade(db_url)
    assert diff(db_url) == []


def test_legacy_create_all_database_is_adopted(db_url):
    """引入迁移之前用 create_all 建的库：不重建、不丢数据，直接标记为第一版再往上升。

    那时的结构就是第一版：先升到 0001 再删掉版本表，模拟当时 create_all 建出来的库。
    （不能直接用现在的 Base.metadata.create_all，它已经包含后面版本加的表和字段。）"""
    migrate.upgrade(db_url, migrate.BASELINE)
    with get_engine(db_url).begin() as conn:
        conn.execute(text("DROP TABLE alembic_version"))
        conn.execute(text("INSERT INTO expenses (id, request_hash, report_id, employee_id, employee_name, "
                          "category, amount, submit_date, note, status, attempts, created_at, updated_at) "
                          "VALUES ('legacy1', '', '', '', '', 'meal', 88, '2026-11-01', '', 'done', 0, "
                          "'2026-11-01 00:00:00', '2026-11-01 00:00:00')"))
    migrate.upgrade(db_url)
    assert migrate.current(db_url) == migrate.head()
    with session(db_url) as s:
        assert s.get(Expense, "legacy1") is not None


def test_check_only_mode_refuses_outdated_schema(db_url):
    with pytest.raises(migrate.SchemaOutdated):
        migrate.ensure_schema(db_url, auto=False)
    migrate.ensure_schema(db_url, auto=True)
    migrate.ensure_schema(db_url, auto=False)           # 已是最新，不报错


def test_offline_sql_export(capsys):
    migrate.sql("sqlite://")
    out = capsys.readouterr().out
    assert "CREATE TABLE expenses" in out and "alembic_version" in out
