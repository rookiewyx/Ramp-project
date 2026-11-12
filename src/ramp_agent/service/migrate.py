"""数据库结构迁移（Alembic）。

为什么不再用 create_all：create_all 只会建不存在的表，已有的表加字段、改类型它不管。
服务上线以后，表结构的每次变化都要有一个迁移脚本，按顺序执行，可以回滚，可以先导出 SQL 给 DBA 审核。

    python -m ramp_agent.service initdb                 # 升级到最新（新库就是建表）
    python -m ramp_agent.service migrate --status       # 当前版本和最新版本
    python -m ramp_agent.service migrate -m "加字段"    # 改了 db.py 的模型后，自动生成迁移脚本
    python -m ramp_agent.service migrate --sql          # 只输出升级 SQL，不执行

本地开发（AUTO_MIGRATE=true，默认）：API 和 worker 启动时自动升级。
部署（docker compose）：由单独的 migrate 容器升级一次，API 和 worker 只检查版本，
不一致就拒绝启动——避免多个进程同时改表。
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect

from .db import get_engine, resolve_url

MIGRATIONS = Path(__file__).parent / "migrations"
# 第一个版本。引入 Alembic 之前用 create_all 建的库，结构和它一致，直接标记为这个版本
BASELINE = "0001"


class SchemaOutdated(RuntimeError):
    pass


def config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", resolve_url(url).replace("%", "%%"))
    return cfg


def head() -> str:
    return ScriptDirectory.from_config(config("sqlite://")).get_current_head()


def current(url: str) -> str | None:
    with get_engine(url).connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def upgrade(url: str, target: str = "head") -> None:
    cfg = config(url)
    with get_engine(url).begin() as conn:
        tables = set(inspect(conn).get_table_names())
        cfg.attributes["connection"] = conn
        if "expenses" in tables and "alembic_version" not in tables:
            command.stamp(cfg, BASELINE)          # 旧库：结构已经是第一版，只补上版本记录
        command.upgrade(cfg, target)


def ensure_schema(url: str, auto: bool) -> None:
    """启动时调用。auto=True 时升级到最新；否则只检查，不是最新就拒绝启动。"""
    if auto:
        upgrade(url)
        return
    cur, latest = current(url), head()
    if cur != latest:
        raise SchemaOutdated(f"数据库结构版本是 {cur}，代码需要 {latest}。"
                             "请先运行 python -m ramp_agent.service initdb")


def revision(url: str, message: str) -> None:
    """对比模型和数据库，自动生成迁移脚本。生成后一定要人工检查一遍再提交。"""
    upgrade(url)
    command.revision(config(url), message=message, autogenerate=True)


def sql(url: str) -> None:
    command.upgrade(config(url), "head", sql=True)
