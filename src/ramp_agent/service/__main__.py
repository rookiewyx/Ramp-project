"""审核服务命令行。

  python -m ramp_agent.service initdb            把数据库升级到最新结构
  python -m ramp_agent.service migrate           查看版本；-m 生成迁移脚本；--sql 导出升级 SQL
  python -m ramp_agent.service api               启动接口（http://127.0.0.1:8080/docs）
  python -m ramp_agent.service worker            启动审核 worker（需要 Redis）
  python -m ramp_agent.service requeue           补投卡住的单子、收尾追问超时的单子（可以放进定时任务，每分钟一次）
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ramp_agent.service")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("initdb", help="把数据库升级到最新结构（新库就是建表）")
    m = sub.add_parser("migrate", help="查看版本 / 生成迁移脚本 / 导出 SQL")
    m.add_argument("-m", "--message", help="改了 db.py 的模型后，用这条说明自动生成迁移脚本")
    m.add_argument("--sql", action="store_true", help="只输出升级 SQL，不执行（给 DBA 审核）")
    a = sub.add_parser("api")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--eager", action="store_true",
                   help="不用 Redis 和 worker，审核在接口进程里同步执行（等同于 TASK_EAGER=true）")
    w = sub.add_parser("worker")
    w.add_argument("--concurrency", type=int, default=8,
                   help="同时审核的笔数。审核主要在等模型返回，并发可以开到 CPU 核数的几倍")
    w.add_argument("--queues", default="default,vlm",
                   help="处理哪些队列：default（文字版 PDF、无附件）、vlm（拍照件）。"
                        "生产上分开起两组 worker，vlm 组的并发按视觉模型配额设置")
    w.add_argument("--pool", choices=["threads", "prefork", "solo"], default="threads",
                   help="threads（默认）：审核是 I/O 密集型，线程池省内存；Windows 只能用 threads 或 solo")
    r = sub.add_parser("requeue")
    r.add_argument("--older-than", type=int, default=60, help="受理超过多少秒还没处理的单子")
    args = p.parse_args(argv)

    import os

    from ..config import get_settings

    if getattr(args, "eager", False):
        os.environ["TASK_EAGER"] = "true"
        get_settings.cache_clear()
    s = get_settings()
    if args.cmd == "initdb":
        from sqlalchemy.engine import make_url

        from .db import resolve_url
        from .migrate import current, upgrade

        upgrade(s.database_url)
        where = make_url(resolve_url(s.database_url)).render_as_string(hide_password=True)   # 不打印密码
        print(f"[service] 数据库已是最新结构（版本 {current(s.database_url)}）：{where}")
    elif args.cmd == "migrate":
        from . import migrate

        if args.sql:
            migrate.sql(s.database_url)
        elif args.message:
            migrate.revision(s.database_url, args.message)
            print("[service] 已生成迁移脚本，在 src/ramp_agent/service/migrations/versions/，提交前请人工检查")
        else:
            print(f"[service] 当前版本 {migrate.current(s.database_url)}，最新版本 {migrate.head()}")
    elif args.cmd == "api":
        import uvicorn

        from ..telemetry import FreshTraceContext
        from .api import create_app

        uvicorn.run(FreshTraceContext(create_app(s)), host=args.host, port=args.port)
    elif args.cmd == "worker":
        from .db import get_engine
        from .migrate import ensure_schema
        from .tasks import app

        ensure_schema(s.database_url, s.auto_migrate)
        pool = args.pool
        if sys.platform == "win32" and pool == "prefork":
            sys.exit("Windows 上不支持 prefork，请用 --pool threads")
        if pool == "prefork":
            get_engine(s.database_url).dispose()    # 连接不能跨 fork 共享，子进程各自重新建立
        else:
            from ..telemetry import setup_tracing  # 线程池不 fork，在主进程初始化即可

            setup_tracing("worker", s)
        app.worker_main(["worker", "--loglevel=INFO", f"--pool={pool}", f"-Q{args.queues}",
                         f"--concurrency={args.concurrency}",
                         f"--hostname={args.queues.replace(',', '+')}@%h"])
    elif args.cmd == "requeue":
        from .tasks import requeue

        from .tasks import expire_waiting

        print(f"[service] 补投 {requeue(args.older_than)} 笔；追问员工超时收尾 {expire_waiting()} 笔")


if __name__ == "__main__":
    main()
