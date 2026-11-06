"""环境自检：依赖是否可导入、镜像是否可达、存储服务是否在线（后两项不在线只告警）。"""

from __future__ import annotations

import importlib
import os
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "fastapi", "uvicorn", "pydantic", "pydantic_settings", "sqlalchemy", "pymysql",
    "alembic", "redis", "openai", "httpx", "tenacity", "pymupdf", "PIL", "yaml",
    "loguru", "opentelemetry.sdk", "huggingface_hub", "pyarrow", "reportlab", "pytest",
]


def check_imports() -> bool:
    ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            print(f"    [OK] {name} {ver if isinstance(ver, str) else ''}".rstrip())
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"    [X]  {name}: {exc}")
    return ok


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def check_mirror() -> None:
    try:
        import httpx
    except ImportError:
        print("    [!]  httpx 未安装，跳过网络检查")
        return

    url = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        r = httpx.head(url, timeout=8, follow_redirects=True)
        print(f"    [OK] {url} 可访问（HTTP {r.status_code}）")
    except Exception as exc:  # noqa: BLE001
        print(f"    [!]  {url} 不可达：{exc}（下载数据集时可能失败）")


def main() -> int:
    print(f"    Python {sys.version.split()[0]}  @ {sys.executable}")
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    print(f"    项目所在盘剩余 {free_gb:.1f} GB")

    print("\n  依赖：")
    ok = check_imports()

    print("\n  网络：")
    check_mirror()

    print("\n  存储服务（未启动属正常，开发服务层时再启动）：")
    for name, port in [("MySQL", 3306), ("Redis", 6379), ("Jaeger OTLP", 4318)]:
        state = "在线" if port_open("127.0.0.1", port) else "未启动"
        print(f"    {name:<12} 127.0.0.1:{port}  {state}")

    print("\n  结果：" + ("依赖全部就绪" if ok else "有依赖缺失，请重新运行 setup.ps1"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
