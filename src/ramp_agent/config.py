"""统一配置：从项目根目录的 .env 读取，字段名与 .env.example 一一对应。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8",
                                      extra="ignore")

    # 文本模型（制度审核用）
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # 视觉模型（扫描件 / 照片识别用）
    vlm_base_url: str = "http://127.0.0.1:8001/v1"
    vlm_api_key: str = "EMPTY"
    vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_timeout_s: float = 60.0
    vlm_max_retries: int = 2
    # 每个进程同时在途的模型请求上限，按模型服务的并发配额设置；0 表示不限制（见 limits.py）
    vlm_max_concurrency: int = 0
    llm_max_concurrency: int = 0

    # 评测平台的数据库；默认是项目里的 SQLite 文件，服务化时可改成 MySQL 地址
    evalhub_url: str = "sqlite:///data/evalhub.db"

    # 存储
    mysql_url: str = "mysql+pymysql://ramp:ramp_dev_pw@127.0.0.1:3306/ramp?charset=utf8mb4"
    redis_url: str = "redis://127.0.0.1:6379/0"

    # ---------------- 审核服务（模块三） ----------------
    # 业务库：默认 SQLite 方便本地跑；docker compose 起了 MySQL 后改成 MYSQL_URL 的值
    database_url: str = "sqlite:///data/ramp.db"
    # 启动时自动把数据库升级到最新结构。本地开发方便；部署时设为 false，由单独的 migrate 步骤执行
    auto_migrate: bool = True
    # 发票文件存储：local 存到 storage_dir；s3 存到任何 S3 兼容存储（RustFS、云厂商对象存储……）
    storage_backend: str = "local"
    storage_dir: str = "data/objects"
    s3_endpoint: str = "127.0.0.1:9000"
    s3_access_key: str = "ramp"
    s3_secret_key: str = "ramp_dev_secret"
    s3_bucket: str = "ramp-invoices"
    s3_secure: bool = False
    # 任务队列：eager=true 时在 API 进程里同步执行，不需要 Redis 和 worker（本地开发、测试用）
    task_eager: bool = False
    task_max_retries: int = 3
    # 每个 worker 的任务速率上限（Celery 写法，如 "30/s"），默认不限。
    # 压测发现按任务限速会把不调模型的单子也一起卡住；保护模型接口请用 VLM/LLM_MAX_CONCURRENCY
    task_rate_limit: str = ""
    stale_after_s: int = 600               # 处理中超过这个时间没完成，视为 worker 崩溃，可被重新领取
    # 发票查重登记：sql（数据库唯一索引，默认，持久）或 redis（SET NX）
    registry_backend: str = "sql"
    max_upload_mb: int = 10
    # 接口鉴权：留空表示不校验（本地开发）；多个 key 用逗号分隔
    api_keys: str = ""

    # ---------------- 链路追踪（OpenTelemetry） ----------------
    otel_enabled: bool = False
    otel_exporter: str = "otlp"                      # otlp：发给 Jaeger；file：写 JSONL，用于离线分析
    otel_exporter_otlp_endpoint: str = "http://127.0.0.1:4318"
    otel_service_name: str = "ramp-agent"
    otel_file: str = "reports/traces.jsonl"

    # ---------------- 调查 Agent（转人工的单子先补证据再重审，见 agent/） ----------------
    agent_enabled: bool = True
    agent_max_steps: int = 8               # 模型最多几轮（每轮可以调用多个工具）
    # 公司系统（出差申请、预审批、发票查验、商户登记）的数据。真实部署换成对接 OA / 税务查验的接口；
    # 这里用 JSON 文件模拟，留空表示没有对接，Agent 只能追问员工
    enterprise_fixture: str = ""
    agent_reply_timeout_h: int = 72        # 追问员工后多久没答复就不再等，交给财务

    # 公司主体信息（用于校验发票抬头，与 policies/expense_policy.md 一致）
    company_name: str = "星河数智科技有限公司"
    company_tax_id: str = "91310000MA1XHSZ00X"


@lru_cache
def get_settings() -> Settings:
    return Settings()
