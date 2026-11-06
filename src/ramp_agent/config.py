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

    # 评测平台的数据库；默认是项目里的 SQLite 文件，服务化时可改成 MySQL 地址
    evalhub_url: str = "sqlite:///data/evalhub.db"

    # 存储
    mysql_url: str = "mysql+pymysql://ramp:ramp_dev_pw@127.0.0.1:3306/ramp?charset=utf8mb4"
    redis_url: str = "redis://127.0.0.1:6379/0"

    # 公司主体信息（用于校验发票抬头，与 policies/expense_policy.md 一致）
    company_name: str = "星河数智科技有限公司"
    company_tax_id: str = "91310000MA1XHSZ00X"


@lru_cache
def get_settings() -> Settings:
    return Settings()
