"""评测平台的表结构。

开发阶段默认存成 data/evalhub.db（SQLite），不需要开 Docker；
服务化时把 EVALHUB_URL 改成 MySQL 地址即可，代码不用动。
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from ..config import ROOT, get_settings


def now() -> datetime:
    return datetime.now().replace(microsecond=0)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    task: Mapped[str] = mapped_column(String(20))           # extraction / policy
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    versions: Mapped[list[DatasetVersion]] = relationship(back_populates="dataset",
                                                          order_by="DatasetVersion.version")


class DatasetVersion(Base):
    """评测集的一个版本。冻结（frozen）后不能再增删用例或改标注，保证同一版本上的分数可以互相比较。"""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(10), default="draft")    # draft / frozen
    source: Mapped[str] = mapped_column(Text, default="")              # 从哪个文件导入 / 从哪个版本派生
    note: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    cases: Mapped[list[Case]] = relationship(back_populates="version", order_by="Case.id")

    @property
    def ref(self) -> str:
        return f"{self.dataset.name}@v{self.version}"


class Case(Base):
    __tablename__ = "cases"
    __table_args__ = (UniqueConstraint("version_id", "key"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    key: Mapped[str] = mapped_column(String(100))            # 业务上的唯一标识，如报销单号、发票号
    payload: Mapped[dict] = mapped_column(JSON)              # 输入（以及识别任务的标准答案）
    tags: Mapped[dict] = mapped_column(JSON, default=dict)   # scenario / origin / family …
    version: Mapped[DatasetVersion] = relationship(back_populates="cases")
    labels: Mapped[list[Label]] = relationship(back_populates="case", order_by="Label.id")


class Label(Base):
    """人工（或生成脚本）给出的标准答案。同一条用例可以有多个标注人，用来算一致性。"""

    __tablename__ = "labels"
    __table_args__ = (UniqueConstraint("case_id", "labeler"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    labeler: Mapped[str] = mapped_column(String(50))
    decision: Mapped[str] = mapped_column(String(10))        # approve / review / reject
    rules: Mapped[list] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(Text, default="")
    unsure: Mapped[bool] = mapped_column(Boolean, default=False)   # 标注人自己也拿不准
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)
    case: Mapped[Case] = relationship(back_populates="labels")


class Run(Base):
    """一次评测运行。记录足够的版本信息，让任何一个数字都能追溯到“当时用的是什么”。"""

    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    task: Mapped[str] = mapped_column(String(20))
    version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    status: Mapped[str] = mapped_column(String(10), default="running")   # running / done / failed
    note: Mapped[str] = mapped_column(Text, default="")
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    code_hash: Mapped[str] = mapped_column(String(64), default="")       # src/ 下全部代码
    prompt_hash: Mapped[str] = mapped_column(String(64), default="")     # 提示词
    policy_hash: Mapped[str] = mapped_column(String(64), default="")     # policies/policy.yaml
    models: Mapped[dict] = mapped_column(JSON, default=dict)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    gold: Mapped[str] = mapped_column(String(50), default="auto")        # 用哪个标注人的答案打分
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    wall_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    version: Mapped[DatasetVersion] = relationship()
    results: Mapped[list[RunResult]] = relationship(back_populates="run", order_by="RunResult.id")


class RunResult(Base):
    __tablename__ = "run_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    output: Mapped[dict] = mapped_column(JSON)
    predicted: Mapped[str | None] = mapped_column(String(20), nullable=True)
    gold: Mapped[str | None] = mapped_column(String(20), nullable=True)   # None = 没有可用答案（未标注或有争议）
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    run: Mapped[Run] = relationship(back_populates="results")
    case: Mapped[Case] = relationship()


def default_url() -> str:
    url = get_settings().evalhub_url
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        path = ROOT / url.removeprefix("sqlite:///")       # 相对路径以项目根目录为准
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path.as_posix()}"
    return url


def make_engine(url: str | None = None):
    url = url or default_url()
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):                       # SQLite 默认不检查外键
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    return engine


@lru_cache
def _factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(make_engine(url), expire_on_commit=False)


def session(url: str | None = None) -> Session:
    return _factory(url)()
