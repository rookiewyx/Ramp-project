from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def sample_invoice():
    """和 gen_synthetic.py 生成的发票同结构的一张样例。"""
    from gen_synthetic import COMPANY, COMPANY_TAX_ID, Invoice

    return Invoice(
        invoice_no="27298069901627204653", issue_date="2026-09-19", seller="禾田简餐",
        buyer_name=COMPANY, buyer_tax_id=COMPANY_TAX_ID,
        items=[{"name": "餐饮服务", "amount": 1095.0}, {"name": "白酒", "amount": 644.0}],
        total=1739.0, tax_rate=0.06,
    )


@pytest.fixture
def sample_pdf(tmp_path, sample_invoice):
    from gen_synthetic import render_pdf

    path = tmp_path / "inv.pdf"
    render_pdf(sample_invoice, path)
    return path


@pytest.fixture
def sample_scan(tmp_path, sample_pdf):
    import random

    from gen_synthetic import render_scan

    path = tmp_path / "inv.jpg"
    render_scan(sample_pdf, path, random.Random(0))
    return path


@pytest.fixture
def make_pdf(tmp_path):
    """生成一张文字版 PDF 发票，返回文件内容（审核服务的测试用）。"""
    from gen_synthetic import COMPANY, COMPANY_TAX_ID, Invoice, render_pdf

    def _make(invoice_no="89000000000000000001", items=(("餐饮服务", 88.0),), issue="2026-10-28"):
        inv = Invoice(invoice_no=invoice_no, issue_date=issue, seller="样例商户", buyer_name=COMPANY,
                      buyer_tax_id=COMPANY_TAX_ID, items=[{"name": n, "amount": a} for n, a in items],
                      total=round(sum(a for _, a in items), 2), tax_rate=0.06)
        path = tmp_path / f"{invoice_no}.pdf"
        render_pdf(inv, path)
        return path.read_bytes()
    return _make


@pytest.fixture
def db_url(tmp_path):
    """业务库地址。默认是临时 SQLite；设置了 RAMP_TEST_DATABASE_URL（CI 里的 MySQL / PostgreSQL）就用它，
    用前用后清空所有表，保证每个测试从空库开始。"""
    import os

    url = os.environ.get("RAMP_TEST_DATABASE_URL")
    if not url:
        yield f"sqlite:///{(tmp_path / 't.db').as_posix()}"
        return
    _drop_all(url)
    yield url
    _drop_all(url)


def _drop_all(url: str) -> None:
    from sqlalchemy import MetaData

    from ramp_agent.service.db import get_engine

    engine = get_engine(url)
    meta = MetaData()
    meta.reflect(engine)
    meta.drop_all(engine)            # 按外键依赖倒序删，连 alembic_version 一起
