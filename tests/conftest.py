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
