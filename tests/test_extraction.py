from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from ramp_agent.extraction import (
    InvoiceExtractor,
    InvoiceFields,
    LineItem,
    Tier,
    VLMOutput,
    validate,
)
from ramp_agent.extraction.normalize import parse_amount, parse_date, parse_rate
from ramp_agent.extraction.pdf_text import has_text_layer, parse_einvoice_text, read_pdf_text
from ramp_agent.extraction.vlm import OpenAICompatibleVLM, parse_json_object


# ---------------- 归一化 ----------------
@pytest.mark.parametrize("raw,expected", [
    ("¥1,739.00", 1739.0), ("5,046.28 CNY", 5046.28), (1739, 1739.0), ("无", None), (None, None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("2026年9月19日", "2026-09-19"), ("2026/09/19", "2026-09-19"), ("2026-02-30", None), ("", None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw,expected", [("6%", 0.06), (6, 0.06), (0.13, 0.13), ("0.09", 0.09)])
def test_parse_rate(raw, expected):
    assert parse_rate(raw) == expected


# ---------------- 校验 ----------------
def good_fields(**kw) -> InvoiceFields:
    base = {"invoice_no": "27298069901627204653", "issue_date": "2026-09-19", "seller": "禾田简餐",
            "buyer_name": "星河数智科技有限公司", "buyer_tax_id": "91310000MA1XHSZ00X",
            "items": [LineItem(name="餐饮服务", amount=1095.0), LineItem(name="白酒", amount=644.0)],
            "total": 1739.0, "tax_rate": 0.06}
    return InvoiceFields(**{**base, **kw})


def test_validate_passes_good_invoice():
    assert validate(good_fields()) == ([], [])


def test_validate_catches_sum_mismatch():
    failed, suspicious = validate(good_fields(total=1793.0))     # 典型的数字看错位
    assert any(f.startswith("sum:") for f in failed)
    assert {"items", "total"} <= set(suspicious)


def test_validate_catches_missing_and_bad_format():
    failed, suspicious = validate(good_fields(invoice_no="1234", buyer_name=None))
    assert "format:invoice_no" in failed and "missing:buyer_name" in failed
    assert {"invoice_no", "buyer_name"} <= set(suspicious)


def test_validate_non_cn_mode_skips_number_format():
    failed, _ = validate(good_fields(invoice_no="201211", buyer_tax_id=None), strict_cn=False)
    assert failed == []


COMPANY_TID = "91310000MA1XHSZ00X"


@pytest.mark.parametrize("read_as", ["91310000MA1XHS200X", "91310000MA1XH5Z00X"])
def test_confusable_tax_id_is_flagged_not_corrected(read_as):
    """评测中真实出现过的误读：Z→2、S→5。要标为可疑交人工，不能静默通过，也不能自动改正。"""
    f = good_fields(buyer_tax_id=read_as)
    failed, suspicious = validate(f, known_tax_ids=(COMPANY_TID,))
    assert "buyer_tax_id" in suspicious and failed[0].startswith("confusable:")
    assert f.buyer_tax_id == read_as


def test_genuinely_different_tax_id_is_not_confusable():
    """真的开错了税号（R4 驳回场景）不该被当成误读。"""
    failed, _ = validate(good_fields(buyer_tax_id="91310000MA1XHSZ7Q1"), known_tax_ids=(COMPANY_TID,))
    assert failed == []


# ---------------- 第一级：文字版 PDF ----------------
def test_pdf_text_layer_parsed_exactly(sample_pdf, sample_invoice):
    text = read_pdf_text(sample_pdf)
    assert has_text_layer(text)
    f = parse_einvoice_text(text)
    assert f.invoice_no == sample_invoice.invoice_no
    assert f.issue_date == sample_invoice.issue_date
    assert f.buyer_tax_id == sample_invoice.buyer_tax_id
    assert f.total == sample_invoice.total and f.tax_rate == 0.06
    assert [(i.name, i.amount) for i in f.items] == [("餐饮服务", 1095.0), ("白酒", 644.0)]


def test_unknown_layout_does_not_crash():
    f = parse_einvoice_text("某某公司收据\n金额：100\n谢谢惠顾")
    assert f.items == [] and f.invoice_no is None
    assert validate(f)[0]          # 解析不出来就过不了校验，路由会升级到视觉模型


# ---------------- 路由 ----------------
class FakeVLM:
    """假的视觉模型：返回预设结果，记录被调用的次数。"""

    def __init__(self, fields: InvoiceFields | None = None, exc: Exception | None = None):
        self.fields, self.exc, self.calls = fields, exc, 0

    def extract(self, image: bytes) -> VLMOutput:
        self.calls += 1
        if self.exc:
            raise self.exc
        return VLMOutput(fields=self.fields, prompt_tokens=800, completion_tokens=120)


def test_text_pdf_never_calls_vlm(sample_pdf):
    vlm = FakeVLM(good_fields())
    r = InvoiceExtractor(vlm).extract(sample_pdf)
    assert r.tier == Tier.PDF_TEXT and r.model_calls == 0 and vlm.calls == 0
    assert not r.needs_review


def test_scan_goes_to_vlm(sample_scan):
    vlm = FakeVLM(good_fields())
    r = InvoiceExtractor(vlm).extract(sample_scan)
    assert r.tier == Tier.VLM and vlm.calls == 1 and r.prompt_tokens == 800
    assert not r.needs_review


def test_vlm_misread_is_flagged_not_silent(sample_scan):
    r = InvoiceExtractor(FakeVLM(good_fields(total=1793.0))).extract(sample_scan)
    assert r.tier == Tier.VLM and r.needs_review and "total" in r.uncertain_fields


def test_vlm_error_fails_gracefully(sample_scan):
    r = InvoiceExtractor(FakeVLM(exc=TimeoutError())).extract(sample_scan)
    assert r.tier == Tier.FAILED and r.error == "vlm_error:TimeoutError" and r.needs_review


def test_force_vlm_routes_text_pdf_to_vlm(sample_pdf):
    vlm = FakeVLM(good_fields())
    r = InvoiceExtractor(vlm, force_vlm=True).extract(sample_pdf)
    assert r.tier == Tier.VLM and vlm.calls == 1


def test_bad_text_parse_falls_back_to_vlm(tmp_path):
    import pymupdf

    path = tmp_path / "odd.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Receipt No. 88 Total 100.00 thank you for shopping here")
    doc.save(path)
    vlm = FakeVLM(good_fields())
    r = InvoiceExtractor(vlm).extract(path)
    assert r.tier == Tier.VLM and r.fallback_from == Tier.PDF_TEXT and vlm.calls == 1


def test_missing_file_and_no_vlm(sample_scan):
    assert InvoiceExtractor().extract(None).error == "no_file"
    r = InvoiceExtractor(vlm=None).extract(sample_scan)
    assert r.tier == Tier.FAILED and r.error == "vlm_unavailable"


# ---------------- 视觉模型客户端（模拟 HTTP，不需要真实服务） ----------------
def test_parse_json_object_tolerates_wrapping():
    assert parse_json_object('好的：\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}


def make_vlm(replies: list[str]) -> tuple[OpenAICompatibleVLM, list[dict]]:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        content = replies[min(len(sent), len(replies)) - 1]
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000}})

    vlm = OpenAICompatibleVLM.__new__(OpenAICompatibleVLM)
    vlm.model = "fake"
    vlm.client = OpenAI(base_url="http://fake/v1", api_key="x", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return vlm, sent


def test_vlm_client_request_and_parsing(sample_scan):
    reply = json.dumps({"invoice_no": "2729 8069 9016 2720 4653", "issue_date": "2026年9月19日",
                        "seller": "禾田简餐", "buyer_name": "星河数智科技有限公司",
                        "buyer_tax_id": "91310000MA1XHSZ00X", "total": "¥1,739.00",
                        "tax_rate": "6%", "items": [{"name": "餐饮服务", "amount": 1095},
                                                    {"name": "白酒", "amount": "644.00"}]},
                       ensure_ascii=False)
    vlm, sent = make_vlm([reply])
    out = vlm.extract(sample_scan.read_bytes())
    assert out.fields == good_fields()          # 归一化后和标准答案完全一致
    assert out.prompt_tokens == 900
    content = sent[0]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vlm_client_retries_once_on_bad_json(sample_scan):
    vlm, sent = make_vlm(["抱歉我看不清", '{"invoice_no": null, "total": 12}'])
    out = vlm.extract(sample_scan.read_bytes())
    assert len(sent) == 2 and out.fields.total == 12.0 and out.prompt_tokens == 1800


def test_missing_tax_id_is_suspicious_on_cn_invoice():
    from ramp_agent.extraction.schema import InvoiceFields
    from ramp_agent.extraction.validate import validate
    f = InvoiceFields(invoice_no="12345678", issue_date="2026-10-20", buyer_name="星河数智科技有限公司",
                      buyer_tax_id="", total=88.0)
    failed, suspicious = validate(f)
    assert "missing:buyer_tax_id" in failed and "buyer_tax_id" in suspicious
    assert "buyer_tax_id" not in validate(f, strict_cn=False)[1]      # 非中国票据（如 CORD 小票）不要求税号
