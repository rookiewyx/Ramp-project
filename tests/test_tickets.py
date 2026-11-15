"""交通票据识别：归一化、可疑字段、评测比较。视觉模型用假的。"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from ramp_agent.evalhub.adapters.extraction import extraction_metrics, run_ticket, ticket_field_equal
from ramp_agent.extraction.tickets import (
    TicketFields,
    check,
    extract_ticket,
    parse_cn_amount,
    station,
    ticket_amount,
    ticket_date,
)


@pytest.mark.parametrize("text, value", [
    ("壹拾元", 10), ("伍元整", 5), ("拾元", 10), ("贰元", 2), ("壹佰元整", 100), ("伍拾元", 50),
    ("贰拾伍元伍角", 25.5), ("壹仟贰佰元", 1200), ("52.00元", None), ("", None),
])
def test_chinese_capital_amount(text, value):
    assert parse_cn_amount(text) == value


def test_amount_and_date_normalization():
    assert [ticket_amount(x) for x in ("52.00元", "￥98.0元", "壹拾元", "51", 43, "18:00", None)] == \
        [52.0, 98.0, 10.0, 51.0, 43.0, None, None]
    assert [ticket_date(x) for x in ("2021年03月05日", "2020-12-30", "21-01-24", "20210305", "2021·3·5",
                                     "2021年04月02日10:29开", "03-05", "#-03-05", "2021-13-40")] == \
        ["2021-03-05", "2020-12-30", "2021-01-24", "2021-03-05", "2021-03-05", "2021-04-02", None, None, None]
    assert station("T2拉萨-贡嘎") == "拉萨贡嘎" == station("拉萨贡嘎") and station("万州北站") == "万州北"
    assert station("巫山(((高铁快巴))") == "巫山"


def test_station_fields_are_compared_leniently_but_codes_strictly():
    assert ticket_field_equal("origin", "万州北", "万州北站")
    assert ticket_field_equal("origin", "拉萨", "T2拉萨-贡嘎")
    assert not ticket_field_equal("origin", "重庆北", "万州北站")
    assert not ticket_field_equal("invoice_no", "20698188", "20698168")
    assert not ticket_field_equal("amount", None, 52.0)


def test_missing_required_field_is_flagged():
    f = TicketFields(doc_type="taxi", invoice_code="150001993910", invoice_no="20698168", date="2021-03-05")
    assert check(f) == ["amount"]                                  # 金额褪色读不出：标记，转人工
    assert check(TicketFields(doc_type="train", origin="万州北", destination="重庆北", date="2021-04-02",
                              amount=98.0, seat_class="二等座")) == []
    assert "doc_type" in check(TicketFields(amount=10))


class FakeVLM:
    def __init__(self, *answers):
        self.answers = list(answers)

    def ask(self, image, prompt):
        return self.answers.pop(0), 900, 60


def _jpg(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buf, format="JPEG")
    p = tmp_path / "t.jpg"
    p.write_bytes(buf.getvalue())
    return p


def test_extract_retries_bad_json_and_scores_silent_vs_missing(tmp_path):
    good = json.dumps({"doc_type": "taxi", "invoice_code": "150001993910", "invoice_no": "20698188",
                       "date": "2021年03月05日", "amount": None})
    r = extract_ticket(FakeVLM("我看不清", good), b"x")
    assert r.model_calls == 2 and r.fields.date == "2021-03-05" and r.uncertain_fields == ["amount"]

    smp = {"id": "a", "path": str(_jpg(tmp_path)), "kind": "taxi", "schema": "ticket",
           "gt": {"invoice_code": "150001993910", "invoice_no": "20698168", "date": "2021-03-05", "amount": 52.0}}
    out = run_ticket(smp, FakeVLM(good))
    assert out["wrong_fields"] == ["invoice_no", "amount"]
    assert out["silent_fields"] == ["invoice_no"]                  # 号码读错了一位：静默错误；金额留空：不算
    m = extraction_metrics([out])
    assert m["doc_accuracy"] == 0 and m["silent_errors"] == 1
    assert m["fields"]["amount"] == {"n": 1, "accuracy": 0.0, "silent": 0}
    assert m["fields"]["date"]["accuracy"] == 1.0 and m["groups"]["taxi"]["type_accuracy"] == 1.0


def test_model_failure_is_reported_not_scored(tmp_path):
    class Broken:
        def ask(self, image, prompt):
            raise TimeoutError("超时")

    out = run_ticket({"id": "a", "path": str(_jpg(tmp_path)), "kind": "taxi", "schema": "ticket",
                      "gt": {"amount": 1.0}}, Broken())
    assert out["tier"] == "failed" and out["wrong_fields"] == []
