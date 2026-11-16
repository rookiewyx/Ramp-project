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


@pytest.fixture
def v2(monkeypatch):
    """默认版本 v3 会把出租车票读两遍；只测单遍逻辑的用例固定在 v2。"""
    from ramp_agent.config import get_settings

    monkeypatch.setattr(get_settings(), "ticket_prompt", "v2")


def test_extract_retries_bad_json_and_scores_silent_vs_missing(tmp_path, v2):
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


def test_taxi_amount_is_cross_checked_against_price_and_distance():
    """褪色的出租车票：模型把里程 22.3 当成了金额（真实票据基线里的错例）。"""
    from ramp_agent.extraction.tickets import taxi_amount_suspicious

    base = dict(doc_type="taxi", invoice_code="131002160106", invoice_no="00726760", date="2021-05-07")
    assert "里程" in taxi_amount_suspicious(TicketFields(**base, amount=22.3, unit_price=3.8, distance_km=22.3))
    assert "一半" in taxi_amount_suspicious(TicketFields(**base, amount=10.0, unit_price=2.5, distance_km=44.0))
    assert taxi_amount_suspicious(TicketFields(**base, amount=86.0, unit_price=3.8, distance_km=22.3)) == ""
    assert taxi_amount_suspicious(TicketFields(**base, amount=14.0, unit_price=2.4, distance_km=1.2)) == ""  # 起步价
    assert taxi_amount_suspicious(TicketFields(**base, amount=52.0)) == ""                                # 没读到单价里程
    assert check(TicketFields(**base, amount=22.3, unit_price=3.8, distance_km=22.3)) == ["amount"]


def test_prompt_version_is_selectable_and_flight_dates_are_separated():
    from ramp_agent.extraction.tickets import PROMPTS

    assert "填开日期" in PROMPTS["v2"] and "填开日期" not in PROMPTS["v1"]
    v2 = json.dumps({"doc_type": "flight", "date": "2021-05-07", "issue_date": "2021-05-21", "amount": 500,
                     "origin": "T3杭州-萧山", "destination": "天津-滨海", "insurance": None})
    vlm = FakeVLM(v2)
    r = extract_ticket(vlm, b"x", version="v2")
    assert r.fields.date == "2021-05-07" and r.fields.issue_date == "2021-05-21" and r.uncertain_fields == []


def test_v3_reads_taxi_twice_and_flags_disagreement(tmp_path):
    """褪色的出租车票：两遍读出来的金额不一样，至少有一遍是错的，标为可疑。其他票种只读一遍。"""
    img = _jpg(tmp_path).read_bytes()
    first = {"doc_type": "taxi", "invoice_code": "131002160106", "invoice_no": "00726760", "date": "2021-05-07",
             "amount": 22.3}
    r = extract_ticket(FakeVLM(json.dumps(first), json.dumps({**first, "amount": 86.0})), img, version="v3")
    assert r.model_calls == 2 and r.uncertain_fields == ["amount"] and r.fields.amount == 22.3
    r = extract_ticket(FakeVLM(json.dumps(first), json.dumps(first)), img, version="v3")
    assert r.model_calls == 2 and r.uncertain_fields == []
    train = json.dumps({"doc_type": "train", "origin": "万州北", "destination": "重庆北", "date": "2021-04-02",
                        "amount": 98.0, "seat_class": "二等座"})
    assert extract_ticket(FakeVLM(train), img, version="v3").model_calls == 1
    assert extract_ticket(FakeVLM(json.dumps(first)), img, version="v2").model_calls == 1


def test_vehicle_class_is_not_a_toll_station():
    from ramp_agent.extraction.tickets import to_ticket

    f = to_ticket({"doc_type": "toll", "entry": "浙江余杭站", "exit": "客1", "amount": 30})
    assert f.entry == "浙江余杭站" and f.exit is None
    assert to_ticket({"doc_type": "toll", "exit": "1036"}).exit is None
