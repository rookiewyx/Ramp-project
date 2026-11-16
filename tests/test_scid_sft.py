"""微调数据：标注 → 训练目标的转换规则。"""

from __future__ import annotations

import json

from build_scid_sft import target

from ramp_agent.extraction.tickets import PROMPT_FT, TYPE_FIELDS, to_ticket


def test_unreadable_fields_become_null_and_xxx_insurance_becomes_zero():
    t, _ = target("taxi", {"发票代码": "131002160106", "发票号码": "#", "日期": "2021年05月07日", "金额": "#86"})
    assert t == {"doc_type": "taxi", "invoice_code": "131002160106", "invoice_no": None,
                 "date": "2021-05-07", "amount": None}
    t, _ = target("flight", {"始发站": "T2拉萨-贡嘎", "到达站": "西安-咸阳", "日期": "2021-04-30",
                             "金额": "￥1650.00", "保险费": "XXX"})
    assert t["insurance"] == 0.0 and t["amount"] == 1650.0 and t["origin"] == "T2拉萨-贡嘎"


def test_incomplete_or_wrong_labels_are_dropped_not_taught_as_null():
    assert target("train", {"始发站": "万州北站", "到达站": "重庆北站", "日期": "2021-04-02", "金额": "98"})[0] is None
    assert target("coach", {"始发站": "济宁北站", "到达站": "曲阜高铁", "日期": "2021-03-05",
                            "金额": "18:00"})[1] == "标注无法解析 amount"
    assert target("coach", {"始发站": "昆明", "到达站": "寻甸", "日期": "2012-03-03", "金额": "35"})[0]
    assert target("coach", {"始发站": "昆明", "到达站": "寻甸", "日期": "2002-03-03", "金额": "35"})[0] is None
    assert target("toll", {"入口": "浙江余杭站", "出口": "客1", "发票代码": "12101", "发票号码": "1620993471",
                           "金额": "30"})[0] is None


def test_training_target_round_trips_through_the_parser():
    """模型照训练目标输出，解析出来的字段和目标一致：训练格式和评测格式是同一套。"""
    t, _ = target("quota", {"发票代码": "131001927754", "发票号码": "21909634", "金额": "壹拾元"})
    f = to_ticket(json.loads(json.dumps(t, ensure_ascii=False)))
    assert (f.doc_type, f.invoice_code, f.invoice_no, f.amount) == ("quota", "131001927754", "21909634", 10.0)
    assert all(k in PROMPT_FT for fields in TYPE_FIELDS.values() for k in fields)
