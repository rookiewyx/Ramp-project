"""生成制度审核的“语义难例集”：只有理解了文字含义才能判对的报销。

合成数据里的违规大多能用代码判断（金额超标、日期不对、关键词命中），体现不出大模型的作用。
这里手工设计了几类必须理解语义的样本，每条都写明期望结论，用来回答
“大模型在这个系统里到底有没有用、有多大用”：
  - 打车超标的理由：成立的（凌晨航班落地）和不成立的（习惯打车）
  - 变相的禁报项目：只写度数和容量的酒、按小时计费的包厢、换了说法的罚款
  - 看起来可疑但其实正常的明细：“电子烟雾报警器”“酒店住宿”“矿泉水”
  - 招待人数只写在说明里：“宴请客户王总、李经理及我方 2 人”

只包含报销单和发票字段，不生成发票文件（评测时直接当作识别结果使用）。
输出 evalsets/policy_challenge.jsonl，体积很小，会提交到仓库。

用法：.venv\\Scripts\\python scripts\\gen_policy_challenge.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evalsets" / "policy_challenge.jsonl"

COMPANY = "星河数智科技有限公司"
TAX_ID = "91310000MA1XHSZ00X"
SUBMIT = date(2026, 10, 20)

# (场景, 类别, 明细[(名称, 金额)], 报销说明, 期望结论, 期望违规条款, 其他字段)
CASES: list[tuple] = [
    # ---- R3 打车超标：理由成立 → 通过 ----
    ("taxi_reason_valid", "taxi", [("客运服务费", 168)], "凌晨1点航班落地，机场大巴和地铁均已停运", "approve", []),
    ("taxi_reason_valid", "taxi", [("客运服务费", 215)], "携带两箱展会物料（约40公斤），无法乘坐地铁", "approve", []),
    ("taxi_reason_valid", "taxi", [("客运服务费", 136)], "台风红色预警，公交地铁全线停运，打车往返客户现场", "approve", []),
    ("taxi_reason_valid", "taxi", [("客运服务费", 182)], "客户生产系统宕机，紧急上门处理，公共交通需1.5小时", "approve", []),
    ("taxi_reason_valid", "taxi", [("客运服务费", 124)], "加班至23:50，末班地铁已过", "approve", []),
    # ---- R3 打车超标：理由不成立 → 转人工 ----
    ("taxi_reason_invalid", "taxi", [("客运服务费", 158)], "习惯打车", "review", ["R3"]),
    ("taxi_reason_invalid", "taxi", [("客运服务费", 233)], "顺路去商场买了点东西", "review", ["R3"]),
    ("taxi_reason_invalid", "taxi", [("客运服务费", 141)], "不想挤地铁", "review", ["R3"]),
    ("taxi_reason_invalid", "taxi", [("客运服务费", 199)], "周末带家人出去玩", "review", ["R3"]),
    ("taxi_reason_invalid", "taxi", [("客运服务费", 176)], "司机推荐走高架，比较快", "review", ["R3"]),
    # ---- R8 变相禁报项目 → 驳回 ----
    ("prohibited_disguised", "entertainment", [("餐饮服务", 680), ("53%vol 酱香型 500ml×2", 520)],
     "招待客户，共 4 人", "reject", ["R8"], {"headcount": 4}),
    ("prohibited_disguised", "other", [("中华（硬）条装", 450)], "客户拜访礼品", "reject", ["R8"]),
    ("prohibited_disguised", "other", [("欢唱包厢 3小时", 680)], "团队活动", "reject", ["R8"]),
    ("prohibited_disguised", "other", [("足疗保健套餐", 398)], "出差缓解疲劳", "reject", ["R8"]),
    ("prohibited_disguised", "other", [("交通违法处理费", 200)], "公务用车违章", "reject", ["R8"]),
    ("prohibited_disguised", "other", [("逾期缴费违约金", 86)], "宽带费逾期", "reject", ["R8"]),
    ("prohibited_disguised", "entertainment", [("餐饮服务", 600), ("精酿扎啤 2L", 168)],
     "招待合作方，共 3 人", "reject", ["R8"], {"headcount": 3}),
    # ---- 看起来可疑、其实正常 → 通过 ----
    ("benign_unusual", "other", [("矿泉水 24瓶", 48)], "部门会议用水", "approve", []),
    ("benign_unusual", "conference", [("会议室租赁 半天", 800)], "季度复盘会", "approve", []),
    ("benign_unusual", "conference", [("茶歇服务", 600)], "客户培训会茶歇", "approve", []),
    ("benign_unusual", "other", [("电子烟雾报警器", 260)], "机房安全设备", "approve", []),
    ("benign_unusual", "hotel", [("酒店住宿 1晚", 380)], "出差住宿", "approve", [],
     {"city": "成都", "nights": 1}),
    ("benign_unusual", "other", [("打印装订费", 96)], "投标文件打印", "approve", []),
    # ---- R2 招待人数只写在说明里 ----
    ("headcount_in_note", "entertainment", [("餐饮服务", 1000)], "宴请客户王总、李经理及我方2人", "approve", []),
    ("headcount_in_note", "entertainment", [("餐饮服务", 1500)], "答谢合作方，共6人", "approve", []),
    ("headcount_in_note", "entertainment", [("餐饮服务", 1200)], "与客户共3人工作餐", "review", ["R2"]),
    ("headcount_in_note", "entertainment", [("餐饮服务", 900)], "招待客户", "review", ["R2"]),
]


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    lines = []
    for n, case in enumerate(CASES, 1):
        scenario, category, items, note, decision, rules = case[:6]
        extra = case[6] if len(case) > 6 else {}
        issue = SUBMIT - timedelta(days=5 + n % 7)
        travel = category in ("taxi", "hotel")
        total = float(sum(a for _, a in items))
        record = {
            "expense_id": f"CH{n:03d}", "report_id": f"CHR{n:03d}", "employee_id": "E900",
            "employee_name": "测试员工", "department": "研发部", "category": category,
            "amount": total, "submit_date": SUBMIT.isoformat(), "note": note,
            "city": extra.get("city", "杭州" if travel else None),
            "trip_start": (issue - timedelta(days=1)).isoformat() if travel else None,
            "trip_end": (issue + timedelta(days=1)).isoformat() if travel else None,
            "nights": extra.get("nights"), "headcount": extra.get("headcount"), "preapproval_id": None,
            "invoice": {
                "invoice_no": f"{99000000000000000000 + n}", "issue_date": issue.isoformat(),
                "seller": "样例商户", "buyer_name": COMPANY, "buyer_tax_id": TAX_ID,
                "items": [{"name": nm, "amount": float(a)} for nm, a in items],
                "total": total, "tax_rate": 0.06, "file": "challenge", "file_type": "none",
            },
            "expected": {"decision": decision, "violations": [{"rule": r, "reason": ""} for r in rules]},
            "scenario": scenario,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {OUT}：{len(lines)} 条")


if __name__ == "__main__":
    main()
