"""生成带标注的合成报销数据：报销单 + 发票文件 + 标准答案。

为什么要合成：公开数据集只有“票据长什么样”，没有“这笔报销按公司制度该不该过”的标注。
这里按 policies/expense_policy.md 的 R1–R9 条款生成报销记录，每条都带：
  - 期望审核结论 expected.decision（approve / review / reject）
  - 触发的条款 expected.violations
  - 发票的真实字段 invoice.fields（用来评测票据识别准确率）

发票文件有两种形态，覆盖分级识别的两条路径：
  - 文字版 PDF（对应电子发票，可直接解析文字）
  - 拍照/扫描 JPG（对应纸质票，需要视觉模型）
所有发票都带“样例数据 非真实发票”水印。

用法：
  .venv\\Scripts\\python scripts\\gen_synthetic.py            # 默认 300 条
  .venv\\Scripts\\python scripts\\gen_synthetic.py -n 1000 --scan-ratio 0.4
  .venv\\Scripts\\python scripts\\gen_synthetic.py --seed 7 --out data/synthetic_test   # 独立测试集
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"

# ---------- 与 policies/expense_policy.md 保持一致 ----------
COMPANY = "星河数智科技有限公司"
COMPANY_TAX_ID = "91310000MA1XHSZ00X"
TIER1 = ["北京", "上海", "广州", "深圳"]   # 用列表不用集合：集合的遍历顺序每次运行可能不同，会让同一个种子生成不同的数据
OTHER_CITIES = ["杭州", "成都", "武汉", "西安", "南京", "长沙", "重庆", "厦门"]
HOTEL_CAP = {"tier1": 600, "other": 400}           # R1
MEAL_DAILY_CAP = 150                               # R2 个人餐饮
ENTERTAIN_PER_HEAD_CAP = 300                       # R2 业务招待
TAXI_CAP = 100                                     # R3
SUBMIT_DEADLINE_DAYS = 90                          # R5
PREAPPROVAL_THRESHOLD = 5000                       # R7
REJECT_RULES = {"R4", "R5_outside_trip", "R6", "R8", "R9"}

# 虚构的员工、部门、商户
EMPLOYEES = [("E001", "张伟", "研发部"), ("E002", "王芳", "市场部"), ("E003", "李强", "销售部"),
             ("E004", "刘洋", "研发部"), ("E005", "陈静", "财务部"), ("E006", "杨帆", "销售部"),
             ("E007", "赵敏", "产品部"), ("E008", "黄磊", "市场部")]
HOTELS = ["云栖商务酒店", "星辰假日酒店", "锦程精选酒店", "悦途连锁酒店", "澜庭酒店"]
RESTAURANTS = ["湘味小馆", "江南食府", "蜀香阁", "粤来轩", "北方饺子馆", "禾田简餐"]
TAXI_COS = ["速行出行科技有限公司", "畅达网约车服务有限公司"]
TRAIN_COS = ["铁路客运服务（样例）"]
AIR_COS = ["蓝天航空服务（样例）"]
EVENT_COS = ["智汇会展服务有限公司", "前沿技术峰会组委会"]


@dataclass
class Invoice:
    invoice_no: str
    issue_date: str
    seller: str
    buyer_name: str
    buyer_tax_id: str
    items: list[dict]
    total: float
    tax_rate: float
    file: str | None = None          # 相对项目根目录的路径；None 表示缺附件
    file_type: str | None = None     # pdf / jpg


@dataclass
class Expense:
    expense_id: str
    report_id: str
    employee_id: str
    employee_name: str
    department: str
    category: str                    # hotel / meal / entertainment / taxi / train / flight / conference / other
    amount: float
    submit_date: str
    note: str = ""
    city: str | None = None
    trip_start: str | None = None
    trip_end: str | None = None
    nights: int | None = None
    headcount: int | None = None
    preapproval_id: str | None = None
    invoice: Invoice | None = None
    expected: dict = field(default_factory=dict)
    scenario: str = ""


def money(x: float) -> float:
    return round(x + 1e-9, 2)


class Generator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.used_invoices: list[Invoice] = []
        self.seq = 0

    def invoice_no(self) -> str:
        return "".join(self.rng.choice("0123456789") for _ in range(20))

    def trip(self, submit: date) -> tuple[str, date, date]:
        city = self.rng.choice(TIER1 + OTHER_CITIES)
        end = submit - timedelta(days=self.rng.randint(2, 30))
        start = end - timedelta(days=self.rng.randint(1, 4))
        return city, start, end

    def make_invoice(self, seller: str, issue: date, items: list[dict], rate: float) -> Invoice:
        total = money(sum(i["amount"] for i in items))
        return Invoice(self.invoice_no(), issue.isoformat(), seller, COMPANY, COMPANY_TAX_ID,
                       items, total, rate)

    # ---------------- 场景 ----------------
    def build(self, report_id: str, emp: tuple[str, str, str], submit: date, scenario: str) -> Expense:
        rng = self.rng
        self.seq += 1
        eid, name, dept = emp
        exp = Expense(f"EXP{self.seq:05d}", report_id, eid, name, dept, "", 0.0,
                      submit.isoformat(), scenario=scenario)
        violations: list[dict] = []

        city, start, end = self.trip(submit)
        tier = "tier1" if city in TIER1 else "other"
        in_trip = start + timedelta(days=rng.randint(0, (end - start).days))

        def trip_fields() -> None:
            exp.city, exp.trip_start, exp.trip_end = city, start.isoformat(), end.isoformat()

        if scenario.startswith("hotel"):
            trip_fields()
            exp.category = "hotel"
            nights = max(1, (end - start).days)
            cap = HOTEL_CAP[tier]
            if scenario == "hotel_ok":
                rate = rng.randint(int(cap * 0.6), cap)
            elif scenario == "hotel_near_cap":      # 难例：上海 580 合规，但超过“其他城市”标准
                exp.city = city = rng.choice(sorted(TIER1))
                rate = rng.randint(560, 600)
            else:                                   # hotel_over
                rate = rng.randint(cap + 30, cap + 400)
                violations.append({"rule": "R1", "reason": f"{city}住宿 {rate} 元/晚，超出标准 {cap} 元"})
            exp.nights = nights
            items = [{"name": f"住宿费 {nights}晚", "amount": money(rate * nights)}]
            exp.invoice = self.make_invoice(rng.choice(HOTELS), end, items, 0.06)

        elif scenario.startswith("meal"):
            trip_fields()
            exp.category = "meal"
            amt = rng.randint(40, MEAL_DAILY_CAP) if scenario == "meal_ok" else rng.randint(
                MEAL_DAILY_CAP + 20, MEAL_DAILY_CAP + 200)
            if scenario == "meal_over":
                violations.append({"rule": "R2", "reason": f"出差个人餐饮 {amt} 元，超出每日 {MEAL_DAILY_CAP} 元"})
            exp.invoice = self.make_invoice(rng.choice(RESTAURANTS), in_trip,
                                            [{"name": "餐饮服务", "amount": float(amt)}], 0.06)

        elif scenario.startswith("entertain"):
            exp.category = "entertainment"
            heads = rng.randint(2, 8)
            exp.headcount = heads
            per = rng.randint(120, ENTERTAIN_PER_HEAD_CAP) if scenario == "entertain_ok" else rng.randint(
                ENTERTAIN_PER_HEAD_CAP + 30, ENTERTAIN_PER_HEAD_CAP + 300)
            items = [{"name": "餐饮服务", "amount": float(per * heads)}]
            if scenario == "entertain_alcohol":
                per = rng.randint(120, 250)
                items = [{"name": "餐饮服务", "amount": float(per * heads)},
                         {"name": "白酒", "amount": float(rng.randint(300, 1200))}]
                violations.append({"rule": "R8", "reason": "发票明细含白酒，烟酒不予报销"})
            elif scenario == "entertain_over":
                violations.append({"rule": "R2", "reason": f"业务招待人均 {per} 元，超出 {ENTERTAIN_PER_HEAD_CAP} 元"})
            exp.note = f"招待客户，共 {heads} 人"
            exp.invoice = self.make_invoice(rng.choice(RESTAURANTS),
                                            submit - timedelta(days=rng.randint(1, 20)), items, 0.06)

        elif scenario.startswith("taxi"):
            trip_fields()
            exp.category = "taxi"
            if scenario == "taxi_ok":
                amt = rng.randint(15, TAXI_CAP)
            else:
                amt = rng.randint(TAXI_CAP + 10, TAXI_CAP + 150)
                if scenario == "taxi_over_noted":   # 难例：超额但写明原因，合规
                    exp.note = "凌晨航班落地，无公共交通，打车回酒店"
                else:                               # taxi_over
                    violations.append({"rule": "R3", "reason": f"市内打车 {amt} 元，超出 {TAXI_CAP} 元且未说明原因"})
            exp.invoice = self.make_invoice(rng.choice(TAXI_COS), in_trip,
                                            [{"name": "客运服务费", "amount": float(amt)}], 0.03)

        elif scenario == "train_ok":
            trip_fields()
            exp.category = "train"
            amt = rng.randint(150, 900)
            exp.invoice = self.make_invoice(TRAIN_COS[0], start,
                                            [{"name": f"高铁二等座 至{city}", "amount": float(amt)}], 0.09)

        elif scenario.startswith("conference"):
            exp.category = "conference"
            amt = rng.randint(PREAPPROVAL_THRESHOLD + 300, PREAPPROVAL_THRESHOLD + 4000)
            if scenario == "conference_preapproved":  # 难例：大额但有预审批，合规
                exp.preapproval_id = f"PA{rng.randint(10000, 99999)}"
            else:
                violations.append({"rule": "R7", "reason": f"单笔 {amt} 元超过 {PREAPPROVAL_THRESHOLD} 元，缺少预审批"})
            exp.note = "参加行业技术峰会"
            exp.invoice = self.make_invoice(rng.choice(EVENT_COS), submit - timedelta(days=rng.randint(5, 40)),
                                            [{"name": "会议注册费", "amount": float(amt)}], 0.06)

        elif scenario == "wrong_buyer":
            trip_fields()
            exp.category = "meal"
            amt = rng.randint(40, MEAL_DAILY_CAP)
            exp.invoice = self.make_invoice(rng.choice(RESTAURANTS), in_trip,
                                            [{"name": "餐饮服务", "amount": float(amt)}], 0.06)
            if rng.random() < 0.5:
                exp.invoice.buyer_name = "个人"
                violations.append({"rule": "R4", "reason": "发票抬头为“个人”，不是公司全称"})
            else:
                exp.invoice.buyer_tax_id = COMPANY_TAX_ID[:-3] + "7Q1"
                violations.append({"rule": "R4", "reason": "发票纳税人识别号与公司不一致"})

        elif scenario == "outside_trip":
            trip_fields()
            exp.category = "hotel"
            rate = rng.randint(200, HOTEL_CAP[tier])
            exp.nights = 1
            outside = start - timedelta(days=rng.randint(3, 10))
            exp.invoice = self.make_invoice(rng.choice(HOTELS), outside,
                                            [{"name": "住宿费 1晚", "amount": float(rate)}], 0.06)
            violations.append({"rule": "R5_outside_trip", "reason": f"开票日 {outside} 不在出差期间 {start}~{end}"})

        elif scenario == "late_submit":
            exp.category = "other"
            amt = rng.randint(50, 400)
            old = submit - timedelta(days=rng.randint(SUBMIT_DEADLINE_DAYS + 5, SUBMIT_DEADLINE_DAYS + 120))
            exp.note = "办公用品采购"
            exp.invoice = self.make_invoice("优选办公用品商行", old,
                                            [{"name": "办公用品", "amount": float(amt)}], 0.13)
            violations.append({"rule": "R5_late", "reason": f"开票日 {old}，超过 {SUBMIT_DEADLINE_DAYS} 天提交期限"})

        elif scenario == "missing_attachment":
            trip_fields()
            exp.category = "taxi"
            amt = rng.randint(15, TAXI_CAP)
            exp.invoice = self.make_invoice(rng.choice(TAXI_COS), in_trip,
                                            [{"name": "客运服务费", "amount": float(amt)}], 0.03)
            violations.append({"rule": "R9", "reason": "未附发票文件"})

        elif scenario == "duplicate":
            # 只挑开票日早于本次提交日的发票，保证数据在时间上合理
            pool = [i for i in self.used_invoices if i.issue_date <= submit.isoformat()]
            src = rng.choice(pool)
            exp.category = "other"
            exp.invoice = Invoice(**{**asdict(src), "items": list(src.items)})
            exp.note = "重复提交（同一张发票）"
            violations.append({"rule": "R6", "reason": f"发票号码 {src.invoice_no} 已报销过"})
        else:
            raise ValueError(scenario)

        exp.amount = exp.invoice.total

        exp.expected = complete_labels(
            violations, submit=submit, amount=exp.amount, category=exp.category,
            headcount=exp.headcount, preapproval_id=exp.preapproval_id, invoice=asdict(exp.invoice))
        if scenario != "duplicate":
            self.used_invoices.append(exp.invoice)
        return exp


def complete_labels(violations: list[dict], *, submit: date, amount: float, category: str,
                    headcount: int | None, preapproval_id: str | None, invoice: dict) -> dict:
    """通用条款兜底检查：任何场景下只要实际触发，都要标出来，避免标签漏标。

    场景只负责“故意制造”的那一条违规；这里补上顺带触发的其他条款。
    例如重复提交的发票会把原发票的问题一起带过来（抬头不符、含白酒等）。
    已经标过的条款不会重复添加，所以对同一条记录反复调用结果不变（--relabel 依赖这一点）。
    """
    violations = list(violations)
    rules = {v["rule"] for v in violations}

    def add(rule: str, reason: str) -> None:
        if rule not in rules:
            violations.append({"rule": rule, "reason": reason})
            rules.add(rule)

    age = (submit - date.fromisoformat(invoice["issue_date"])).days
    if age > SUBMIT_DEADLINE_DAYS:
        add("R5_late", f"开票 {age} 天后才提交，超过 {SUBMIT_DEADLINE_DAYS} 天")
    if amount > PREAPPROVAL_THRESHOLD and not preapproval_id:
        add("R7", f"单笔 {amount} 元超过 {PREAPPROVAL_THRESHOLD} 元，缺少预审批")
    if invoice["buyer_name"] != COMPANY or invoice["buyer_tax_id"] != COMPANY_TAX_ID:
        add("R4", "发票抬头或税号与公司不一致")
    if any("白酒" in i["name"] for i in invoice["items"]):
        add("R8", "发票明细含白酒，烟酒不予报销")
    if category == "entertainment" and headcount and amount / headcount > ENTERTAIN_PER_HEAD_CAP:
        add("R2", f"业务招待人均 {amount / headcount:.0f} 元，超出 {ENTERTAIN_PER_HEAD_CAP} 元")

    decision = "reject" if rules & REJECT_RULES else ("review" if rules else "approve")
    return {"decision": decision, "violations": violations}


def fix_duplicate_labels(records: list[dict]) -> None:
    """R6 以“真正提交过的发票”为准。

    缺附件的报销（R9）其实没有提交发票，被退回后员工补上发票重新提交，是第一次真正报销这张发票，不算重复；
    再往后又提交同一张，才算重复。按提交顺序逐条检查并修正标签。
    """
    seen: set[str] = set()
    for r in records:
        inv = r["invoice"]
        rules = [v for v in r["expected"]["violations"] if v["rule"] != "R6" or inv["invoice_no"] in seen]
        if len(rules) != len(r["expected"]["violations"]):
            names = {v["rule"] for v in rules}
            decision = "reject" if names & REJECT_RULES else ("review" if names else "approve")
            r["expected"] = {"decision": decision, "violations": rules}
        if inv["file"] is not None:
            seen.add(inv["invoice_no"])


def relabel(path: Path) -> None:
    """只重算已有数据的标签，不改动发票和报销信息，已有的识别评测结果仍然可以复用。"""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    before = [r["expected"] for r in records]
    for r in records:
        r["expected"] = complete_labels(
            r["expected"]["violations"], submit=date.fromisoformat(r["submit_date"]),
            amount=r["amount"], category=r["category"], headcount=r["headcount"],
            preapproval_id=r["preapproval_id"], invoice=r["invoice"])
    fix_duplicate_labels(records)
    changed = 0
    for r, old in zip(records, before):
        new = r["expected"]
        if new != old:
            changed += 1
            print(f"[relabel] {r['expense_id']} {r['scenario']}: {old['decision']} -> "
                  f"{new['decision']} {[v['rule'] for v in new['violations']]}")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"[relabel] {path}: {len(records)} 条，更新 {changed} 条标签")


SCENARIO_WEIGHTS = {
    # 合规（约 55%，含难例）
    "hotel_ok": 9, "meal_ok": 9, "taxi_ok": 8, "train_ok": 7, "entertain_ok": 6,
    "hotel_near_cap": 4, "taxi_over_noted": 3, "conference_preapproved": 3,
    # 需人工审核
    "hotel_over": 6, "meal_over": 5, "entertain_over": 4, "taxi_over": 5, "conference": 3, "late_submit": 4,
    # 建议驳回
    "wrong_buyer": 5, "outside_trip": 4, "entertain_alcohol": 4, "missing_attachment": 4, "duplicate": 5,
}


# ---------------- 渲染发票 ----------------
def render_pdf(inv: Invoice, path: Path) -> None:
    from reportlab.lib.colors import Color, black
    from reportlab.lib.pagesizes import A5, landscape
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    font = "STSong-Light"   # reportlab 内置中文字体，无需字体文件
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font))
    w, h = landscape(A5)
    c = canvas.Canvas(str(path), pagesize=(w, h))

    c.setFont(font, 16)
    c.drawCentredString(w / 2, h - 40, "电子发票（普通发票）")
    c.setFont(font, 9)
    c.drawString(w - 210, h - 60, f"发票号码：{inv.invoice_no}")
    c.drawString(w - 210, h - 74, f"开票日期：{inv.issue_date}")

    c.rect(30, h - 150, w - 60, 66)
    c.drawString(40, h - 102, f"购买方名称：{inv.buyer_name}")
    c.drawString(40, h - 118, f"纳税人识别号：{inv.buyer_tax_id}")
    c.drawString(w / 2 + 10, h - 102, f"销售方名称：{inv.seller}")

    y = h - 175
    c.drawString(40, y, "项目名称")
    c.drawString(260, y, "金额")
    c.drawString(340, y, "税率")
    c.drawString(400, y, "税额")
    c.line(30, y - 6, w - 30, y - 6)
    for it in inv.items:
        y -= 20
        net = it["amount"] / (1 + inv.tax_rate)
        c.drawString(40, y, it["name"])
        c.drawString(260, y, f"{net:.2f}")
        c.drawString(340, y, f"{int(inv.tax_rate * 100)}%")
        c.drawString(400, y, f"{it['amount'] - net:.2f}")
    c.line(30, 70, w - 30, 70)
    c.setFont(font, 11)
    c.drawString(40, 50, f"价税合计（小写）：¥{inv.total:.2f}")

    # 水印：明确标注为合成数据
    c.saveState()
    c.setFillColor(Color(0.85, 0.2, 0.2, alpha=0.18))
    c.setFont(font, 30)
    c.translate(w / 2, h / 2)
    c.rotate(20)
    c.drawCentredString(0, 0, "样例数据 非真实发票 SYNTHETIC")
    c.restoreState()
    c.setFillColor(black)
    c.save()


def render_scan(pdf: Path, jpg: Path, rng: random.Random) -> None:
    """把 PDF 渲染成“拍照件”：轻微旋转、模糊、噪点、偏色，逼真地走视觉模型路径。"""
    import pymupdf as fitz
    from PIL import Image, ImageEnhance, ImageFilter

    page = fitz.open(pdf)[0]
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = img.rotate(rng.uniform(-3, 3), expand=True, fillcolor=(235, 232, 225))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.05))
    noise = Image.effect_noise(img.size, rng.uniform(8, 20)).convert("RGB")
    img = Image.blend(img, noise, 0.06)
    img.save(jpg, quality=rng.randint(60, 85))


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成报销数据")
    ap.add_argument("-n", type=int, default=300, help="报销条数")
    ap.add_argument("--scan-ratio", type=float, default=0.3, help="发票为拍照件的比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="输出目录。换个种子生成到另一个目录，可以作为独立的测试集")
    ap.add_argument("--relabel", action="store_true",
                    help="不重新生成，只按最新的标注规则更新 --out 目录里已有数据的标签")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.relabel:
        relabel(out / "expenses.jsonl")
        return

    gen = Generator(args.seed)
    rng = gen.rng
    inv_dir = out / "invoices"
    inv_dir.mkdir(parents=True, exist_ok=True)

    names, weights = zip(*SCENARIO_WEIGHTS.items())
    today = date(2026, 10, 31)
    expenses: list[Expense] = []
    report_no = 0
    while len(expenses) < args.n:
        report_no += 1
        rid = f"R{report_no:04d}"
        emp = rng.choice(EMPLOYEES)
        submit = today - timedelta(days=rng.randint(0, 120))
        for _ in range(rng.randint(1, 4)):
            if len(expenses) >= args.n:
                break
            sc = rng.choices(names, weights)[0]
            if sc == "duplicate" and not any(i.issue_date <= submit.isoformat() for i in gen.used_invoices):
                sc = "meal_ok"
            expenses.append(gen.build(rid, emp, submit, sc))

    rendered: dict[str, tuple[str, str]] = {}   # 重复发票复用同一个文件
    for exp in expenses:
        inv = exp.invoice
        if exp.scenario == "missing_attachment":
            inv.file, inv.file_type = None, None
            continue
        if inv.invoice_no in rendered:
            inv.file, inv.file_type = rendered[inv.invoice_no]
            continue
        pdf = inv_dir / f"{inv.invoice_no}.pdf"
        render_pdf(inv, pdf)
        if rng.random() < args.scan_ratio:
            jpg = pdf.with_suffix(".jpg")
            render_scan(pdf, jpg, rng)
            pdf.unlink()
            inv.file, inv.file_type = os.path.relpath(jpg, ROOT).replace("\\", "/"), "jpg"
        else:
            inv.file, inv.file_type = os.path.relpath(pdf, ROOT).replace("\\", "/"), "pdf"
        rendered[inv.invoice_no] = (inv.file, inv.file_type)

    records = [asdict(exp) for exp in expenses]
    fix_duplicate_labels(records)
    for exp, rec in zip(expenses, records):
        exp.expected = rec["expected"]
    (out / "expenses.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    stats = {"total": len(expenses), "reports": report_no, "decision": {}, "rule": {},
             "file_type": {}, "scenario": {}}
    for exp in expenses:
        d = exp.expected["decision"]
        stats["decision"][d] = stats["decision"].get(d, 0) + 1
        stats["scenario"][exp.scenario] = stats["scenario"].get(exp.scenario, 0) + 1
        ft = exp.invoice.file_type or "missing"
        stats["file_type"][ft] = stats["file_type"].get(ft, 0) + 1
        for v in exp.expected["violations"]:
            stats["rule"][v["rule"]] = stats["rule"].get(v["rule"], 0) + 1
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[done] {out / 'expenses.jsonl'}")


if __name__ == "__main__":
    main()
