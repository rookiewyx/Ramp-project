"""标注页面的后端（FastAPI）。

盲标：接口只返回当前标注人自己的标注，不返回系统结论、生成脚本的答案或其他人的标注，
避免标注人被已有答案带偏。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..policy import load_policy
from . import datasets as dsm
from .db import Case, Dataset, DatasetVersion, session

STATIC = Path(__file__).parent / "static"
RESERVED = {"designer", "generator", "finance"}   # 留给导入的预置标注（出题人、生成器、线上财务复核），网页上不能用


class LabelIn(BaseModel):
    labeler: str = Field(min_length=1, max_length=50)
    decision: str
    rules: list[str] = Field(default_factory=list)
    note: str = ""
    unsure: bool = False


def _check_labeler(name: str) -> str:
    name = name.strip()
    if not name or name.lower() in RESERVED:
        raise HTTPException(400, f"标注人名不能为空，也不能用 {sorted(RESERVED)}")
    return name


def _own_label(case: Case, labeler: str) -> dict | None:
    lb = next((x for x in case.labels if x.labeler == labeler), None)
    return None if lb is None else {"decision": lb.decision, "rules": lb.rules, "note": lb.note,
                                    "unsure": lb.unsure}


def _view(payload: dict) -> dict:
    """只把审核需要看的信息给标注人：去掉场景名、文件路径等可能暗示答案或无关的字段。"""
    inv = payload.get("invoice", {})
    return {
        "expense_id": payload.get("expense_id"), "category": payload.get("category"),
        "amount": payload.get("amount"), "submit_date": payload.get("submit_date"),
        "note": payload.get("note", ""), "city": payload.get("city"),
        "trip_start": payload.get("trip_start"), "trip_end": payload.get("trip_end"),
        "nights": payload.get("nights"), "headcount": payload.get("headcount"),
        "preapproval_id": payload.get("preapproval_id"),
        "has_attachment": inv.get("file") is not None,
        "invoice": {k: inv.get(k) for k in ("invoice_no", "issue_date", "seller", "buyer_name",
                                            "buyer_tax_id", "items", "total")},
        **_investigation_view(payload),
    }


def _investigation_view(payload: dict) -> dict:
    """调查评测集：识别可疑的字段、公司系统里的记录、员工被追问时的答复。员工编号换成“本人 / 其他员工”。"""
    if "world" not in payload:
        return {}
    me = payload.get("employee_id")
    w = payload.get("world") or {}

    def who(r: dict) -> dict:
        return {**{k: v for k, v in r.items() if k != "employee_id"},
                "owner": "本人" if r.get("employee_id", me) == me else "其他员工"}

    return {"uncertain_fields": payload.get("uncertain_fields") or [],
            "world": {"travel_requests": [who(r) for r in w.get("travel_requests", [])],
                      "preapprovals": [who(r) for r in w.get("preapprovals", [])],
                      "invoices": w.get("invoices", []), "merchants": w.get("merchants", [])},
            "employee_reply": payload.get("employee_reply")}


def create_app(db_url: str | None = None) -> FastAPI:
    app = FastAPI(title="RAMP 评测平台 · 标注")

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return (STATIC / "label.html").read_text(encoding="utf-8")

    @app.get("/api/policy")
    def policy() -> dict:
        p = load_policy()
        return {"company": p.company.model_dump(),
                "rules": [{"id": k, "title": v.title, "clause": v.clause, "outcome": v.outcome}
                          for k, v in p.rules.items()]}

    @app.get("/api/datasets")
    def list_versions() -> list[dict]:
        with session(db_url) as s:
            out = []
            for ds in s.scalars(select(Dataset).where(Dataset.task == "policy").order_by(Dataset.id)):
                for v in ds.versions:
                    out.append({"version_id": v.id, "ref": v.ref, "status": v.status,
                                "total": len(v.cases)})
            return out

    @app.get("/api/versions/{vid}/cases")
    def cases(vid: int, labeler: str) -> dict:
        labeler = _check_labeler(labeler)
        with session(db_url) as s:
            v = s.get(DatasetVersion, vid)
            if v is None:
                raise HTTPException(404, "版本不存在")
            items = []
            for c in v.cases:
                own = _own_label(c, labeler)
                items.append({"case_id": c.id, "key": c.key,
                              "decision": own["decision"] if own else None,
                              "unsure": own["unsure"] if own else False})
            return {"ref": v.ref, "status": v.status, "cases": items, **dsm.progress(v, labeler)}

    @app.get("/api/cases/{cid}")
    def case(cid: int, labeler: str) -> dict:
        labeler = _check_labeler(labeler)
        with session(db_url) as s:
            c = s.get(Case, cid)
            if c is None:
                raise HTTPException(404, "用例不存在")
            return {"case_id": c.id, "key": c.key, "frozen": c.version.status == "frozen",
                    "claim": _view(c.payload), "label": _own_label(c, labeler)}

    @app.put("/api/cases/{cid}/label")
    def put_label(cid: int, body: LabelIn) -> dict:
        labeler = _check_labeler(body.labeler)
        with session(db_url) as s:
            c = s.get(Case, cid)
            if c is None:
                raise HTTPException(404, "用例不存在")
            try:
                dsm.save_label(s, c, labeler, body.decision, body.rules, body.note.strip(), body.unsure)
            except dsm.EvalHubError as exc:
                raise HTTPException(409, str(exc)) from exc
            s.commit()
            return {"ok": True, **dsm.progress(c.version, labeler)}

    return app
