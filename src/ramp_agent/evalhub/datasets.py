"""评测集：导入、冻结、派生新版本、解析标准答案。"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import ROOT
from .db import Case, Dataset, DatasetVersion, Label, now

DECISIONS = ("approve", "review", "reject")


class EvalHubError(Exception):
    pass


# ---------------- 定位版本 ----------------
def get_version(s: Session, ref: str) -> DatasetVersion:
    """'policy-hard@v2' / 'policy-hard@2' 指定版本；'policy-hard' 取最新版本。"""
    name, _, ver = ref.partition("@")
    ds = s.scalar(select(Dataset).where(Dataset.name == name))
    if ds is None:
        raise EvalHubError(f"没有名为 {name} 的评测集")
    if not ds.versions:
        raise EvalHubError(f"{name} 还没有任何版本")
    if not ver:
        return ds.versions[-1]
    num = int(ver.lstrip("vV"))
    for v in ds.versions:
        if v.version == num:
            return v
    raise EvalHubError(f"{name} 没有 v{num}")


def _new_version(s: Session, name: str, task: str, source: str, description: str = "") -> DatasetVersion:
    ds = s.scalar(select(Dataset).where(Dataset.name == name))
    if ds is None:
        ds = Dataset(name=name, task=task, description=description)
        s.add(ds)
        s.flush()
    elif ds.task != task:
        raise EvalHubError(f"{name} 是 {ds.task} 任务的评测集，不能导入 {task} 用例")
    if ds.versions and ds.versions[-1].status == "draft":
        raise EvalHubError(f"{ds.versions[-1].ref} 还是草稿，请先冻结（freeze）再建新版本")
    v = DatasetVersion(dataset=ds, version=(ds.versions[-1].version + 1) if ds.versions else 1,
                       source=source)
    s.add(v)
    s.flush()
    return v


def _require_draft(v: DatasetVersion) -> None:
    if v.status == "frozen":
        raise EvalHubError(f"{v.ref} 已冻结，不能修改；需要改动请用 fork 派生新版本")


# ---------------- 导入 ----------------
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rel(p: Path) -> str:
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:            # Windows 下跨盘符
        return str(p)


def import_policy(s: Session, path: Path, name: str, *, labeler: str = "generator",
                  with_labels: bool = True, description: str = "") -> DatasetVersion:
    """导入报销单格式（和 data/synthetic/expenses.jsonl 相同）的审核评测集。

    - 记录里的 expected 作为 labeler（默认 generator）的标注导入；with_labels=False 时不导入，留给人工标注。
    - 记录里的 designer_label（难题集的出题人预判）作为 designer 的标注导入，标注页面不会展示给其他标注人。
    """
    v = _new_version(s, name, "policy", _rel(path), description)
    for r in _read_jsonl(path):
        tags = {"scenario": r.get("scenario", ""), **(r.get("tags") or {})}
        payload = {k: val for k, val in r.items() if k not in ("expected", "designer_label", "tags")}
        case = Case(version=v, key=r["expense_id"], payload=payload, tags=tags)
        s.add(case)
        if with_labels and r.get("expected"):
            e = r["expected"]
            s.add(Label(case=case, labeler=labeler, decision=e["decision"],
                        rules=sorted({x["rule"] for x in e.get("violations", [])})))
        if r.get("designer_label"):
            d = r["designer_label"]
            s.add(Label(case=case, labeler="designer", decision=d["decision"],
                        rules=d.get("rules", []), note=d.get("note", "")))
    s.flush()
    return v


def import_extraction(s: Session, path: Path, name: str, description: str = "") -> DatasetVersion:
    """从合成报销数据里取出发票文件和发票字段，作为识别评测集；标准答案存在 payload['gt'] 里。"""
    from .adapters.extraction import samples_from_records

    v = _new_version(s, name, "extraction", _rel(path), description)
    for smp in samples_from_records(_read_jsonl(path)):
        s.add(Case(version=v, key=smp["id"], payload=smp, tags={"kind": smp["kind"]}))
    s.flush()
    return v


# ---------------- 冻结 / 派生 ----------------
def content_hash(v: DatasetVersion) -> str:
    h = hashlib.sha256()
    for c in sorted(v.cases, key=lambda c: c.key):
        h.update(json.dumps([c.key, c.payload, sorted(
            (lb.labeler, lb.decision, sorted(lb.rules)) for lb in c.labels)],
            ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


def freeze(s: Session, v: DatasetVersion, note: str = "") -> DatasetVersion:
    _require_draft(v)
    if not v.cases:
        raise EvalHubError(f"{v.ref} 没有用例，不能冻结")
    v.status, v.frozen_at, v.content_hash = "frozen", now(), content_hash(v)
    if note:
        v.note = note
    return v


def fork(s: Session, ref: str, note: str = "") -> DatasetVersion:
    """以某个已冻结版本为底，复制全部用例和标注，得到一个新的草稿版本，用来追加或修正。"""
    base = get_version(s, ref)
    if base.status != "frozen":
        raise EvalHubError(f"{base.ref} 还是草稿，直接在上面改即可")
    v = _new_version(s, base.dataset.name, base.dataset.task, f"fork:{base.ref}")
    v.note = note
    for c in base.cases:
        nc = Case(version=v, key=c.key, payload=c.payload, tags=c.tags)
        s.add(nc)
        for lb in c.labels:
            s.add(Label(case=nc, labeler=lb.labeler, decision=lb.decision, rules=lb.rules,
                        note=lb.note, unsure=lb.unsure))
    s.flush()
    return v


# ---------------- 标注 ----------------
def save_label(s: Session, case: Case, labeler: str, decision: str, rules: list[str] | None = None,
               note: str = "", unsure: bool = False) -> Label:
    _require_draft(case.version)
    if decision not in DECISIONS:
        raise EvalHubError(f"结论只能是 {DECISIONS}")
    lb = next((x for x in case.labels if x.labeler == labeler), None)
    if lb is None:
        lb = Label(case=case, labeler=labeler, decision=decision)
        s.add(lb)
    lb.decision, lb.rules, lb.note, lb.unsure = decision, sorted(set(rules or [])), note, unsure
    s.flush()
    return lb


# ---------------- 标准答案 ----------------
@dataclass
class Gold:
    decision: str | None          # None：没有可用答案
    rules: list[str]
    status: str                   # labeled / unlabeled / disputed / unsure


def resolve_gold(case: Case, gold: str = "auto") -> Gold:
    """确定一条用例的标准答案。

    gold='某人'：只用这个标注人的答案。
    gold='auto'：所有非 designer 的标注人（没有时才用 designer）意见一致就采用；
                 不一致记为 disputed，不参与打分，留待讨论。
    标注人自己勾了“拿不准”的，记为 unsure，同样不参与打分。
    """
    labels = case.labels
    if gold != "auto":
        labels = [lb for lb in labels if lb.labeler == gold]
    else:
        humans = [lb for lb in labels if lb.labeler != "designer"]
        labels = humans or labels
    if not labels:
        return Gold(None, [], "unlabeled")
    if any(lb.unsure for lb in labels):
        return Gold(None, [], "unsure")
    decisions = {lb.decision for lb in labels}
    if len(decisions) > 1:
        return Gold(None, [], "disputed")
    rules = Counter(r for lb in labels for r in lb.rules)
    return Gold(labels[0].decision, sorted(r for r, n in rules.items() if n * 2 > len(labels)), "labeled")


def progress(v: DatasetVersion, labeler: str) -> dict:
    done = sum(1 for c in v.cases if any(lb.labeler == labeler for lb in c.labels))
    return {"total": len(v.cases), "done": done}


def agreement(v: DatasetVersion) -> dict:
    """标注人两两之间的一致性，以及意见不一致的用例。"""
    from itertools import combinations

    from .stats import cohen_kappa

    labelers = sorted({lb.labeler for c in v.cases for lb in c.labels})
    pairs = []
    for x, y in combinations(labelers, 2):
        a, b, diff = [], [], []
        for c in v.cases:
            lx = next((lb for lb in c.labels if lb.labeler == x), None)
            ly = next((lb for lb in c.labels if lb.labeler == y), None)
            if lx and ly:
                a.append(lx.decision)
                b.append(ly.decision)
                if lx.decision != ly.decision:
                    diff.append({"key": c.key, x: lx.decision, y: ly.decision,
                                 f"{x}_note": lx.note, f"{y}_note": ly.note})
        pairs.append({"pair": [x, y], "n": len(a), "agree": len(a) - len(diff),
                      "kappa": cohen_kappa(a, b), "disagreements": diff})
    return {"labelers": labelers, "pairs": pairs}
