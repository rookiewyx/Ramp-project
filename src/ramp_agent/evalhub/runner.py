"""运行器：在某个评测集版本上跑一次识别或审核，结果连同版本信息一起入库。"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path

from sqlalchemy.orm import Session

from ..config import ROOT, get_settings
from .datasets import EvalHubError, resolve_gold
from .db import DatasetVersion, Run, RunResult


# ---------------- 版本指纹 ----------------
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_hash() -> str:
    """src/ 下全部 .py 的内容哈希。项目没用 git 时，靠它判断两次运行的代码是否相同。"""
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def prompt_hash() -> str:
    from .. import pii
    from ..extraction import vlm
    from ..policy import semantic

    # 脱敏会改变发给模型的文字，也算进提示词哈希
    text = (inspect.getsource(semantic.system_prompt) + inspect.getsource(semantic.build_prompt)
            + inspect.getsource(semantic.masked_request) + inspect.getsource(pii) + vlm.PROMPT)
    return _sha(text.encode())


def policy_hash() -> str:
    return _sha((ROOT / "policies" / "policy.yaml").read_bytes().replace(b"\r\n", b"\n"))


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
                             timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------- 运行 ----------------
def _start(s: Session, v: DatasetVersion, task: str, note: str, gold: str, params: dict,
           models: dict) -> Run:
    if v.dataset.task != task:
        raise EvalHubError(f"{v.ref} 是 {v.dataset.task} 评测集，不能跑 {task}")
    if v.status != "frozen":
        print(f"[evalhub] 提醒：{v.ref} 还是草稿，结果不能和其他运行严格比较；正式评测前请先冻结")
    run = Run(task=task, version=v, note=note, gold=gold, params=params, models=models,
              git_commit=git_commit(), code_hash=code_hash(), prompt_hash=prompt_hash(),
              policy_hash=policy_hash())
    s.add(run)
    s.flush()
    return run


def run_policy_eval(s: Session, v: DatasetVersion, *, judge=None, note: str = "", gold: str = "auto",
                    extraction_run: int | None = None, extraction_report: Path | None = None,
                    workers: int = 4) -> Run:
    from .adapters.policy import policy_metrics, run_policy

    lookup = None
    source = "oracle"
    if extraction_run is not None:
        ext = s.get(Run, extraction_run)
        if ext is None or ext.task != "extraction":
            raise EvalHubError(f"运行 #{extraction_run} 不是识别任务的运行")
        table = {r.case.key: r.output for r in ext.results}
        lookup, source = table.get, f"run#{extraction_run}"
    elif extraction_report is not None:
        table = {}
        for line in extraction_report.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            table[row["id"]] = row
        lookup, source = table.get, f"report:{extraction_report.name}"

    settings = get_settings()
    run = _start(s, v, "policy", note, gold,
                 params={"invoice_fields": source, "llm": judge is not None, "workers": workers},
                 models={"llm": settings.llm_model if judge is not None else None})
    cases = list(v.cases)
    t0 = time.perf_counter()
    outputs = run_policy([c.payload for c in cases], judge=judge, extraction_lookup=lookup,
                         workers=workers)
    run.wall_seconds = round(time.perf_counter() - t0, 2)

    items = []
    for c, out in zip(cases, outputs):
        g = resolve_gold(c, gold)
        s.add(RunResult(run=run, case=c, output=out, predicted=out["decision"], gold=g.decision,
                        correct=None if g.decision is None else out["decision"] == g.decision,
                        llm_calls=out["llm_calls"], tokens=out["tokens"]))
        items.append({"gold": g.decision, "gold_rules": g.rules, "predicted": out["decision"],
                      "violated": out["violated"], "tags": c.tags, "llm_calls": out["llm_calls"],
                      "tokens": out["tokens"]})
    run.metrics = policy_metrics(items)
    run.status = "done"
    s.flush()
    return run


def run_extraction_eval(s: Session, v: DatasetVersion, *, vlm=None, note: str = "",
                        force_vlm: bool = False, workers: int = 4) -> Run:
    from ..extraction import InvoiceExtractor
    from .adapters.extraction import extraction_metrics, run_extraction

    settings = get_settings()
    run = _start(s, v, "extraction", note, "payload",
                 params={"force_vlm": force_vlm, "vlm": vlm is not None, "workers": workers},
                 models={"vlm": settings.vlm_model if vlm is not None else None})
    cases = list(v.cases)
    t0 = time.perf_counter()
    outputs = run_extraction([c.payload for c in cases], InvoiceExtractor(vlm, force_vlm=force_vlm),
                             workers=workers)
    run.wall_seconds = round(time.perf_counter() - t0, 2)
    for c, out in zip(cases, outputs):
        failed = out["tier"] == "failed"
        s.add(RunResult(run=run, case=c, output=out, predicted=out["tier"],
                        gold=None if failed else "all_fields",
                        correct=None if failed else not out["wrong_fields"],
                        latency_ms=out["latency_ms"], llm_calls=out["model_calls"], tokens=out["tokens"]))
    run.metrics = extraction_metrics(outputs)
    run.status = "done"
    s.flush()
    return run
