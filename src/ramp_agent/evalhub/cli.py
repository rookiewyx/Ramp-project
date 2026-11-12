"""评测平台命令行。

  .venv\\Scripts\\python -m ramp_agent.evalhub <命令> ...
  （重新运行一次 setup.ps1 或 `uv pip install -e .` 之后，也可以直接用 evalhub <命令>）

常用流程：
  init                                                                        导入项目自带的评测集
  import   evalsets/policy_hard_v1.jsonl --dataset policy-hard --no-labels   导入为草稿
  serve                                                                       打开标注页面
  agreement policy-hard                                                       标注一致性
  freeze   policy-hard                                                        冻结
  run      policy --dataset policy-hard --note "基线"                          跑一次并入库
  compare  1 2                                                                对比两次运行
  gate     1 2                                                                回归门禁（不通过时退出码为 1）
  export   policy-hard -o evalsets/policy_hard_v1.labeled.jsonl               导出带标注的评测集（入库到 git）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import openai
from loguru import logger
from sqlalchemy import select

from ..config import ROOT
from . import datasets as dsm
from .compare import compare, format_comparison, gate
from .db import Dataset, Run, session


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


# ---------------- 模型连接 ----------------
def _probe(client, what: str) -> bool:
    try:
        client.with_options(timeout=10, max_retries=0).models.list()
    except openai.AuthenticationError:
        sys.exit(f"[evalhub] {what}的 API Key 无效，请检查 .env")
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        print(f"[evalhub] {what} {client.base_url} 连不上（{type(exc).__name__}），本次不使用")
        return False
    except openai.APIStatusError:
        pass
    return True


def connect_llm():
    from ..policy import LLMJudge

    j = LLMJudge()
    return j if _probe(j.client, "大模型") else None


def connect_agent(kind: str | None):
    """调查 Agent：llm 用 .env 里的文本模型；scripted 是写死流程的对照组，不调模型。"""
    from ..agent import Investigator, ScriptedPlanner

    if not kind:
        return None
    if kind == "scripted":
        return Investigator(ScriptedPlanner(), "scripted")
    a = Investigator.from_settings()
    if not _probe(a.client, "大模型（调查 Agent）"):
        sys.exit("[evalhub] --agent llm 需要可用的大模型")
    return a


def connect_vlm():
    from ..extraction import OpenAICompatibleVLM

    v = OpenAICompatibleVLM()
    return v if _probe(v.client, "视觉模型") else None


# ---------------- 命令 ----------------
def cmd_list(a) -> None:
    with session() as s:
        for ds in s.scalars(select(Dataset).order_by(Dataset.id)):
            print(f"{ds.name}（{ds.task}）{ds.description}")
            for v in ds.versions:
                labelers = sorted({lb.labeler for c in v.cases for lb in c.labels})
                print(f"  v{v.version} [{v.status}] {len(v.cases)} 条  标注人：{', '.join(labelers) or '无'}"
                      f"  来源：{v.source}{'  ' + v.note if v.note else ''}")
        runs = s.scalars(select(Run).order_by(Run.id.desc()).limit(a.runs)).all()
        if runs:
            print("\n最近的运行：")
        for r in runs:
            m = r.metrics
            key = (f"准确率 {m.get('accuracy')}  误通过 {m.get('false_approve')}  误驳回 {m.get('false_reject')}"
                   if r.task == "policy" else
                   f"整张全对 {m.get('doc_accuracy')}  静默错误 {m.get('silent_errors')}")
            print(f"  #{r.id} {r.started_at:%m-%d %H:%M} {r.task:<10} {r.version.ref:<22} {key}  {r.note}")


def cmd_import(a) -> None:
    with session() as s:
        if a.task == "policy":
            v = dsm.import_policy(s, _path(a.file), a.dataset, labeler=a.labeler,
                                  with_labels=not a.no_labels, description=a.description)
        else:
            v = dsm.import_extraction(s, _path(a.file), a.dataset, description=a.description)
        s.commit()
        print(f"[evalhub] 导入 {v.ref}（草稿）：{len(v.cases)} 条")
        if a.freeze:
            dsm.freeze(s, v)
            s.commit()
            print(f"[evalhub] 已冻结 {v.ref}")


# 项目自带的评测集：(名称, 任务, 文件, 是否带标注, 是否冻结, 说明)
STANDARD = [
    ("policy-dev", "policy", "data/synthetic/expenses.jsonl", True, True, "合成开发集（seed 42）"),
    ("policy-test", "policy", "data/synthetic_test/expenses.jsonl", True, True, "合成独立测试集（seed 7）"),
    ("policy-challenge", "policy", "evalsets/policy_challenge.jsonl", True, True, "语义难例集（出题人与提示词同源，偏易）"),
    ("policy-hard", "policy", "evalsets/policy_hard_v1.jsonl", False, False, "难题集 v1，待人工盲标"),
    ("policy-hard2", "policy", "evalsets/policy_hard_v2.jsonl", False, False,
     "难题集第二批（留出集）：盲标、冻结后只跑一次，不据此修改系统"),
    ("investigate-dev", "policy", "evalsets/investigate_dev.jsonl", True, True,
     "调查开发集：带公司系统数据和员工答复，出题人标注，调试调查 Agent 用"),
    ("investigate", "policy", "evalsets/investigate_v1.jsonl", False, False,
     "调查留出集：盲标、冻结后只跑一次"),
    ("extract-dev", "extraction", "data/synthetic/expenses.jsonl", False, True, "合成开发集的发票"),
    ("extract-test", "extraction", "data/synthetic_test/expenses.jsonl", False, True, "独立测试集的发票"),
]


def cmd_init(a) -> None:
    """导入项目自带的评测集。已存在的跳过，可以重复运行。"""
    with session() as s:
        for name, task, file, labels, frozen, desc in STANDARD:
            if s.scalar(select(Dataset).where(Dataset.name == name)) is not None:
                print(f"[evalhub] {name} 已存在，跳过")
                continue
            path = _path(file)
            if not path.exists():
                print(f"[evalhub] {name}：找不到 {file}，跳过")
                continue
            if task == "policy":
                v = dsm.import_policy(s, path, name, with_labels=labels, description=desc)
            else:
                v = dsm.import_extraction(s, path, name, description=desc)
            if frozen:
                dsm.freeze(s, v)
            s.commit()
            print(f"[evalhub] {v.ref}：{len(v.cases)} 条，{'已冻结' if frozen else '草稿（待标注）'}  {desc}")


def cmd_freeze(a) -> None:
    with session() as s:
        v = dsm.get_version(s, a.ref)
        dsm.freeze(s, v, a.note)
        s.commit()
        print(f"[evalhub] 已冻结 {v.ref}，内容指纹 {v.content_hash[:12]}")


def cmd_fork(a) -> None:
    with session() as s:
        v = dsm.fork(s, a.ref, a.note)
        s.commit()
        print(f"[evalhub] 已派生 {v.ref}（草稿），包含 {len(v.cases)} 条用例")


def cmd_agreement(a) -> None:
    with session() as s:
        v = dsm.get_version(s, a.ref)
        res = dsm.agreement(v)
        if len(res["labelers"]) < 2:
            print(f"[evalhub] {v.ref} 只有标注人 {res['labelers']}，至少需要两位才能计算一致性")
            return
        for p in res["pairs"]:
            x, y = p["pair"]
            print(f"{x} vs {y}：共同标注 {p['n']} 条，一致 {p['agree']} 条，Cohen's kappa = {p['kappa']}")
            for d in p["disagreements"]:
                print(f"  {d['key']}：{x}={d[x]}  {y}={d[y]}"
                      + (f"  （{x}：{d[x + '_note']}）" if d[x + "_note"] else "")
                      + (f"  （{y}：{d[y + '_note']}）" if d[y + "_note"] else ""))


def cmd_run(a) -> None:
    from .runner import run_extraction_eval, run_policy_eval

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    with session() as s:
        v = dsm.get_version(s, a.dataset)
        if a.task == "policy":
            judge = None if a.no_llm else connect_llm()
            run = run_policy_eval(s, v, judge=judge, note=a.note, gold=a.gold,
                                  extraction_run=a.extraction_run,
                                  extraction_report=_path(a.extraction_report) if a.extraction_report else None,
                                  workers=a.workers, agent=connect_agent(a.agent))
        else:
            vlm = None if a.no_vlm else connect_vlm()
            if a.force_vlm and vlm is None:
                sys.exit("--force-vlm 需要可用的视觉模型")
            run = run_extraction_eval(s, v, vlm=vlm, note=a.note, force_vlm=a.force_vlm, workers=a.workers)
        s.commit()
        print(f"[evalhub] 运行 #{run.id} 完成，用时 {run.wall_seconds}s")
        _show(run, a.limit)


def _show(run: Run, limit: int = 15) -> None:
    m = run.metrics
    print(f"\n运行 #{run.id}  {run.task}  {run.version.ref}  {run.note}")
    print(f"代码 {run.code_hash[:10]}  提示词 {run.prompt_hash[:10]}  制度 {run.policy_hash[:10]}  "
          f"git {run.git_commit[:10] if run.git_commit else '无'}  模型 {run.models}  参数 {run.params}")
    if run.task == "policy":
        if m.get("accuracy") is None:
            print(f"没有可打分的用例（未标注 / 有争议 / 拿不准共 {m['unscored']} 条）")
            return
        lo, hi = m["accuracy_ci95"]
        print(f"结论准确率 {m['accuracy']:.1%}（95% 区间 {lo:.1%}–{hi:.1%}，{m['scored']} 条参与打分，"
              f"{m['unscored']} 条未标注/有争议/拿不准）")
        print(f"误通过 {m['false_approve']}  误驳回 {m['false_reject']}  多转人工 {m['extra_review']}  "
              f"大模型调用 {m['llm_calls']} 次  tokens {m['tokens']}")
        if "agent" in m:
            _show_agent(m["agent"])
        else:
            print("\n| 分组 | 条数 | 准确率 |\n|---|---|---|")
            for k, g in m["groups"].items():
                print(f"| {k} | {g['n']} | {g['accuracy']:.0%} |")
    else:
        lo, hi = m["doc_accuracy_ci95"] or (0, 0)
        print(f"整张全对 {m['doc_accuracy']}（95% 区间 {lo:.1%}–{hi:.1%}）  静默错误 {m['silent_errors']}  "
              f"标记复核 {m['flagged']}  路径 {m['tiers']}  模型调用 {m['model_calls']}  tokens {m['tokens']}")
    wrong = [r for r in run.results if r.correct is False]
    if wrong:
        print(f"\n判错 {len(wrong)} 条（最多显示 {limit} 条）：")
        for r in wrong[:limit]:
            tag = r.case.tags.get("family") or r.case.tags.get("scenario") or r.case.tags.get("kind", "")
            extra = (f"违规 {r.output.get('violated')} 不确定 {r.output.get('uncertain')}" if run.task == "policy"
                     else f"读错 {r.output.get('wrong_fields')}")
            ag = r.output.get("agent")
            if ag:
                extra += f"；初审 {r.output.get('decision_before')}，调查 {ag['status']}" + (
                    f"（{ag['skip_reason']}）" if ag["status"] == "skipped" else
                    f"，工具 {[x['tool'] for x in ag['steps']]}")
            print(f"  {r.case.key} [{tag}] 标准答案 {r.gold}，系统 {r.predicted}；{extra}")


def _show_agent(g: dict) -> None:
    pi = g["per_investigation"]
    print(f"\n调查 Agent：调查 {g['investigated']} 条，跳过 {g['skipped']} 条，状态 {g['status']}")
    print(f"准确率 {g['accuracy_before']:.1%} → {g['accuracy_after']:.1%}   "
          f"转人工率 {g['review_rate_before']:.1%} → {g['review_rate_after']:.1%}   "
          f"误通过 {g['false_approve_before']} → {g['false_approve_after']}   "
          f"误驳回 {g['false_reject_before']} → {g['false_reject_after']}")
    print(f"改对 {g['fixed']} 条，改错 {g['broken']} 条；追问员工 {g['asked_employee']} 次；"
          f"证据采纳 {g['findings_accepted']}、拒绝 {g['findings_rejected']}；出错 {g['errors']}")
    if pi["tool_calls"] is not None:
        print(f"每次调查平均：工具 {pi['tool_calls']} 次，模型 {pi['llm_calls']} 次，tokens {pi['tokens']}，"
              f"耗时 {pi['latency_ms']} ms")
    print("\n| 分组 | 条数 | 调查前 | 调查后 |\n|---|---|---|---|")
    for k, x in g["groups"].items():
        print(f"| {k} | {x['n']} | {x['before']:.0%} | {x['after']:.0%} |")


def cmd_show(a) -> None:
    with session() as s:
        run = s.get(Run, a.run)
        if run is None:
            sys.exit(f"没有运行 #{a.run}")
        _show(run, a.limit)
        if a.case:
            r = next((x for x in run.results if x.case.key == a.case), None)
            print(json.dumps(r.output if r else None, ensure_ascii=False, indent=2))


def _two_runs(s, a):
    base, new = s.get(Run, a.base), s.get(Run, a.new)
    if base is None or new is None:
        sys.exit("运行编号不存在，用 list 查看")
    return compare(base, new)


def cmd_compare(a) -> None:
    with session() as s:
        print(format_comparison(_two_runs(s, a), a.limit))


def cmd_gate(a) -> None:
    with session() as s:
        cmp = _two_runs(s, a)
        if a.waive and not a.reason:
            sys.exit("--waive 必须同时用 --reason 写明为什么允许这些用例改坏")
        reasons = gate(cmp, a.max_broken, set(a.waive))
        print(format_comparison(cmp, a.limit))
        if a.waive:
            print(f"\n[门禁] 豁免 {'、'.join(a.waive)}，理由：{a.reason}")
        if reasons:
            print("\n[门禁] 不通过：\n  " + "\n  ".join(reasons))
            sys.exit(1)
        print("\n[门禁] 通过")


def cmd_export(a) -> None:
    """导出某个版本的用例和全部标注。数据库在 data/ 下不入 git，标注成果靠导出文件保存到仓库里。"""
    with session() as s:
        v = dsm.get_version(s, a.ref)
        out = _path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps({**c.payload, "tags": c.tags, "labels": [
                    {"labeler": lb.labeler, "decision": lb.decision, "rules": lb.rules, "note": lb.note,
                     "unsure": lb.unsure} for lb in c.labels]}, ensure_ascii=False) + "\n" for c in v.cases)
        print(f"[evalhub] 已导出 {v.ref}（{len(v.cases)} 条）到 {out}")


def cmd_harvest(a) -> None:
    from ..config import get_settings
    from .harvest import harvest

    source = a.source or get_settings().database_url
    with session() as s:
        rep_ = harvest(source, s, a.dataset, audit=a.audit, seed=a.seed, dry_run=a.dry_run)
        s.commit()
    print(rep_.format())
    if rep_.added and not a.dry_run:
        print(f"下一步：evalhub serve 盲标 {rep_.version}（抽检的单子没有财务结论，必须标），"
              f"然后 evalhub agreement {a.dataset}")


def cmd_serve(a) -> None:
    import uvicorn

    from .web import create_app

    print(f"[evalhub] 标注页面：http://{a.host}:{a.port}/  （Ctrl+C 退出）")
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalhub", description="RAMP 评测平台")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="导入项目自带的评测集（可重复运行）")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("list", help="列出评测集、版本和最近的运行")
    p.add_argument("--runs", type=int, default=10)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("import", help="导入 jsonl 为评测集的新版本（草稿）")
    p.add_argument("file")
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", choices=["policy", "extraction"], default="policy")
    p.add_argument("--labeler", default="generator", help="记录里 expected 字段的标注人名")
    p.add_argument("--no-labels", action="store_true", help="不导入 expected，留给人工标注")
    p.add_argument("--freeze", action="store_true", help="导入后直接冻结")
    p.add_argument("--description", default="")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("freeze", help="冻结一个草稿版本")
    p.add_argument("ref")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("fork", help="从已冻结版本派生新草稿")
    p.add_argument("ref")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_fork)

    p = sub.add_parser("agreement", help="标注人之间的一致性")
    p.add_argument("ref")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("run", help="在某个评测集版本上运行并入库")
    p.add_argument("task", choices=["policy", "extraction"])
    p.add_argument("--dataset", required=True, help="名称或 名称@v版本")
    p.add_argument("--note", default="")
    p.add_argument("--gold", default="auto", help="用哪位标注人的答案打分，默认 auto")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--force-vlm", action="store_true")
    p.add_argument("--extraction-run", type=int, help="端到端：用这次识别运行的结果作为发票字段")
    p.add_argument("--extraction-report", help="端到端：用旧的识别评测报告 .jsonl 作为发票字段")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--agent", choices=["llm", "scripted"],
                   help="初审之后再做调查：llm 调查 Agent；scripted 写死流程的对照组")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("show", help="查看一次运行")
    p.add_argument("run", type=int)
    p.add_argument("--case", help="打印某条用例的完整输出")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_show)

    for name, func, help_ in (("compare", cmd_compare, "对比两次运行"), ("gate", cmd_gate, "回归门禁")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("base", type=int)
        p.add_argument("new", type=int)
        p.add_argument("--limit", type=int, default=20)
        if name == "gate":
            p.add_argument("--max-broken", type=int, default=0)
            p.add_argument("--waive", action="append", default=[], metavar="用例",
                           help="允许改坏的用例，可重复；必须配合 --reason")
            p.add_argument("--reason", default="")
        p.set_defaults(func=func)

    p = sub.add_parser("export", help="导出用例和标注")
    p.add_argument("ref")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("harvest", help="数据闭环：把线上财务复核和抽检的单子收进评测集")
    p.add_argument("--source", help="业务库地址，默认用 .env 里的 DATABASE_URL")
    p.add_argument("--dataset", default="policy-prod")
    p.add_argument("--audit", type=int, default=20, help="从没人复核的自动结论里随机抽检多少条")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser("serve", help="启动标注页面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    try:
        a.func(a)
    except dsm.EvalHubError as exc:
        sys.exit(f"[evalhub] {exc}")


if __name__ == "__main__":
    main()
