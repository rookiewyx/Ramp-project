"""把合成数据里的几笔报销提交给审核服务，看接口的实际效果。

先启动服务（另开一个窗口）：
    .venv\\Scripts\\python -m ramp_agent.service api --eager
再运行：
    .venv\\Scripts\\python scripts\\demo_service.py -n 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["category", "amount", "submit_date", "employee_id", "employee_name", "report_id", "note",
          "city", "trip_start", "trip_end", "nights", "headcount", "preapproval_id"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key", default="")
    p.add_argument("--data", default="data/synthetic/expenses.jsonl")
    p.add_argument("-n", type=int, default=10)
    a = p.parse_args()

    headers = {"X-API-Key": a.api_key} if a.api_key else {}
    records = [json.loads(line) for line in (ROOT / a.data).read_text(encoding="utf-8").splitlines()][: a.n]
    ids = {}
    with httpx.Client(base_url=a.url, headers=headers, timeout=120) as c:
        for r in records:
            form = {k: str(r[k]) for k in FIELDS if r.get(k) is not None}
            form["amount"] = f"{float(r['amount']):.2f}"
            f = r["invoice"].get("file")
            files = None
            if f:
                path = ROOT / f
                files = {"file": (path.name, path.read_bytes())}
            resp = c.post("/v1/expenses", data=form, files=files,
                          headers={"Idempotency-Key": f"demo-{r['expense_id']}"})
            resp.raise_for_status()
            ids[resp.json()["id"]] = r

        # 用了 worker 时审核在后台进行，轮询到全部完成
        deadline = time.time() + 300
        results = {}
        while len(results) < len(ids) and time.time() < deadline:
            for eid in ids:
                if eid not in results:
                    d = c.get(f"/v1/expenses/{eid}").json()
                    if d["status"] in ("done", "failed"):
                        results[eid] = d
            time.sleep(0.5)

        hit = Counter()
        for eid, d in results.items():
            want = ids[eid]["expected"]["decision"]
            ok = d["decision"] == want
            hit[ok] += 1
            rv = d.get("review") or {}
            print(f"{ids[eid]['expense_id']}  {ids[eid]['category']:<13} 系统 {d['decision'] or d['status']:<7}"
                  f" 期望 {want:<7} {'✓' if ok else '✗'}  识别 {rv.get('extraction_tier', '-'):<8}"
                  f" 大模型 {rv.get('llm_calls', 0)} 次  {rv.get('latency_ms', 0)} ms")
        print(f"\n一致 {hit[True]}/{len(results)}")
        print(json.dumps(c.get("/v1/stats").json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
