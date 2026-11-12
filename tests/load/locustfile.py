"""审核服务压测。

两类用户：
  员工（9 成）：提交一笔报销，然后每 0.5 秒查一次，直到审核完成。
      除了接口本身的耗时，额外上报一项 “review (end-to-end)”：从提交到拿到结论的总时间，
      这才是员工真正感受到的等待时间。
  财务（1 成）：看待办列表，打开一笔，给出复核结论。

发票从合成数据里取（文字版 PDF 和拍照件都有，比例和真实数据一致）。

用法（服务、worker、模拟模型都启动后）：
    .venv\\Scripts\\locust -f tests\\load\\locustfile.py --host http://127.0.0.1:8080 \\
        --headless -u 30 -r 5 -t 3m --csv reports\\load\\run1
可选环境变量：
    RAMP_LOAD_DATA   合成数据（默认 data/synthetic/expenses.jsonl）
    RAMP_API_KEY     接口鉴权的 key
    RAMP_PACING_S    每个员工两次提交之间的间隔（秒，默认 10），用来控制到达速率：
                     到达速率 ≈ 员工数 × 0.9 ÷ 间隔
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from pathlib import Path

from locust import HttpUser, between, constant_pacing, events, task

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / os.environ.get("RAMP_LOAD_DATA", "data/synthetic/expenses.jsonl")
API_KEY = os.environ.get("RAMP_API_KEY", "")
PACING = float(os.environ.get("RAMP_PACING_S", "10"))
FIELDS = ["category", "submit_date", "employee_id", "employee_name", "report_id", "note", "city",
          "trip_start", "trip_end", "nights", "headcount", "preapproval_id"]
POOL: list[tuple[dict, tuple[str, bytes] | None]] = []


@events.init.add_listener
def load_pool(environment, **_):
    """启动时把发票读进内存，压测过程中不碰磁盘。"""
    for line in DATA.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        form = {k: str(r[k]) for k in FIELDS if r.get(k) is not None}
        form["amount"] = f"{float(r['amount']):.2f}"
        f = r["invoice"].get("file")
        file = (Path(f).name, (ROOT / f).read_bytes()) if f and (ROOT / f).exists() else None
        POOL.append((form, file))
    print(f"[load] 载入 {len(POOL)} 笔报销，其中 {sum(1 for _, f in POOL if f and f[0].endswith('.jpg'))} 张拍照件")


def headers(**extra) -> dict:
    return {**({"X-API-Key": API_KEY} if API_KEY else {}), **extra}


class Employee(HttpUser):
    weight = 9
    wait_time = constant_pacing(PACING)

    @task
    def submit_and_wait(self):
        form, file = random.choice(POOL)
        t0 = time.perf_counter()
        files = {"file": file} if file else None
        with self.client.post("/v1/expenses", data=form, files=files, name="POST /v1/expenses",
                              headers=headers(**{"Idempotency-Key": uuid.uuid4().hex}),
                              catch_response=True) as r:
            if r.status_code != 202:
                r.failure(f"{r.status_code} {r.text[:200]}")
                return
            eid = r.json()["id"]
        deadline = t0 + 180
        while time.perf_counter() < deadline:
            d = self.client.get(f"/v1/expenses/{eid}", name="GET /v1/expenses/{id}", headers=headers()).json()
            if d["status"] in ("done", "failed"):
                ok = d["status"] == "done"
                events.request.fire(request_type="E2E", name="review (end-to-end)",
                                    response_time=(time.perf_counter() - t0) * 1000, response_length=0,
                                    exception=None if ok else RuntimeError(d.get("error")), context={})
                return
            time.sleep(0.5)
        events.request.fire(request_type="E2E", name="review (end-to-end)",
                            response_time=(time.perf_counter() - t0) * 1000, response_length=0,
                            exception=TimeoutError("180 秒内没有审核完成"), context={})


class Finance(HttpUser):
    weight = 1
    wait_time = between(3, 6)

    @task
    def review_queue(self):
        todo = self.client.get("/v1/expenses?decision=review&pending=true&limit=20",
                               name="GET /v1/expenses (待办)", headers=headers()).json()
        if not todo:
            return
        e = random.choice(todo)
        self.client.get(f"/v1/expenses/{e['id']}", name="GET /v1/expenses/{id}", headers=headers())
        self.client.post(f"/v1/expenses/{e['id']}/override", name="POST /v1/expenses/{id}/override",
                         json={"reviewer": "压测财务", "decision": "review", "note": ""}, headers=headers())
