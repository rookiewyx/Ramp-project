"""压测用的模拟模型服务（OpenAI 兼容接口）。

压测要回答的是“我们的系统能扛多少”，不是“百炼能扛多少”。直接打真实模型有三个问题：
花钱、被限流、模型延迟的波动会盖住系统本身的瓶颈。所以用这个服务代替，按实测数据模拟：
  - 视觉模型：中位数 3.5 s（eval_log 里 qwen3-vl-plus 的实测平均值）
  - 文本模型：中位数 1.2 s
延迟服从对数正态分布（有长尾，和真实接口一样）。还可以模拟故障率和服务方的并发上限（超过返回 429）。

视觉模型返回一张格式正确、能通过校验的发票；文本模型对问到的明细都答“不违规”、理由成立、人数 4。
压测只关心耗时，不关心结论对不对。

用法：
    .venv\\Scripts\\python scripts\\mock_models.py --port 8009 --vlm-ms 3500 --llm-ms 1200
然后在 .env 或环境变量里把 LLM_BASE_URL 和 VLM_BASE_URL 指向 http://127.0.0.1:8009/v1，API_KEY 随便填。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

COMPANY, TAX_ID = "星河数智科技有限公司", "91310000MA1XHSZ00X"
ITEMS_RE = re.compile(r'(\[\{"index".*?\}\])')


def build_app(vlm_ms: float, llm_ms: float, sigma: float, fail_rate: float, max_concurrent: int) -> FastAPI:
    app = FastAPI(title="mock models")
    state = {"inflight": 0, "served": 0, "rejected": 0, "failed": 0}
    rng = random.Random(7)

    def delay(median_ms: float) -> float:
        return median_ms / 1000 * math.exp(rng.gauss(0, sigma))

    def reply(content: str, model: str, pt: int, ct: int) -> dict:
        return {"id": f"mock-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        msgs = body.get("messages", [])
        vision = any(isinstance(m.get("content"), list)
                     and any(c.get("type") == "image_url" for c in m["content"]) for m in msgs)
        if max_concurrent and state["inflight"] >= max_concurrent:
            state["rejected"] += 1
            return JSONResponse(status_code=429, content={"error": {"message": "rate limited (mock)"}})
        state["inflight"] += 1
        try:
            await asyncio.sleep(delay(vlm_ms if vision else llm_ms))
            if rng.random() < fail_rate:
                state["failed"] += 1
                return JSONResponse(status_code=500, content={"error": {"message": "internal error (mock)"}})
            state["served"] += 1
            if vision:
                inv = {"invoice_no": f"{rng.randrange(10**19, 10**20)}", "issue_date": "2026-10-28",
                       "seller": "样例商户", "buyer_name": COMPANY, "buyer_tax_id": TAX_ID,
                       "items": [{"name": "餐饮服务", "amount": 88.0}], "total": 88.0, "tax_rate": 0.06}
                return reply(json.dumps(inv, ensure_ascii=False), body.get("model", "mock-vl"), 1300, 150)
            text = "\n".join(m["content"] for m in msgs if isinstance(m.get("content"), str))
            ans: dict = {"taxi_reason": {"valid": True, "reason": "模拟"}, "headcount": 4}
            m = ITEMS_RE.search(text)
            if m:
                listing = json.loads(m.group(1))
                ans["items"] = [{"index": x["index"], "name": x["name"], "prohibited": False,
                                 "category": "无", "reason": "模拟"} for x in listing]
            return reply(json.dumps(ans, ensure_ascii=False), body.get("model", "mock-llm"), 300, 60)
        finally:
            state["inflight"] -= 1

    @app.get("/stats")
    def stats():
        return state

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8009)
    p.add_argument("--vlm-ms", type=float, default=3500, help="视觉模型延迟中位数（毫秒）")
    p.add_argument("--llm-ms", type=float, default=1200, help="文本模型延迟中位数（毫秒）")
    p.add_argument("--sigma", type=float, default=0.35, help="对数正态分布的 sigma，越大长尾越长")
    p.add_argument("--fail-rate", type=float, default=0.0, help="返回 500 的比例")
    p.add_argument("--max-concurrent", type=int, default=0, help="服务方并发上限，超过返回 429；0 表示不限")
    a = p.parse_args()
    uvicorn.run(build_app(a.vlm_ms, a.llm_ms, a.sigma, a.fail_rate, a.max_concurrent),
                host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
