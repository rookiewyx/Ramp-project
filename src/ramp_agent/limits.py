"""模型调用并发上限。

模型服务都有并发或速率配额，超过就返回 429。压测发现：worker 并发 32、配额只有 10 时，
85% 的请求被拒，客户端重试、任务重试层层叠加，端到端 P95 从 5 秒涨到 41 秒，
还有 4% 的报销在重试用尽后被降级成转人工。

解决办法是在调用方排队，而不是把请求打过去等着被拒：每个进程对视觉模型、文本模型分别限制同时在途的请求数。
等待时间记在链路里（model.wait_ms），能直接看出是不是配额不够。

上限是每个 worker 进程的；开多个 worker 进程时，各进程上限之和不要超过配额。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from opentelemetry import trace


@lru_cache
def _semaphore(name: str, limit: int) -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(limit)


@contextmanager
def model_slot(name: str, limit: int) -> Iterator[None]:
    """占一个调用名额；limit <= 0 表示不限制。"""
    if limit <= 0:
        yield
        return
    sem = _semaphore(name, limit)
    t0 = time.perf_counter()
    sem.acquire()
    trace.get_current_span().set_attribute("model.wait_ms", round((time.perf_counter() - t0) * 1000, 1))
    try:
        yield
    finally:
        sem.release()
