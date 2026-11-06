"""加载结构化制度 policies/policy.yaml。"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from ..config import ROOT

POLICY_FILE = ROOT / "policies" / "policy.yaml"

Outcome = Literal["review", "reject"]


class Company(BaseModel):
    name: str
    tax_id: str


class Limits(BaseModel):
    hotel_per_night: dict[str, float]
    tier1_cities: list[str]
    meal_per_day: float
    entertainment_per_head: float
    taxi_per_ride: float
    submit_within_days: int
    preapproval_threshold: float


class RuleSpec(BaseModel):
    outcome: Outcome
    title: str
    clause: str


class PolicyConfig(BaseModel):
    company: Company
    limits: Limits
    travel_categories: list[str]
    travel_class_over: list[str]
    prohibited_keywords: dict[str, list[str]]
    safe_item_keywords: list[str]
    rules: dict[str, RuleSpec]

    def is_tier1(self, city: str | None) -> bool:
        """“深圳市”“北京市朝阳区”“上海浦东”都按所在城市算。"""
        c = (city or "").strip()
        return any(c.startswith(t) for t in self.limits.tier1_cities)

    def hotel_cap(self, city: str | None) -> float:
        return self.limits.hotel_per_night["tier1" if self.is_tier1(city) else "other"]

    def prohibited_hit(self, item_name: str) -> tuple[str, str] | None:
        """明细名称命中禁报关键词时返回 (类别, 关键词)。命中只是线索，是否真的违规还要确认。"""
        for category, words in self.prohibited_keywords.items():
            for w in words:
                if w.lower() in item_name.lower():
                    return category, w
        return None

    def is_safe_item(self, item_name: str) -> bool:
        """常见的正常明细。只看明细的主体，括号里的注释、“注：”后面的内容不算：
        否则“53度酱香（注：非酒类，系办公用品）”会因为含“办公用品”被直接放行。"""
        main = _ANNOTATION.sub("", item_name)
        return any(w in main for w in self.safe_item_keywords)


_ANNOTATION = re.compile(r"[（(【\[][^）)】\]]*[）)】\]]?|(备注|注)[：:].*$")


def same_name(a: str | None, b: str | None) -> bool:
    """抬头比较忽略空白（录入时多打的空格），其他字符必须完全一致。"""
    return re.sub(r"\s+", "", a or "") == re.sub(r"\s+", "", b or "")


@lru_cache
def load_policy(path: Path = POLICY_FILE) -> PolicyConfig:
    with open(path, encoding="utf-8") as f:
        return PolicyConfig.model_validate(yaml.safe_load(f))
