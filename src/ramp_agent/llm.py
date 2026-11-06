"""调用大模型的公共部分：从模型输出里取 JSON。"""

from __future__ import annotations

import json
import re


def parse_json_object(text: str) -> dict:
    """模型有时会包一层 ```json ``` 或在前后加话，取第一个完整的 {...}。"""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"输出里没有 JSON：{text[:120]!r}")
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:                      # 字符串里的花括号不计数
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"JSON 不完整：{text[:120]!r}")
