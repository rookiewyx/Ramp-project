"""检查文本大模型配置：发一个审核用的语义判断请求，打印模型的回答、耗时和 tokens。

用法：.venv\\Scripts\\python scripts\\check_llm.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ramp_agent.config import get_settings
from ramp_agent.policy import LLMJudge, SemanticRequest


def main() -> None:
    s = get_settings()
    key = s.llm_api_key
    masked = "（未填写）" if not key or key == "EMPTY" else f"{key[:5]}...{key[-4:]}"
    print(f"地址  {s.llm_base_url}\n模型  {s.llm_model}\n密钥  {masked}\n")

    req = SemanticRequest(category="entertainment", amount=1200, note="宴请客户王总、李经理及我方2人",
                          items_to_check=["53%vol 酱香型 500ml×2", "电子烟雾报警器"], need_headcount=True)
    print("提问：明细“53%vol 酱香型 500ml×2”“电子烟雾报警器”是否属于禁报项目；说明里招待了几个人")
    t0 = time.perf_counter()
    try:
        res = LLMJudge(s).judge(req)
    except openai.AuthenticationError:
        sys.exit("[失败] API Key 无效：检查 LLM_API_KEY，以及密钥和地址是否属于同一地域")
    except openai.NotFoundError as exc:
        sys.exit(f"[失败] 地址或模型名不对（404）：检查 LLM_BASE_URL 和 LLM_MODEL\n{exc}")
    except openai.APIConnectionError as exc:
        sys.exit(f"[失败] 连不上 {s.llm_base_url}：{exc}")
    except openai.APIStatusError as exc:
        sys.exit(f"[失败] 服务返回 {exc.status_code}：{exc.message}")
    except ValueError as exc:
        sys.exit(f"[失败] 模型没有按要求输出 JSON：{exc}")

    print(f"\n[成功] 用时 {time.perf_counter() - t0:.1f}s，tokens：输入 {res.prompt_tokens} / 输出 {res.completion_tokens}")
    for name, v in res.items.items():
        print(f"  {name}：{'禁报' if v.prohibited else '正常'}（{v.category}）{v.reason}")
    print(f"  招待人数：{res.headcount}")
    print("\n期望：酱香型 → 禁报（烟酒）；电子烟雾报警器 → 正常；人数 4")


if __name__ == "__main__":
    main()
