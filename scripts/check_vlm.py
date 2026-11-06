"""检查视觉模型配置：拿一张合成的拍照发票调用一次，打印识别结果、校验结果、耗时和 tokens。

用法：
  .venv\\Scripts\\python scripts\\check_vlm.py
  .venv\\Scripts\\python scripts\\check_vlm.py --image 某张发票.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ramp_agent.config import get_settings
from ramp_agent.extraction import OpenAICompatibleVLM, validate


def pick_sample() -> tuple[Path, dict | None]:
    """从合成数据里挑第一张拍照件，同时带上标准答案方便对比。"""
    for line in (ROOT / "data/synthetic/expenses.jsonl").read_text(encoding="utf-8").splitlines():
        inv = json.loads(line)["invoice"]
        if inv["file_type"] == "jpg":
            return ROOT / inv["file"], inv
    sys.exit("合成数据里没有拍照件，请先运行 scripts/gen_synthetic.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="要识别的图片，默认用合成数据里的第一张拍照件")
    args = ap.parse_args()

    s = get_settings()
    key = s.vlm_api_key
    masked = "（未填写）" if not key or key == "EMPTY" else f"{key[:5]}...{key[-4:]}"
    print(f"地址  {s.vlm_base_url}\n模型  {s.vlm_model}\n密钥  {masked}\n")

    path, gt = (Path(args.image), None) if args.image else pick_sample()
    print(f"识别  {path}")
    t0 = time.perf_counter()
    try:
        out = OpenAICompatibleVLM(s).extract(path.read_bytes())
    except openai.AuthenticationError:
        sys.exit("[失败] API Key 无效：检查 VLM_API_KEY 是否完整复制，以及密钥和地址是否属于同一地域")
    except openai.NotFoundError as exc:
        sys.exit(f"[失败] 地址或模型名不对（404）：检查 VLM_BASE_URL 和 VLM_MODEL\n{exc}")
    except openai.APIConnectionError as exc:
        sys.exit(f"[失败] 连不上 {s.vlm_base_url}：{exc}")
    except openai.APIStatusError as exc:
        sys.exit(f"[失败] 服务返回 {exc.status_code}：{exc.message}")
    cost = time.perf_counter() - t0

    print(f"\n[成功] 用时 {cost:.1f}s，tokens：输入 {out.prompt_tokens} / 输出 {out.completion_tokens}")
    print(json.dumps(out.fields.model_dump(), ensure_ascii=False, indent=2))
    failed, suspicious = validate(out.fields)
    print("\n校验：" + ("全部通过" if not failed else f"未通过 {failed}，可疑字段 {suspicious}"))

    if gt:
        wrong = [k for k in ("invoice_no", "issue_date", "seller", "buyer_name", "buyer_tax_id", "total")
                 if getattr(out.fields, k) != gt[k]]
        print("对比标准答案：" + ("主要字段全部正确" if not wrong else
                                f"读错 {wrong}；标准答案 { {k: gt[k] for k in wrong} }"))


if __name__ == "__main__":
    main()
