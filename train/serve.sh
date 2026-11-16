#!/usr/bin/env bash
# 合并 LoRA 并用 vLLM 起 OpenAI 兼容接口。项目里把 VLM_BASE_URL 指过来就能用，其余代码不改。
#   bash serve.sh <检查点目录>      # 微调后的模型，服务名 ramp-ticket
#   bash serve.sh base               # 不微调的基础模型（对照组），服务名 base
set -euo pipefail
MODEL=${MODEL:-Qwen3.5-4B}
PORT=${PORT:-8000}

if [ "${1:-}" = "base" ]; then
    WEIGHTS="$HOME/models/$MODEL"; NAME=base
else
    CKPT=${1:?用法：bash serve.sh <检查点目录> 或 bash serve.sh base}
    if [ ! -d "${CKPT}-merged" ]; then
        swift export --adapters "$CKPT" --merge_lora true
    fi
    WEIGHTS="${CKPT}-merged"; NAME=ramp-ticket
fi

# 图片大小交给模型自带的处理器决定：训练时也没有另设上限，两边一致
vllm serve "$WEIGHTS" \
    --served-model-name "$NAME" \
    --port "$PORT" \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.85
