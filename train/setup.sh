#!/usr/bin/env bash
# 训练机器上装环境（在 AutoDL 之类的租用机器上执行一次），参考 ms-swift 的 Qwen3.5 最佳实践：
# https://swift.readthedocs.io/en/v4.4/BestPractices/Qwen3_5-Best-Practice.html
# 镜像：PyTorch 2.7 以上、CUDA 12.8 以上、Python 3.12（RTX PRO 6000 是 Blackwell 架构，旧 CUDA 不支持）。
#
# 和官方步骤的三处不同：
#   1. 先装 vLLM：它会把 PyTorch 换成自己要求的版本。要编译的包必须在这之后装，
#      否则是对着旧 PyTorch 编译的，换版本后会报符号找不到；
#   2. 不用 flash-attn：它是可选的（训练用 PyTorch 自带的 sdpa，推理用 vLLM 自带的注意力）。
#      镜像里如果预装了，是对着旧 PyTorch 编译的，vLLM 一加载就报 undefined symbol，所以要卸掉；
#   3. causal-conv1d 要现场编译，需要 nvcc。AutoDL 的部分镜像没有 nvcc，那就跳过：
#      Qwen3.5 会走 PyTorch 的慢路径（RTX PRO 6000 上 batch 16 约 6.7 秒/步，4B 训 2 个 epoch 约 2 小时）。
set -euo pipefail

# AutoDL 的学术资源加速（访问 GitHub、Hugging Face），别的平台没有这个文件就跳过
source /etc/network_turbo 2>/dev/null || true

pip install -U "vllm>=0.17.0"
pip install -U "ms-swift>=4.4" modelscope "transformers>=5.9" "qwen_vl_utils>=0.0.14" peft liger-kernel
# 换了 PyTorch 之后，镜像里预编译的扩展都对不上了（报 undefined symbol），先卸掉
pip uninstall -y flash-attn flash_attn causal-conv1d causal_conv1d 2>/dev/null || true

# Qwen3.5 的线性注意力层用这两个包（没有也能跑，但很慢）。causal-conv1d 要现场编译，几分钟到十几分钟
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
if command -v nvcc >/dev/null; then
    MAX_JOBS=8 pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
else
    echo "没有 nvcc，跳过 causal-conv1d：训练能跑，只是慢一些"
fi

# 基础模型从魔搭下载（国内机器比 Hugging Face 稳）。MODEL 可以换成 Qwen/Qwen3.5-2B
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
modelscope download --model "$MODEL" --local_dir "$HOME/models/$(basename "$MODEL")"

python - <<'PY'
import torch, transformers, vllm, swift
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "算力", torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__, "vllm", vllm.__version__, "ms-swift", swift.__version__)
import importlib.util
import fla  # noqa: F401
print("causal_conv1d", "ok（快速路径）" if importlib.util.find_spec("causal_conv1d") else "没装（慢路径）")
assert importlib.util.find_spec("flash_attn") is None, "flash_attn 还在：vLLM 会加载它并崩掉，先 pip uninstall"
PY
