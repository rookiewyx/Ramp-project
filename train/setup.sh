#!/usr/bin/env bash
# 训练机器上装环境（在 AutoDL 之类的租用机器上执行一次）。按 ms-swift 的 Qwen3.5 最佳实践：
# https://swift.readthedocs.io/en/v4.4/BestPractices/Qwen3_5-Best-Practice.html
# 镜像建议选 PyTorch 2.x + CUDA 12.x + Python 3.12（flash-linear-attention 推荐 3.12）。
set -euo pipefail

pip install -U "ms-swift>=4.4" modelscope
pip install -U "transformers>=5.9" "qwen_vl_utils>=0.0.14" peft liger-kernel
# Qwen3.5 的线性注意力层需要下面两个包；要现场编译，可能要十几分钟
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
pip install "flash-attn==2.8.3" --no-build-isolation
pip install -U "vllm>=0.17.0"

# 基础模型从魔搭下载（国内机器比 Hugging Face 稳）。MODEL 可以换成 Qwen/Qwen3.5-2B
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
modelscope download --model "$MODEL" --local_dir "$HOME/models/$(basename "$MODEL")"

python -c "import swift, vllm, transformers; print('ok', swift.__version__, vllm.__version__, transformers.__version__)"
nvidia-smi
