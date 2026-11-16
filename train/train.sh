#!/usr/bin/env bash
# LoRA 微调。数据是 scripts/build_scid_sft.py 生成的 sft.zip，解压在 $HOME/sft。
#   bash train.sh                    # Qwen3.5-4B
#   MODEL=Qwen3.5-2B bash train.sh   # 2B 对照
set -euo pipefail
MODEL=${MODEL:-Qwen3.5-4B}
DATA=${DATA:-$HOME/sft}
OUT=${OUT:-$HOME/output/$MODEL}
EPOCHS=${EPOCHS:-2}
BS=${BS:-8}                # 96 GB 显存可以开到 16；24 GB 的卡用 2，再把 GRAD_ACC 调到 8
GRAD_ACC=${GRAD_ACC:-2}

cd "$DATA"                 # 训练数据里的图片路径是相对路径 images/xxx.jpg
# 视觉部分也加 LoRA（--freeze_vit false）：现在的错误主要出在“看”这一步（褪色、盖章压字）
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model "$HOME/models/$MODEL" \
    --tuner_type lora \
    --dataset train.jsonl \
    --val_dataset val.jsonl \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BS" \
    --per_device_eval_batch_size "$BS" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_checkpointing true \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --load_from_cache_file true \
    --output_dir "$OUT" 2>&1 | tee "$HOME/train_$MODEL.log"

echo "训练完成。最好的检查点：$(ls -d "$OUT"/v*/checkpoint-* | tail -1)"
