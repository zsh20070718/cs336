#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/personal/zsh/toys/cs336/assignment-1
source /mnt/data/personal/zsh/miniconda3/etc/profile.d/conda.sh
conda activate cs336

echo "Starting training..."

PYTHONUNBUFFERED=1 python -u -m uv run cs336_basics/train.py \
  --train_tokens_path data/tinystories_train_tokens_bpe10k.npy \
  --valid_tokens_path data/tinystories_valid_tokens_bpe10k.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --batch_size 8 \
  --max_steps 20000 \
  --eval_every 500 \
  --eval_batches 50 \
  --log_every 20 \
  --save_every 500 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_iters 1000 \
  --cosine_cycle_iters 20000 \
  --device cuda \
  --sleep_per_step 0.03 \
  --ckpt_path checkpoints/overnight_bpe10k_$(date +%F_%H-%M).pt \
  2>&1 | tee logs/overnight_$(date +%F_%H-%M).log