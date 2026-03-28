#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p logs checkpoints data

echo "[smoke] encode OWT text with existing tokenizer -> npy"
PYTHONUNBUFFERED=1 uv run python -m cs336_basics.encode_text_tokens \
  --train_txt data/owt_train.txt \
  --valid_txt data/owt_valid.txt \
  --tokenizer_pkl data/tinystories_bpe10k_tokenizer.pkl \
  --train_tokens_out data/owt_train_tokens_bpe10k.npy \
  --valid_tokens_out data/owt_valid_tokens_bpe10k.npy \
  --meta_out data/owt_bpe10k_meta.json

VOCAB_SIZE="$(uv run python -c "import json; print(json.load(open(\"data/owt_bpe10k_meta.json\"))[\"vocab_size\"])")"

echo "[smoke] train (small budget)"
PYTHONUNBUFFERED=1 uv run python -m cs336_basics.train \
  --train_tokens_path data/owt_train_tokens_bpe10k.npy \
  --valid_tokens_path data/owt_valid_tokens_bpe10k.npy \
  --vocab_size "${VOCAB_SIZE}" \
  --context_length 128 \
  --d_model 256 \
  --num_layers 2 \
  --num_heads 8 \
  --d_ff 768 \
  --batch_size 4 \
  --max_steps 12 \
  --eval_every 4 \
  --eval_batches 2 \
  --log_every 2 \
  --save_every 100000 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_iters 2 \
  --cosine_cycle_iters 12 \
  --device cuda \
  --sleep_per_step 0 \
  --ckpt_path checkpoints/owt_smoke.pt \
  2>&1 | tee "logs/owt_smoke_$(date +%F_%H-%M).log"

echo "[smoke] done"
