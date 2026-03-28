#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p logs checkpoints

if [[ ! -f data/owt_bpe10k_meta.json ]]; then
  echo "Missing data/owt_bpe10k_meta.json — run encode first or train_owt_smoke.sh"
  exit 1
fi

VOCAB_SIZE="$(uv run python -c "import json; print(json.load(open(\"data/owt_bpe10k_meta.json\"))[\"vocab_size\"])")"

echo "[smoke train-only] vocab_size=${VOCAB_SIZE}"
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
  --ckpt_path checkpoints/owt_smoke_train_only.pt \
  2>&1 | tee "logs/owt_smoke_train_only_$(date +%F_%H-%M).log"
