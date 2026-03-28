#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p logs checkpoints data

echo "[owt] encode (skip if train + valid npy and meta already exist — delete to force re-encode)"
if [[ ! -f data/owt_train_tokens_bpe10k.npy ]] || [[ ! -f data/owt_valid_tokens_bpe10k.npy ]] || [[ ! -f data/owt_bpe10k_meta.json ]]; then
  PYTHONUNBUFFERED=1 uv run python -m cs336_basics.encode_text_tokens \
    --train_txt data/owt_train.txt \
    --valid_txt data/owt_valid.txt \
    --tokenizer_pkl data/tinystories_bpe10k_tokenizer.pkl \
    --train_tokens_out data/owt_train_tokens_bpe10k.npy \
    --valid_tokens_out data/owt_valid_tokens_bpe10k.npy \
    --meta_out data/owt_bpe10k_meta.json
else
  echo "[owt] found existing npy + meta; not re-encoding. Remove those files to re-encode."
fi

VOCAB_SIZE="$(uv run python -c "import json; print(json.load(open(\"data/owt_bpe10k_meta.json\"))[\"vocab_size\"])")"

echo "[owt] training vocab_size=${VOCAB_SIZE}"
PYTHONUNBUFFERED=1 uv run python -m cs336_basics.train \
  --train_tokens_path data/owt_train_tokens_bpe10k.npy \
  --valid_tokens_path data/owt_valid_tokens_bpe10k.npy \
  --vocab_size "${VOCAB_SIZE}" \
  --context_length 256 \
  --d_model 512 \
  --num_layers 12 \
  --num_heads 16 \
  --d_ff 1344 \
  --batch_size 64 \
  --max_steps 50000 \
  --eval_every 500 \
  --eval_batches 50 \
  --log_every 20 \
  --save_every 500 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_iters 1000 \
  --cosine_cycle_iters 50000 \
  --device cuda \
  --sleep_per_step 0 \
  --ckpt_path "checkpoints/owt_overnight_bpe10k_$(date +%F_%H-%M).pt" \
  2>&1 | tee "logs/owt_overnight_$(date +%F_%H-%M).log"
