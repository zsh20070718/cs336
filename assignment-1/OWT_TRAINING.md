# OWT training (reuse tokenizer, encode only)

## Prereqs

- Raw text: `data/owt_train.txt`, `data/owt_valid.txt` (UTF-8).
- Saved BPE tokenizer: default `data/tinystories_bpe10k_tokenizer.pkl` (same format as `generate.py` / `train_tokenizer.py` output).
- From `assignment-1/`, use `uv` ([README.md](./README.md)).

## Encode text to token `.npy`

```sh
uv run python -m cs336_basics.encode_text_tokens \
  --train_txt data/owt_train.txt \
  --valid_txt data/owt_valid.txt \
  --tokenizer_pkl data/tinystories_bpe10k_tokenizer.pkl \
  --train_tokens_out data/owt_train_tokens_bpe10k.npy \
  --valid_tokens_out data/owt_valid_tokens_bpe10k.npy \
  --meta_out data/owt_bpe10k_meta.json
```

On success you get `vocab_size` in the meta JSON and a round-trip check on the first `--verify_roundtrip_chars` characters of the train file (default 4096). Disable with `--verify_roundtrip_chars 0`.

## Smoke tests

**Full pipeline (encode + short train):**

```sh
chmod +x train_owt_smoke.sh
./train_owt_smoke.sh
```

Success: no exception; logs show `[train]` and `[eval]` lines; `checkpoints/owt_smoke.pt` written.

**Train-only smoke** (requires existing `data/owt_*_tokens_bpe10k.npy` and `data/owt_bpe10k_meta.json`):

```sh
chmod +x train_owt_smoke_train_only.sh
./train_owt_smoke_train_only.sh
```

## Overnight run

```sh
chmod +x train_owt.sh
./train_owt.sh
```

If token npy files already exist, the script skips re-encoding; delete the npy files to force a fresh encode.

## `train.py` flags

Read `vocab_size` from `data/owt_bpe10k_meta.json` — it must match the tokenizer embedding matrix size.

```sh
VOCAB_SIZE=$(uv run python -c "import json; print(json.load(open('data/owt_bpe10k_meta.json'))['vocab_size'])")
uv run python -m cs336_basics.train \
  --train_tokens_path data/owt_train_tokens_bpe10k.npy \
  --valid_tokens_path data/owt_valid_tokens_bpe10k.npy \
  --vocab_size "$VOCAB_SIZE" \
  ...
```

## Resume

```sh
uv run python -m cs336_basics.train \
  ...same args... \
  --resume \
  --ckpt_path checkpoints/your_checkpoint.pt
```

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `Missing tokenizer pickle` | Path to `--tokenizer_pkl`; generate with `train_tokenizer.py` or copy an existing `.pkl`. |
| Round-trip check failed | Wrong tokenizer for this text, or corrupt/cut UTF-8; fix file or tokenizer. |
| CUDA OOM | Lower `--batch_size`, `--context_length`, or model width (`--d_model`, `--num_layers`). |
| `vocab_size` mismatch | Always use the size from `owt_bpe10k_meta.json` after encoding with that tokenizer. |
| Slow encode | OWT files are large; encoding is single-pass line-by-line; ensure disk is local/fast. |
