from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import run_train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train BPE tokenizer and save tokenized TinyStories")
    parser.add_argument("--data_dir", type = str, default = "data")
    parser.add_argument("--train_txt", type = str, default = "TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_txt", type = str, default = "TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_size", type = int, default = 10000)
    parser.add_argument("--special_token", type = str, default = "<|endoftext|>")
    parser.add_argument("--train_tokens_out", type = str, default = "tinystories_train_tokens_bpe10k.npy")
    parser.add_argument("--valid_tokens_out", type = str, default = "tinystories_valid_tokens_bpe10k.npy")
    parser.add_argument("--tokenizer_pkl_out", type = str, default = "tinystories_bpe10k_tokenizer.pkl")
    parser.add_argument("--meta_out", type = str, default = "tokenizer_meta.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    train_txt = data_dir / args.train_txt
    valid_txt = data_dir / args.valid_txt

    if not train_txt.exists():
        raise FileNotFoundError(f"Missing train file: {train_txt}")
    if not valid_txt.exists():
        raise FileNotFoundError(f"Missing valid file: {valid_txt}")

    special_tokens = [args.special_token]

    print("Training BPE...")
    vocab, merges = run_train_bpe(
        input_path = train_txt,
        vocab_size = args.vocab_size,
        special_tokens = special_tokens,
    )
    tokenizer = Tokenizer(vocab, merges, special_tokens = special_tokens)

    print("Encoding train...")
    train_tokens = np.array(
        tokenizer.encode(train_txt.read_text(encoding = "utf-8")),
        dtype = np.int32,
    )
    print("Encoding valid...")
    valid_tokens = np.array(
        tokenizer.encode(valid_txt.read_text(encoding = "utf-8")),
        dtype = np.int32,
    )

    train_out = data_dir / args.train_tokens_out
    valid_out = data_dir / args.valid_tokens_out
    tokenizer_pkl_out = data_dir / args.tokenizer_pkl_out
    meta_out = data_dir / args.meta_out

    np.save(train_out, train_tokens)
    np.save(valid_out, valid_tokens)

    with open(tokenizer_pkl_out, "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": special_tokens,
            },
            f,
        )

    meta = {
        "type": "bpe_utf8",
        "vocab_size": args.vocab_size,
        "dtype": "int32",
        "special_tokens": special_tokens,
        "train_tokens_shape": list(train_tokens.shape),
        "valid_tokens_shape": list(valid_tokens.shape),
        "train_tokens_path": str(train_out),
        "valid_tokens_path": str(valid_out),
        "tokenizer_pkl_path": str(tokenizer_pkl_out),
    }
    meta_out.write_text(json.dumps(meta, indent = 2), encoding = "utf-8")

    print("Saved:")
    print(train_out)
    print(valid_out)
    print(tokenizer_pkl_out)
    print(meta_out)
    print(f"train_tokens={train_tokens.shape[0]} valid_tokens={valid_tokens.shape[0]}")


if __name__ == "__main__":
    main()
