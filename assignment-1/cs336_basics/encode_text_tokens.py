from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def load_tokenizer_from_pkl(
    tokenizer_pkl_path: Path,
    special_token: str,
) -> Tokenizer:
    with open(tokenizer_pkl_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, tuple) and len(payload) == 2:
        vocab, merges = payload
        special_tokens = [special_token]
    elif isinstance(payload, dict):
        vocab = payload["vocab"]
        merges = payload["merges"]
        special_tokens = payload.get("special_tokens", [special_token])
    else:
        raise ValueError(f"Unsupported tokenizer cache format at {tokenizer_pkl_path}")
    return Tokenizer(
        vocab = vocab,
        merges = merges,
        special_tokens = special_tokens,
    )


def encode_file_line_by_line(
    path: Path,
    tokenizer: Tokenizer,
) -> np.ndarray:
    ids: list[int] = []
    with open(path, encoding = "utf-8") as f:
        for line in f:
            ids.extend(tokenizer.encode(line))
    return np.array(ids, dtype = np.int32)


def verify_roundtrip(
    tokenizer: Tokenizer,
    text_path: Path,
    max_chars: int,
) -> None:
    if max_chars <= 0:
        return
    sample = text_path.read_text(encoding = "utf-8")[:max_chars]
    if not sample:
        return
    print("Sample done")
    roundtrip = tokenizer.decode(tokenizer.encode(sample))
    print("Roundtrip done")
    if roundtrip != sample:
        raise RuntimeError(
            "Tokenizer round-trip check failed (decode(encode(sample)) != sample). "
            "Pick a tokenizer trained on UTF-8 text compatible with this corpus.",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Encode text files to int32 token npy using a saved BPE tokenizer")
    parser.add_argument("--train_txt", type = str, required = True)
    parser.add_argument("--valid_txt", type = str, required = True)
    parser.add_argument("--tokenizer_pkl", type = str, default = "data/tinystories_bpe10k_tokenizer.pkl")
    parser.add_argument("--special_token", type = str, default = "<|endoftext|>")
    parser.add_argument("--train_tokens_out", type = str, required = True)
    parser.add_argument("--valid_tokens_out", type = str, required = True)
    parser.add_argument("--meta_out", type = str, default = None)
    parser.add_argument(
        "--verify_roundtrip_chars",
        type = int,
        default = 4096,
        help = "If >0, check decode(encode(first N chars of train_txt)) == prefix; 0 disables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_txt = Path(args.train_txt)
    valid_txt = Path(args.valid_txt)
    tokenizer_pkl = Path(args.tokenizer_pkl)
    if not tokenizer_pkl.exists():
        raise FileNotFoundError(f"Missing tokenizer pickle: {tokenizer_pkl}")
    if not train_txt.exists():
        raise FileNotFoundError(f"Missing train file: {train_txt}")
    if not valid_txt.exists():
        raise FileNotFoundError(f"Missing valid file: {valid_txt}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer_from_pkl(
        tokenizer_pkl_path = tokenizer_pkl,
        special_token = args.special_token,
    )
    vocab_size = len(tokenizer.vocab)
    print("Loading done")

    verify_roundtrip(
        tokenizer = tokenizer,
        text_path = train_txt,
        max_chars = args.verify_roundtrip_chars,
    )

    print("Encoding train...")
    train_tokens = encode_file_line_by_line(train_txt, tokenizer)
    print("Encoding valid...")
    valid_tokens = encode_file_line_by_line(valid_txt, tokenizer)

    train_out = Path(args.train_tokens_out)
    valid_out = Path(args.valid_tokens_out)
    train_out.parent.mkdir(parents = True, exist_ok = True)
    valid_out.parent.mkdir(parents = True, exist_ok = True)

    np.save(train_out, train_tokens)
    np.save(valid_out, valid_tokens)

    meta = {
        "type": "bpe_utf8_encoded",
        "vocab_size": vocab_size,
        "dtype": "int32",
        "special_token": args.special_token,
        "tokenizer_pkl": str(tokenizer_pkl.resolve()),
        "train_txt": str(train_txt.resolve()),
        "valid_txt": str(valid_txt.resolve()),
        "train_tokens_shape": list(train_tokens.shape),
        "valid_tokens_shape": list(valid_tokens.shape),
        "train_tokens_path": str(train_out.resolve()),
        "valid_tokens_path": str(valid_out.resolve()),
    }
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents = True, exist_ok = True)
        meta_path.write_text(json.dumps(meta, indent = 2), encoding = "utf-8")
        print(f"wrote meta {meta_path}")

    print(f"vocab_size={vocab_size}")
    print(f"train_tokens={train_tokens.shape[0]} -> {train_out}")
    print(f"valid_tokens={valid_tokens.shape[0]} -> {valid_out}")


if __name__ == "__main__":
    main()
