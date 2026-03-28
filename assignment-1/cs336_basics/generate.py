from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import run_train_bpe
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Generate text from a CS336 checkpoint")
    parser.add_argument(
        "--ckpt_path",
        type = str,
        default = "checkpoints/owt_overnight_bpe10k_2026-03-28_09-41.pt",
    )
    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--prompt", type = str, default = "Once upon a time")
    parser.add_argument("--max_new_tokens", type = int, default = 256)
    parser.add_argument("--temperature", type = float, default = 0.8)
    parser.add_argument("--top_p", type = float, default = 0.95)
    parser.add_argument("--seed", type = int, default = 42)

    parser.add_argument("--context_length", type = int, default = 256)
    parser.add_argument("--num_heads", type = int, default = 16)
    parser.add_argument("--d_ff", type = int, default = 1344)
    parser.add_argument("--rope_theta", type = float, default = 10_000.0)

    parser.add_argument("--special_token", type = str, default = "<|endoftext|>")
    parser.add_argument(
        "--bpe_train_text_path",
        type = str,
        default = None,
        help = "Corpus path for BPE training when tokenizer cache is missing.",
    )
    parser.add_argument("--tokenizer_cache_path", type = str, default = "data/tinystories_bpe10k_tokenizer.pkl")
    parser.add_argument("--output_text_path", type = str, default = "checkpoints/owt_generated.txt")
    parser.add_argument("--output_ids_path", type = str, default = "checkpoints/owt_generated_ids.txt")
    return parser.parse_args()


def infer_model_hparams(
    model_state: dict[str, torch.Tensor],
    num_heads: int,
    d_ff: int,
    context_length: int,
    rope_theta: float,
) -> dict[str, int | float]:
    vocab_size, d_model = model_state["token_embeddings.weight"].shape
    num_layers = len(
        {
            key.split(".")[1]
            for key in model_state
            if key.startswith("layers.")
        }
    )
    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "d_ff": int(d_ff),
        "context_length": int(context_length),
        "rope_theta": float(rope_theta),
    }


def load_or_build_tokenizer(
    tokenizer_cache_path: Path,
    bpe_train_text_path: Path | None,
    vocab_size: int,
    special_token: str,
) -> Tokenizer:
    if tokenizer_cache_path.exists():
        with open(tokenizer_cache_path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, tuple) and len(payload) == 2:
            vocab, merges = payload
            cached_special_tokens = [special_token]
        elif isinstance(payload, dict):
            vocab = payload["vocab"]
            merges = payload["merges"]
            cached_special_tokens = payload.get("special_tokens", [special_token])
        else:
            raise ValueError(f"Unsupported tokenizer cache format at {tokenizer_cache_path}")
        return Tokenizer(vocab = vocab, merges = merges, special_tokens = cached_special_tokens)

    if bpe_train_text_path is None:
        raise ValueError(
            "Tokenizer cache not found and --bpe_train_text_path was not set. "
            "Pass --bpe_train_text_path to train BPE, or point --tokenizer_cache_path to a .pkl.",
        )

    vocab, merges = run_train_bpe(
        input_path = bpe_train_text_path,
        vocab_size = vocab_size,
        special_tokens = [special_token],
    )
    tokenizer_cache_path.parent.mkdir(parents = True, exist_ok = True)
    with open(tokenizer_cache_path, "wb") as f:
        pickle.dump(
            {"vocab": vocab, "merges": merges, "special_tokens": [special_token]},
            f,
        )
    return Tokenizer(vocab = vocab, merges = merges, special_tokens = [special_token])


def top_p_sample(
    logits: torch.Tensor,
    top_p: float,
) -> int:
    probs = torch.softmax(logits, dim = -1)
    sorted_probs, sorted_indices = torch.sort(probs, descending = True)
    cumulative = torch.cumsum(sorted_probs, dim = -1)
    keep_mask = cumulative <= top_p
    keep_mask[0] = True

    filtered_probs = sorted_probs * keep_mask
    filtered_probs = filtered_probs / filtered_probs.sum()
    sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples = 1)
    sampled_token = sorted_indices[sampled_sorted_idx]
    return int(sampled_token.item())


def generate_ids(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    eos_id: int | None,
) -> list[int]:
    generated = list(prompt_ids)
    model.eval()
    for _ in range(max_new_tokens):
        x = torch.tensor([generated[-context_length:]], dtype = torch.long, device = next(model.parameters()).device)
        with torch.no_grad():
            logits = model(x)[0, -1, :]

        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            scaled = logits / temperature
            next_id = top_p_sample(scaled, top_p = top_p)

        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return generated


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt = torch.load(args.ckpt_path, map_location = "cpu", weights_only = False)
    model_state = ckpt["model"]

    hparams = infer_model_hparams(
        model_state = model_state,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        context_length = args.context_length,
        rope_theta = args.rope_theta,
    )
    model = TransformerLM(
        vocab_size = int(hparams["vocab_size"]),
        context_length = int(hparams["context_length"]),
        d_model = int(hparams["d_model"]),
        num_layers = int(hparams["num_layers"]),
        num_heads = int(hparams["num_heads"]),
        d_ff = int(hparams["d_ff"]),
        rope_theta = float(hparams["rope_theta"]),
        device = device,
    )
    model.load_state_dict(model_state)
    model.to(device)

    tokenizer = load_or_build_tokenizer(
        tokenizer_cache_path = Path(args.tokenizer_cache_path),
        bpe_train_text_path = Path(args.bpe_train_text_path) if args.bpe_train_text_path else None,
        vocab_size = int(hparams["vocab_size"]),
        special_token = args.special_token,
    )

    prompt_ids = tokenizer.encode(args.prompt)
    special_id = tokenizer.encode(args.special_token)
    eos_id = special_id[0] if len(special_id) == 1 else None

    generated_ids = generate_ids(
        model = model,
        prompt_ids = prompt_ids,
        max_new_tokens = args.max_new_tokens,
        context_length = int(hparams["context_length"]),
        temperature = args.temperature,
        top_p = args.top_p,
        eos_id = eos_id,
    )
    generated_text = tokenizer.decode(generated_ids)

    output_text_path = Path(args.output_text_path)
    output_ids_path = Path(args.output_ids_path)
    output_text_path.parent.mkdir(parents = True, exist_ok = True)
    output_ids_path.parent.mkdir(parents = True, exist_ok = True)

    output_text_path.write_text(generated_text, encoding = "utf-8")
    output_ids_path.write_text(" ".join(str(i) for i in generated_ids), encoding = "utf-8")

    print(f"[hparams] {hparams}")
    print(f"[prompt_tokens] {len(prompt_ids)}")
    print(f"[generated_total_tokens] {len(generated_ids)}")
    print(f"[saved_text] {output_text_path}")
    print(f"[saved_ids] {output_ids_path}")
    print("----- GENERATED TEXT BEGIN -----")
    print(generated_text)
    print("----- GENERATED TEXT END -----")


if __name__ == "__main__":
    main()
