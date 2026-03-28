from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from cs336_basics.generate import infer_model_hparams, load_or_build_tokenizer, top_p_sample
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Interactive text chat with CS336 checkpoint")
    parser.add_argument(
        "--ckpt_path",
        type = str,
        default = "checkpoints/owt_overnight_bpe10k_2026-03-28_09-41.pt",
    )
    parser.add_argument("--tokenizer_cache_path", type = str, default = "data/tinystories_bpe10k_tokenizer.pkl")
    parser.add_argument("--bpe_train_text_path", type = str, default = None)
    parser.add_argument("--special_token", type = str, default = "<|endoftext|>")

    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--seed", type = int, default = 42)

    parser.add_argument("--context_length", type = int, default = 256)
    parser.add_argument("--num_heads", type = int, default = 16)
    parser.add_argument("--d_ff", type = int, default = 1344)
    parser.add_argument("--rope_theta", type = float, default = 10_000.0)

    parser.add_argument("--temperature", type = float, default = 0.9)
    parser.add_argument("--top_p", type = float, default = 0.92)
    parser.add_argument("--max_new_tokens", type = int, default = 160)
    parser.add_argument("--max_turns", type = int, default = 20)
    parser.add_argument("--history_chars", type = int, default = 4000)
    parser.add_argument("--system", type = str, default = None, help = "Optional system preamble (plain text).")
    parser.add_argument(
        "--repetition_penalty",
        type = float,
        default = 1.15,
        help = ">1.0 discourages re-sampling recent tokens (1.0 disables).",
    )
    parser.add_argument(
        "--repetition_window",
        type = int,
        default = 64,
        help = "How many assistant tokens count toward repetition penalty.",
    )
    parser.add_argument(
        "--max_same_token_run",
        type = int,
        default = 8,
        help = "Stop assistant generation if the same token repeats this many times in a row (0 disables).",
    )
    return parser.parse_args()


def apply_repetition_penalty(
    logits: torch.Tensor,
    recent_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty <= 1.0 or not recent_ids:
        return logits
    out = logits.clone()
    for tid in set(recent_ids):
        if out[tid] > 0:
            out[tid] /= penalty
        else:
            out[tid] *= penalty
    return out


def common_prefix_len(
    a: str,
    b: str,
) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def trailing_same_id_run(ids: list[int]) -> int:
    if not ids:
        return 0
    last = ids[-1]
    run = 0
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == last:
            run += 1
        else:
            break
    return run


def sample_next_id(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    scaled = logits / temperature
    return top_p_sample(scaled, top_p = top_p)


def stream_reply_tokens(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: int | None,
    special_token: str,
    repetition_penalty: float,
    repetition_window: int,
    max_same_token_run: int,
) -> tuple[list[int], str]:
    generated = list(prompt_ids)
    new_ids: list[int] = []
    prev_decoded = ""
    model.eval()

    for _ in range(max_new_tokens):
        x = torch.tensor([generated[-context_length:]], dtype = torch.long, device = next(model.parameters()).device)
        with torch.no_grad():
            logits = model(x)[0, -1, :]

        window = new_ids[-repetition_window:] if repetition_window > 0 else []
        logits = apply_repetition_penalty(logits, window, repetition_penalty)
        next_id = sample_next_id(logits, temperature = temperature, top_p = top_p)
        generated.append(next_id)
        new_ids.append(next_id)

        if max_same_token_run > 0 and trailing_same_id_run(new_ids) >= max_same_token_run:
            reply_text_raw = tokenizer.decode(new_ids)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return new_ids, reply_text_raw.strip()

        reply_text_raw = tokenizer.decode(new_ids)
        if special_token in reply_text_raw:
            reply_text = reply_text_raw.split(special_token, maxsplit = 1)[0]
            delta = reply_text[common_prefix_len(prev_decoded, reply_text) :]
            sys.stdout.write(delta)
            sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            return new_ids, reply_text.strip()

        delta = reply_text_raw[common_prefix_len(prev_decoded, reply_text_raw) :]
        sys.stdout.write(delta)
        sys.stdout.flush()
        prev_decoded = reply_text_raw

        if eos_id is not None and next_id == eos_id:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return new_ids, reply_text_raw.strip()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return new_ids, tokenizer.decode(new_ids).strip()


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

    eos_id_list = tokenizer.encode(args.special_token)
    eos_id = eos_id_list[0] if len(eos_id_list) == 1 else None

    history = ""
    if args.system:
        history = f"System: {args.system.strip()}"
        print("System message loaded. Commands: /exit to quit.")
    else:
        print(
            "Chat started (Q/A style prompt; base LM on web text, not instruction-tuned). "
            "Commands: /exit to quit.",
        )

    for turn_idx in range(args.max_turns):
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Bye.")
            break

        if history:
            history += f"\n\nQ: {user_text}\nA:"
        else:
            history = f"Q: {user_text}\nA:"
        if len(history) > args.history_chars:
            history = history[-args.history_chars :]

        prompt_ids = tokenizer.encode(history)

        print("\nAssistant: ", end = "", flush = True)
        reply_ids, reply_text = stream_reply_tokens(
            model = model,
            tokenizer = tokenizer,
            prompt_ids = prompt_ids,
            context_length = int(hparams["context_length"]),
            max_new_tokens = args.max_new_tokens,
            temperature = args.temperature,
            top_p = args.top_p,
            eos_id = eos_id,
            special_token = args.special_token,
            repetition_penalty = args.repetition_penalty,
            repetition_window = args.repetition_window,
            max_same_token_run = args.max_same_token_run,
        )

        history += f" {reply_text}"
        if eos_id is not None and len(reply_ids) > 0 and reply_ids[-1] == eos_id:
            print("[Boundary reached: <|endoftext|>]")
            break

        if turn_idx == args.max_turns - 1:
            print("[Boundary reached: max_turns]")


if __name__ == "__main__":
    main()
