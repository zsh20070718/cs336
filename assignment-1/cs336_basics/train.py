from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train CS336 Transformer LM")

    parser.add_argument("--train_tokens_path", type = str, required = True)
    parser.add_argument("--valid_tokens_path", type = str, required = True)
    parser.add_argument("--token_dtype", type = str, default = "int32")

    parser.add_argument("--vocab_size", type = int, required = True)
    parser.add_argument("--context_length", type = int, default = 256)
    parser.add_argument("--d_model", type = int, default = 512)
    parser.add_argument("--num_layers", type = int, default = 4)
    parser.add_argument("--num_heads", type = int, default = 16)
    parser.add_argument("--d_ff", type = int, default = 1344)
    parser.add_argument("--rope_theta", type = float, default = 10_000.0)

    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--max_steps", type = int, default = 5000)
    parser.add_argument("--lr_max", type = float, default = 3e-4)
    parser.add_argument("--lr_min", type = float, default = 3e-5)
    parser.add_argument("--warmup_iters", type = int, default = 200)
    parser.add_argument("--cosine_cycle_iters", type = int, default = 5000)
    parser.add_argument("--beta1", type = float, default = 0.9)
    parser.add_argument("--beta2", type = float, default = 0.95)
    parser.add_argument("--eps", type = float, default = 1e-8)
    parser.add_argument("--weight_decay", type = float, default = 0.1)
    parser.add_argument("--grad_clip", type = float, default = 1.0)

    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--eval_every", type = int, default = 200)
    parser.add_argument("--eval_batches", type = int, default = 20)
    parser.add_argument("--log_every", type = int, default = 20)
    parser.add_argument("--save_every", type = int, default = 500)
    parser.add_argument("--ckpt_path", type = str, default = "checkpoints/latest.pt")
    parser.add_argument("--resume", action = "store_true")

    return parser.parse_args()


def load_tokens_memmap(
    path: str,
    token_dtype: str,
) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        return np.load(path_obj, mmap_mode = "r")

    dtype = np.dtype(token_dtype)
    return np.memmap(path_obj, dtype = dtype, mode = "r")


@torch.no_grad()
def evaluate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        x, y = get_batch(
            dataset = dataset,
            batch_size = batch_size,
            context_length = context_length,
            device = device,
        )
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device

    train_tokens = load_tokens_memmap(args.train_tokens_path, args.token_dtype)
    valid_tokens = load_tokens_memmap(args.valid_tokens_path, args.token_dtype)

    model = TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta,
        device = device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr = args.lr_max,
        betas = (args.beta1, args.beta2),
        eps = args.eps,
        weight_decay = args.weight_decay,
    )

    start_step = 0
    if args.resume and os.path.exists(args.ckpt_path):
        loaded_step = load_checkpoint(args.ckpt_path, model, optimizer)
        start_step = loaded_step + 1
        print(f"[resume] loaded checkpoint from step={loaded_step}")

    Path(args.ckpt_path).parent.mkdir(parents = True, exist_ok = True)

    model.train()
    run_start_time = time.time()

    for step in range(start_step, args.max_steps):
        step_start = time.time()

        lr = get_lr_cosine_schedule(
            it = step,
            max_learning_rate = args.lr_max,
            min_learning_rate = args.lr_min,
            warmup_iters = args.warmup_iters,
            cosine_cycle_iters = args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(
            dataset = train_tokens,
            batch_size = args.batch_size,
            context_length = args.context_length,
            device = device,
        )
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        step_elapsed = time.time() - step_start
        tokens_per_sec = (args.batch_size * args.context_length) / max(step_elapsed, 1e-9)

        if step % args.log_every == 0:
            elapsed = time.time() - run_start_time
            print(
                f"[train] step={step} lr={lr:.6e} loss={loss.item():.6f} "
                f"tokens_per_sec={tokens_per_sec:.2f} elapsed_sec={elapsed:.2f}"
            )

        if step % args.eval_every == 0:
            val_loss = evaluate_loss(
                model = model,
                dataset = valid_tokens,
                batch_size = args.batch_size,
                context_length = args.context_length,
                device = device,
                eval_batches = args.eval_batches,
            )
            elapsed = time.time() - run_start_time
            print(
                f"[eval] step={step} lr={lr:.6e} train_loss={loss.item():.6f} "
                f"val_loss={val_loss:.6f} elapsed_sec={elapsed:.2f}"
            )

        if step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, args.ckpt_path)
            print(f"[ckpt] saved step={step} path={args.ckpt_path}")

    save_checkpoint(model, optimizer, args.max_steps - 1, args.ckpt_path)
    print(f"[done] final checkpoint saved path={args.ckpt_path}")


if __name__ == "__main__":
    main()
