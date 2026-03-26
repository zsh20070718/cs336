from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

import os
from typing import BinaryIO, IO

import torch
import torch.nn as nn


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(dataset.shape[0])
    high = n - context_length
    starts = np.random.randint(0, high, size=batch_size)

    x = np.empty((batch_size, context_length), dtype=np.int64)
    y = np.empty((batch_size, context_length), dtype=np.int64)
    for b in range(batch_size):
        i = int(starts[b])
        x[b] = dataset[i : i + context_length]
        y[b] = dataset[i + 1 : i + context_length + 1]

    return (
        torch.tensor(x, device=device, dtype=torch.long),
        torch.tensor(y, device=device, dtype=torch.long),
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src, map_location = "cpu", weights_only = False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["iteration"])
