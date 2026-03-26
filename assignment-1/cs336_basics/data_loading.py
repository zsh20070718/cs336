from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


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