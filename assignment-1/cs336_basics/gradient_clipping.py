from __future__ import annotations

from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm_sq = sum((g.detach() ** 2).sum() for g in grads)
    total_norm = total_norm_sq.sqrt()

    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g.mul_(scale)

