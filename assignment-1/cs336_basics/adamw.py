from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]

                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha = 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                denom = v.sqrt().add(eps)
                p.data.addcdiv_(m, denom, value = -alpha_t)

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss

