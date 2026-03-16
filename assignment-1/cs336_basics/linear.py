import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, device=device, dtype=dtype)
        )
        _std = 2.0 / (in_features + out_features)
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=_std,
            a=-3 * _std,
            b=3 * _std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight