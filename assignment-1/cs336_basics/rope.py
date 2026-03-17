import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device = None
    ):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        half = d_k // 2
        positions = torch.arange(max_seq_len, device = device)
        pair_idx = torch.arange(half, device = device, dtype = torch.float32)
        inv_freq = torch.pow(theta, -2 * pair_idx / d_k)
        angles = positions[:, None].float() * inv_freq
        cos_buf = angles.cos()
        sin_buf = angles.sin()
        self.register_buffer("_cos", cos_buf, persistent = False)
        self.register_buffer("_sin", sin_buf, persistent = False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        cos_pos = self._cos[token_positions]
        sin_pos = self._sin[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = cos_pos * x_even - sin_pos * x_odd
        out_odd = sin_pos * x_even + cos_pos * x_odd
        out = torch.stack([out_even, out_odd], dim = -1).flatten(-2)
        return out
