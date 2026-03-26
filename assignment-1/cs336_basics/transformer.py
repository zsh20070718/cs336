from typing import Any, Callable, Optional

import math
import torch
import torch.nn as nn

from jaxtyping import Float
from torch import Tensor


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device = None,
        dtype = None,
    ):
        super().__init__()
        from cs336_basics.linear import Linear

        self.w1 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device = device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device = device, dtype = dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.w1(x)
        silu = hidden * torch.sigmoid(hidden)
        return self.w2(silu * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device = None,
        dtype = None,
    ):
        super().__init__()
        from cs336_basics.linear import Linear
        from cs336_basics.rope import RoPE

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.k_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.v_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.output_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.rope = RoPE(self.head_dim, theta, max_seq_len, device = device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        token_positions = torch.arange(sequence_length, device = x.device, dtype = torch.long)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype = torch.bool, device = x.device)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        from cs336_basics.softmax import softmax

        attn_weights = softmax(scores, dim = -1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.output_proj(context)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device = None,
        dtype = None,
    ):
        super().__init__()
        from cs336_basics.rmsnorm import RMSNorm

        self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.attn = CausalSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            max_seq_len = max_seq_len,
            theta = theta,
            device = device,
            dtype = dtype,
        )
        self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)
        self.ffn = SwiGLU(d_model, d_ff, device = device, dtype = dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device = None,
        dtype = None,
    ):
        super().__init__()
        from cs336_basics.embedding import Embedding
        from cs336_basics.linear import Linear
        from cs336_basics.rmsnorm import RMSNorm

        self.token_embeddings = Embedding(vocab_size, d_model, device = device, dtype = dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = d_model,
                    num_heads = num_heads,
                    d_ff = d_ff,
                    max_seq_len = context_length,
                    theta = rope_theta,
                    device = device,
                    dtype = dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device = device, dtype = dtype)
        self.lm_head = Linear(d_model, vocab_size, device = device, dtype = dtype)

    def forward(
        self,
        in_indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    residual = in_features
    transformer_input = in_features

    from cs336_basics.attention import multihead_self_attention_with_rope
    from cs336_basics.positionwise_feedforward import run_swiglu
    from cs336_basics.rmsnorm import RMSNorm

    ln1_weights = weights["ln1.weight"]
    rmsnorm1 = RMSNorm(d_model, device=ln1_weights.device, dtype=ln1_weights.dtype)
    rmsnorm1.load_state_dict({"weight": ln1_weights})
    x_norm1 = rmsnorm1(transformer_input)

    sequence_length = in_features.size(-2)
    token_positions = torch.arange(sequence_length, device=in_features.device, dtype=torch.long)

    attn_out = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x_norm1,
        token_positions=token_positions,
    )
    transformer_input = residual + attn_out

    residual = transformer_input

    ln2_weights = weights["ln2.weight"]
    rmsnorm2 = RMSNorm(d_model, device=ln2_weights.device, dtype=ln2_weights.dtype)
    rmsnorm2.load_state_dict({"weight": ln2_weights})
    x_norm2 = rmsnorm2(transformer_input)

    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=x_norm2,
    )
    transformer_input = residual + ffn_out

    return transformer_input


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: torch.Tensor,
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    from cs336_basics.embedding import Embedding
    from cs336_basics.rmsnorm import RMSNorm

    token_emb_w = weights["token_embeddings.weight"]
    embedding = Embedding(vocab_size, d_model, device=token_emb_w.device, dtype=token_emb_w.dtype)
    embedding.load_state_dict({"weight": token_emb_w})
    x = embedding(in_indices)

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        block_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=block_weights,
            in_features=x,
        )

    ln_final_w = weights["ln_final.weight"]
    ln_final = RMSNorm(d_model, device=ln_final_w.device, dtype=ln_final_w.dtype)
    ln_final.load_state_dict({"weight": ln_final_w})
    x = ln_final(x)

    lm_head_w = weights["lm_head.weight"]
    logits = x @ lm_head_w.T
    return logits