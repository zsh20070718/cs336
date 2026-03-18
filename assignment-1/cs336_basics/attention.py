"""
Scaled dot-product attention (Vaswani et al., 2017, §3.2.1).

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""
from __future__ import annotations
from jaxtyping import Float
from torch import Tensor

import math
import torch
from .softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Args:
        Q: (..., n_queries, d_k) query tensor
        K: (..., n_keys, d_k) key tensor
        V: (..., n_keys, d_v) value tensor
        mask: (..., n_queries, n_keys) boolean; True = attend, False = mask out.
              If None, no masking.

    Returns:
        (..., n_queries, d_v) attention output.
    """
    d_k = Q.size(-1)
    # scores: (..., n_queries, n_keys). Only transpose last two dims for batching.
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Canonically: True = attend, False = do not attend.
        # Set positions that must not be attended to -inf so softmax gives 0.
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # Project the input features into the query, key, and value spaces
    # Project the input features into the query, key, and value spaces
    # Project the input features into the query, key, and value spaces
    # Project the input features into the query, key, and value spaces
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    d_head = d_k // num_heads
    assert num_heads * d_head == d_k

    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    from einops import rearrange    
    sequence_length = Q.shape[-2]
    Q = rearrange(Q, "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head", num_heads=num_heads)
    K = rearrange(K, "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head", num_heads=num_heads)
    V = rearrange(V, "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head", num_heads=num_heads)

    mask = torch.zeros(sequence_length, sequence_length)
    for i in range(sequence_length):
        for j in range(sequence_length):
            mask[i, j] = i >= j
    t = scaled_dot_product_attention(Q, K, V, mask.bool())
    t = rearrange(t, "... num_heads sequence_length d_head -> ... sequence_length (num_heads d_head)")
    t = t @ o_proj_weight.T

    return t