from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    x_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - x_max
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True)) + x_max
    log_probs = inputs - logsumexp

    batch_size = inputs.size(-2)
    nll = -log_probs[torch.arange(batch_size), targets]
    return nll.mean()

