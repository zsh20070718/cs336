import torch
from torch import nn
from jaxtyping import Float
from torch import Tensor


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    from cs336_basics.linear import Linear
    w1 = Linear(d_model, d_ff, device = w1_weight.device, dtype = w1_weight.dtype)
    w1.load_state_dict({"weight": w1_weight.T})
    w2 = Linear(d_ff, d_model, device = w2_weight.device, dtype = w2_weight.dtype)
    w2.load_state_dict({"weight": w2_weight.T})
    w3 = Linear(d_model, d_ff, device = w3_weight.device, dtype = w3_weight.dtype)
    w3.load_state_dict({"weight": w3_weight.T})
    return w2(SiLU(w1(in_features)) * w3(in_features))