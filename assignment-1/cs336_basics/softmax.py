import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim = dim, keepdim = True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim = dim, keepdim = True)
