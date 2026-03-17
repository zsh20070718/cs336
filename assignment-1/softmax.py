import torch

def softmax(x: torch.Tensor):
    x_max = x.max(dim = -1, keepdim = True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim = -1, keepdim = True)