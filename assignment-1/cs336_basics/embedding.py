import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        _std = 1
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std= _std,
            a=-3 * _std,
            b=3 * _std,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]