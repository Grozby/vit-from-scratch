import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        model_dim: int,
        number_patches: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.position_embeddings = nn.Parameter(
            torch.nn.init.normal_(
                tensor=torch.empty(
                    1,
                    number_patches + 1,
                    model_dim,
                ),
                std=0.02,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings
