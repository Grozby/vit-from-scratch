import torch
import torch.nn as nn


class ClassToken(nn.Module):

    def __init__(
        self,
        model_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_token = nn.Parameter(torch.zeros(
            1,
            1,
            model_dim,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = x.shape
        class_token = self.class_token.repeat(batch_size, 1, 1, 1)
        # Add at the start for each image!
        return torch.cat(
            (class_token, x),
            dim=1,
        )
