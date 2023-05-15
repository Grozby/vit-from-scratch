from typing import Union, Tuple

import torch
import torch.nn as nn

from pytorch.vision_transformer.class_token import ClassToken
from pytorch.vision_transformer.positional_embedding import PositionalEmbedding


def to_tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)


class ToFlattenedPatches(nn.Module):

    def __init__(
        self,
        model_dim: int,
        image_shape: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        channels: int = 3,
        dropout_rate: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        image_shape = to_tuple(image_shape)
        patch_size = to_tuple(patch_size)

        self.model_dim = model_dim
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.channels = channels
        self.dropout_rate = dropout_rate

        image_height, image_width = image_shape
        patch_height, patch_width = patch_size

        self.number_patches_height = image_height // patch_height
        self.number_patches_width = image_width // patch_width
        self.number_patches = (self.number_patches_height *
                               self.number_patches_width)

        self.patchyfier = nn.Conv2d(
            in_channels=channels,
            out_channels=model_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        self.class_token = ClassToken(model_dim=model_dim)
        self.positional_embedding = PositionalEmbedding(
            model_dim=model_dim,
            number_patches=self.number_patches,
        )
        self.embedding_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.patchyfier(x)
        batch_size, model_dim, n_patches_width, n_patches_height = x.shape
        x = torch.reshape(
            x,
            shape=[
                batch_size,
                (n_patches_width * n_patches_height),
                model_dim,
            ],
        )
        x = self.class_token(x)
        x = self.positional_embedding(x)
        x = self.embedding_dropout(x)
        return x
