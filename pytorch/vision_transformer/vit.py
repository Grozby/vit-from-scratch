from typing import Tuple, Union

import torch
import torch.nn as nn

from pytorch.transformer.transformer_encoder_layer import TransformerEncoderLayer
from pytorch.vision_transformer.to_flattened_patches import ToFlattenedPatches


class ViT(nn.Module):

    def __init__(
        self,
        model_dim: int,
        feed_forward_hidden_dim: int,
        number_attention_heads: int,
        number_stacks: int,
        number_classes: int,
        image_shape: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        channels: int = 3,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        embedding_dropout_rate: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_dim = model_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.number_attention_heads = number_attention_heads
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.channels = channels
        self.attention_dropout_rate = attention_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate
        self.embedding_dropout_rate = embedding_dropout_rate

        self.patchyfier = ToFlattenedPatches(
            model_dim=model_dim,
            image_shape=image_shape,
            patch_size=patch_size,
            channels=channels,
            dropout_rate=embedding_dropout_rate,
        )
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                model_dim=model_dim,
                mlp_hidden_dim=feed_forward_hidden_dim,
                number_attention_heads=number_attention_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                mlp_activation_function=nn.GELU(),
                order_layer_norm="before",
            ) for _ in range(number_stacks)
        ])
        self.linear = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(
                in_features=model_dim,
                out_features=number_classes,
            ),
        )

    def forward(self, x: torch.Tensor):
        x = self.patchyfier(x)
        for e in self.encoder:
            x = e(x)
        x = x[:, 0]
        return self.linear(x)
