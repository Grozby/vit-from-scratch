from typing import Optional

import torch
import torch.nn as nn

from pytorch.transformer.transformer_encoder_layer import TransformerEncoderLayer


class Transformer(nn.Module):

    def __init__(
        self,
        model_dim: int,
        feed_forward_hidden_dim: int,
        number_attention_heads: int,
        number_stacks: int,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        number_classes: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_dim = model_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.number_attention_heads = number_attention_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                model_dim=model_dim,
                mlp_hidden_dim=feed_forward_hidden_dim,
                number_attention_heads=number_attention_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
            ) for _ in range(number_stacks)
        ])

        self.decoder = nn.ModuleList([
            TransformerEncoderLayer(
                model_dim=model_dim,
                mlp_hidden_dim=feed_forward_hidden_dim,
                number_attention_heads=number_attention_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
            ) for _ in range(number_stacks)
        ])

        out_features = model_dim if number_classes is None else number_classes
        self.linear = nn.Linear(
            in_features=model_dim,
            out_features=out_features,
        )
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x: torch.Tensor):
        for e in self.encoder:
            x = e(x)
        x_decoder = x
        for d in self.decoder:
            x = d(x=x, x_decoder=x_decoder)

        x = self.linear(x)
        return self.log_softmax(x)
