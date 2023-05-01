from typing import Optional

import torch
import torch.nn as nn

from transformer.multi_head_attention import MultiHeadAttention
from transformer.transformer_mlp import TransformerMLP


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        model_dim: int,
        mlp_hidden_dim: int,
        number_attention_heads: int,
        dropout_rate: float = 0.0,
        mlp_activation_function: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_dim = model_dim
        self.feed_forward_hidden_dim = mlp_hidden_dim
        self.number_attention_heads = number_attention_heads
        self.dropout_rate = dropout_rate

        self.multi_attention_head = MultiHeadAttention(
            model_dim=self.model_dim,
            number_attention_heads=self.number_attention_heads,
            dropout_rate=self.dropout_rate,
        )
        self.attention_layer_norm = nn.LayerNorm(self.model_dim)
        self.mlp = TransformerMLP(
            in_features=self.model_dim,
            hidden_features=self.feed_forward_hidden_dim,
            dropout_rate=self.dropout_rate,
            activation_function=mlp_activation_function,
        )
        self.mlp_layer_norm = nn.LayerNorm(self.model_dim)

    def forward(self, x: torch.Tensor):
        attention = self.multi_attention_head(
            queries=x,
            keys=x,
            values=x,
        )
        x = self.layer_norm_attention(x + attention)
        return self.mlp_layer_norm(x + self.mlp(x))