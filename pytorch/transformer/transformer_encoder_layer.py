from typing import Optional

import torch
import torch.nn as nn

from pytorch.transformer.multi_head_attention import MultiHeadAttention
from pytorch.transformer.transformer_mlp import TransformerMLP


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        model_dim: int,
        mlp_hidden_dim: int,
        number_attention_heads: int,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        mlp_activation_function: Optional[nn.Module] = None,
        order_layer_norm: str = "after",
        *args,
        **kwargs,
    ):
        assert order_layer_norm in ["before", "after"]
        super().__init__(*args, **kwargs)
        self.model_dim = model_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.number_attention_heads = number_attention_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate
        self.order_layer_norm = order_layer_norm

        self.multi_attention_head = MultiHeadAttention(
            model_dim=model_dim,
            number_attention_heads=number_attention_heads,
            dropout_rate=attention_dropout_rate,
        )
        self.attention_layer_norm = nn.LayerNorm(model_dim)
        self.mlp = TransformerMLP(
            in_features=model_dim,
            hidden_features=mlp_hidden_dim,
            dropout_rate=mlp_dropout_rate,
            activation_function=mlp_activation_function,
        )
        self.mlp_layer_norm = nn.LayerNorm(model_dim)

    def forward_layer_norm_after(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        attention = self.multi_attention_head(
            queries=x,
            keys=x,
            values=x,
        )
        x = self.layer_norm_attention(x + attention)
        return self.mlp_layer_norm(x + self.mlp(x))

    def forward_layer_norm_before(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        previous = x
        x = self.attention_layer_norm(x)
        x = self.multi_attention_head(
            queries=x,
            keys=x,
            values=x,
        )
        previous = x = x + previous
        x = self.mlp_layer_norm(x)
        x = self.mlp(x)
        return x + previous

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.order_layer_norm == "after":
            return self.forward_layer_norm_after(x, *args, **kwargs)
        elif self.order_layer_norm == "before":
            return self.forward_layer_norm_before(x, *args, **kwargs)
        else:
            raise NotImplementedError(
                f"`self.order_layer_norm` can be either `after` or `before`, "
                f"not {self.order_layer_norm}")
