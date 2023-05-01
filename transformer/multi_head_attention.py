import torch
import torch.nn as nn

from transformer.attention import AttentionScaledDotProduct


class MultiHeadAttention(nn.Module):

    def __init(
        self,
        model_dim: int,
        number_attention_heads: int,
        dropout_rate: float = 0.0,
        *args,
        **kwargs,
    ):
        assert model_dim % number_attention_heads == 0
        super().__init(*args, **kwargs)
        self.model_dim = model_dim
        self.embedding_dim = model_dim // number_attention_heads
        self.dropout_rate = dropout_rate
        self.h = number_attention_heads

        self.attention_heads = nn.ModuleList([
            AttentionScaledDotProduct(
                model_dim=self.model_dim,
                embedding_dim=self.embedding_dim,
                dropout_rate=self.dropout_rate,
            ) for _ in range(number_attention_heads)
        ])
        self.linear = nn.Linear(
            in_features=self.model_dim,
            out_features=self.model_dim,
            bias=False,
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(
            torch.concatenate(
                *[
                    attention(
                        queries=queries,
                        keys=keys,
                        values=values,
                    ) for attention in self.attention_heads
                ],
                dim=-1,
            ))
