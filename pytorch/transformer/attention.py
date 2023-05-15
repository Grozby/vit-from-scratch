import torch
import torch.nn as nn


class AttentionScaledDotProduct(nn.Module):

    def __init__(
        self,
        model_dim: int,
        embedding_dim: int,
        dropout_rate: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim

        self.linear_queries = nn.Linear(
            in_features=model_dim,
            out_features=embedding_dim,
        )
        self.linear_keys = nn.Linear(
            in_features=model_dim,
            out_features=embedding_dim,
        )
        self.linear_values = nn.Linear(
            in_features=model_dim,
            out_features=embedding_dim,
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        q = self.linear_queries(queries)
        k = self.linear_keys(keys)
        v = self.linear_values(values)

        attention = torch.softmax(
            (q @ k.transpose(-2, -1)) / torch.sqrt(q.shape[-1]),
            dim=-1,
        )
        attention = self.dropout(attention)
        return attention @ v
