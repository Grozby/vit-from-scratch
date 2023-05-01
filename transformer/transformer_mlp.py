from typing import Optional

import torch
import torch.nn as nn


class TransformerMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout_rate: float = 0.0,
        activation_function: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dropout_rate = dropout_rate

        self.first_linear = nn.Linear(
            in_features=in_features,
            out_features=hidden_features,
        )
        self.second_linear = nn.Linear(
            in_features=hidden_features,
            out_features=in_features,
        )
        if activation_function is None:
            activation_function = nn.ReLU()
        self.activation = activation_function

    def forward(self, inputs: torch.Tensor, *args, is_training: bool = True):
        x = self.first_linear(inputs)
        x = self.activation(x)
        x = torch.dropout(x, p=self.dropout_rate, train=is_training)
        x = self.second_linear(x)
        x = torch.dropout(x, p=self.dropout_rate, train=is_training)
        return x
