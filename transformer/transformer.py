import torch
import torch.nn as nn

from transformer.transformer_encoder_layer import TransformerEncoderLayer


class Transformer(nn.Module):

    def __init__(
        self,
        model_dim: int,
        feed_forward_hidden_dim: int,
        number_attention_heads: int,
        number_stacks: int,
        dropout_rate: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_dim = model_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.number_attention_heads = number_attention_heads
        self.dropout_rate = dropout_rate

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                model_dim=self.model_dim,
                mlp_hidden_dim=feed_forward_hidden_dim,
                number_attention_heads=number_attention_heads,
            ) for _ in range(number_stacks)
        ])

        self.decoder = nn.ModuleList([
            TransformerEncoderLayer(
                model_dim=self.model_dim,
                mlp_hidden_dim=feed_forward_hidden_dim,
                number_attention_heads=number_attention_heads,
            ) for _ in range(number_stacks)
        ])

        self.linear = nn.Linear(
            in_features=self.model_dim,
            out_features=self.model_dim,
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
