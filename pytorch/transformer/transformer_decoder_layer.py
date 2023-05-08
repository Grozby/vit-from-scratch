import torch
import torch.nn as nn

from pytorch.transformer.multi_head_attention import MultiHeadAttention
from pytorch.transformer.transformer import TransformerEncoderLayer


class TransformerDecoderLayer(TransformerEncoderLayer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_multi_attention_head = MultiHeadAttention(
            model_dim=self.model_dim,
            number_attention_heads=self.number_attention_heads,
            dropout_rate=self.dropout_rate,
        )
        self.masked_attention_layer_norm = nn.LayerNorm(self.model_dim)

    def forward_layer_norm_after(
        self,
        x: torch.Tensor,
        x_decoder: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        masked_attention = self.masked_multi_attention_head(
            queries=x,
            keys=x,
            values=x,
        )
        x = self.masked_attention_layer_norm(x + masked_attention)

        attention = self.multi_attention_head(
            queries=x,
            keys=x_decoder,
            values=x_decoder,
        )
        x = self.attention_layer_norm(x + attention)

        return self.mlp_layer_norm(x + self.mlp(x))


    def forward_layer_norm_before(
        self,
        x: torch.Tensor,
        x_decoder: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        previous = x
        x = self.masked_attention_layer_norm(x)
        x = self.masked_multi_attention_head(
            queries=x,
            keys=x,
            values=x,
        )
        previous = x = previous + x

        x = self.attention_layer_norm(x)
        x = self.multi_attention_head(
            queries=x,
            keys=x_decoder,
            values=x_decoder,
        )
        previous = x = previous + x

        x = self.mlp_layer_norm(x)
        x = self.mlp(x)
        return previous + x
