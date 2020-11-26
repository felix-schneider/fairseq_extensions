from typing import Optional

from torch import Tensor

from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer


class MyTransformerEncoderLayer(TransformerEncoderLayer):

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        # Diff: removed the check of attn_mask at the beginning
        # I will be the judge of what attn_mask can and cannot contain.

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class MyTransformerDecoderLayer(TransformerDecoderLayer):
    # Future placeholder
    pass
