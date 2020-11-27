from .fast_compatible_multihead_attention import FastCompatibleMultiheadAttention
from .my_transformer_layer import MyTransformerEncoderLayer, MyTransformerDecoderLayer


class FastTransformerEncoderLayer(MyTransformerEncoderLayer):
    def build_self_attention(self, embed_dim, args):
        if self.quant_noise > 0:
            raise NotImplementedError("Quantization is not supported")
        return FastCompatibleMultiheadAttention(
            embed_dim, args.encoder_attention_heads, args.attention_dropout,
            bias=True, self_attention=True
        )


class FastTransformerDecoderLayer(MyTransformerDecoderLayer):

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        if self.quant_noise > 0:
            raise NotImplementedError("Quantization is not supported")
        if add_bias_kv or add_zero_attn:
            raise NotImplementedError
        return FastCompatibleMultiheadAttention(
            embed_dim, args.decoder_attention_heads, dropout=self.dropout_module.p,
            bias=True, self_attention=True
        )

    def build_encoder_attention(self, embed_dim, args):
        if self.quant_noise > 0:
            raise NotImplementedError("Quantization is not supported")
        return FastCompatibleMultiheadAttention(
            embed_dim, args.decoder_attention_heads, getattr(args, "encoder_embed_dim", None),
            self.dropout_module.p, bias=True, encoder_decoder_attention=True
        )
