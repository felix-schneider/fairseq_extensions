from fairseq.models import register_model, register_model_architecture
from ..modules.fast_transformer_layer import FastTransformerEncoderLayer, FastTransformerDecoderLayer
from .my_transformer import MyTransformerModel, MyTransformerEncoder, MyTransformerDecoder, my_base_architecture


@register_model("fast_transformer")
class FastTransformerModel(MyTransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FastTransformerEncoder(args, src_dict, embed_tokens)


class FastTransformerEncoder(MyTransformerEncoder):
    def build_encoder_layer(self, args):
        return FastTransformerEncoderLayer(args)


class FastTransformerDecoder(MyTransformerDecoder):
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return FastTransformerDecoderLayer(args, no_encoder_attn)


@register_model_architecture("fast_transformer", "fast_transformer")
def fast_base_architecture(args):
    my_base_architecture(args)
