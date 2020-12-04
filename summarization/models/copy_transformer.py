from typing import Optional, Dict, List, Any, Tuple

import torch
from torch import nn, Tensor
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from ...models.my_transformer import MyTransformerModel, my_base_architecture
from ...models.fast_transformer import FastTransformerEncoder, FastTransformerDecoder


@register_model("copy_transformer")
class CopyTransformer(MyTransformerModel):

    @staticmethod
    def add_args(parser):
        MyTransformerModel.add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='N',
                            help='number of attention heads to be used for '
                                 'pointing')
        parser.add_argument('--alignment-layer', type=int, metavar='I',
                            help='layer number to be used for pointing (0 '
                                 'corresponding to the bottommost layer)')
        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary) - args.source_position_markers
        padding_idx = dictionary.pad()
        unk_idx = dictionary.unk()

        emb = ExtraOOVEmbedding(num_embeddings, embed_dim, padding_idx, unk_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FastTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return CopyTransformerDecoder(args, tgt_dict, embed_tokens)


class CopyTransformerDecoder(FastTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer

        self.copy_prob_layer = nn.Linear(self.output_embed_dim, 1)
        nn.init.zeros_(self.copy_prob_layer.bias)

        self.dict_size = len(dictionary)
        self.num_oov_types = args.source_position_markers
        self.num_embeddings = self.dict_size - self.num_oov_types

    def forward(self,
                prev_output_tokens,
                encoder_out: Optional[EncoderOut] = None,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                features_only: bool = False,
                full_context_alignment: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                src_lengths: Optional[Any] = None,
                return_all_hiddens: bool = False,
                **extra_args):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=self.alignment_layer,
            alignment_heads=self.alignment_heads
        )

        if not features_only:
            copy_probs = torch.sigmoid(self.copy_prob_layer(x))  # B x T x 1
            x = self.output_layer(x, extra["attn"][0], encoder_out.src_tokens, copy_probs)

        return x, extra

    def output_layer(self, features, attn, src_tokens, copy_probs):
        logits = super().output_layer(features)

        bsz, tgt_len, n = logits.size()
        src_len = src_tokens.size(1)
        assert n == self.num_embeddings
        assert src_tokens.size(0) == bsz
        assert list(attn.size()) == [bsz, tgt_len, src_len]

        probs = super().get_normalized_probs((logits, None), log_probs=False, sample=None)
        padding = probs.new_zeros((bsz, tgt_len, self.num_oov_types))
        probs = torch.cat((probs * copy_probs, padding), dim=2)

        attn_probs = probs.new_zeros(bsz, tgt_len, self.dict_size)
        attn_probs.scatter_add_(dim=2,
                                index=src_tokens.unsqueeze(1).expand_as(attn),
                                src=attn * (1 - copy_probs))
        return probs + attn_probs

    def get_normalized_probs(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
                             log_probs: bool, sample: Optional[Dict[str, Tensor]] = None):
        probs = net_output[0]
        if log_probs:
            return probs.clamp(1e-10, 1.0).log()
        else:
            return probs


class ExtraOOVEmbedding(nn.Embedding):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int,
                 unk_idx: int
                 ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.unk_idx = unk_idx
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(self.weight[padding_idx], 0)

    def forward(self, input: Tensor):
        input = torch.where(
            input >= self.num_embeddings, input.new_full(input.size(), self.unk_idx), input
        )
        return super().forward(input)


@register_model_architecture("copy_transformer", "copy_transformer")
def base_copy_architecture(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", -1)
    my_base_architecture(args)
    if args.alignment_layer < 0:
        args.alignment_layer = args.decoder_layers + args.alignment_layer
