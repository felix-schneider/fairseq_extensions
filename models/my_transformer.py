"""
Some convenience additions to transformers. Basis for all my other models
"""

from typing import Optional, Dict, List, Any

import torch
from torch import Tensor

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, TransformerModel, base_architecture
from ..modules.my_transformer_layer import MyTransformerDecoderLayer, MyTransformerEncoderLayer


@register_model("my_transformer")
class MyTransformerModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MyTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return MyTransformerDecoder(args, tgt_dict, embed_tokens)


class MyTransformerEncoder(TransformerEncoder):
    def build_encoder_layer(self, args):
        return MyTransformerEncoderLayer(args)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False,
                token_embeddings: Optional[torch.Tensor] = None):
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # Diff: Padding mask is optional if there is not padding
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = []

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # Diff: Why would you include the keys for src_tokens and src_lengths in the encoder and not use them?!
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [src_tokens],
            "src_lengths": [src_lengths],
        }


class MyTransformerDecoder(TransformerDecoder):
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return MyTransformerDecoderLayer(args, no_encoder_attn)

    def forward(self, prev_output_tokens, encoder_out: Optional[Dict[str, List[Tensor]]] = None,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None, features_only: bool = False,
                full_context_alignment: bool = False, alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None, src_lengths: Optional[Any] = None,
                return_all_hiddens: bool = False, **extra_args):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            **extra_args  # diff: pass extra args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out: Optional[Dict[str, List[Tensor]]],
                         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                         full_context_alignment: bool = False, alignment_layer: Optional[int] = None,
                         alignment_heads: Optional[int] = None,
                         **unused):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )


@register_model_architecture("my_transformer", "my_transformer")
def my_base_architecture(args):
    base_architecture(args)
