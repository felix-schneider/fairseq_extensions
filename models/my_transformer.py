"""
Some convenience additions to transformers. Basis for all my other models
"""

from typing import Optional, Dict, List, Any

import torch
from torch import Tensor

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
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
        x = x.transpose(0, 1).contiguous()

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

        # Diff: Why would you include the fields for src_tokens and src_lengths in the encoder and not use them?!
        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )


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

    def extract_features(self,
                         prev_output_tokens,
                         encoder_out: Optional[EncoderOut],
                         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                         full_context_alignment: bool = False,
                         alignment_layer: Optional[int] = None,
                         alignment_heads: Optional[int] = None,
                         **unused):
        # diff: if alignment_layer is None, we want no alignment

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1).contiguous()

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn  # diff: remove type conversions

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("my_transformer", "my_transformer")
def my_base_architecture(args):
    base_architecture(args)
