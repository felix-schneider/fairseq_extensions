import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from apex.contrib.multihead_attn.fast_encdec_multihead_attn_func import fast_encdec_attn_func
from apex.contrib.multihead_attn.fast_self_multihead_attn_func import fast_self_attn_func
from torch import nn
from torch import Tensor

from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import MultiheadAttention


@with_incremental_state
class FastCompatibleMultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            dropout=0.0,
            bias=True,
            self_attention=False,
            encoder_decoder_attention=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim

        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.bias = bias

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention
        assert kdim is None or self.encoder_decoder_attention

        if self.self_attention:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * self.embed_dim, self.embed_dim))
        else:
            self.in_proj_weight_q = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim))
            self.in_proj_weight_kv = nn.Parameter(torch.empty(2 * self.embed_dim, self.kdim))

        self.out_proj_weight = nn.Parameter(torch.empty(self.embed_dim, self.embed_dim))

        if self.self_attention:
            if self.bias:
                self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim))
            else:
                self.register_parameter("in_proj_bias", None)
                self.in_proj_bias = None
        else:
            if self.bias:
                self.in_proj_bias_q = nn.Parameter(torch.empty(self.embed_dim))
                self.in_proj_bias_kv = nn.Parameter(torch.empty(2 * self.embed_dim))
            else:
                self.register_parameter('in_proj_bias_q', None)
                self.register_parameter('in_proj_bias_kv', None)
                self.in_proj_bias_q = None
                self.in_proj_bias_kv = None
        if self.bias:
            self.out_proj_bias = nn.Parameter(torch.empty(self.embed_dim))
        else:
            self.register_parameter('out_proj_bias', None)
            self.out_proj_bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Scaling all input weights by 1 / sqrt(2)
        if self.self_attention:
            # in_proj_weight has shape [3 * hidden, hidden] but it should be
            # initialized like a [hidden, hidden] matrix.
            nn.init.uniform_(self.in_proj_weight,
                             a=-math.sqrt(3 / self.embed_dim / 2),
                             b=math.sqrt(3 / self.embed_dim / 2))
            if self.bias:
                nn.init.constant_(self.in_proj_bias, 0.)
        else:
            nn.init.xavier_uniform_(self.in_proj_weight_q)
            # in_proj_weight_kv has shape [2 * hidden, kdim] but it should be
            # initialized like a [hidden, kdim] matrix.
            nn.init.uniform_(self.in_proj_weight_kv,
                             a=-math.sqrt(3 / (self.embed_dim + self.kdim)),
                             b=math.sqrt(3 / (self.embed_dim + self.kdim)))
            if self.bias:
                nn.init.constant_(self.in_proj_bias_q, 0.)
                nn.init.constant_(self.in_proj_bias_kv, 0.)

        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.bias:
            nn.init.constant_(self.out_proj_bias, 0.)

    def forward(
            self,
            query: torch.Tensor,
            key,
            value,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            incremental_state=None,
            static_kv=False,
            before_softmax=False,
            need_head_weights=False
    ):
        if need_head_weights:
            need_weights = True

        if (
                query.is_cuda
                and query.dtype == torch.float16
                and not (self.encoder_decoder_attention and (self.bias or attn_mask is not None))
                and self.kdim == self.embed_dim
                and incremental_state is None
                and not (key_padding_mask is not None and attn_mask is not None)
                and not need_weights
                and not before_softmax
                and not (attn_mask is not None and not self.bias)
        ):
            # print("Using CUDA implementation")
            if self.self_attention:
                outputs = fast_self_attn_func(
                    attn_mask is not None, self.training, self.num_heads, query,
                    self.in_proj_weight, self.out_proj_weight, self.in_proj_bias, self.out_proj_bias,
                    key_padding_mask.to(torch.uint8) if key_padding_mask is not None else attn_mask,
                    attn_mask is not None, self.dropout
                )
            else:
                outputs = fast_encdec_attn_func(
                    attn_mask is not None, self.training, self.num_heads, query, key,
                    self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight,
                    key_padding_mask.to(torch.uint8) if key_padding_mask is not None else attn_mask,
                    self.dropout
                )
            return outputs, None

        # print("Using python implementation")
        tgt_len, bsz, embed_dim = query.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    # assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if saved_state is not None and "prev_key" in saved_state and static_kv:
            key = None

        if self.self_attention:
            if self.in_proj_bias is not None:
                input_lin_results = torch.addmm(self.in_proj_bias,
                                                query.view(tgt_len * bsz, embed_dim),
                                                self.in_proj_weight.transpose(0, 1),
                                                beta=1., alpha=1.)
            else:
                input_lin_results = torch.mm(query.view(tgt_len * bsz, embed_dim),
                                             self.in_proj_weight.transpose(0, 1))
            input_lin_results = input_lin_results.view(tgt_len, bsz * self.num_heads, 3, self.head_dim)
            queries = input_lin_results[:, :, 0, :]
            keys = input_lin_results[:, :, 1, :]
            values = input_lin_results[:, :, 2, :]
        else:
            if self.in_proj_bias_q is not None:
                input_lin_q_results = torch.addmm(self.in_proj_bias_q,
                                                  query.view(tgt_len * bsz, embed_dim),
                                                  self.in_proj_weight_q.transpose(0, 1),
                                                  beta=1., alpha=1.)
            else:
                input_lin_q_results = torch.mm(query.view(tgt_len * bsz, embed_dim),
                                               self.in_proj_weight_q.transpose(0, 1))
            input_lin_q_results = input_lin_q_results.view(tgt_len, bsz, self.in_proj_weight_q.size(0))
            queries = input_lin_q_results.view(tgt_len, bsz * self.num_heads, self.head_dim)

            if key is not None:
                src_len, bsz, kv_dim = key.size()
                assert kv_dim == self.kdim
                if self.in_proj_bias_kv is not None:
                    input_lin_kv_results = torch.addmm(self.in_proj_bias_kv,
                                                       key.view(src_len * bsz, kv_dim),
                                                       self.in_proj_weight_kv.transpose(0, 1),
                                                       beta=1., alpha=1.)
                else:
                    input_lin_kv_results = torch.mm(key.view(src_len * bsz, kv_dim),
                                                    self.in_proj_weight_kv.transpose(0, 1))
                input_lin_kv_results = input_lin_kv_results.view(src_len, bsz, self.in_proj_weight_kv.size(0))
                input_lin_kv_results = input_lin_kv_results.view(src_len, bsz * self.num_heads, 2, self.head_dim)
                keys = input_lin_kv_results[:, :, 0, :]
                values = input_lin_kv_results[:, :, 1, :]
            else:
                keys = None
                values = None

        if saved_state is not None:
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(-1, bsz * self.num_heads, self.head_dim)
                if static_kv:
                    keys = prev_key
                else:
                    assert keys is not None
                    keys = torch.cat([prev_key, keys], dim=0)

            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(-1, bsz * self.num_heads, self.head_dim)
                if static_kv:
                    values = prev_value
                else:
                    assert values is not None
                    values = torch.cat([prev_value, values], dim=1)

            prev_key_padding_mask = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert keys is not None and values is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=keys.size(0),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = keys.view(-1, bsz, self.num_heads, self.head_dim)
            saved_state["prev_value"] = values.view(-1, bsz, self.num_heads, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert keys is not None and values is not None
        src_len = keys.size(0)

        # [bsz*num_heads, tgt_len, src_Len]
        matmul1_results = torch.bmm(queries.transpose(0, 1),
                                    keys.transpose(0, 1).transpose(1, 2)) * self.scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                matmul1_results += attn_mask
            else:
                matmul1_results = matmul1_results.view(bsz, self.num_heads, tgt_len, src_len)
                matmul1_results += attn_mask.unsqueeze(1)
                matmul1_results = matmul1_results.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            matmul1_results = matmul1_results.view(bsz, self.num_heads, tgt_len, src_len)
            mask = key_padding_mask.to(torch.bool)
            matmul1_results = matmul1_results.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            matmul1_results = matmul1_results.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return matmul1_results, values

        softmax_results_float = F.softmax(matmul1_results, dim=-1, dtype=torch.float32)
        softmax_results = softmax_results_float.to(matmul1_results.dtype)

        dropout_results = F.dropout(softmax_results, p=self.dropout, training=self.training)

        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1)) \
            .transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.out_proj_bias is not None:
            outputs = torch.addmm(self.out_proj_bias,
                                  matmul2_results.view(tgt_len * bsz, embed_dim),
                                  self.out_proj_weight.transpose(0, 1),
                                  beta=1., alpha=1.)
        else:
            outputs = torch.mm(matmul2_results.view(tgt_len * bsz, embed_dim),
                               self.out_proj_weight.transpose(0, 1))

        outputs = outputs.view(tgt_len, bsz, embed_dim)
        attn_weights = None
        if need_weights:
            # [bsz * num_heads, tgt_len, src_len]
            attn_weights = softmax_results_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(0, 1)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return outputs, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                            0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
