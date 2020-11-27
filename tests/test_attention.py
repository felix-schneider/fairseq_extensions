import numpy as np
import torch

from ..modules.fast_compatible_multihead_attention import FastCompatibleMultiheadAttention


def compare(a, b, **kwargs):
    np.testing.assert_allclose(a.detach().cpu().numpy(), b.detach().cpu().numpy(), **kwargs)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype) \
        .expand(*length.size(), max_len)\
        .lt(length.unsqueeze(-1))
    if dtype is not None:
        mask = mask.to(dtype=dtype)
    return mask


seq_len = 10
bsz = 5
embed_dim = 512
num_heads = 8

print("Test self-attention")
attn = FastCompatibleMultiheadAttention(
    embed_dim, num_heads, dropout=0.0, bias=True, self_attention=True
)

attn.half().cuda()

inputs = torch.rand((seq_len, bsz, embed_dim)).half().cuda()
input_lengths = torch.randint(2, 10, (bsz,)).cuda()
padding_mask = (~length_to_mask(input_lengths, seq_len)).cuda()

outputs_cuda, _ = attn(inputs, inputs, inputs, key_padding_mask=padding_mask)
outputs_python, _ = attn(inputs, inputs, inputs, key_padding_mask=padding_mask, need_weights=True)

compare(outputs_cuda, outputs_python, atol=1e-3)
