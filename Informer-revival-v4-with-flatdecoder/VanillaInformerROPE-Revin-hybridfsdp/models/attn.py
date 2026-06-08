"""
models/attn.py

CHANGES vs previous version:

1. FullAttention.forward: added `phase=None` kwarg.
   _flash_attention and _standard_attention pass phase to
   BlockTriangularCausalMask so the mask aligns to each sample's
   real timestep boundaries.

2. ProbAttention.forward: added `phase=None` kwarg, threaded to
   _update_context which passes it to ProbMask.

3. AttentionLayer.forward: added `phase=None` kwarg, passed through
   to self.inner_attention.

When phase=None (encoder self-attn, cross-attn), mask construction
falls back to phase=0 (uniform mask). No regression.

When mask_flag=False (encoder self-attn, cross-attn), phase is
irrelevant — no mask is built regardless.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

from utils.masking import TriangularCausalMask, ProbMask, BlockTriangularCausalMask


class FullAttention(nn.Module):
    """
    FlashAttention-powered Scaled Dot-Product Attention.

    Phase-aware: when mask_flag=True and attn_mask is None, constructs
    BlockTriangularCausalMask with per-sample phase offset.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False,
                 channel_period=1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout_p = attention_dropout
        self.dropout = nn.Dropout(attention_dropout)
        self.sdpa_available = hasattr(F, 'scaled_dot_product_attention')
        self.channel_period = channel_period

    def forward(self, queries, keys, values, attn_mask, phase=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        if self.sdpa_available and not self.output_attention:
            return self._flash_attention(
                queries, keys, values, attn_mask, scale, phase)
        else:
            return self._standard_attention(
                queries, keys, values, attn_mask, scale, phase)

    def _flash_attention(self, queries, keys, values, attn_mask, scale,
                         phase=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)

        sdpa_mask = None
        if self.mask_flag:
            if attn_mask is not None:
                # Explicit mask from caller (e.g. cross-attention)
                sdpa_mask = torch.zeros(
                    B, H, L, S, device=queries.device, dtype=queries.dtype)
                sdpa_mask.masked_fill_(attn_mask.mask, float('-inf'))
            else:
                # Phase-aware block-triangular causal mask
                block_mask = BlockTriangularCausalMask(
                    B, L, self.channel_period,
                    device=queries.device,
                    phase=phase                                       # NEW
                )
                sdpa_mask = torch.zeros(
                    B, H, L, S, device=queries.device, dtype=queries.dtype)
                sdpa_mask.masked_fill_(block_mask.mask, float('-inf'))

        dropout_p = self.dropout_p if self.training else 0.0

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=scale
            )

        out = out.transpose(1, 2).contiguous()
        return (out, None)

    def _standard_attention(self, queries, keys, values, attn_mask, scale,
                            phase=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                # Phase-aware block-triangular causal mask
                attn_mask = BlockTriangularCausalMask(
                    B, L, self.channel_period,
                    device=queries.device,
                    phase=phase                                       # NEW
                )
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """
    ProbSparse Self-Attention.

    Phase-aware: passes phase to ProbMask in _update_context.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False,
                 channel_period=1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.channel_period = channel_period

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        index_flat = index_sample.reshape(-1)
        index_for_gather = index_flat[None, None, :, None].expand(
            B, H, L_Q * sample_k, E)
        K_sample_flat = torch.gather(K, dim=2, index=index_for_gather)
        K_sample = K_sample_flat.reshape(B, H, L_Q, sample_k, E)

        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2),
            K_sample.transpose(-2, -1)
        ).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[
            torch.arange(B, device=Q.device)[:, None, None],
            torch.arange(H, device=Q.device)[None, :, None],
            M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(
                B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask,
                        phase=None):
        """
        CHANGE: added phase kwarg, passed to ProbMask.
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(
                B, H, L_Q, index, scores,
                device=V.device,
                channel_period=self.channel_period,
                phase=phase                                           # NEW
            )
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B, device=V.device)[:, None, None],
            torch.arange(H, device=V.device)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones(
                [B, H, L_V, L_V], device=attn.device) / L_V).type_as(attn)
            attns[
                torch.arange(B, device=V.device)[:, None, None],
                torch.arange(H, device=V.device)[None, :, None],
                index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, phase=None):
        """
        CHANGE: added phase kwarg, passed to _update_context.
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask,
            phase=phase                                               # NEW
        )

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Attention Layer wrapper.

    CHANGE: forward() accepts `phase=None` kwarg, passed to inner_attention.
    """
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, phase=None):
        """
        CHANGE: added phase=None kwarg, passed to self.inner_attention.

        For decoder self-attention: phase=dec_phase (per-sample).
        For encoder self-attention: phase=None (mask_flag=False, irrelevant).
        For cross-attention: phase=None (mask_flag=False, irrelevant).
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask,
            phase=phase                                               # NEW
        )

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
