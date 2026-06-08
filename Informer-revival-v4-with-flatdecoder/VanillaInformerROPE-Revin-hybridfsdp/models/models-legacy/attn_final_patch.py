"""
attn.py - OPTIMIZED Attention Mechanisms with FlashAttention + Fast ProbSparse

FIXES FROM ORIGINAL:
1. ProbAttention._prob_QK: K_expand REPLACED with torch.gather
   - ELIMINATES the O(B*H*L_Q*L_K*E) intermediate tensor
   - At L=30816: 31.1 TB (K_expand) -> 55.6 GB (gather) = 560x savings
2. torch.randint and torch.arange created on GPU (was CPU — caused sync)
3. FlashAttention path unchanged (already optimal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    """
    FlashAttention-powered Scaled Dot-Product Attention

    Uses PyTorch 2.0+ SDPA which automatically selects the best kernel:
    1. FlashAttention2 (fastest, requires SM80+ GPUs like A100/H100)
    2. Memory-Efficient Attention (xFormers-style)
    3. Math fallback (standard attention)
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout_p = attention_dropout
        self.dropout = nn.Dropout(attention_dropout)
        self.sdpa_available = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        if self.sdpa_available and not self.output_attention:
            return self._flash_attention(queries, keys, values, attn_mask, scale)
        else:
            return self._standard_attention(queries, keys, values, attn_mask, scale)

    def _flash_attention(self, queries, keys, values, attn_mask, scale):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)

        sdpa_mask = None
        if self.mask_flag and attn_mask is not None:
            sdpa_mask = torch.zeros(B, H, L, S, device=queries.device, dtype=queries.dtype)
            sdpa_mask.masked_fill_(attn_mask.mask, float('-inf'))

        use_causal = self.mask_flag and attn_mask is None and L == S
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
                is_causal=use_causal,
                scale=scale
            )

        out = out.transpose(1, 2).contiguous()
        return (out, None)

    def _standard_attention(self, queries, keys, values, attn_mask, scale):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """
    ProbSparse Self-Attention from Informer paper.

    OPTIMIZATIONS over original:
    1. torch.gather replaces K_expand — ~560x memory reduction at L=30816
    2. All index tensors created on GPU — eliminates CPU-GPU sync
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        ProbSparse Q·K computation using torch.gather (memory-efficient).

        Goal: For each query position q, randomly sample `sample_k` keys,
              compute Q[q] · K[sampled_keys] to estimate sparsity, then
              select the top-n_top queries with highest sparsity scores.

        Args:
            Q: [B, H, L_Q, D]  — queries (head-first layout after transpose)
            K: [B, H, L_K, D]  — keys
            sample_k: int      — keys to sample per query (factor * ceil(ln(L_K)))
            n_top: int         — top queries to select   (factor * ceil(ln(L_Q)))

        Returns:
            Q_K: [B, H, n_top, L_K]  — full attention scores for top queries
            M_top: [B, H, n_top]     — indices of the selected top queries

        ================================================================
        EQUIVALENCE PROOF: K_expand (original) vs torch.gather (new)
        ================================================================

        Both methods compute the same target tensor:

            K_sample[b, h, q, s, e] = K[b, h, index_sample[q, s], e]

        where index_sample is [L_Q, sample_k] with random ints in [0, L_K).

        --- ORIGINAL (K_expand) ---

        Step A: Build expanded view
            K:                      [B, H,      L_K, E]
            K.unsqueeze(-3):        [B, H,   1, L_K, E]
            .expand(B,H,L_Q,L_K,E):[B, H, L_Q, L_K, E]   (strided view, stride=0 on dim-2)

            K_expand[b,h,q,k,e] = K[b,h,k,e]  for ALL q
            (Every q-slice is the same copy of K.)

        Step B: Advanced indexing
            arange(L_Q).unsqueeze(1):  [L_Q,   1]          (values: 0,1,...,L_Q-1)
            index_sample:              [L_Q,   sample_k]    (random key indices)

            These broadcast to [L_Q, sample_k] and index dims 2 and 3:

            K_sample[b, h, q, s, e]
                = K_expand[b, h,  arange[q],  index_sample[q,s],  e]
                = K_expand[b, h,     q,       index_sample[q,s],  e]
                = K[b, h, index_sample[q,s], e]                       (by Step A)

            BUT: advanced indexing on expand() materializes the FULL
            B*H*L_Q*L_K*E tensor before extracting the sample_k entries.
            At L=30816, B=16, H=8, E=64: that's 31.1 TB in float32. OOM.

        --- NEW (torch.gather) ---

        Step 1: Flatten the 2D sample index into 1D
            index_sample:  [L_Q, sample_k]
            index_flat = index_sample.reshape(-1)    shape: [L_Q * sample_k]

            Row-major layout means:
                index_flat[q * sample_k + s] = index_sample[q, s]

        Step 2: Shape the index for torch.gather
            index_for_gather = index_flat[None, None, :, None].expand(B, H, L_Q*sample_k, E)

            Shape: [B, H, L_Q * sample_k, E]
            Every (b,h,e)-slice sees the same index values (strided view, no copy).

            index_for_gather[b, h, i, e] = index_flat[i]
                                          = index_sample[i // sample_k, i % sample_k]

        Step 3: torch.gather(K, dim=2, index_for_gather)
            K:                [B, H,  L_K,           E]
            index_for_gather: [B, H,  L_Q*sample_k,  E]

            By the definition of torch.gather along dim=2:
                output[b, h, i, e] = K[b, h,  index_for_gather[b,h,i,e],  e]
                                   = K[b, h,  index_sample[i//sample_k, i%sample_k],  e]

            Shape: [B, H, L_Q * sample_k, E]

        Step 4: Reshape to [B, H, L_Q, sample_k, E]
            K_sample[b, h, q, s, e]
                = output[b, h,  q*sample_k + s,  e]          (by reshape)
                = K[b, h,  index_sample[q, s],  e]           (by Step 3)

        --- CONCLUSION ---

            K_sample[b,h,q,s,e] = K[b, h, index_sample[q,s], e]

            Identical result for both methods.                        QED

            Memory: gather allocates only B*H*(L_Q*sample_k)*E output
            (plus the strided index view). With sample_k ≈ 55 vs L_K = 30816,
            that's 560x less memory than materializing K_expand.
        ================================================================
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # =====================================================================
        # STEP 1: Sample random key indices (on GPU to avoid CPU-GPU sync)
        # =====================================================================
        # index_sample[q, s] = random key position for query q, sample s
        # Shape: [L_Q, sample_k], values in [0, L_K)
        index_sample = torch.randint(
            L_K, (L_Q, sample_k), device=K.device
        )

        # =====================================================================
        # STEP 2: Gather sampled keys via torch.gather (replaces K_expand)
        #
        # Original was:
        #   K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        #   K_sample = K_expand[:, :, arange(L_Q).unsqueeze(1), index_sample, :]
        #
        # Problem: advanced indexing materializes the full L_Q × L_K expansion.
        # Solution: flatten sample indices, gather, reshape.
        # =====================================================================

        # (a) Flatten [L_Q, sample_k] -> [L_Q * sample_k]
        #     index_flat[q * sample_k + s] = index_sample[q, s]
        index_flat = index_sample.reshape(-1)              # [L_Q * sample_k]

        # (b) Expand for gather: broadcast across B, H, E dims
        #     index_for_gather[b, h, i, e] = index_flat[i]  for all b, h, e
        index_for_gather = index_flat[None, None, :, None].expand(
            B, H, L_Q * sample_k, E
        )                                                   # [B, H, L_Q*sample_k, E]

        # (c) Gather from K along dim=2 (the L_K dimension)
        #     output[b,h,i,e] = K[b, h, index_for_gather[b,h,i,e], e]
        #                     = K[b, h, index_sample[i//sample_k, i%sample_k], e]
        K_sample_flat = torch.gather(
            K, dim=2, index=index_for_gather
        )                                                   # [B, H, L_Q*sample_k, E]

        # (d) Reshape to separate the L_Q and sample_k dimensions
        #     K_sample[b,h,q,s,e] = K_sample_flat[b, h, q*sample_k+s, e]
        #                         = K[b, h, index_sample[q,s], e]
        K_sample = K_sample_flat.reshape(
            B, H, L_Q, sample_k, E
        )                                                   # [B, H, L_Q, sample_k, E]

        # =====================================================================
        # STEP 3: Compute sparsity measurement via Q · K_sample^T
        # =====================================================================
        # For each query q, dot-product with its sample_k sampled keys:
        #   Q_K_sample[b,h,q,s] = sum_e Q[b,h,q,e] * K_sample[b,h,q,s,e]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2),                # [B, H, L_Q, 1, E]
            K_sample.transpose(-2, -1)      # [B, H, L_Q, E, sample_k]
        ).squeeze(-2)                       # [B, H, L_Q, sample_k]

        # =====================================================================
        # STEP 4: Select top-n_top queries by sparsity score
        #
        # M[q] = max_s(Q_K_sample[q,s]) - mean_s(Q_K_sample[q,s])
        # High M → query's attention is peaked (sparse) → informative query
        # =====================================================================
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]     # [B, H, n_top]

        # =====================================================================
        # STEP 5: Compute full Q·K^T ONLY for the selected top queries
        # =====================================================================
        Q_reduce = Q[
            torch.arange(B, device=Q.device)[:, None, None],
            torch.arange(H, device=Q.device)[None, :, None],
            M_top, :
        ]                                           # [B, H, n_top, E]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # [B, H, n_top, L_K]

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        Initialize context with either mean (no mask) or cumsum (causal mask).
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)  # causal mask requires self-attention
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        Update context at the top-query positions with proper attention.
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B, device=V.device)[:, None, None],
            torch.arange(H, device=V.device)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V], device=attn.device) / L_V).type_as(attn)
            attns[
                torch.arange(B, device=V.device)[:, None, None],
                torch.arange(H, device=V.device)[None, :, None],
                index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Attention Layer wrapper

    Handles projection of queries, keys, values and wraps the attention mechanism.
    Works with both FullAttention (FlashAttention-powered) and ProbAttention.
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

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
