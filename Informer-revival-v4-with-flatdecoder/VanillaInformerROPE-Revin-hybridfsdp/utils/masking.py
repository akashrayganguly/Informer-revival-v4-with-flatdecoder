"""
utils/masking.py

CHANGES vs previous version:

1. BlockTriangularCausalMask: added `phase` parameter (default=None → phase=0).
   Block index for position j is now (phase + j) // channel_period instead of
   j // channel_period. When phase is a (B,) tensor, each sample in the batch
   gets its own mask aligned to its actual timestep boundaries.

   Without this fix, 86% of training windows (all with dec_phase != 0) had
   misaligned block boundaries, allowing causal leakage at every timestep
   boundary in the decoder self-attention.

2. ProbMask: added `phase` parameter with identical logic. Ensures ProbAttention's
   sparse masking also respects per-sample timestep boundaries.

When channel_period=1 or phase=None/0: both classes produce identical output
to the previous version. No regression for ETT datasets or channel_period=1 use.
"""

import torch


class TriangularCausalMask():
    """Original triangular causal mask. Unchanged."""
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class BlockTriangularCausalMask():
    """
    Block-triangular causal mask for (N*channel_period, m+1) restructured data.

    Phase-aware: each sample in the batch can have a different starting channel
    (phase offset), and the mask aligns block boundaries to real timestep
    boundaries for that sample.

    Block index for position j of sample b:
        block_idx[b, j] = (phase[b] + j) // channel_period

    Masked when: block_idx[b, j_key] > block_idx[b, i_query]
    i.e. key position j is in a strictly later timestep block than query position i.

    Shape: (B, 1, L, L), dtype bool, True = masked (blocked).

    When channel_period=1:
        (phase + j) // 1 = phase + j. block_idx[j] > block_idx[i] ↔ j > i.
        Equivalent to TriangularCausalMask regardless of phase.

    When phase=None or phase=0 for all samples:
        Equivalent to previous version (no regression).
    """
    def __init__(self, B, L, channel_period=1, device="cpu", phase=None):
        with torch.no_grad():
            j = torch.arange(L, device=device)                       # (L,)

            if phase is None:
                # ── No phase: uniform mask, same for all B samples ────
                block_idx = j // channel_period                       # (L,)
                mask_2d = (
                    block_idx.unsqueeze(0) > block_idx.unsqueeze(1)
                )                                                     # (L, L)
                self._mask = mask_2d.unsqueeze(0).unsqueeze(0).expand(
                    B, 1, L, L
                ).clone()
            else:
                # ── Per-sample phase: mask differs across batch ───────
                # phase: (B,) LongTensor — channel at position 0
                if not isinstance(phase, torch.Tensor):
                    phase = torch.tensor(phase, device=device, dtype=torch.long)
                phase = phase.to(device=device, dtype=torch.long)
                if phase.dim() == 0:
                    phase = phase.unsqueeze(0).expand(B)

                # block_idx[b, j] = (phase[b] + j) // channel_period
                block_idx = (
                    phase.unsqueeze(1) + j.unsqueeze(0)
                ) // channel_period                                   # (B, L)

                # mask[b, i, j] = True when key block > query block
                query_blocks = block_idx.unsqueeze(2)                 # (B, L, 1)
                key_blocks = block_idx.unsqueeze(1)                   # (B, 1, L)
                mask_3d = (key_blocks > query_blocks)                 # (B, L, L)
                self._mask = mask_3d.unsqueeze(1)                     # (B, 1, L, L)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    """
    Sparse attention mask for ProbAttention.

    Phase-aware: uses (phase + j) // channel_period for block indices.

    When channel_period=1 or phase=None: identical to original .triu(1) behaviour.
    """
    def __init__(self, B, H, L, index, scores, device="cpu",
                 channel_period=1, phase=None):
        L_K = scores.shape[-1]

        if phase is None:
            # ── No phase: uniform block-triangular mask ───────────────
            q_block = torch.arange(L, device=device) // channel_period
            k_block = torch.arange(L_K, device=device) // channel_period
            _mask = (
                k_block.unsqueeze(0) > q_block.unsqueeze(1)
            )                                                         # (L, L_K)
            _mask_ex = _mask[None, None, :].expand(B, H, L, L_K)
        else:
            # ── Per-sample phase ──────────────────────────────────────
            if not isinstance(phase, torch.Tensor):
                phase = torch.tensor(phase, device=device, dtype=torch.long)
            phase = phase.to(device=device, dtype=torch.long)
            if phase.dim() == 0:
                phase = phase.unsqueeze(0).expand(B)

            j_q = torch.arange(L, device=device)
            j_k = torch.arange(L_K, device=device)
            q_block = (
                phase.unsqueeze(1) + j_q.unsqueeze(0)
            ) // channel_period                                       # (B, L)
            k_block = (
                phase.unsqueeze(1) + j_k.unsqueeze(0)
            ) // channel_period                                       # (B, L_K)

            # mask[b, i, j] = True when key block > query block
            _mask = (
                k_block.unsqueeze(1) > q_block.unsqueeze(2)
            )                                                         # (B, L, L_K)
            _mask_ex = _mask[:, None, :, :].expand(B, H, L, L_K)

        # Select rows corresponding to the top-k query indices
        indicator = _mask_ex[
            torch.arange(B, device=device)[:, None, None],
            torch.arange(H, device=device)[None, :, None],
            index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
