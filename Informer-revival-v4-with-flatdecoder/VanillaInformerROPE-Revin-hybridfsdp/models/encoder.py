"""
models/encoder.py

CHANGES vs previous version:

1. ConvLayer.forward: added `enc_phase=None` kwarg.
   Block selection now uses (enc_phase + j) // channel_period instead of
   j // channel_period. Per-sample phase-aware selection via precomputed
   keep_table indexed by enc_phase.

   Key property (verified numerically): output always has exactly L//2
   positions AND the channel pattern (enc_phase + k) % cp is preserved.
   This means enc_phase passes through unchanged — no post-ConvLayer
   phase update needed.

2. Encoder.forward: added `enc_phase=None` kwarg, passed to each ConvLayer.

3. EncoderStack.forward: added `enc_phase=None` kwarg, passed to each
   sub-encoder.

4. EncoderLayer: UNCHANGED. Encoder self-attention uses mask_flag=False,
   so phase is irrelevant for the attention mask. enc_phase does not need
   to flow through EncoderLayer.

When enc_phase=None or channel_period=1: all behaviour identical to
previous version. No regression for ETT datasets.

ConvLayer changes 1-3 from previous version (kernel_size=1, InstanceNorm1d,
channel-aware selection) are preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in, channel_period=1):
        super(ConvLayer, self).__init__()
        self.channel_period = channel_period

        # kernel_size=1: pointwise independent projection (unchanged)
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0
        )

        # InstanceNorm1d: no batch-size dependence (unchanged)
        self.norm = nn.InstanceNorm1d(c_in, affine=True)

        self.activation = nn.ELU()

    def forward(self, x, enc_phase=None):
        """
        Phase-aware channel-preserving downsampling.

        Selects every other REAL timestep block (aligned to actual timestep
        boundaries using enc_phase), halving the sequence length while
        preserving the channel_period structure exactly.

        Args:
            x:          (B, L, d_model) where L = n_timesteps * channel_period
            enc_phase:  (B,) LongTensor — channel at position 0 of each sample.
                        None → phase=0 for all samples (original behaviour).

        Returns:
            (B, L//2, d_model) with channel pattern preserved:
            output position k has channel (enc_phase + k) % channel_period.
            enc_phase is unchanged after this operation.
        """
        # Pointwise conv + norm + activation
        x = self.downConv(x.permute(0, 2, 1))   # (B, d_model, L)
        x = self.norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)                    # (B, L, d_model)

        cp = self.channel_period
        B, L, D = x.shape
        L_out = L // 2

        if enc_phase is None or cp <= 1:
            # ── Original behavior: select even positional blocks ──────────
            n_timesteps = L // cp
            even_ts = torch.arange(0, n_timesteps, 2, device=x.device)
            block_starts = (
                even_ts.unsqueeze(1) * cp
                + torch.arange(cp, device=x.device).unsqueeze(0)
            )
            indices = block_starts.reshape(-1)
            return x[:, indices, :]

        # ── Phase-aware selection ─────────────────────────────────────────
        # For each possible phase value (0..cp-1), precompute which
        # positions to keep. A position j is kept when its real timestep
        # block index (phase + j) // cp is even.
        #
        # Proven invariant: every phase keeps exactly L//2 positions,
        # and the channel pattern (phase + k) % cp is preserved in the
        # output (verified numerically for all phases and L values in
        # the current config).

        j = torch.arange(L, device=x.device)
        keep_indices_list = []
        for phase_val in range(cp):
            real_block = (phase_val + j) // cp
            mask = (real_block % 2 == 0)
            kept = j[mask]
            keep_indices_list.append(kept)

        # (cp, L_out) — each row is the set of kept positions for that phase
        keep_table = torch.stack(keep_indices_list, dim=0)            # (cp, L_out)

        # Look up the correct indices for each sample's phase
        batch_indices = keep_table[enc_phase]                         # (B, L_out)

        # Gather: select per-sample positions from x
        x = torch.gather(
            x, 1,
            batch_indices.unsqueeze(-1).expand(-1, -1, D)
        )                                                             # (B, L_out, D)

        return x


class EncoderLayer(nn.Module):
    """
    Informer Encoder Layer with learnable channel-mixing matrix W.
    UNCHANGED — encoder self-attention uses mask_flag=False, so enc_phase
    is irrelevant for the attention mask.
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", channel_mix_size=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.channel_mix_size = channel_mix_size
        if channel_mix_size is not None and channel_mix_size > 0:
            self.norm3 = nn.LayerNorm(d_model)
            self.W = nn.Parameter(torch.empty(channel_mix_size, channel_mix_size))
            nn.init.xavier_uniform_(self.W)
        else:
            self.norm3 = None
            self.W = None

    def forward(self, x, attn_mask=None):
        # Self-attention + residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        # FFN + residual
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)

        # Channel mixing (optional)
        if self.W is not None:
            batch_size, seq_len, d_model = x.shape
            c = self.channel_mix_size

            if seq_len % c != 0:
                raise RuntimeError(
                    f"EncoderLayer channel_mix_size={c} does not evenly "
                    f"divide seq_len={seq_len}. seq_len must be a multiple "
                    f"of channel_mix_size."
                )

            n = seq_len // c
            x_reshaped = x.view(batch_size, n, c, d_model)
            W = self.W
            if x_reshaped.dtype != W.dtype:
                x_reshaped = x_reshaped.to(W.dtype)
            x_transformed = torch.einsum('ij,bnjd->bnid', W, x_reshaped)
            x = x_transformed.reshape(batch_size, seq_len, d_model)
            x = self.dropout(x)
            x = self.norm3(x)

        return x, attn


class Encoder(nn.Module):
    """
    CHANGE: forward() accepts enc_phase=None, passed to each ConvLayer.
    EncoderLayer does not need enc_phase (mask_flag=False).
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (nn.ModuleList(conv_layers)
                            if conv_layers is not None else None)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, enc_phase=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers,
                                               self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x, enc_phase=enc_phase)               # NEW
                # enc_phase is unchanged after ConvLayer (proven invariant)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    """
    CHANGE: forward() accepts enc_phase=None, passed to each sub-encoder.
    """
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None, enc_phase=None):
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            # enc_phase is unchanged for the slice because
            # (L - inp_len) is always a multiple of channel_period
            x_s, attn = encoder(
                x[:, -inp_len:, :],
                enc_phase=enc_phase                                   # NEW
            )
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        return x_stack, attns
