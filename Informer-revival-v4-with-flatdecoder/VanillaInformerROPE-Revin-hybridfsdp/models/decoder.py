"""
models/decoder.py

CHANGES vs previous version:

1. DecoderLayer.forward: added `dec_phase=None` kwarg.
   Passed to self_attention as `phase=dec_phase` so the block-triangular
   causal mask aligns to each sample's real timestep boundaries.
   Cross-attention does not receive phase (mask_flag=False, no mask built).

2. Decoder.forward: added `dec_phase=None` kwarg.
   Threaded to each DecoderLayer.

When dec_phase=None: mask falls back to phase=0 (uniform mask). No regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", channel_mix_size=None):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # Channel-mixing extension (unchanged)
        self.channel_mix_size = channel_mix_size
        if channel_mix_size is not None and channel_mix_size > 0:
            self.norm4 = nn.LayerNorm(d_model)
            self.W = nn.Parameter(torch.empty(channel_mix_size, channel_mix_size))
            nn.init.xavier_uniform_(self.W)
        else:
            self.norm4 = None
            self.W = None

    def forward(self, x, cross, x_mask=None, cross_mask=None,
                dec_phase=None):
        """
        CHANGE: added dec_phase=None kwarg.
        Passed to self_attention as phase=dec_phase so the causal mask
        uses per-sample timestep-aligned block boundaries.

        Cross-attention does not receive phase — mask_flag=False means
        no causal mask is constructed.
        """
        # Decoder self-attention (causal, mask_flag=True)
        # Phase-aware BlockTriangularCausalMask constructed inside
        # self_attention when x_mask is None.
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            phase=dec_phase                                           # NEW
        )[0])
        x = self.norm1(x)

        # Cross-attention (mask_flag=False, no causal mask applied)
        # phase not needed here
        x = x + self.dropout(self.cross_attention(
            x, cross, cross, attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm3(x + y)

        # Channel mixing (optional, unchanged)
        if self.W is not None:
            batch_size, seq_len, d_model = x.shape
            c = self.channel_mix_size

            if seq_len % c != 0:
                raise RuntimeError(
                    f"DecoderLayer channel_mix_size={c} does not evenly "
                    f"divide decoder seq_len={seq_len} (label_len + pred_len). "
                    f"The total decoder sequence length must be a multiple "
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
            x = self.norm4(x)

        return x


class Decoder(nn.Module):
    """
    CHANGE: forward() accepts dec_phase=None, threaded to each DecoderLayer.
    """
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None,
                dec_phase=None):
        for layer in self.layers:
            x = layer(x, cross,
                      x_mask=x_mask,
                      cross_mask=cross_mask,
                      dec_phase=dec_phase)                            # NEW
        if self.norm is not None:
            x = self.norm(x)
        return x
