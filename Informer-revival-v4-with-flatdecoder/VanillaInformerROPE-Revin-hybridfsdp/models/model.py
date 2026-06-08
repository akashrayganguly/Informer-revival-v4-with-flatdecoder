"""
models/model.py — PatchTST-style SHARED per-channel flatten head.

CHANGES vs decoder version:

1. Decoder, cross-attention, and dec_embedding REMOVED.
   decoder.py is no longer imported.

2. Output head: ONE shared Linear(T * d_model, P * c_out) applied
   independently to each channel's encoder positions and interleaved.
       T = L_enc / channel_period   (timesteps-per-channel in enc out)
       P = pred_len / channel_period (timesteps-per-channel to predict)

   Step-by-step for input enc_out (B, L_enc, d_model) with enc_phase (B,):

     (a) Reshape (B, L_enc, D) -> (B, T, cp, D). At slot k of axis-2 the
         channel is (phase + k) % cp, because position j = t*cp + k has
         channel (phase + j) % cp = (phase + k) % cp (preserved by encoder).

     (b) Re-align axis-2 so slot c contains channel c by gathering with
         slot_for_channel[b, c] = (c - phase[b]) % cp.

     (c) Permute to (B, cp, T, D), flatten last two dims to (B*cp, T*D),
         apply the SHARED head Linear(T*D, P*C) -> (B*cp, P*C).

     (d) View as (B, cp, P, C), permute to (B, P, cp, C) (slot c = channel c).

     (e) Interleave back into phase order: gather along axis-2 with
         take_idx[b, k_in_block] = (phase[b] + k_in_block) % cp.

     (f) View (B, P, cp, C) -> (B, P*cp, C) = (B, pred_len, c_out).

   Result: prediction step k of sample b carries channel (phase[b] + k) % cp,
   which is exactly what per-channel RevIN denorm expects — no changes
   needed there.

3. forward() still accepts x_dec, x_mark_dec, dec_self_mask, dec_enc_mask,
   dec_phase positionally/keyword for back-compat with the existing
   exp_informer.py — they are simply ignored.

4. Asserts on construction: L_enc % cp == 0  and  pred_len % cp == 0.
   For the current config (seq_len=2352, e_layers=3, distil=True, cp=7):
       L_enc = 2352 // 4 = 588,  T = 84
       pred_len = 672,           P = 96
       Head params: 84 * 16 * (96 * 1) + 96 = 129,120

5. InformerStack: same trick. Each sub-encoder's output starts with
   channel `phase` and has length divisible by cp (assert-checked).
   Because L_i % cp == 0, the concatenated EncoderStack output also
   follows the global (phase + j) % cp pattern, so the same reshape +
   gather logic works without modification.

All other code paths (encoder, embedding, RevIN, masking, phase
threading) are unchanged.
"""

import torch
import torch.nn as nn

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from utils.RevIN import RevIN


def _shared_per_channel_head(enc_out, enc_phase, cp, pred_len, c_out,
                             head, head_dropout):
    """
    Apply ONE shared Linear head per channel and interleave outputs so
    prediction step k of sample b has channel (enc_phase[b] + k) % cp.

    Args:
        enc_out:      (B, L_enc, d_model)
        enc_phase:    (B,) LongTensor — channel at encoder position 0.
                      May be None → treated as all zeros.
        cp:           channel_period
        pred_len:     total prediction length (must be divisible by cp)
        c_out:        output features per position
        head:         nn.Linear(T*d_model, P*c_out)  — SHARED across channels
        head_dropout: nn.Dropout

    Returns:
        (B, pred_len, c_out)
    """
    B, L_enc, D = enc_out.shape
    T = L_enc // cp
    P = pred_len // cp

    if enc_phase is None:
        phase = torch.zeros(B, dtype=torch.long, device=enc_out.device)
    else:
        phase = enc_phase.long().to(enc_out.device)

    # (a) Reshape: at slot k of axis-2, channel is (phase + k) % cp.
    x = enc_out.view(B, T, cp, D)

    # (b) Re-align axis-2: slot c should contain channel c.
    #     slot_for_channel[b, c] = (c - phase[b]) % cp
    c_arr = torch.arange(cp, device=enc_out.device)
    slot_for_channel = (c_arr.unsqueeze(0) - phase.unsqueeze(1)) % cp   # (B, cp)
    gather_idx = slot_for_channel.unsqueeze(1).unsqueeze(-1).expand(B, T, cp, D)
    x_aligned = torch.gather(x, dim=2, index=gather_idx)                # (B, T, cp, D)

    # (c) Apply SHARED head across channels:
    #     (B, T, cp, D) -> (B, cp, T, D) -> (B*cp, T*D) -> head -> (B*cp, P*c_out)
    x_aligned = x_aligned.permute(0, 2, 1, 3).contiguous()              # (B, cp, T, D)
    x_flat = x_aligned.view(B * cp, T * D)
    x_flat = head_dropout(x_flat)
    out_flat = head(x_flat)                                             # (B*cp, P*c_out)
    out = out_flat.view(B, cp, P, c_out)                                # (B, cp, P, C)

    # (d) Permute so slot c at any time = channel c.
    out = out.permute(0, 2, 1, 3).contiguous()                          # (B, P, cp, C)

    # (e) Interleave by gathering: slot k_in_block <- channel (phase + k_in_block) % cp.
    k_arr = torch.arange(cp, device=enc_out.device)
    take_idx = (phase.unsqueeze(1) + k_arr.unsqueeze(0)) % cp           # (B, cp)
    take_idx = take_idx.unsqueeze(1).unsqueeze(-1).expand(B, P, cp, c_out)
    out = torch.gather(out, dim=2, index=take_idx)                      # (B, P, cp, C)

    # (f) Flatten time blocks back to (B, pred_len, c_out)
    return out.contiguous().view(B, P * cp, c_out)


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=512, dropout=0.0, attn='prob', embed='fixed', freq='h',
                 activation='gelu', output_attention=False, distil=True,
                 mix=True, device=torch.device('cuda:0'), use_revin=True,
                 channel_mix_size=None, channel_period=1, max_len=200000):
        super().__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.label_len = label_len          # accepted but unused
        self.c_out = c_out
        self.channel_period = channel_period
        self.output_attention = output_attention
        self.distil = distil

        # ── Encoder embedding ─────────────────────────────────────────────
        self.enc_embedding = DataEmbedding(
            c_in=enc_in, d_model=d_model, embed_type=embed, freq=freq,
            dropout=dropout, channel_period=channel_period, max_len=max_len,
        )

        Attn = ProbAttention if attn == 'prob' else FullAttention

        # ── Encoder (unchanged) ───────────────────────────────────────────
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout,
                             output_attention=output_attention,
                             channel_period=channel_period),
                        d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                    channel_mix_size=channel_mix_size,
                ) for _ in range(e_layers)
            ],
            [ConvLayer(d_model, channel_period=channel_period)
             for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # ── Compute encoder output length after distillation ──────────────
        if distil:
            L_enc = seq_len
            for _ in range(e_layers - 1):
                L_enc = L_enc // 2
        else:
            L_enc = seq_len
        self.L_enc = L_enc

        # ── Shape compatibility for the shared per-channel head ───────────
        assert L_enc % channel_period == 0, (
            f"L_enc ({L_enc}) must be divisible by channel_period "
            f"({channel_period}) for the shared per-channel head."
        )
        assert out_len % channel_period == 0, (
            f"pred_len ({out_len}) must be divisible by channel_period "
            f"({channel_period}) for the shared per-channel head."
        )
        T_per_channel = L_enc // channel_period
        P_per_channel = out_len // channel_period

        # ── SHARED head (weights reused across all cp channels) ───────────
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(T_per_channel * d_model,
                              P_per_channel * c_out, bias=True)

        # ── RevIN (unchanged) ─────────────────────────────────────────────
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(enc_in, channel_period=channel_period)

    def forward(self, x_enc, x_mark_enc,
                x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                enc_phase=None, dec_phase=None):
        """
        x_dec, x_mark_dec, dec_self_mask, dec_enc_mask, dec_phase are
        accepted but ignored (decoder removed). Kept for back-compat with
        exp_informer.py which still builds dec_inp.
        """
        # RevIN encoder-side normalization (unchanged)
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm', enc_phase=enc_phase)

        # Encoder (unchanged)
        enc_out = self.enc_embedding(
            x_enc, x_mark_enc, phase_offset=enc_phase)
        enc_out, attns = self.encoder(
            enc_out, attn_mask=enc_self_mask, enc_phase=enc_phase)
        # enc_out: (B, L_enc, d_model), with channel at position j of sample b
        # being (enc_phase[b] + j) % channel_period — preserved by ConvLayer.

        # SHARED per-channel head with phase-aware interleaving
        out = _shared_per_channel_head(
            enc_out, enc_phase,
            cp=self.channel_period,
            pred_len=self.pred_len,
            c_out=self.c_out,
            head=self.head,
            head_dropout=self.head_dropout,
        )
        # out: (B, pred_len, c_out). Step k of sample b has channel
        # (enc_phase[b] + k) % cp — matches RevIN denorm expectation.

        # RevIN denormalization (unchanged)
        if self.use_revin:
            out = self.revin(out, 'denorm', enc_phase=enc_phase)

        if self.output_attention:
            return out, attns
        return out


class InformerStack(nn.Module):
    """
    InformerStack with shared per-channel flatten head. Decoder removed.

    Each sub-encoder consumes seq_len // 2^i positions and (with distil)
    outputs (seq_len // 2^i) // 2^(el_i - 1) positions. EncoderStack
    concatenates them along the time axis. Because each sub-encoder's
    output starts at channel `phase` and its length is divisible by cp
    (asserted in __init__), the concatenated output globally follows
    (phase + j) % cp — so the same _shared_per_channel_head works.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1],
                 d_layers=2, d_ff=512, dropout=0.0, attn='prob', embed='fixed',
                 freq='h', activation='gelu', output_attention=False,
                 distil=True, mix=True, device=torch.device('cuda:0'),
                 use_revin=True, channel_mix_size=None, channel_period=1,
                 max_len=200000):
        super().__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.label_len = label_len
        self.c_out = c_out
        self.channel_period = channel_period
        self.output_attention = output_attention
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(enc_in, channel_period=channel_period)

        self.enc_embedding = DataEmbedding(
            c_in=enc_in, d_model=d_model, embed_type=embed, freq=freq,
            dropout=dropout, channel_period=channel_period, max_len=max_len,
        )

        Attn = ProbAttention if attn == 'prob' else FullAttention
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout,
                                 output_attention=output_attention,
                                 channel_period=channel_period),
                            d_model, n_heads, mix=False),
                        d_model, d_ff, dropout=dropout,
                        activation=activation,
                        channel_mix_size=channel_mix_size,
                    ) for _ in range(el)
                ],
                [ConvLayer(d_model, channel_period=channel_period)
                 for _ in range(el - 1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model),
            ) for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)

        # ── Compute per-sub-encoder output lengths, assert each is div by cp ──
        per_sub_lens = []
        if distil:
            for i, el in zip(inp_lens, e_layers):
                inp_len = seq_len // (2 ** i)
                out_len_i = inp_len // (2 ** (el - 1))
                per_sub_lens.append(out_len_i)
        else:
            for i in inp_lens:
                per_sub_lens.append(seq_len // (2 ** i))
        for k, L_i in enumerate(per_sub_lens):
            assert L_i % channel_period == 0, (
                f"InformerStack sub-encoder {k} produces length {L_i} "
                f"which is not divisible by channel_period {channel_period}. "
                f"The shared per-channel head requires per-sub divisibility "
                f"so the concatenated output keeps the global "
                f"(phase + j) % cp pattern."
            )
        L_enc = sum(per_sub_lens)
        self.L_enc = L_enc

        assert L_enc % channel_period == 0
        assert out_len % channel_period == 0
        T_per_channel = L_enc // channel_period
        P_per_channel = out_len // channel_period

        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(T_per_channel * d_model,
                              P_per_channel * c_out, bias=True)

    def forward(self, x_enc, x_mark_enc,
                x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                enc_phase=None, dec_phase=None):
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm', enc_phase=enc_phase)

        enc_out = self.enc_embedding(
            x_enc, x_mark_enc, phase_offset=enc_phase)
        enc_out, attns = self.encoder(
            enc_out, attn_mask=enc_self_mask, enc_phase=enc_phase)

        out = _shared_per_channel_head(
            enc_out, enc_phase,
            cp=self.channel_period,
            pred_len=self.pred_len,
            c_out=self.c_out,
            head=self.head,
            head_dropout=self.head_dropout,
        )

        if self.use_revin:
            out = self.revin(out, 'denorm', enc_phase=enc_phase)

        if self.output_attention:
            return out, attns
        return out
