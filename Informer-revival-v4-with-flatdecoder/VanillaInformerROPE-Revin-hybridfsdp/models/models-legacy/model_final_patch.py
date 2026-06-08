"""
model.py - Informer and InformerStack with ROPE + RevIN

FIX: channel_period default changed from None to 321. Passing None to
DataEmbedding propagates None to RotaryChannelEmbedding which crashes
on torch.arange(0, None).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

from utils.RevIN import RevIN


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'),
                 use_revin=True,
                 channel_mix_size=None,
                 channel_period=321,   # FIX: was None, which crashes DataEmbedding
                 max_len=200000):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding - with channel_period parameter
        self.enc_embedding = DataEmbedding(
            c_in=enc_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )
        self.dec_embedding = DataEmbedding(
            c_in=dec_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout,
                             output_attention=output_attention),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    channel_mix_size=channel_mix_size,
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout,
                             output_attention=False),
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    channel_mix_size=channel_mix_size,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.use_revin:
            dec_out = self.revin(dec_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'),
                 use_revin=True,
                 channel_mix_size=None,
                 channel_period=321,   # FIX: was 321 already, kept consistent
                 max_len=200000):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding - with channel_period parameter
        self.enc_embedding = DataEmbedding(
            c_in=enc_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )
        self.dec_embedding = DataEmbedding(
            c_in=dec_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout,
                                 output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                        channel_mix_size=channel_mix_size,
                    ) for l in range(el)
                ],
                [
                    ConvLayer(d_model) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout,
                             output_attention=False),
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    channel_mix_size=channel_mix_size,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.use_revin:
            dec_out = self.revin(dec_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
