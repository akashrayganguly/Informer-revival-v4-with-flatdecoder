import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    Informer Encoder Layer with learnable channel-mixing matrix W.

    FSDP / Hybrid-Shard notes
    -------------------------
    * The auto-wrap policy wraps each EncoderLayer as one FSDP unit.
      All sub-modules (attention, conv1/2, norms, W) live in the same
      shard, so the einsum always sees the fully-materialised W during
      forward — no special handling needed.
    * `use_orig_params=True` is required in the FSDP constructor for
      bare `nn.Parameter` objects (self.W) to work correctly.
    * Under `MixedPrecision`, FSDP casts parameters to param_dtype
      before forward.  The einsum inputs (x after norm2, which is fp32
      from LayerNorm) and W (cast to param_dtype by FSDP) may differ.
      We explicitly cast the einsum operands to the same dtype to avoid
      silent promotion or errors.

    Parameters
    ----------
    attention : nn.Module
        Attention layer (ProbAttention or FullAttention wrapped in AttentionLayer).
    d_model : int
        Hidden dimension.
    d_ff : int, optional
        Feed-forward inner dimension (default: 4 * d_model).
    dropout : float
        Dropout probability.
    activation : str
        'relu' or 'gelu'.
    channel_mix_size : int or None
        Size of the learnable mixing matrix W.  When not None, a
        (channel_mix_size, channel_mix_size) parameter is created and
        applied group-wise along the sequence dimension.  seq_len must
        be divisible by channel_mix_size at runtime.
        Set to None to disable channel mixing (original Informer behaviour).
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", channel_mix_size=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # ---------- standard informer components ----------
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # ---------- channel-mixing extension ----------
        self.channel_mix_size = channel_mix_size
        if channel_mix_size is not None and channel_mix_size > 0:
            self.norm3 = nn.LayerNorm(d_model)
            # nn.Linear is better supported by FSDP than a bare Parameter
            # for sharding/unsharding, but a bare Parameter also works
            # when use_orig_params=True.  We keep nn.Parameter for
            # mathematical clarity (it's a square mixing matrix, not a
            # bias-equipped linear map).
            self.W = nn.Parameter(torch.empty(channel_mix_size, channel_mix_size))
            nn.init.xavier_uniform_(self.W)
        else:
            self.norm3 = None
            self.W = None

    def forward(self, x, attn_mask=None):
        """
        Parameters
        ----------
        x : Tensor [B, L, D]
        attn_mask : optional mask

        Returns
        -------
        x : Tensor [B, L', D]   (L' = L when no distillation pool follows)
        attn : attention weights or None
        """
        # ---- self-attention + residual ----
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        # ---- FFN + residual ----
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)

        # ---- channel mixing (optional) ----
        if self.W is not None:
            batch_size, seq_len, d_model = x.shape
            c = self.channel_mix_size

            if seq_len % c != 0:
                raise RuntimeError(
                    f"EncoderLayer channel_mix_size={c} does not evenly "
                    f"divide seq_len={seq_len}.  seq_len must be a multiple "
                    f"of channel_mix_size."
                )

            n = seq_len // c

            # Reshape: [B, n, c, D]
            x_reshaped = x.view(batch_size, n, c, d_model)

            # Ensure matching dtypes under mixed-precision.
            # FSDP casts self.W to param_dtype; x may still be fp32
            # from LayerNorm.  Cast x to match W.
            W = self.W
            if x_reshaped.dtype != W.dtype:
                x_reshaped = x_reshaped.to(W.dtype)

            # Apply W group-wise: [c, c] × [B, n, c, D] → [B, n, c, D]
            x_transformed = torch.einsum('ij,bnjd->bnid', W, x_reshaped)

            # Reshape back: [B, L, D]
            x = x_transformed.reshape(batch_size, seq_len, d_model)
            x = self.dropout(x)
            x = self.norm3(x)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
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
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns
