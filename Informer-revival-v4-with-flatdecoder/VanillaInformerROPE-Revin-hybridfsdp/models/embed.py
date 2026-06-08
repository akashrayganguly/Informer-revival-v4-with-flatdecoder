"""
models/embed.py

CHANGES vs previous version:

1. NEW: AbsoluteTemporalEmbedding — fixed sinusoidal positional encoding
   indexed by timestep = (phase_offset + j) // channel_period.
   Replaces RotaryPositionalEmbeddingFixed. Non-learnable, ADDITIVE
   (added to the embedding vector, not applied as a rotation).

2. NEW: AbsoluteChannelEmbedding — fixed sinusoidal positional encoding
   indexed by channel = (phase_offset + j) % channel_period.
   Replaces RotaryChannelEmbeddingFixed. Non-learnable, ADDITIVE.

3. DataEmbedding.forward: now applies learnable ROPE (temporal + channel)
   followed by additive absolute sinusoidal embeddings (temporal + channel):

       x = value_embedding(x)
       x = rpe(x, phase_offset)                     # learnable ROPE temporal
       x = learnable_channel_embedding(x, phase_offset)  # learnable ROPE channel
       x = x + abs_temporal(x, phase_offset)         # additive sinusoidal temporal
       x = x + abs_channel(x, phase_offset)          # additive sinusoidal channel

   The ROPE fixed classes (RotaryPositionalEmbeddingFixed,
   RotaryChannelEmbeddingFixed) are kept in the file for reference but
   are no longer instantiated by DataEmbedding.

All other classes unchanged: TokenEmbedding (kernel_size=1),
RotaryPositionalEmbedding (learnable), RotaryChannelEmbeddingLearnable,
RotaryChannelEmbeddingFixed (kept, unused), RotaryPositionalEmbeddingFixed
(kept, unused), PositionalEmbedding, FixedEmbedding, TemporalEmbedding,
TimeFeatureEmbedding.

When phase_offset=None: all behaviour identical to uniform phase=0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =============================================================================
# UTILITY FUNCTIONS  (unchanged)
# =============================================================================

def _compute_rope_embeddings(
    max_len: int,
    d_model: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
    inv_freq = 1.0 / (base ** (
        torch.arange(0, d_model, 2, dtype=dtype, device=device) / d_model
    ))
    positions = torch.arange(0, max_len, dtype=dtype, device=device)
    sinusoid_inp = torch.outer(positions, inv_freq)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def _rotate_half_optimized(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    sin_embed: torch.Tensor,
    cos_embed: torch.Tensor
) -> torch.Tensor:
    return x * cos_embed + _rotate_half_optimized(x) * sin_embed


# =============================================================================
# ABSOLUTE SINUSOIDAL TEMPORAL EMBEDDING (NEW)
# =============================================================================

class AbsoluteTemporalEmbedding(nn.Module):
    """
    Fixed (non-learnable) absolute sinusoidal positional embedding for
    temporal positions.  ADDITIVE — returns a tensor to be added to the
    embedding vector.

    Phase-aware timestep indexing:
        timestep_index[b, j] = (phase_offset[b] + j) // channel_period

    This gives every position within the same real timestep the same
    temporal embedding, and consecutive real timesteps receive consecutive
    sinusoidal indices — matching the ROPE fixed temporal logic exactly,
    but as an additive embedding instead of a rotation.

    Standard sinusoidal formula (Vaswani et al.):
        PE(pos, 2i)   = sin(pos / base^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / base^(2i / d_model))

    Args:
        d_model:         Embedding dimension (must be even).
        max_len:         Maximum number of distinct timestep indices.
        base:            Frequency base (default 10000.0).
        channel_period:  Number of channels c. Positions within the same
                         real timestep share one sinusoidal index.
    """

    def __init__(self, d_model: int, max_len: int = 200000,
                 base: float = 10000.0, channel_period: int = 1):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.channel_period = channel_period

        # Precompute full sinusoidal table: (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(base) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=True)  # (max_len, d_model)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        """
        Args:
            x:            (B, seq_len, d_model) — used only for shape/device.
            phase_offset: None or (B,) LongTensor — channel at position 0.

        Returns:
            (B, seq_len, d_model) embedding to be ADDED to x.
        """
        batch, seq_len, _ = x.size()
        j = torch.arange(seq_len, device=x.device)

        if phase_offset is None:
            # Uniform phase=0 for all samples
            timestep_indices = j // self.channel_period              # (seq_len,)

            max_index = timestep_indices[-1].item() if seq_len > 0 else 0
            if max_index >= self.max_len:
                raise ValueError(
                    f"Timestep index {max_index} >= max_len {self.max_len}. "
                    f"Increase max_len to at least {max_index + 1}."
                )

            # (seq_len, d_model) → (1, seq_len, d_model) → broadcast to (B,...)
            embed = self.pe[timestep_indices].unsqueeze(0)
            if embed.dtype != x.dtype:
                embed = embed.to(x.dtype)
            return embed.expand(batch, -1, -1)
        else:
            # Per-sample phase-aware timestep indexing
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(
                    device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            # (B, seq_len)
            timestep_indices = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) // self.channel_period

            max_index = timestep_indices.max().item()
            if max_index >= self.max_len:
                raise ValueError(
                    f"Timestep index {max_index} >= max_len {self.max_len}. "
                    f"Increase max_len to at least {max_index + 1}."
                )

            embed = self.pe[timestep_indices]  # (B, seq_len, d_model)
            if embed.dtype != x.dtype:
                embed = embed.to(x.dtype)
            return embed

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, max_len={self.max_len}, '
                f'base={self.base}, channel_period={self.channel_period}')


# =============================================================================
# ABSOLUTE SINUSOIDAL CHANNEL EMBEDDING (NEW)
# =============================================================================

class AbsoluteChannelEmbedding(nn.Module):
    """
    Fixed (non-learnable) absolute sinusoidal embedding for channel identity.
    ADDITIVE — returns a tensor to be added to the embedding vector.

    Phase-aware channel indexing:
        channel_index[b, j] = (phase_offset[b] + j) % channel_period

    Each of the `channel_period` channels receives a unique d_model-dimensional
    sinusoidal fingerprint.  This replaces RotaryChannelEmbeddingFixed with
    an additive embedding.

    Uses a separate frequency base (default 50000.0) from the temporal
    embedding to give the channel dimension its own spectral signature.

    Args:
        c_in:            Number of input features (kept for API compat).
        d_model:         Embedding dimension (must be even).
        channel_period:  Number of channels c.
        base:            Frequency base (default 50000.0).
    """

    def __init__(self, c_in: int, d_model: int,
                 channel_period: int = 321, base: float = 50000.0):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base

        # Precompute sinusoidal table: (channel_period, d_model)
        pe = torch.zeros(channel_period, d_model, dtype=torch.float32)
        position = torch.arange(
            0, channel_period, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(base) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=True)  # (cp, d_model)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        """
        Args:
            x:            (B, seq_len, d_model) — used only for shape/device.
            phase_offset: None or (B,) LongTensor — channel at position 0.

        Returns:
            (B, seq_len, d_model) embedding to be ADDED to x.
        """
        batch, seq_len, _ = x.size()
        j = torch.arange(seq_len, device=x.device, dtype=torch.long)

        if phase_offset is None:
            channel_indices = j % self.channel_period               # (seq_len,)
            embed = self.pe[channel_indices].unsqueeze(0)           # (1, seq_len, d)
            if embed.dtype != x.dtype:
                embed = embed.to(x.dtype)
            return embed.expand(batch, -1, -1)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(
                    device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            # (B, seq_len)
            channel_indices = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) % self.channel_period

            embed = self.pe[channel_indices]  # (B, seq_len, d_model)
            if embed.dtype != x.dtype:
                embed = embed.to(x.dtype)
            return embed

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, c_in={self.c_in}, '
                f'period={self.channel_period}, base={self.base}')


# =============================================================================
# ROTARY POSITIONAL EMBEDDING — LEARNABLE  (unchanged)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Learnable Rotary Positional Embedding with timestep-granularity indexing.
    Unchanged from previous version.
    """

    def __init__(self, d_model: int, max_len: int = 200000,
                 base: float = 10000.0, channel_period: int = 1):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.channel_period = channel_period

        sin_embed, cos_embed = _compute_rope_embeddings(max_len, d_model, base)
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device)

        if phase_offset is None:
            timestep_indices = j // self.channel_period

            max_index = timestep_indices[-1].item()
            if max_index >= self.max_len:
                raise ValueError(
                    f"Timestep index {max_index} >= max_len {self.max_len}. "
                    f"Increase max_len to at least {max_index + 1}."
                )

            sin_base = self.sin_embed[timestep_indices, :]
            cos_base = self.cos_embed[timestep_indices, :]
            sin_embed = sin_base.unsqueeze(0).repeat_interleave(2, dim=-1)
            cos_embed = cos_base.unsqueeze(0).repeat_interleave(2, dim=-1)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(
                    device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            timestep_indices = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) // self.channel_period

            max_index = timestep_indices.max().item()
            if max_index >= self.max_len:
                raise ValueError(
                    f"Timestep index {max_index} >= max_len {self.max_len}. "
                    f"Increase max_len to at least {max_index + 1}."
                )

            sin_base = self.sin_embed[timestep_indices, :]
            cos_base = self.cos_embed[timestep_indices, :]
            sin_embed = sin_base.repeat_interleave(2, dim=-1)
            cos_embed = cos_base.repeat_interleave(2, dim=-1)

        if sin_embed.dtype != x.dtype:
            sin_embed = sin_embed.to(x.dtype)
            cos_embed = cos_embed.to(x.dtype)

        return _apply_rotary_emb(x, sin_embed, cos_embed)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, max_len={self.max_len}, '
                f'channel_period={self.channel_period}')


# =============================================================================
# ROTARY POSITIONAL EMBEDDING — FIXED  (kept for reference, unused by DataEmbedding)
# =============================================================================

class RotaryPositionalEmbeddingFixed(nn.Module):
    """
    Fixed (non-learnable) Rotary Positional Embedding.
    KEPT FOR REFERENCE — no longer used by DataEmbedding.
    Replaced by AbsoluteTemporalEmbedding.
    """

    def __init__(self, d_model: int, max_len: int = 200000,
                 base: float = 10000.0, channel_period: int = 1):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.channel_period = channel_period

        sin_embed, cos_embed = _compute_rope_embeddings(max_len, d_model, base)
        self.register_buffer("sin_embed", sin_embed, persistent=True)
        self.register_buffer("cos_embed", cos_embed, persistent=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device)

        if phase_offset is None:
            timestep_indices = j // self.channel_period

            max_index = timestep_indices[-1].item()
            if max_index >= self.max_len:
                raise ValueError(
                    f"Timestep index {max_index} >= max_len {self.max_len}. "
                    f"Increase max_len to at least {max_index + 1}."
                )

            sin_base = self.sin_embed[timestep_indices, :]
            cos_base = self.cos_embed[timestep_indices, :]
            sin_embed = sin_base.unsqueeze(0).repeat_interleave(2, dim=-1)
            cos_embed = cos_base.unsqueeze(0).repeat_interleave(2, dim=-1)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(
                    device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            timestep_indices = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) // self.channel_period

            max_index = timestep_indices.max().item()
            if max_index >= self.max_len:
                raise ValueError(
                    f"Timestep index {max_index} >= max_len {self.max_len}. "
                    f"Increase max_len to at least {max_index + 1}."
                )

            sin_base = self.sin_embed[timestep_indices, :]
            cos_base = self.cos_embed[timestep_indices, :]
            sin_embed = sin_base.repeat_interleave(2, dim=-1)
            cos_embed = cos_base.repeat_interleave(2, dim=-1)

        if sin_embed.dtype != x.dtype:
            sin_embed = sin_embed.to(x.dtype)
            cos_embed = cos_embed.to(x.dtype)

        return _apply_rotary_emb(x, sin_embed, cos_embed)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, max_len={self.max_len}, '
                f'channel_period={self.channel_period}')


# =============================================================================
# ROTARY CHANNEL EMBEDDING — LEARNABLE  (unchanged)
# =============================================================================

class RotaryChannelEmbeddingLearnable(nn.Module):
    """Learnable Rotary Channel Embedding with per-sample phase correction.
    UNCHANGED."""

    def __init__(self, c_in: int, d_model: int,
                 channel_period: int = 321, max_len: int = 2000,
                 base: float = 50000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base
        self.max_len = max_len

        sin_embed, cos_embed = _compute_rope_embeddings(
            channel_period, d_model, base)
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device, dtype=torch.long)

        if phase_offset is None:
            positions = j % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
            cos_e = self.cos_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            positions = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(2, dim=-1)
            cos_e = self.cos_embed[positions].repeat_interleave(2, dim=-1)

        if sin_e.dtype != x.dtype:
            sin_e = sin_e.to(x.dtype)
            cos_e = cos_e.to(x.dtype)

        return _apply_rotary_emb(x, sin_e, cos_e)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, c_in={self.c_in}, '
                f'period={self.channel_period}, base={self.base}')


# =============================================================================
# ROTARY CHANNEL EMBEDDING — FIXED  (kept for reference, unused by DataEmbedding)
# =============================================================================

class RotaryChannelEmbeddingFixed(nn.Module):
    """Fixed (non-learnable) Rotary Channel Embedding.
    KEPT FOR REFERENCE — no longer used by DataEmbedding.
    Replaced by AbsoluteChannelEmbedding."""

    def __init__(self, c_in: int, d_model: int,
                 channel_period: int = 321, max_len: int = 2000,
                 base: float = 50000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base

        sin_embed, cos_embed = _compute_rope_embeddings(
            channel_period, d_model, base)
        self.register_buffer("sin_embed", sin_embed, persistent=True)
        self.register_buffer("cos_embed", cos_embed, persistent=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device, dtype=torch.long)

        if phase_offset is None:
            positions = j % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
            cos_e = self.cos_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            positions = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(2, dim=-1)
            cos_e = self.cos_embed[positions].repeat_interleave(2, dim=-1)

        if sin_e.dtype != x.dtype:
            sin_e = sin_e.to(x.dtype)
            cos_e = cos_e.to(x.dtype)

        return _apply_rotary_emb(x, sin_e, cos_e)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, c_in={self.c_in}, '
                f'period={self.channel_period}, base={self.base}')


# =============================================================================
# STANDARD EMBEDDINGS  (unchanged)
# =============================================================================

class PositionalEmbedding(nn.Module):
    """Unchanged."""
    def __init__(self, d_model: int, max_len: int = 200000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]


class TokenEmbedding(nn.Module):
    """Token Embedding: kernel_size=1, pointwise projection. Unchanged."""
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=1,
            padding=0
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    """Unchanged."""
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model, dtype=torch.float32)
        position = torch.arange(0, c_in, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Unchanged."""
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        super().__init__()
        minute_size, hour_size, weekday_size, day_size, month_size = 4, 24, 7, 32, 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = (self.minute_embed(x[:, :, 4])
                    if hasattr(self, 'minute_embed') else 0.)
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Unchanged."""
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        super().__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


# =============================================================================
# MAIN DATA EMBEDDING
# =============================================================================

class DataEmbedding(nn.Module):
    """
    Combined Data Embedding.

    CHANGE: Replaced fixed ROPE embeddings with additive absolute sinusoidal
    embeddings.

    Previous combination (all ROPE, multiplicative):
        x = value_embedding(x)
        x = rpe(x) + rpe_fixed(x)                    # two rotated copies summed
        x = fixed_channel(x) + learnable_channel(x)  # two rotated copies summed

    New combination (learnable ROPE + additive absolute sinusoidal):
        x = value_embedding(x)
        x = rpe(x)                        # learnable ROPE temporal (rotation)
        x = learnable_channel(x)          # learnable ROPE channel (rotation)
        x = x + abs_temporal(x)           # fixed sinusoidal temporal (additive)
        x = x + abs_channel(x)            # fixed sinusoidal channel (additive)

    The learnable ROPE components are kept because they adapt during training.
    The fixed ROPE components are replaced with standard additive sinusoidal
    embeddings that inject position/channel information additively rather
    than via rotation.
    """

    def __init__(self, c_in: int, d_model: int,
                 embed_type: str = 'fixed', freq: str = 'h',
                 dropout: float = 0.1, channel_period: int = 321,
                 max_len: int = 200000):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.channel_period = channel_period

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(
            d_model=d_model, max_len=max_len)

        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)

        # ── NEW: Absolute sinusoidal embeddings (replace fixed ROPE) ─────
        self.abs_temporal = AbsoluteTemporalEmbedding(
            d_model=d_model, max_len=max_len,
            base=10000.0, channel_period=channel_period)
        self.abs_channel = AbsoluteChannelEmbedding(
            c_in=c_in, d_model=d_model,
            channel_period=channel_period, base=50000.0)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor,
                phase_offset=None) -> torch.Tensor:
        x = self.value_embedding(x)

        # Additive absolute sinusoidal: temporal + channel
        x = x + self.abs_temporal(x, phase_offset=phase_offset)
        x = x + self.abs_channel(x, phase_offset=phase_offset)

        return self.dropout(x)
