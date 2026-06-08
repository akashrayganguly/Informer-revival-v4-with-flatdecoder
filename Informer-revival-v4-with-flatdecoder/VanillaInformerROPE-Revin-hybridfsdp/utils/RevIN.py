"""
utils/RevIN.py - Per-Channel Instance Normalization (FSDP & BF16 Optimized)

CHANGES vs latest repo:

1. Removed all_reduce from _get_statistics (CRITICAL):
   The original per-channel RevIN called dist.all_reduce on mean_per_channel
   and var_per_channel, averaging per-instance statistics across all samples
   on all 24 GPUs. This converted instance normalization into a form of batch
   normalization — each sample was normalized by the average statistics of
   ~48 samples rather than its own window's statistics. The entire mathematical
   basis of RevIN is that each sample normalizes itself. The all_reduce was
   removed entirely. RevIN statistics are now strictly local per sample.

   The global-mode all_reduce (when channel_period<=1 or enc_phase is None)
   is also removed for the same reason. Standard RevIN implementations
   (PatchTST, iTransformer) perform no cross-GPU reduction on RevIN statistics.

2. affine=False default:
   The affine parameters (affine_weight, affine_bias) had shape (num_features,)
   = (enc_in,) = (m+1,) — one value per lag column. After per-channel Z-scoring
   by StandardScaler, the affine was learning a correction per lag offset shared
   across all 7 channels. This is the wrong inductive bias: lag columns of
   different channels should share the same lag-position treatment, but channels
   may genuinely need different post-normalization corrections.

   With affine=False, the per-channel StandardScaler handles the global
   distributional differences and RevIN handles the instance-level mean/stdev
   removal, without introducing spurious per-lag-column parameters.

   The affine functionality is kept in the code (controlled by self.affine)
   for ablation or if explicitly set to True. The forward() interface is
   unchanged. The _normalize and _denormalize methods skip affine when
   self.affine=False.

The per-channel statistics computation structure (one-hot gather, channel_at_pos
indexing, (B, cp, F) storage) is unchanged — it was correct in the latest repo.
The global fallback (when channel_period<=1 or enc_phase is None) is also
unchanged in logic, only the all_reduce is removed.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-3, affine=False,
                 distributed=True, channel_period=1):
        """
        Reversible Instance Normalization with per-channel statistics.

        Args:
            num_features:    Number of features/columns (enc_in = m+1)
            eps:             Numerical stability epsilon (1e-3 for BF16)
            affine:          If True, learnable affine per feature column.
                             Default changed to False — see module docstring.
            distributed:     Kept for API compatibility. No longer used for
                             RevIN statistics (all_reduce removed). May be
                             used by subclasses.
            channel_period:  Number of channels c in (N*c, m+1) data.
                             1 = original global statistics (ETT-safe default).
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.distributed = distributed   # kept for API compat, not used in stats
        self.channel_period = channel_period

        # Per-channel statistics: set during normalize(), read during denormalize()
        self.mean_per_channel = None    # (B, cp, F) or None
        self.stdev_per_channel = None   # (B, cp, F) or None

        # Global statistics: used when channel_period<=1 or enc_phase is None
        self.mean = None                # (B, 1, F) or None
        self.stdev = None               # (B, 1, F) or None

        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, enc_phase=None):
        """
        Compute mean and std per channel (or globally for channel_period<=1).

        CHANGE: all dist.all_reduce calls removed. RevIN statistics are
        local per sample — each sample normalizes by its own window's
        channel-specific statistics. Cross-GPU reduction is wrong here
        because it averages instance-level statistics across different samples.

        Args:
            x:          (B, seq_len, num_features)
            enc_phase:  (B,) LongTensor — channel at position 0 of each
                        sample. None triggers global statistics fallback.
        """
        cp = self.channel_period

        if enc_phase is None or cp <= 1:
            # ── GLOBAL STATISTICS (original behavior, no all_reduce) ──────
            dim2reduce = tuple(range(1, x.ndim - 1))
            mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            variance = torch.var(x, dim=dim2reduce, keepdim=True,
                                 unbiased=False)
            # REMOVED: dist.all_reduce — global RevIN stats are per-instance
            self.mean = mean.detach()
            self.stdev = torch.sqrt(variance + self.eps).detach()
            self.mean_per_channel = None
            self.stdev_per_channel = None
            return

        # ── PER-CHANNEL STATISTICS ────────────────────────────────────────
        B, T, F = x.shape

        # channel_at_pos[b, j] = which channel is at position j of sample b
        j = torch.arange(T, device=x.device)
        channel_at_pos = (
            enc_phase.to(x.device).unsqueeze(1) + j.unsqueeze(0)
        ) % cp                                                      # (B, T)

        # One-hot: (B, T, cp)
        one_hot = torch.zeros(B, T, cp, device=x.device, dtype=x.dtype)
        one_hot.scatter_(2, channel_at_pos.unsqueeze(-1), 1.0)

        # Count per channel per sample: (B, cp)
        count = one_hot.sum(dim=1).clamp(min=1.0)

        # Mean per channel: (B, cp, F)
        x_sum = torch.bmm(one_hot.transpose(1, 2), x)              # (B, cp, F)
        mean_per_channel = x_sum / count.unsqueeze(-1)

        # Variance per channel: (B, cp, F)
        b_idx = torch.arange(B, device=x.device)[:, None].expand(B, T)
        mean_at_pos = mean_per_channel[b_idx, channel_at_pos, :]   # (B, T, F)
        x_centered = x - mean_at_pos
        var_sum = torch.bmm(one_hot.transpose(1, 2), x_centered ** 2)
        var_per_channel = var_sum / count.unsqueeze(-1)

        # REMOVED: dist.all_reduce on mean_per_channel and var_per_channel.
        # RevIN is instance normalization — statistics must be local per sample.
        # Averaging across GPUs converts it to batch normalization, which is
        # wrong: sample A on GPU 0 would be normalized by the average statistics
        # of all samples across all GPUs, losing the instance-level adaptation.

        self.mean_per_channel = mean_per_channel.detach()           # (B, cp, F)
        self.stdev_per_channel = torch.sqrt(
            var_per_channel + self.eps).detach()                    # (B, cp, F)
        self.mean = None
        self.stdev = None

    def _normalize(self, x, enc_phase=None):
        """
        Apply normalization using stored statistics.

        Can be called directly (without going through forward()) to normalize
        the decoder label portion in model.py using dec_phase.

        Args:
            x:          (B, seq_len, num_features) — any window of the data
            enc_phase:  (B,) LongTensor — channel at position 0 of this window
        """
        cp = self.channel_period

        if self.mean_per_channel is None or enc_phase is None or cp <= 1:
            # ── GLOBAL NORMALIZATION ───────────────────────────────────────
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = (x * self.affine_weight.view(1, 1, -1)
                     + self.affine_bias.view(1, 1, -1))
            return x

        # ── PER-CHANNEL NORMALIZATION ─────────────────────────────────────
        B, T, F = x.shape
        j = torch.arange(T, device=x.device)
        channel_at_pos = (
            enc_phase.to(x.device).unsqueeze(1) + j.unsqueeze(0)
        ) % cp                                                      # (B, T)

        b_idx = torch.arange(B, device=x.device)[:, None].expand(B, T)
        mean_at_pos = self.mean_per_channel[b_idx, channel_at_pos, :]
        stdev_at_pos = self.stdev_per_channel[b_idx, channel_at_pos, :]

        x = (x - mean_at_pos) / stdev_at_pos

        if self.affine:
            x = (x * self.affine_weight.view(1, 1, -1)
                 + self.affine_bias.view(1, 1, -1))

        return x

    def _denormalize(self, x, enc_phase=None):
        """
        Reverse normalization using stored statistics.

        Handles c_out < enc_in: slices mean/stdev to first F features.
        Prediction step k has channel (enc_phase + k) % cp, valid because
        seq_len % cp == 0 by construction.

        Args:
            x:          (B, pred_len, c_out) where c_out <= num_features
            enc_phase:  (B,) LongTensor — channel at encoder position 0
        """
        cp = self.channel_period
        F = x.shape[-1]

        if self.mean_per_channel is None or enc_phase is None or cp <= 1:
            # ── GLOBAL DENORMALIZATION ─────────────────────────────────────
            if self.affine:
                weight = self.affine_weight[:F].view(1, 1, -1)
                bias = self.affine_bias[:F].view(1, 1, -1)
                x = (x - bias) / (weight + self.eps * self.eps)
            x = x * self.stdev[:, :, :F] + self.mean[:, :, :F]
            return x

        # ── PER-CHANNEL DENORMALIZATION ───────────────────────────────────
        B, pred_len, _ = x.shape

        # Prediction step k has channel (enc_phase + k) % cp
        k_idx = torch.arange(pred_len, device=x.device)
        channel_at_step = (
            enc_phase.to(x.device).unsqueeze(1) + k_idx.unsqueeze(0)
        ) % cp                                                      # (B, pred_len)

        b_idx = torch.arange(B, device=x.device)[:, None].expand(B, pred_len)

        # Slice to first F features — handles c_out < enc_in cleanly
        mean_at_step = self.mean_per_channel[
            b_idx, channel_at_step, :F]                            # (B, pred_len, F)
        stdev_at_step = self.stdev_per_channel[
            b_idx, channel_at_step, :F]                            # (B, pred_len, F)

        if self.affine:
            weight = self.affine_weight[:F].view(1, 1, -1)
            bias = self.affine_bias[:F].view(1, 1, -1)
            x = (x - bias) / (weight + self.eps * self.eps)

        x = x * stdev_at_step + mean_at_step
        return x

    def normalize(self, x, enc_phase=None):
        """Compute statistics then normalize."""
        self._get_statistics(x, enc_phase)
        return self._normalize(x, enc_phase)

    def denormalize(self, x, enc_phase=None):
        """Denormalize using previously stored statistics."""
        if self.mean_per_channel is None and self.mean is None:
            raise RuntimeError(
                "RevIN.denormalize() called before normalize(). "
                "Call normalize() first."
            )
        return self._denormalize(x, enc_phase)

    def forward(self, x, mode: str, enc_phase=None):
        """
        Args:
            x:          Input tensor (B, seq_len, num_features)
            mode:       'norm' to normalize, 'denorm' to denormalize
            enc_phase:  (B,) LongTensor — channel at position 0. None = global.
        """
        if mode == 'norm':
            return self.normalize(x, enc_phase)
        elif mode == 'denorm':
            return self.denormalize(x, enc_phase)
        else:
            raise NotImplementedError(
                f"Mode '{mode}' not supported. Use 'norm' or 'denorm'.")
