"""
utils/tools.py - Utility Functions with FSDP Support

KEY CHANGE: StandardScaler now supports per-channel statistics.

PROBLEM WITH ORIGINAL:
    StandardScaler.fit(data) called data.mean(0) and data.std(0), computing
    one mean and std per column averaged across ALL rows. For (N*c, m+1)
    restructured data, this mixes all c channel distributions into a single
    statistic. Channel 0 (mean 0.3) and channel 6 (mean 0.8) both get
    subtracted by the mixture mean ~0.55 — wrong for both.

FIX:
    fit(data, channel_period=c): for each channel ch in 0..c-1, extract
    rows data[ch::c, :] (all rows belonging to that channel) and compute
    their mean and std independently. Stores mean_channels (c, F) and
    std_channels (c, F).

    transform(data): row j has channel j % c. Applies mean_channels[j%c, :]
    and std_channels[j%c, :] to each row. Vectorized with numpy/torch.

    inverse_transform(data, channel_indices): reverses transform. For
    predictions of shape (N_saved, pred_len, c_out), channel_indices of
    shape (N_saved, pred_len) gives the channel for each (sample, step)
    pair. Applies the correct channel's statistics to each element.

    BACKWARD COMPAT: channel_period=1 (default) gives original global
    behavior exactly. ETT datasets call fit() without channel_period.

REMOVES NEED FOR EXTERNAL MINMAXSCALER:
    With per-channel Z-score normalization, feed RAW data directly.
    inverse_transform in test() brings predictions back to raw units.
    Do NOT use both external MinMaxScaler AND this per-channel scaler —
    that double-normalizes and only the Z-score is inverted in test().
"""

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


class StandardScaler():
    """
    Standard Scaler with optional per-channel statistics.

    Usage for (N*c, m+1) restructured data:
        scaler = StandardScaler()
        scaler.fit(train_data, channel_period=7)   # per-channel
        data_normalized = scaler.transform(full_data)
        # inverse in test():
        pred_raw = scaler.inverse_transform(pred_array, channel_indices)
    """

    def __init__(self):
        # Global statistics (used when channel_period=1)
        self.mean = 0.
        self.std = 1.

        # Per-channel statistics (used when channel_period > 1)
        self.channel_period = 1
        self.mean_channels = None   # (channel_period, F) ndarray
        self.std_channels = None    # (channel_period, F) ndarray

    def fit(self, data, channel_period=1):
        """
        Fit scaler on training data.

        Args:
            data:           numpy array (N_train [* channel_period], F)
            channel_period: number of channels c. If > 1, row j belongs to
                            channel j % channel_period. Computes separate
                            mean/std per channel.
        """
        self.channel_period = channel_period

        if channel_period <= 1:
            # ── ORIGINAL GLOBAL BEHAVIOR ──────────────────────────────────
            self.mean = data.mean(0)
            self.std = data.std(0)
            self.mean_channels = None
            self.std_channels = None
            return

        # ── PER-CHANNEL STATISTICS ────────────────────────────────────────
        cp = channel_period
        F = data.shape[1]
        self.mean_channels = np.zeros((cp, F), dtype=np.float64)
        self.std_channels = np.ones((cp, F), dtype=np.float64)

        for ch in range(cp):
            # Rows at positions ch, ch+cp, ch+2*cp, ... all belong to channel ch
            rows = data[ch::cp, :]                      # (N_train, F)
            self.mean_channels[ch] = rows.mean(0)
            std = rows.std(0)
            # Prevent division by zero for constant channels
            self.std_channels[ch] = np.where(std == 0, 1.0, std)

        # Also set global mean/std as channel-averages for backward compat
        # (used by any code that reads scaler.mean directly)
        self.mean = self.mean_channels.mean(0)          # (F,)
        self.std = self.std_channels.mean(0)            # (F,)

    def transform(self, data):
        """
        Normalize data using fitted statistics.

        For channel_period > 1: row j gets normalized by channel (j % cp)'s
        statistics. Row indices here are absolute indices in df_data starting
        from 0 — consistent with how the dataset is structured.

        Args:
            data: numpy array or torch.Tensor of shape (N_rows, F)

        Returns:
            Normalized array/tensor of same shape
        """
        if self.channel_period <= 1 or self.mean_channels is None:
            # ── ORIGINAL GLOBAL TRANSFORM ─────────────────────────────────
            if torch.is_tensor(data):
                mean = torch.from_numpy(self.mean).type_as(data).to(data.device)
                std = torch.from_numpy(self.std).type_as(data).to(data.device)
            else:
                mean, std = self.mean, self.std
            return (data - mean) / std

        # ── PER-CHANNEL TRANSFORM ─────────────────────────────────────────
        cp = self.channel_period
        n_rows = data.shape[0]

        if torch.is_tensor(data):
            row_ch = torch.arange(n_rows, device=data.device) % cp
            mean_t = torch.from_numpy(self.mean_channels).float().to(data.device)
            std_t = torch.from_numpy(self.std_channels).float().to(data.device)
            mean_at_row = mean_t[row_ch, :]             # (n_rows, F)
            std_at_row = std_t[row_ch, :]
            return (data - mean_at_row) / std_at_row
        else:
            row_ch = np.arange(n_rows) % cp
            mean_at_row = self.mean_channels[row_ch, :] # (n_rows, F)
            std_at_row = self.std_channels[row_ch, :]
            return (data - mean_at_row) / std_at_row

    def inverse_transform(self, data, channel_indices=None):
        """
        Reverse normalization.

        For per-channel mode, channel_indices specifies which channel's
        statistics to use for each element.

        Args:
            data:             numpy array of shape (..., F) — predictions or
                              ground truth
            channel_indices:  numpy int array of shape (...) — same leading
                              dimensions as data minus the last (F) dimension.
                              Entry [j, k] gives the channel index for
                              prediction step k of sample j.
                              None triggers original global behavior.

        Returns:
            Array in original (pre-normalization) units, same shape as data.

        Example (test() usage):
            pred_raw = scaler.inverse_transform(
                preds_to_save,           # (N_saved, pred_len, c_out)
                channel_indices          # (N_saved, pred_len)
            )
        """
        if (self.channel_period <= 1 or self.mean_channels is None
                or channel_indices is None):
            # ── ORIGINAL GLOBAL INVERSE TRANSFORM ─────────────────────────
            if torch.is_tensor(data):
                mean = torch.from_numpy(self.mean).type_as(data).to(data.device)
                std = torch.from_numpy(self.std).type_as(data).to(data.device)
            else:
                mean, std = self.mean, self.std
            if data.shape[-1] != mean.shape[-1]:
                mean = mean[-1:]
                std = std[-1:]
            return data * std + mean

        # ── PER-CHANNEL INVERSE TRANSFORM ────────────────────────────────
        # channel_indices: (...) integer array giving channel for each element
        # data: (..., F) — the F features are the delayed columns (c_out of them)
        F = data.shape[-1]
        ch_idx = np.asarray(channel_indices, dtype=np.int32)

        # mean_at[..., f] = mean_channels[channel_indices[...], f]
        mean_at = self.mean_channels[ch_idx, :F]       # (..., F)
        std_at = self.std_channels[ch_idx, :F]         # (..., F)

        return data * std_at + mean_at


class EarlyStopping:
    """
    Early stopping with FSDP support.
    Unchanged from original — see original docstring.
    """

    def __init__(self, patience=7, verbose=False, delta=0,
                 use_fsdp=False, global_rank=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.use_fsdp = use_fsdp
        self.global_rank = global_rank

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self._should_print():
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

        if self.use_fsdp and dist.is_initialized():
            device = next(model.parameters()).device
            early_stop_tensor = torch.tensor(int(self.early_stop), device=device)
            dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
            self.early_stop = bool(early_stop_tensor.item())

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose and self._should_print():
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> '
                  f'{val_loss:.6f}).  Saving model ...')

        if self.use_fsdp and isinstance(model, FSDP):
            with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                state_dict = model.state_dict()
                if self._should_print():
                    torch.save(state_dict, path + '/' + 'checkpoint.pth')
        else:
            if self._should_print():
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(),
                               path + '/' + 'checkpoint.pth')
                else:
                    torch.save(model.state_dict(),
                               path + '/' + 'checkpoint.pth')

        self.val_loss_min = val_loss

    def _should_print(self):
        if self.use_fsdp:
            return self.global_rank == 0
        return True


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'cosine':
        import math
        min_lr = args.learning_rate * 0.01
        lr = (min_lr + (args.learning_rate - min_lr) *
              (1 + math.cos(math.pi * epoch / args.train_epochs)) / 2)
        lr_adjust = {epoch: lr}
    else:
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        should_print = (not hasattr(args, 'use_fsdp')
                        or not args.use_fsdp
                        or args.global_rank == 0)
        if should_print:
            print('Updating learning rate to {}'.format(lr))


def visual(true, preds=None, name='./pic/test.pdf'):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def save_checkpoint_fsdp(model, optimizer, epoch, path, args):
    if hasattr(args, 'use_fsdp') and args.use_fsdp and isinstance(model, FSDP):
        with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            model_state_dict = model.state_dict()
            if args.global_rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, path)
    else:
        if not hasattr(args, 'global_rank') or args.global_rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': None,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if isinstance(model, torch.nn.DataParallel):
                checkpoint['model_state_dict'] = model.module.state_dict()
            else:
                checkpoint['model_state_dict'] = model.state_dict()
            torch.save(checkpoint, path)


def load_checkpoint_fsdp(model, optimizer, path, args):
    checkpoint = torch.load(path, map_location='cpu')
    if hasattr(args, 'use_fsdp') and args.use_fsdp and isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)


def clip_grad_norm_fsdp(model, max_norm, args):
    if max_norm <= 0:
        return
    if hasattr(args, 'use_fsdp') and args.use_fsdp and isinstance(model, FSDP):
        model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_parameter_count(model, args):
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model, args):
    should_print = (not hasattr(args, 'use_fsdp')
                    or not args.use_fsdp
                    or args.global_rank == 0)
    if not should_print:
        return
    total_params = get_parameter_count(model, args)
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print('=' * 50)
    print('Model Summary')
    print('=' * 50)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print('=' * 50)
