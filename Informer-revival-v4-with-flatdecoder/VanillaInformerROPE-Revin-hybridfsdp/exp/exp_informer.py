"""
exp/exp_informer.py - Uniform Loss + Eval-Mode Diagnostic

CHANGES vs previous version:

1. REMOVED difficulty weighting. All channels now receive equal weight
   (1/cp each). The loss is simply ch_mse_tensor.mean() — equivalent to
   uniform MSE across channels. The end-of-epoch weight update block is
   removed entirely.

2. ADDED eval-mode diagnostic at the end of each training epoch, before
   vali() calls. Switches model to eval (dropout off), runs 50 batches of
   the training loader, computes per-channel MSE, all-reduces across FSDP
   ranks, and prints side-by-side with the train-mode per-channel MSE.
   This isolates the effect of dropout noise on training loss by comparing
   the same data with dropout on vs off.

3. vali() returns (scalar_loss, per_ch_mse) — unchanged.

4. test() all_gathers predictions across FSDP ranks — unchanged.

All other logic unchanged.
"""

import gc
import json
import os
import time
import warnings
import numpy as np
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, BackwardPrefetch, CPUOffload,
    StateDictType, FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)

try:
    from torch.amp import GradScaler
    _GRADSCALER_DEVICE = True
except ImportError:
    from torch.cuda.amp import GradScaler
    _GRADSCALER_DEVICE = False

from data.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute,
                               Dataset_Custom, Dataset_Pred)
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from utils.tools import EarlyStopping, adjust_learning_rate
from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                       SequentialLR, ReduceLROnPlateau)
from utils.metrics import metric

warnings.filterwarnings('ignore')


# =============================================================================
# CUDA STREAM DATA PREFETCHER (unchanged)
# =============================================================================
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = tuple(
                item.to(self.device, non_blocking=True)
                if isinstance(item, torch.Tensor) else item
                for item in batch
            )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        for item in batch:
            if isinstance(item, torch.Tensor):
                item.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


# =============================================================================
# AMP CONFIGURATION (unchanged)
# =============================================================================
def get_amp_config():
    if not torch.cuda.is_available():
        return {'supported': False, 'dtype': torch.float32,
                'use_scaler': False, 'name': 'FP32'}
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return {'supported': True, 'dtype': torch.bfloat16,
                'use_scaler': False, 'name': 'BF16'}
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:
        return {'supported': True, 'dtype': torch.float16,
                'use_scaler': True, 'name': 'FP16'}
    return {'supported': False, 'dtype': torch.float32,
            'use_scaler': False, 'name': 'FP32'}


def get_num_gpus_per_node():
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE')
    if local_world_size:
        return int(local_world_size)
    slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
    if slurm_gpus:
        return int(slurm_gpus)
    return torch.cuda.device_count()


def setup_hybrid_sharding(args):
    if not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_gpus_per_node = get_num_gpus_per_node()
    num_nodes = world_size // num_gpus_per_node
    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"HYBRID SHARDING SETUP")
        print(f"  World: {world_size}, Nodes: {num_nodes}, "
              f"GPUs/node: {num_gpus_per_node}")
        print(f"{'='*60}")
    if num_nodes <= 1:
        if global_rank == 0:
            print("  Single node — HYBRID_SHARD equivalent to FULL_SHARD")
        return None
    try:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh(
            "cuda", (num_nodes, num_gpus_per_node),
            mesh_dim_names=("replicate", "shard")
        )
        if global_rank == 0:
            print(f"  DeviceMesh created: ({num_nodes}, {num_gpus_per_node})")
        return mesh
    except Exception as e:
        if global_rank == 0:
            print(f"  DeviceMesh creation failed: {e}")
        return None


class PerformanceMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optimizer_time = 0.0
        self.count = 0

    def update(self, data_t, fwd_t, bwd_t, opt_t):
        self.data_time += data_t
        self.forward_time += fwd_t
        self.backward_time += bwd_t
        self.optimizer_time += opt_t
        self.count += 1

    def summary(self):
        if self.count == 0:
            return "No data"
        total = (self.data_time + self.forward_time +
                 self.backward_time + self.optimizer_time)
        if total == 0:
            return "No time recorded"
        return (f"Data:{self.data_time/total*100:.0f}% "
                f"Fwd:{self.forward_time/total*100:.0f}% "
                f"Bwd:{self.backward_time/total*100:.0f}% "
                f"Opt:{self.optimizer_time/total*100:.0f}%")


# =============================================================================
# MAIN EXPERIMENT CLASS
# =============================================================================
class Exp_Informer(Exp_Basic):

    def __init__(self, args):
        self.gradient_accumulation_steps = getattr(
            args, 'gradient_accumulation_steps', 1)
        self.amp_config = get_amp_config()
        self.device_mesh = None
        self.perf_monitor = PerformanceMonitor()
        self.use_prefetcher = (getattr(args, 'use_prefetcher', True)
                               and torch.cuda.is_available())
        super(Exp_Informer, self).__init__(args)
        if self._should_print():
            self._print_config()

    def _print_config(self):
        args = self.args
        print(f"\n{'='*60}")
        print(f"EXPERIMENT CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Model: {args.model}")
        print(f"  Data: {args.data}")
        print(f"  Seq/Label/Pred: {args.seq_len}/{args.label_len}/{args.pred_len}")
        print(f"  d_model: {args.d_model}, n_heads: {args.n_heads}")
        print(f"  E-layers: {args.e_layers}, D-layers: {args.d_layers}")
        print(f"  FSDP: {getattr(args, 'use_fsdp', False)}")
        if getattr(args, 'use_fsdp', False):
            print(f"  Sharding Strategy: "
                  f"{getattr(args, 'fsdp_sharding_strategy', 'FULL_SHARD')}")
            print(f"  World Size: {getattr(args, 'world_size', 1)}")
        print(f"  AMP: {self.amp_config['name']}")
        print(f"  Channel Period: {getattr(args, 'channel_period', 1)}")
        print(f"  Per-channel loss: uniform (equal weights)")
        print(f"  Eval-mode diagnostic: enabled (50 train batches)")
        print(f"  Block-triangular decoder mask: enabled")
        print(f"  RevIN: {getattr(args, 'use_revin', True)} "
              f"(affine=False, no all_reduce)")
        print(f"  Test metric gathering: {'all_gather (global)' if getattr(args, 'use_fsdp', False) else 'single-process'}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

    def _build_model(self):
        args = self.args
        model_dict = {'informer': Informer, 'informerstack': InformerStack}
        if args.model not in model_dict:
            raise ValueError(f"Unknown model: {args.model}")
        ModelClass = model_dict[args.model]
        e_layers = (args.s_layers
                    if args.model == 'informerstack'
                    and hasattr(args, 's_layers')
                    else args.e_layers)

        base_kwargs = {
            'enc_in': args.enc_in, 'dec_in': args.dec_in,
            'c_out': args.c_out,
            'seq_len': args.seq_len, 'label_len': args.label_len,
            'out_len': args.pred_len,
            'factor': getattr(args, 'factor', 5),
            'd_model': getattr(args, 'd_model', 512),
            'n_heads': getattr(args, 'n_heads', 8),
            'e_layers': e_layers,
            'd_layers': getattr(args, 'd_layers', 1),
            'd_ff': getattr(args, 'd_ff', 2048),
            'dropout': getattr(args, 'dropout', 0.05),
            'attn': getattr(args, 'attn', 'prob'),
            'embed': getattr(args, 'embed', 'timeF'),
            'freq': getattr(args, 'freq', 'h'),
            'activation': getattr(args, 'activation', 'gelu'),
            'output_attention': getattr(args, 'output_attention', False),
            'distil': getattr(args, 'distil', True),
            'mix': getattr(args, 'mix', True),
            'device': self.device,
        }

        import inspect
        sig = inspect.signature(ModelClass.__init__)
        model_params = list(sig.parameters.keys())
        extended_kwargs = {}
        if 'use_revin' in model_params:
            extended_kwargs['use_revin'] = getattr(args, 'use_revin', True)
        if 'channel_mix_size' in model_params:
            extended_kwargs['channel_mix_size'] = getattr(
                args, 'channel_mix_size', None)
        if 'channel_period' in model_params:
            extended_kwargs['channel_period'] = getattr(
                args, 'channel_period', 1)
        if 'max_len' in model_params:
            extended_kwargs['max_len'] = getattr(args, 'max_len', 200000)

        model = ModelClass(**{**base_kwargs, **extended_kwargs}).float()

        if self._should_print():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model Parameters: {total_params:,} total, "
                  f"{trainable_params:,} trainable")

        if getattr(args, 'use_fsdp', False) and torch.cuda.is_available():
            model = self._wrap_model_with_fsdp(model)
        elif (getattr(args, 'use_multi_gpu', False)
              and torch.cuda.device_count() > 1):
            model = nn.DataParallel(model, device_ids=args.device_ids)

        return model

    def _apply_activation_checkpointing(self, model):
        check_fn = lambda m: isinstance(m, (EncoderLayer, DecoderLayer))
        wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)
        if self._should_print():
            enc_count = sum(
                1 for m in model.modules() if isinstance(m, EncoderLayer))
            dec_count = sum(
                1 for m in model.modules() if isinstance(m, DecoderLayer))
            print(f"  Activation Checkpointing: {enc_count} encoder, "
                  f"{dec_count} decoder layers")

    def _wrap_model_with_fsdp(self, model):
        args = self.args
        if self._should_print():
            print(f"\nWrapping model with FSDP...")
        if getattr(args, 'fsdp_activation_checkpointing', False):
            self._apply_activation_checkpointing(model)
        strategy_name = getattr(args, 'fsdp_sharding_strategy', 'FULL_SHARD')
        strategy = getattr(ShardingStrategy, strategy_name)
        is_hybrid = strategy_name in ['HYBRID_SHARD', '_HYBRID_SHARD_ZERO2']
        if is_hybrid:
            self.device_mesh = setup_hybrid_sharding(args)
        mp_policy = None
        if getattr(args, 'use_amp', False) and self.amp_config['supported']:
            mp_policy = MixedPrecision(
                param_dtype=self.amp_config['dtype'],
                reduce_dtype=self.amp_config['dtype'],
                buffer_dtype=self.amp_config['dtype'],
            )
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={EncoderLayer, DecoderLayer},
        )
        fsdp_kwargs = {
            'auto_wrap_policy': wrap_policy,
            'sharding_strategy': strategy,
            'mixed_precision': mp_policy,
            'device_id': torch.cuda.current_device(),
            'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
            'forward_prefetch': True,
            'limit_all_gathers': True,
            'use_orig_params': True,
        }
        if getattr(args, 'fsdp_cpu_offload', False):
            fsdp_kwargs['cpu_offload'] = CPUOffload(offload_params=True)
        if is_hybrid and self.device_mesh is not None:
            fsdp_kwargs['device_mesh'] = self.device_mesh
        model = FSDP(model, **fsdp_kwargs)
        if self._should_print():
            print(f"  Sharding Strategy: {strategy_name}")
            print(f"  Mixed Precision: "
                  f"{self.amp_config['name'] if mp_policy else 'Disabled'}")
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom, 'ECL': Dataset_Custom,
            'Solar': Dataset_Custom, 'custom': Dataset_Custom,
        }
        Data = data_dict.get(args.data, Dataset_Custom)
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle, drop_last, batch_size = False, True, args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle, drop_last, batch_size = False, False, 1
            Data = Dataset_Pred
            freq = getattr(args, 'detail_freq', args.freq)
        else:
            shuffle, drop_last, batch_size = True, True, args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=getattr(args, 'inverse', False),
            timeenc=timeenc,
            freq=freq,
            cols=getattr(args, 'cols', None),
            channel_period=getattr(args, 'channel_period', 1),
            stride=getattr(args, 'stride', 1),
        )

        if self._should_print():
            print(f'{flag} dataset: {len(data_set)} samples '
                  f'(stride={getattr(args, "stride", 1)})')

        num_workers = getattr(args, 'num_workers', 6)
        prefetch_factor = (getattr(args, 'prefetch_factor', 4)
                           if num_workers > 0 else None)
        use_persistent = (flag in ['train', 'val']) and num_workers > 0

        if getattr(args, 'use_fsdp', False):
            fsdp_drop_last = drop_last
            if flag in ['test', 'pred']:
                fsdp_drop_last = False

            sampler = DistributedSampler(
                data_set, shuffle=shuffle, drop_last=fsdp_drop_last)
            data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=False,
                sampler=sampler, num_workers=num_workers,
                drop_last=fsdp_drop_last, pin_memory=True,
                persistent_workers=use_persistent,
                prefetch_factor=prefetch_factor,
            )
        else:
            data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, drop_last=drop_last,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=use_persistent,
            )

        return data_set, data_loader

    def _select_optimizer(self):
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in [
                    'bias', 'layernorm', 'batchnorm',
                    'norm1', 'norm2', 'norm3', 'norm4',
                    'affine_weight', 'affine_bias']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        weight_decay = getattr(self.args, 'weight_decay', 0.01)
        if self._should_print():
            n_decay = sum(p.numel() for p in decay_params)
            n_no_decay = sum(p.numel() for p in no_decay_params)
            print(f"  AdamW param groups: {n_decay:,} decay, "
                  f"{n_no_decay:,} no-decay")
        return optim.AdamW(
            [{'params': decay_params, 'weight_decay': weight_decay},
             {'params': no_decay_params, 'weight_decay': 0.0}],
            lr=self.args.learning_rate,
        )

    def _select_criterion(self):
        return nn.MSELoss()

    # =========================================================================
    # GATHER PREDICTIONS ACROSS FSDP RANKS (unchanged)
    # =========================================================================
    def _gather_predictions(self, local_tensor, real_total_samples):
        if not (getattr(self.args, 'use_fsdp', False) and dist.is_initialized()):
            return local_tensor.cpu().numpy()

        world_size = dist.get_world_size()

        local_n = torch.tensor([local_tensor.shape[0]],
                               device=self.device, dtype=torch.long)
        all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
        dist.all_gather(all_n, local_n)
        max_n = max(n.item() for n in all_n)

        if local_tensor.shape[0] < max_n:
            pad_shape = list(local_tensor.shape)
            pad_shape[0] = max_n - local_tensor.shape[0]
            padding = torch.zeros(pad_shape, dtype=local_tensor.dtype,
                                  device=local_tensor.device)
            local_tensor = torch.cat([local_tensor, padding], dim=0)

        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor)

        all_preds = torch.cat(gathered, dim=0)

        total_gathered = max_n * world_size
        reordered = torch.empty_like(all_preds[:total_gathered])
        for r in range(world_size):
            n_r = all_n[r].item()
            for i in range(n_r):
                orig_idx = r + i * world_size
                if orig_idx < total_gathered:
                    reordered[orig_idx] = gathered[r][i]

        result = reordered[:real_total_samples]

        del gathered, all_preds, reordered

        if getattr(self.args, 'global_rank', 0) == 0:
            return result.cpu().numpy()
        else:
            del result
            return None

    # =========================================================================
    # VALIDATION — returns (scalar_loss, per_ch_mse)  (unchanged)
    # =========================================================================
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        cp = getattr(self.args, 'channel_period', 1)
        pred_len = self.args.pred_len

        total_loss = torch.tensor(0.0, device=self.device)
        per_ch_sq_sum = torch.zeros(cp, device=self.device)
        per_ch_count = torch.zeros(cp, device=self.device)
        n_batches = 0

        with torch.no_grad():
            for batch in vali_loader:
                pred, true = self._process_one_batch(vali_data, *batch)

                loss = criterion(pred[:, :, :1], true[:, :, :1])
                total_loss += loss.detach()
                n_batches += 1

                enc_phase_v = batch[4].long().to(self.device)
                k_idx = torch.arange(pred_len, device=self.device)
                channel_at_step = (
                    enc_phase_v.unsqueeze(1) + k_idx.unsqueeze(0)
                ) % cp

                sq_err = (
                    pred[:, :, :1].squeeze(-1) -
                    true[:, :, :1].squeeze(-1)
                ) ** 2

                for ch in range(cp):
                    mask = (channel_at_step == ch).float()
                    per_ch_sq_sum[ch] += (sq_err * mask).sum().detach()
                    per_ch_count[ch] += mask.sum().detach()

        avg_loss = total_loss / max(n_batches, 1)

        if getattr(self.args, 'use_fsdp', False) and dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(per_ch_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(per_ch_count, op=dist.ReduceOp.SUM)

        per_ch_mse = per_ch_sq_sum / per_ch_count.clamp(min=1.0)

        self.model.train()
        return avg_loss.float().item(), per_ch_mse.float()

    # =========================================================================
    # TRAINING — CHANGED: uniform loss + eval-mode diagnostic
    # =========================================================================
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if self._should_print() and not os.path.exists(path):
            os.makedirs(path)

        if getattr(self.args, 'use_fsdp', False) and dist.is_initialized():
            dist.barrier()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True,
            use_fsdp=getattr(self.args, 'use_fsdp', False),
            global_rank=getattr(self.args, 'global_rank', 0)
        )
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        cp = getattr(self.args, 'channel_period', 1)
        pred_len = self.args.pred_len

        grad_accum = self.gradient_accumulation_steps
        scheduler = None
        lradj = getattr(self.args, 'lradj', 'type1')
        _plateau_scheduler = False

        if lradj == 'cosine':
            steps_per_epoch = max(1, train_steps // grad_accum)
            total_steps = steps_per_epoch * self.args.train_epochs
            warmup_ratio = getattr(self.args, 'warmup_ratio', 0.05)
            warmup_steps = max(1, int(total_steps * warmup_ratio))
            cosine_steps = max(1, total_steps - warmup_steps)
            min_lr = self.args.learning_rate * 0.01
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0,
                total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=cosine_steps, eta_min=min_lr)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps])
        elif lradj == 'plateau':
            _plateau_factor = getattr(self.args, 'plateau_factor', 0.3)
            _plateau_patience = getattr(self.args, 'plateau_patience', 3)
            _min_lr = getattr(self.args, 'min_lr', 1e-6)
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=_plateau_factor,
                patience=_plateau_patience, min_lr=_min_lr, verbose=False)
            _plateau_scheduler = True

        use_amp = (getattr(self.args, 'use_amp', False)
                   and self.amp_config['supported'])
        use_scaler = use_amp and self.amp_config['use_scaler']
        if use_scaler:
            scaler = GradScaler('cuda') if _GRADSCALER_DEVICE else GradScaler()
        else:
            scaler = None

        for epoch in range(self.args.train_epochs):
            train_loss_sum = torch.tensor(0.0, device=self.device)
            train_per_ch_sq_sum = torch.zeros(cp, device=self.device)
            train_per_ch_count = torch.zeros(cp, device=self.device)
            train_loss_count = 0

            if (getattr(self.args, 'use_fsdp', False)
                    and hasattr(train_loader, 'sampler')):
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            self.perf_monitor.reset()
            epoch_start = time.time()
            iter_start = time.time()

            if self.use_prefetcher:
                data_iter = DataPrefetcher(train_loader, self.device)
            else:
                data_iter = train_loader

            for i, batch in enumerate(data_iter):
                data_time = time.time() - iter_start
                is_accum_step = (i + 1) % grad_accum != 0
                is_last_step = (i + 1) == train_steps
                should_sync = not is_accum_step or is_last_step

                if (getattr(self.args, 'use_fsdp', False)
                        and isinstance(self.model, FSDP)
                        and not should_sync):
                    sync_ctx = self.model.no_sync()
                else:
                    sync_ctx = nullcontext()

                fwd_start = time.time()
                with sync_ctx:
                    if self.use_prefetcher:
                        pred, true = self._process_one_batch_prefetched(
                            train_data, *batch)
                    else:
                        pred, true = self._process_one_batch(
                            train_data, *batch)

                    enc_phase_train = batch[4].long().to(
                        self.device, non_blocking=True)

                    k_idx = torch.arange(pred_len, device=self.device)
                    channel_at_step = (
                        enc_phase_train.unsqueeze(1) + k_idx.unsqueeze(0)
                    ) % cp                                           # (B, pred_len)

                    sq_err = (
                        pred[:, :, :1].squeeze(-1) -
                        true[:, :, :1].squeeze(-1)
                    ) ** 2                                           # (B, pred_len)

                    # Per-channel MSE for this batch
                    ch_mse_list = []
                    for ch in range(cp):
                        mask = (channel_at_step == ch).float()
                        denom = mask.sum().clamp(min=1.0)
                        ch_mse = (sq_err * mask).sum() / denom
                        ch_mse_list.append(ch_mse)
                    ch_mse_tensor = torch.stack(ch_mse_list)         # (cp,)

                    # ── CHANGE: Uniform loss — equal weight to all channels ──
                    loss = ch_mse_tensor.mean()

                    loss_scaled = loss / grad_accum
                    fwd_time = time.time() - fwd_start

                    with torch.no_grad():
                        train_loss_sum += loss.detach().float()
                        train_loss_count += 1
                        for ch in range(cp):
                            mask = (channel_at_step == ch).float()
                            train_per_ch_sq_sum[ch] += (
                                sq_err * mask).sum().detach()
                            train_per_ch_count[ch] += mask.sum().detach()

                    bwd_start = time.time()
                    if use_scaler:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    bwd_time = time.time() - bwd_start

                opt_time = 0
                if should_sync:
                    opt_start = time.time()
                    if use_scaler:
                        max_grad_norm = getattr(self.args, 'max_grad_norm', 0)
                        if max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            if isinstance(self.model, FSDP):
                                self.model.clip_grad_norm_(max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        max_grad_norm = getattr(self.args, 'max_grad_norm', 0)
                        if max_grad_norm > 0:
                            if isinstance(self.model, FSDP):
                                self.model.clip_grad_norm_(max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None and not _plateau_scheduler:
                        scheduler.step()
                    opt_time = time.time() - opt_start

                self.perf_monitor.update(
                    data_time, fwd_time, bwd_time, opt_time)

                if (i + 1) % 100 == 0 and self._should_print():
                    speed = (time.time() - epoch_start) / (i + 1)
                    eta = speed * (train_steps - i - 1)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1} | Iter {i+1}/{train_steps} | "
                          f"Loss: {loss.item():.6f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Speed: {speed:.2f}s/iter | ETA: {eta:.0f}s")

                iter_start = time.time()

            epoch_time = time.time() - epoch_start

            # ── Aggregate training losses ─────────────────────────────────
            train_loss = (train_loss_sum /
                          max(train_loss_count, 1)).item()
            train_per_ch_mse = (train_per_ch_sq_sum /
                                train_per_ch_count.clamp(min=1.0))

            if getattr(self.args, 'use_fsdp', False) and dist.is_initialized():
                dist.all_reduce(train_loss_sum, op=dist.ReduceOp.AVG)
                dist.all_reduce(train_per_ch_sq_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_per_ch_count, op=dist.ReduceOp.SUM)
                train_loss = (train_loss_sum /
                              max(train_loss_count, 1)).item()
                train_per_ch_mse = (train_per_ch_sq_sum /
                                    train_per_ch_count.clamp(min=1.0))

            # =================================================================
            # EVAL-MODE DIAGNOSTIC ON TRAINING DATA
            #
            # Purpose: isolate the effect of dropout noise on training loss.
            # Runs 50 batches of the same train_loader with model.eval()
            # (dropout off) and compares per-channel MSE side-by-side with
            # the train-mode (dropout on) per-channel MSE accumulated above.
            #
            # If the train-mode MSE is consistently higher than eval-mode MSE
            # on the SAME data, the gap is caused by dropout noise, not by
            # overfitting or underfitting.
            # =================================================================
            self.model.eval()
            eval_per_ch_sq_sum = torch.zeros(cp, device=self.device)
            eval_per_ch_count = torch.zeros(cp, device=self.device)

            with torch.no_grad():
                for diag_i, diag_batch in enumerate(train_loader):
                    if diag_i >= 50:
                        break
                    pred_diag, true_diag = self._process_one_batch(
                        train_data, *diag_batch)

                    enc_phase_diag = diag_batch[4].long().to(self.device)
                    k_idx_diag = torch.arange(pred_len, device=self.device)
                    ch_at_step_diag = (
                        enc_phase_diag.unsqueeze(1) + k_idx_diag.unsqueeze(0)
                    ) % cp

                    sq_err_diag = (
                        pred_diag[:, :, :1].squeeze(-1) -
                        true_diag[:, :, :1].squeeze(-1)
                    ) ** 2

                    for ch in range(cp):
                        mask_d = (ch_at_step_diag == ch).float()
                        eval_per_ch_sq_sum[ch] += (sq_err_diag * mask_d).sum()
                        eval_per_ch_count[ch] += mask_d.sum()

            if getattr(self.args, 'use_fsdp', False) and dist.is_initialized():
                dist.all_reduce(eval_per_ch_sq_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_per_ch_count, op=dist.ReduceOp.SUM)

            eval_per_ch_mse = (eval_per_ch_sq_sum /
                               eval_per_ch_count.clamp(min=1.0))

            self.model.train()
            # ── END EVAL-MODE DIAGNOSTIC ──────────────────────────────────

            vali_loss, per_ch_mse_val = self.vali(
                vali_data, vali_loader, criterion)
            test_loss, per_ch_mse_test = self.vali(
                test_data, test_loader, criterion)

            if self._should_print():
                print(f"\n  Epoch {epoch+1} Complete | Time: {epoch_time:.0f}s")
                print(f"    Train Loss: {train_loss:.7f}")
                print(f"    Vali Loss:  {vali_loss:.7f}")
                print(f"    Test Loss:  {test_loss:.7f}")
                print(f"    Train per-channel MSE:")
                for ch in range(cp):
                    print(f"      ch{ch}: {train_per_ch_mse[ch].item():.6f}")
                print(f"    Eval-mode MSE on train data (50 batches):")
                print(f"      {'ch':<6} {'train-mode':<14} "
                      f"{'eval-mode':<14} {'ratio(T/E)':<12}")
                for ch in range(cp):
                    t_mse = train_per_ch_mse[ch].item()
                    e_mse = eval_per_ch_mse[ch].item()
                    ratio = t_mse / e_mse if e_mse > 0 else float('inf')
                    print(f"      ch{ch:<3} {t_mse:<14.6f} "
                          f"{e_mse:<14.6f} {ratio:<12.4f}")
                print(f"    Val per-channel MSE:")
                for ch in range(cp):
                    print(f"      ch{ch}: {per_ch_mse_val[ch].item():.6f}")
                print(f"    Test per-channel MSE:")
                for ch in range(cp):
                    print(f"      ch{ch}: {per_ch_mse_test[ch].item():.6f}")
                print(f"    Timing: {self.perf_monitor.summary()}\n")

            # ── Test-best checkpoint ──────────────────────────────────────
            if not hasattr(self, '_best_test_loss'):
                self._best_test_loss = float('inf')
                self._best_test_epoch = 0

            if test_loss < self._best_test_loss:
                self._best_test_loss = test_loss
                self._best_test_epoch = epoch + 1
                test_best_path = path + '_testbest'
                if self._should_print():
                    print(f'  Test loss improved --> {test_loss:.7f} '
                          f'(epoch {self._best_test_epoch}). '
                          f'Saving test-best model...')
                    if not os.path.exists(test_best_path):
                        os.makedirs(test_best_path)
                if (getattr(self.args, 'use_fsdp', False)
                        and isinstance(self.model, FSDP)):
                    with FSDP.state_dict_type(
                        self.model, StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True,
                                            rank0_only=True)
                    ):
                        state_dict = self.model.state_dict()
                        if self._should_print():
                            torch.save(state_dict,
                                       test_best_path + '/checkpoint.pth')
                else:
                    if self._should_print():
                        state = (self.model.module.state_dict()
                                 if isinstance(self.model, nn.DataParallel)
                                 else self.model.state_dict())
                        torch.save(state, test_best_path + '/checkpoint.pth')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if self._should_print():
                    print("Early stopping triggered")
                break

            if _plateau_scheduler:
                prev_lr = optimizer.param_groups[0]['lr']
                scheduler.step(vali_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if self._should_print():
                    if new_lr < prev_lr:
                        print(f"    ReduceLROnPlateau: LR reduced "
                              f"{prev_lr:.2e} → {new_lr:.2e}")
                    else:
                        print(f"    Current LR: {new_lr:.2e}")
            elif scheduler is None:
                adjust_learning_rate(optimizer, epoch + 1, self.args)
            else:
                if self._should_print():
                    print(f"    Current LR: "
                          f"{optimizer.param_groups[0]['lr']:.2e}")

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self._load_checkpoint(best_model_path)

        return self.model

    # =========================================================================
    # BATCH PROCESSING (unchanged)
    # =========================================================================
    def _process_one_batch(self, dataset_object,
                           batch_x, batch_y, batch_x_mark, batch_y_mark,
                           enc_phase, dec_phase):
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
        enc_phase = enc_phase.long().to(self.device, non_blocking=True)
        dec_phase = dec_phase.long().to(self.device, non_blocking=True)

        if getattr(self.args, 'padding', 0) == 0:
            dec_inp = torch.zeros(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]],
                dtype=batch_y.dtype)
        else:
            dec_inp = torch.ones(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]],
                dtype=batch_y.dtype)

        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
        dec_inp = dec_inp.to(self.device, non_blocking=True)

        if getattr(self.args, 'output_attention', False):
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                enc_phase=enc_phase, dec_phase=dec_phase)[0]
        else:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                enc_phase=enc_phase, dec_phase=dec_phase)

        if getattr(self.args, 'inverse', False):
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(
            self.device, non_blocking=True)

        if outputs.shape[-1] < batch_y.shape[-1]:
            batch_y = batch_y[:, :, :outputs.shape[-1]]

        if outputs.dtype != batch_y.dtype:
            batch_y = batch_y.to(outputs.dtype)

        return outputs, batch_y

    def _process_one_batch_prefetched(self, dataset_object,
                                      batch_x, batch_y, batch_x_mark,
                                      batch_y_mark, enc_phase, dec_phase):
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        enc_phase = enc_phase.long()
        dec_phase = dec_phase.long()

        if getattr(self.args, 'padding', 0) == 0:
            dec_inp = torch.zeros(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]],
                device=self.device, dtype=batch_y.dtype)
        else:
            dec_inp = torch.ones(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]],
                device=self.device, dtype=batch_y.dtype)

        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

        if getattr(self.args, 'output_attention', False):
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                enc_phase=enc_phase, dec_phase=dec_phase)[0]
        else:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                enc_phase=enc_phase, dec_phase=dec_phase)

        if getattr(self.args, 'inverse', False):
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        true = batch_y[:, -self.args.pred_len:, f_dim:]

        if outputs.shape[-1] < true.shape[-1]:
            true = true[:, :, :outputs.shape[-1]]

        if outputs.dtype != true.dtype:
            true = true.to(outputs.dtype)

        return outputs, true

    # =========================================================================
    # TEST (unchanged)
    # =========================================================================
    def test(self, setting, load=False):
        test_data, test_loader = self._get_data(flag='test')

        if load:
            checkpoint_path = os.path.join(
                self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)

        self.model.eval()

        preds_list, trues_list = [], []
        preds_list_to_save, trues_list_to_save = [], []

        with torch.no_grad():
            for batch in test_loader:
                pred, true = self._process_one_batch(test_data, *batch)
                preds_list.append(pred[:, :, :1].detach().float())
                trues_list.append(true[:, :, :1].detach().float())
                preds_list_to_save.append(pred.detach().float())
                trues_list_to_save.append(true.detach().float())

        local_preds = torch.cat(preds_list, dim=0)
        local_trues = torch.cat(trues_list, dim=0)
        local_preds_to_save = torch.cat(preds_list_to_save, dim=0)
        local_trues_to_save = torch.cat(trues_list_to_save, dim=0)

        del preds_list, trues_list, preds_list_to_save, trues_list_to_save

        real_total_samples = len(test_data)
        use_fsdp = getattr(self.args, 'use_fsdp', False)

        if use_fsdp and dist.is_initialized():
            preds = self._gather_predictions(local_preds, real_total_samples)
            trues = self._gather_predictions(local_trues, real_total_samples)
            preds_to_save = self._gather_predictions(
                local_preds_to_save, real_total_samples)
            trues_to_save = self._gather_predictions(
                local_trues_to_save, real_total_samples)
            del local_preds, local_trues, local_preds_to_save, local_trues_to_save
        else:
            preds = local_preds.cpu().numpy()
            trues = local_trues.cpu().numpy()
            preds_to_save = local_preds_to_save.cpu().numpy()
            trues_to_save = local_trues_to_save.cpu().numpy()
            del local_preds, local_trues, local_preds_to_save, local_trues_to_save

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._should_print():
            print(f'Test shape: {preds.shape}, {trues.shape}')
            print(f'Test samples: {preds.shape[0]} '
                  f'(real dataset size: {real_total_samples})')

            mae, mse, rmse, mape, mspe = metric(preds, trues)

            print(f'\nTest Results (global, all ranks gathered):')
            print(f'  MSE:  {mse:.7f}')
            print(f'  MAE:  {mae:.7f}')
            print(f'  RMSE: {rmse:.7f}')

            folder = f'./results/{setting}/'
            os.makedirs(folder, exist_ok=True)
            np.save(f'{folder}metrics.npy',
                    np.array([mae, mse, rmse, mape, mspe]))

            cp = getattr(self.args, 'channel_period', 1)
            stride = getattr(self.args, 'stride', 1)
            world_size = getattr(self.args, 'world_size', 1)
            border1 = test_data.border1
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            N_saved = len(preds)

            row_step = stride
            j_idx = np.arange(N_saved, dtype=np.int64)
            k_idx = np.arange(pred_len, dtype=np.int64)
            abs_rows = (border1
                        + j_idx[:, None] * row_step
                        + seq_len
                        + k_idx[None, :])
            channel_indices = (abs_rows % cp).astype(np.int32)

            has_per_channel_scaler = (
                hasattr(test_data, 'scaler')
                and test_data.scaler is not None
                and hasattr(test_data.scaler, 'channel_period')
                and test_data.scaler.channel_period > 1
                and test_data.scaler.mean_channels is not None
            )

            if has_per_channel_scaler:
                pred_inv = test_data.scaler.inverse_transform(
                    preds_to_save, channel_indices)
                true_inv = test_data.scaler.inverse_transform(
                    trues_to_save, channel_indices)
                np.save(f'{folder}pred.npy', pred_inv)
                np.save(f'{folder}true.npy', true_inv)
            elif (hasattr(test_data, 'scaler')
                  and test_data.scaler is not None):
                c = preds_to_save.shape[-1]
                if c == len(test_data.scaler.mean):
                    np.save(f'{folder}pred.npy',
                            test_data.scaler.inverse_transform(preds_to_save))
                    np.save(f'{folder}true.npy',
                            test_data.scaler.inverse_transform(trues_to_save))
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    mean_s = (test_data.scaler.mean[f_dim:f_dim + c]
                              if f_dim >= 0
                              else test_data.scaler.mean[f_dim:])
                    std_s = (test_data.scaler.std[f_dim:f_dim + c]
                             if f_dim >= 0
                             else test_data.scaler.std[f_dim:])
                    np.save(f'{folder}pred.npy', preds_to_save * std_s + mean_s)
                    np.save(f'{folder}true.npy', trues_to_save * std_s + mean_s)
            else:
                np.save(f'{folder}pred.npy', preds_to_save)
                np.save(f'{folder}true.npy', trues_to_save)

            np.save(f'{folder}channel_indices.npy', channel_indices)

            metadata = {
                'setting': setting,
                'border1': int(border1),
                'channel_period': int(cp),
                'stride': int(stride),
                'seq_len': int(seq_len),
                'label_len': int(self.args.label_len),
                'pred_len': int(pred_len),
                'n_saved_samples': int(N_saved),
                'use_fsdp': bool(use_fsdp),
                'world_size': int(world_size),
                'predictions_gathered': bool(use_fsdp and dist.is_initialized()),
                'row_step_between_saved_samples': int(row_step),
                'per_channel_inverse_applied': bool(has_per_channel_scaler),
                'pred_npy_units': (
                    'raw_original' if has_per_channel_scaler
                    else 'globally_scaled'),
                'channel_index_formula': (
                    '(border1 + j * stride + seq_len + k) % channel_period'),
            }
            with open(f'{folder}metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f'\n  channel_indices.npy: shape={channel_indices.shape}')
            print(f'  Per-channel inverse applied: {has_per_channel_scaler}')
            print(f'  pred.npy units: {metadata["pred_npy_units"]}')
            print(f'  Predictions gathered from all ranks: '
                  f'{metadata["predictions_gathered"]}')
            print(f'  Saved to: {folder}')
        else:
            mae, mse, rmse = 0.0, 0.0, 0.0

        if use_fsdp and dist.is_initialized():
            metrics_tensor = torch.tensor(
                [mse, mae], device=self.device, dtype=torch.float64)
            dist.broadcast(metrics_tensor, src=0)
            mse = metrics_tensor[0].item()
            mae = metrics_tensor[1].item()

        del test_loader, test_data
        gc.collect()

        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            checkpoint_path = os.path.join(
                self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)

        self.model.eval()
        preds_list = []

        with torch.no_grad():
            for batch in pred_loader:
                pred, _ = self._process_one_batch(pred_data, *batch)
                preds_list.append(pred.detach().float())

        local_preds = torch.cat(preds_list, dim=0)
        del preds_list

        use_fsdp = getattr(self.args, 'use_fsdp', False)
        real_total_samples = len(pred_data)

        if use_fsdp and dist.is_initialized():
            preds = self._gather_predictions(local_preds, real_total_samples)
            del local_preds
        else:
            preds = local_preds.cpu().numpy()
            del local_preds

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._should_print() and preds is not None:
            folder = f'./results/{setting}/'
            os.makedirs(folder, exist_ok=True)
            if hasattr(pred_data, 'scaler') and pred_data.scaler is not None:
                np.save(f'{folder}real_prediction.npy',
                        pred_data.scaler.inverse_transform(preds))
            else:
                np.save(f'{folder}real_prediction.npy', preds)

        return preds

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            if self._should_print():
                print(f"Checkpoint not found: {path}")
            return
        if self._should_print():
            print(f"Loading checkpoint: {path}")
        if (getattr(self.args, 'use_fsdp', False)
                and isinstance(self.model, FSDP)):
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                if getattr(self.args, 'global_rank', 0) == 0:
                    state_dict = torch.load(path, map_location='cpu')
                else:
                    state_dict = None
                if dist.is_initialized():
                    object_list = [state_dict]
                    dist.broadcast_object_list(object_list, src=0)
                    state_dict = object_list[0]
                if state_dict is not None:
                    self.model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(path, map_location=self.device)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

    def _should_print(self):
        if hasattr(self.args, 'use_fsdp') and self.args.use_fsdp:
            return getattr(self.args, 'global_rank', 0) == 0
        return True
