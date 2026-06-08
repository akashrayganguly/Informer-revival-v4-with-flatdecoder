"""
exp_informer.py - Informer Experiment Class with FSDP Support

Inherits from Exp_Basic and adds:
- FSDP model wrapping with HYBRID_SHARD support
- Mixed precision training (BF16/FP16 auto-detection)
- Gradient accumulation with no_sync optimization
- Data prefetching with CUDA streams
- Activation checkpointing
- FlashAttention integration

CRITICAL FIXES:
1. Proper inheritance from Exp_Basic
2. Fixed _process_one_batch_prefetched bug (batch_y_gpu -> batch_y)
3. Model parameters match model.py signature exactly
4. FSDP checkpoint saving/loading with collective operations
"""

import os
import time
import warnings
import inspect
import numpy as np
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

# Activation checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.cuda.amp import GradScaler

# Local imports
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


# =============================================================================
# CUDA STREAM DATA PREFETCHER
# =============================================================================
class DataPrefetcher:
    """
    Prefetches data to GPU using a separate CUDA stream.
    Overlaps CPU→GPU data transfer with GPU computation.
    """

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
                item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item
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
# AMP CONFIGURATION
# =============================================================================
def get_amp_config():
    """Auto-detect best mixed precision configuration for current hardware."""
    if not torch.cuda.is_available():
        return {'supported': False, 'dtype': torch.float32, 'use_scaler': False, 'name': 'FP32'}

    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return {'supported': True, 'dtype': torch.bfloat16, 'use_scaler': False, 'name': 'BF16'}

    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:
        return {'supported': True, 'dtype': torch.float16, 'use_scaler': True, 'name': 'FP16'}

    return {'supported': False, 'dtype': torch.float32, 'use_scaler': False, 'name': 'FP32'}


# =============================================================================
# GPU DETECTION FOR HYBRID SHARDING
# =============================================================================
def get_num_gpus_per_node():
    """Detect number of GPUs per node from environment or hardware."""
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE')
    if local_world_size:
        return int(local_world_size)
    slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
    if slurm_gpus:
        return int(slurm_gpus)
    return torch.cuda.device_count()


def setup_hybrid_sharding(args):
    """Setup DeviceMesh for HYBRID_SHARD strategy."""
    if not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_gpus_per_node = get_num_gpus_per_node()
    num_nodes = world_size // num_gpus_per_node

    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"HYBRID SHARDING SETUP")
        print(f"  World size: {world_size}")
        print(f"  Nodes: {num_nodes}")
        print(f"  GPUs/node: {num_gpus_per_node}")
        print(f"{'='*60}")

    if num_nodes <= 1:
        if global_rank == 0:
            print("  Single node detected - HYBRID_SHARD equivalent to FULL_SHARD")
        return None

    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda",
            (num_nodes, num_gpus_per_node),
            mesh_dim_names=("replicate", "shard")
        )

        if global_rank == 0:
            print(f"  DeviceMesh created: ({num_nodes}, {num_gpus_per_node})")
            print(f"  Strategy: Shard within node, replicate across nodes")

        return mesh
    except Exception as e:
        if global_rank == 0:
            print(f"  DeviceMesh creation failed: {e}")
            print(f"  Falling back to standard FSDP")
        return None


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================
class PerformanceMonitor:
    """Track timing breakdown of training loop components."""
    
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
        total = self.data_time + self.forward_time + self.backward_time + self.optimizer_time
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
    """
    Informer Experiment with FSDP support and performance optimizations.
    
    Inherits from Exp_Basic for device management.
    
    Features:
    - FSDP with HYBRID_SHARD support for multi-node training
    - Mixed precision (BF16/FP16) with auto-detection
    - Gradient accumulation with communication optimization
    - CUDA stream data prefetching
    - Activation checkpointing for memory efficiency
    """

    def __init__(self, args):
        """
        Initialize experiment.
        
        Args:
            args: Configuration object with all hyperparameters
        """
        # Initialize performance settings BEFORE calling parent __init__
        # (because parent calls _build_model which needs these)
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.amp_config = get_amp_config()
        self.device_mesh = None
        self.perf_monitor = PerformanceMonitor()
        self.use_prefetcher = getattr(args, 'use_prefetcher', True) and torch.cuda.is_available()
        
        # Call parent __init__ which:
        # 1. Calls _acquire_device()
        # 2. Calls _build_model().to(self.device)
        super(Exp_Informer, self).__init__(args)
        
        # Print configuration (rank 0 only)
        if self._should_print():
            self._print_config()

    def _print_config(self):
        """Print experiment configuration."""
        args = self.args
        print(f"\n{'='*60}")
        print(f"EXPERIMENT CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Model: {args.model}")
        print(f"  Data: {args.data}")
        print(f"  Seq/Label/Pred: {args.seq_len}/{args.label_len}/{args.pred_len}")
        print(f"  d_model: {args.d_model}, n_heads: {args.n_heads}")
        print(f"  E-layers: {args.e_layers}, D-layers: {args.d_layers}")
        print(f"")
        print(f"  FSDP: {getattr(args, 'use_fsdp', False)}")
        if getattr(args, 'use_fsdp', False):
            print(f"  Sharding Strategy: {getattr(args, 'fsdp_sharding_strategy', 'FULL_SHARD')}")
            print(f"  World Size: {getattr(args, 'world_size', 1)}")
        print(f"  AMP: {self.amp_config['name']}")
        print(f"  Grad Accumulation: {self.gradient_accumulation_steps}")
        print(f"  Data Prefetcher: {self.use_prefetcher}")
        print(f"  Channel Mix Size: {getattr(args, 'channel_mix_size', None)}")
        print(f"  Channel Period: {getattr(args, 'channel_period', 321)}")
        print(f"  Use RevIN: {getattr(args, 'use_revin', True)}")
        print(f"  Device: {self.device}")
        print(f"")
        print(f"  FAITHFUL VECTOR LOSS:")
        print(f"    Backprop: Full vector (all features)")
        print(f"    Reported: First element only (:,:,:1)")
        print(f"{'='*60}\n")

    def _build_model(self):
        """
        Build the Informer model with FSDP wrapping if enabled.
        
        Supports BOTH:
        - Original model.py (without use_revin, channel_mix_size, channel_period, max_len)
        - Modified model.py (with those extra parameters)
        
        Note: Parent class calls .to(self.device) after this returns,
        which is fine for FSDP (it handles device placement internally).
        """
        args = self.args
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack
        }
        
        if args.model not in model_dict:
            raise ValueError(f"Unknown model: {args.model}. Choose from {list(model_dict.keys())}")
        
        ModelClass = model_dict[args.model]
        
        # For InformerStack, e_layers should be a list
        if args.model == 'informerstack':
            e_layers = args.s_layers if hasattr(args, 's_layers') else [3, 2, 1]
        else:
            e_layers = args.e_layers
        
        # Base parameters (always present in model.py)
        base_kwargs = {
            'enc_in': args.enc_in,
            'dec_in': args.dec_in,
            'c_out': args.c_out,
            'seq_len': args.seq_len,
            'label_len': args.label_len,
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
        
        # Check if model supports extended parameters by inspecting signature
        import inspect
        sig = inspect.signature(ModelClass.__init__)
        model_params = list(sig.parameters.keys())
        
        # Extended parameters (may not exist in original model.py)
        extended_kwargs = {}
        if 'use_revin' in model_params:
            extended_kwargs['use_revin'] = getattr(args, 'use_revin', True)
        if 'channel_mix_size' in model_params:
            extended_kwargs['channel_mix_size'] = getattr(args, 'channel_mix_size', None)
        if 'channel_period' in model_params:
            extended_kwargs['channel_period'] = getattr(args, 'channel_period', 321)
        if 'max_len' in model_params:
            extended_kwargs['max_len'] = getattr(args, 'max_len', 200000)
        
        # Merge kwargs
        model_kwargs = {**base_kwargs, **extended_kwargs}
        
        # Build model
        model = ModelClass(**model_kwargs).float()
        
        if self._should_print():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Apply FSDP or DataParallel wrapping
        if getattr(args, 'use_fsdp', False) and torch.cuda.is_available():
            model = self._wrap_model_with_fsdp(model)
        elif getattr(args, 'use_multi_gpu', False) and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        
        return model

    def _apply_activation_checkpointing(self, model):
        """Apply activation checkpointing to encoder and decoder layers."""
        check_fn = lambda m: isinstance(m, (EncoderLayer, DecoderLayer))
        wrapper = functools.partial(
            checkpoint_wrapper, 
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            model, 
            checkpoint_wrapper_fn=wrapper, 
            check_fn=check_fn
        )

        if self._should_print():
            enc_count = sum(1 for m in model.modules() if isinstance(m, EncoderLayer))
            dec_count = sum(1 for m in model.modules() if isinstance(m, DecoderLayer))
            print(f"  Activation Checkpointing: {enc_count} encoder, {dec_count} decoder layers")

    def _wrap_model_with_fsdp(self, model):
        """
        Wrap model with FSDP for distributed training.
        
        Supports:
        - FULL_SHARD: Full parameter sharding (max memory savings)
        - SHARD_GRAD_OP: Shard gradients and optimizer states
        - HYBRID_SHARD: Shard within node, replicate across nodes
        - NO_SHARD: No sharding (equivalent to DDP)
        """
        args = self.args
        
        if self._should_print():
            print(f"\nWrapping model with FSDP...")

        # Apply activation checkpointing BEFORE FSDP wrapping
        if getattr(args, 'fsdp_activation_checkpointing', False):
            self._apply_activation_checkpointing(model)

        # Get sharding strategy
        strategy_name = getattr(args, 'fsdp_sharding_strategy', 'FULL_SHARD')
        strategy = getattr(ShardingStrategy, strategy_name)
        is_hybrid = strategy_name in ['HYBRID_SHARD', '_HYBRID_SHARD_ZERO2']

        # Setup DeviceMesh for hybrid sharding
        if is_hybrid:
            self.device_mesh = setup_hybrid_sharding(args)

        # Setup mixed precision policy
        mp_policy = None
        if getattr(args, 'use_amp', False) and self.amp_config['supported']:
            mp_policy = MixedPrecision(
                param_dtype=self.amp_config['dtype'],
                reduce_dtype=self.amp_config['dtype'],
                buffer_dtype=self.amp_config['dtype'],
            )

        # Auto-wrap policy: each EncoderLayer/DecoderLayer is one FSDP unit
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={EncoderLayer, DecoderLayer},
        )

        # Build FSDP kwargs
        fsdp_kwargs = {
            'auto_wrap_policy': wrap_policy,
            'sharding_strategy': strategy,
            'mixed_precision': mp_policy,
            'device_id': torch.cuda.current_device(),
            'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
            'forward_prefetch': True,
            'limit_all_gathers': True,
            'use_orig_params': True,  # Required for nn.Parameter (channel mixing W)
        }

        # Add CPU offload if enabled
        if getattr(args, 'fsdp_cpu_offload', False):
            fsdp_kwargs['cpu_offload'] = CPUOffload(offload_params=True)

        # Add DeviceMesh for hybrid sharding
        if is_hybrid and self.device_mesh is not None:
            fsdp_kwargs['device_mesh'] = self.device_mesh

        # Wrap model
        model = FSDP(model, **fsdp_kwargs)

        if self._should_print():
            print(f"  Sharding Strategy: {strategy_name}")
            print(f"  DeviceMesh: {'Yes' if self.device_mesh else 'No'}")
            print(f"  Mixed Precision: {self.amp_config['name'] if mp_policy else 'Disabled'}")
            print(f"  Forward Prefetch: Yes")
            print(f"  use_orig_params: True")

        return model

    def _get_data(self, flag):
        """
        Get dataset and dataloader for specified split.
        
        Args:
            flag: 'train', 'val', 'test', or 'pred'
            
        Returns:
            data_set: Dataset instance
            data_loader: DataLoader instance
        """
        args = self.args

        # Dataset mapping
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        
        Data = data_dict.get(args.data, Dataset_Custom)
        timeenc = 0 if args.embed != 'timeF' else 1

        # Configure based on split
        if flag == 'test':
            shuffle, drop_last, batch_size = False, True, args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle, drop_last, batch_size = False, False, 1
            Data = Dataset_Pred
            freq = getattr(args, 'detail_freq', args.freq)
        else:  # train or val
            shuffle, drop_last, batch_size = True, True, args.batch_size
            freq = args.freq

        # Create dataset
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
            cols=getattr(args, 'cols', None)
        )

        if self._should_print():
            print(f'{flag} dataset: {len(data_set)} samples')

        # DataLoader settings
        num_workers = getattr(args, 'num_workers', 4)
        prefetch_factor = getattr(args, 'prefetch_factor', 4) if num_workers > 0 else None

        # Create dataloader with FSDP support
        if getattr(args, 'use_fsdp', False):
            sampler = DistributedSampler(
                data_set,
                shuffle=shuffle,
                drop_last=drop_last
            )
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,  # Sampler handles shuffling
                sampler=sampler,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=prefetch_factor,
            )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=True if torch.cuda.is_available() else False,
            )

        return data_set, data_loader

    def _select_optimizer(self):
        """Select optimizer."""
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """Select loss criterion."""
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        """
        Validation loop.
        
        Uses "faithful vector" loss: reports loss only on first element (:,:,:1)
        
        OPTIMIZED: 
        - All loss computation stays on GPU
        - No .float() conversion (BF16 compatible)
        - Single GPU→CPU transfer at the end
        
        Args:
            vali_data: Validation dataset
            vali_loader: Validation dataloader
            criterion: Loss function
            
        Returns:
            Average validation loss (on first element only)
        """
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        with torch.no_grad():
            for batch in vali_loader:
                pred, true = self._process_one_batch(vali_data, *batch)
                # Faithful vector: report loss only on first element
                # No .float() needed - criterion works with BF16
                loss = criterion(pred[:, :, :1], true[:, :, :1])
                total_loss += loss.detach()
                count += 1

        avg_loss = total_loss / max(count, 1)
        
        # Synchronize loss across ranks in FSDP mode
        if getattr(self.args, 'use_fsdp', False) and dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

        self.model.train()
        # Single GPU→CPU transfer here
        return avg_loss.item()

    def train(self, setting):
        """
        Main training loop with FSDP support.
        
        FAITHFUL VECTOR LOSS:
        - Backpropagation: Uses full vector loss (all features)
        - Reporting: Uses first element loss only (:,:,:1)
        
        Args:
            setting: Experiment setting string for checkpointing
            
        Returns:
            Trained model
        """
        # Get data
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Create checkpoint directory
        path = os.path.join(self.args.checkpoints, setting)
        if self._should_print() and not os.path.exists(path):
            os.makedirs(path)

        # Synchronize before training
        if getattr(self.args, 'use_fsdp', False) and dist.is_initialized():
            dist.barrier()

        train_steps = len(train_loader)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            use_fsdp=getattr(self.args, 'use_fsdp', False),
            global_rank=getattr(self.args, 'global_rank', 0)
        )

        # Initialize optimizer and criterion
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        # Setup AMP
        use_amp = getattr(self.args, 'use_amp', False) and self.amp_config['supported']
        use_scaler = use_amp and self.amp_config['use_scaler']
        scaler = GradScaler() if use_scaler else None

        grad_accum = self.gradient_accumulation_steps

        if self._should_print():
            print(f"\n{'='*60}")
            print(f"TRAINING START")
            print(f"{'='*60}")
            print(f"  Steps per epoch: {train_steps}")
            print(f"  Gradient accumulation: {grad_accum}")
            print(f"  Data prefetcher: {self.use_prefetcher}")
            print(f"  AMP enabled: {use_amp}")
            print(f"  Loss: Full vector for backprop, first element for reporting")
            print(f"{'='*60}\n")

        # Training loop
        for epoch in range(self.args.train_epochs):
            # Track both full loss and first-element loss
            train_loss_full_sum = torch.tensor(0.0, device=self.device)  # For debugging
            train_loss_first_sum = torch.tensor(0.0, device=self.device)  # For reporting
            train_loss_count = 0

            # Set epoch for distributed sampler
            if getattr(self.args, 'use_fsdp', False) and hasattr(train_loader, 'sampler'):
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            self.perf_monitor.reset()
            epoch_start = time.time()
            iter_start = time.time()

            # Use data prefetcher if enabled
            if self.use_prefetcher:
                data_iter = DataPrefetcher(train_loader, self.device)
            else:
                data_iter = train_loader

            for i, batch in enumerate(data_iter):
                data_time = time.time() - iter_start

                # Determine if this is an accumulation step
                is_accum_step = (i + 1) % grad_accum != 0
                is_last_step = (i + 1) == train_steps
                should_sync = not is_accum_step or is_last_step

                # Use no_sync context for gradient accumulation (FSDP optimization)
                if getattr(self.args, 'use_fsdp', False) and isinstance(self.model, FSDP) and not should_sync:
                    sync_ctx = self.model.no_sync()
                else:
                    sync_ctx = nullcontext()

                fwd_start = time.time()

                with sync_ctx:
                    # Forward pass
                    if self.use_prefetcher:
                        pred, true = self._process_one_batch_prefetched(train_data, *batch)
                    else:
                        pred, true = self._process_one_batch(train_data, *batch)

                    # ==========================================================
                    # FAITHFUL VECTOR LOSS:
                    # - loss: Full vector (used for backpropagation)
                    # - actual_loss: First element only (used for reporting)
                    # No .float() needed - criterion works with BF16
                    # ==========================================================
                    loss = criterion(pred, true)  # Full vector
                    actual_loss = criterion(pred[:, :, :1], true[:, :, :1])  # First element
                    
                    loss_scaled = loss / grad_accum

                    fwd_time = time.time() - fwd_start

                    # Track losses (no gradient)
                    with torch.no_grad():
                        train_loss_full_sum += loss.detach()
                        train_loss_first_sum += actual_loss.detach()
                        train_loss_count += 1

                    # Backward pass (uses full vector loss)
                    bwd_start = time.time()
                    if use_scaler:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    bwd_time = time.time() - bwd_start

                # Optimizer step (only on sync steps)
                opt_time = 0
                if should_sync:
                    opt_start = time.time()
                    
                    if use_scaler:
                        # Gradient clipping with scaler
                        max_grad_norm = getattr(self.args, 'max_grad_norm', 0)
                        if max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            if isinstance(self.model, FSDP):
                                self.model.clip_grad_norm_(max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Gradient clipping without scaler
                        max_grad_norm = getattr(self.args, 'max_grad_norm', 0)
                        if max_grad_norm > 0:
                            if isinstance(self.model, FSDP):
                                self.model.clip_grad_norm_(max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    opt_time = time.time() - opt_start

                self.perf_monitor.update(data_time, fwd_time, bwd_time, opt_time)

                # Logging (report first-element loss, not full vector loss)
                if (i + 1) % 100 == 0 and self._should_print():
                    speed = (time.time() - epoch_start) / (i + 1)
                    eta = speed * (train_steps - i - 1)
                    # Report actual_loss (first element) as per faithful vector convention
                    print(f"  Epoch {epoch+1} | Iter {i+1}/{train_steps} | "
                          f"Loss (1st elem): {actual_loss.item():.7f} | "
                          f"Loss (full): {loss.item():.7f} | "
                          f"Speed: {speed:.2f}s/iter | ETA: {eta:.0f}s")
                    print(f"    Timing: {self.perf_monitor.summary()}")

                iter_start = time.time()

            # Epoch end - compute metrics
            epoch_time = time.time() - epoch_start
            
            # Report first-element loss (faithful vector convention)
            train_loss = (train_loss_first_sum / max(train_loss_count, 1)).item()
            train_loss_full = (train_loss_full_sum / max(train_loss_count, 1)).item()

            # Synchronize training loss across ranks
            if getattr(self.args, 'use_fsdp', False) and dist.is_initialized():
                loss_tensor = torch.tensor(train_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = loss_tensor.item()
                
                loss_full_tensor = torch.tensor(train_loss_full, device=self.device)
                dist.all_reduce(loss_full_tensor, op=dist.ReduceOp.AVG)
                train_loss_full = loss_full_tensor.item()

            # Validation (uses first-element loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            if self._should_print():
                print(f"\n  Epoch {epoch+1} Complete | Time: {epoch_time:.0f}s")
                print(f"    Train Loss (1st elem): {train_loss:.7f}")
                print(f"    Train Loss (full vec): {train_loss_full:.7f}")
                print(f"    Vali Loss:  {vali_loss:.7f}")
                print(f"    Test Loss:  {test_loss:.7f}")
                print(f"    Timing: {self.perf_monitor.summary()}\n")

            # Early stopping (uses first-element vali loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if self._should_print():
                    print("Early stopping triggered")
                break

            # Learning rate adjustment
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        # Load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self._load_checkpoint(best_model_path)

        return self.model

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Process one batch (data on CPU, transfer to GPU).
        
        OPTIMIZED: No unnecessary .float() conversions - model handles dtype via AMP.
        
        Args:
            dataset_object: Dataset for inverse transform
            batch_x: Encoder input [B, seq_len, features]
            batch_y: Decoder target [B, label_len + pred_len, features]
            batch_x_mark: Encoder time features
            batch_y_mark: Decoder time features
            
        Returns:
            outputs: Model predictions [B, pred_len, c_out]
            batch_y: Ground truth [B, pred_len, c_out]
        """
        # Move to device - use .float() only here for input consistency
        # The model's AMP context will handle BF16 conversion internally
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        batch_y = batch_y.float()  # Keep on CPU for now, only move target slice later
        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

        # Prepare decoder input (label + zeros for prediction)
        if getattr(self.args, 'padding', 0) == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], 
                                  dtype=batch_y.dtype)
        else:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], 
                                 dtype=batch_y.dtype)

        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
        dec_inp = dec_inp.to(self.device, non_blocking=True)

        # Forward pass - model handles AMP/BF16 internally
        if getattr(self.args, 'output_attention', False):
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # Inverse transform if needed
        if getattr(self.args, 'inverse', False):
            outputs = dataset_object.inverse_transform(outputs)

        # Get target for loss - move only the slice we need
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=True)

        return outputs, batch_y

    def _process_one_batch_prefetched(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Process one batch (data already on GPU from prefetcher).
        
        OPTIMIZED: 
        - Data already on GPU from prefetcher
        - No unnecessary .float() conversions - model handles dtype via AMP
        
        FIXED: Previous version had undefined 'batch_y_gpu' variable.
        """
        # Data is already on GPU from prefetcher
        # Only ensure float type for model input (model's AMP handles BF16 internally)
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # Prepare decoder input - create directly on GPU with same dtype
        if getattr(self.args, 'padding', 0) == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]],
                                  device=self.device, dtype=batch_y.dtype)
        else:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]],
                                 device=self.device, dtype=batch_y.dtype)

        # FIXED: Use batch_y (not undefined batch_y_gpu)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

        # Forward pass - model handles AMP/BF16 internally
        if getattr(self.args, 'output_attention', False):
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # Inverse transform if needed
        if getattr(self.args, 'inverse', False):
            outputs = dataset_object.inverse_transform(outputs)

        # Get target - data already on GPU
        f_dim = -1 if self.args.features == 'MS' else 0
        true = batch_y[:, -self.args.pred_len:, f_dim:]

        return outputs, true

    def test(self, setting, load=False):
        """
        Test the model.
        
        FAITHFUL VECTOR: Uses only first element (:,:,:1) for metrics.
        
        OPTIMIZED:
        - Predictions collected on GPU
        - Single GPU→CPU transfer at the end
        - No per-batch CPU transfers
        
        Args:
            setting: Experiment setting string
            load: Whether to load checkpoint before testing
        """
        test_data, test_loader = self._get_data(flag='test')
        
        if load:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Collect predictions on GPU
        preds_list = []
        trues_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                pred, true = self._process_one_batch(test_data, *batch)
                # Faithful vector: use only first element for metrics
                # Keep on GPU - just detach from computation graph
                preds_list.append(pred[:, :, :1].detach())
                trues_list.append(true[:, :, :1].detach())

        # Concatenate on GPU, then single transfer to CPU
        preds = torch.cat(preds_list, dim=0).cpu().numpy()
        trues = torch.cat(trues_list, dim=0).cpu().numpy()
        
        # Clear GPU memory
        del preds_list, trues_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self._should_print():
            print(f'Test shape: {preds.shape}, {trues.shape}')

        # Calculate metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        if self._should_print():
            print(f'\nTest Results (Faithful Vector - 1st element):')
            print(f'  MSE:  {mse:.7f}')
            print(f'  MAE:  {mae:.7f}')
            print(f'  RMSE: {rmse:.7f}')
            
            # Save results
            folder = f'./results/{setting}/'
            os.makedirs(folder, exist_ok=True)
            np.save(f'{folder}metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            
            # Save with inverse transform if scaler available
            if hasattr(test_data, 'scaler') and test_data.scaler is not None:
                np.save(f'{folder}pred.npy', test_data.scaler.inverse_transform(preds))
                np.save(f'{folder}true.npy', test_data.scaler.inverse_transform(trues))
            else:
                np.save(f'{folder}pred.npy', preds)
                np.save(f'{folder}true.npy', trues)

        return mse, mae

    def predict(self, setting, load=False):
        """
        Make predictions on new data.
        
        OPTIMIZED:
        - Predictions collected on GPU
        - Single GPU→CPU transfer at the end
        
        Args:
            setting: Experiment setting string
            load: Whether to load checkpoint before predicting
        """
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)

        self.model.eval()
        
        # Collect predictions on GPU
        preds_list = []

        with torch.no_grad():
            for batch in pred_loader:
                pred, _ = self._process_one_batch(pred_data, *batch)
                preds_list.append(pred.detach())

        # Concatenate on GPU, then single transfer to CPU
        preds = torch.cat(preds_list, dim=0).cpu().numpy()
        
        # Clear GPU memory
        del preds_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._should_print():
            folder = f'./results/{setting}/'
            os.makedirs(folder, exist_ok=True)
            
            # Save with inverse transform if scaler available
            if hasattr(pred_data, 'scaler') and pred_data.scaler is not None:
                np.save(f'{folder}real_prediction.npy', pred_data.scaler.inverse_transform(preds))
            else:
                np.save(f'{folder}real_prediction.npy', preds)

        return preds

    def _load_checkpoint(self, path):
        """
        Load model checkpoint.
        
        CRITICAL: In FSDP mode, all ranks must participate in loading.
        """
        if not os.path.exists(path):
            if self._should_print():
                print(f"Checkpoint not found: {path}")
            return

        if self._should_print():
            print(f"Loading checkpoint: {path}")

        if getattr(self.args, 'use_fsdp', False) and isinstance(self.model, FSDP):
            # FSDP checkpoint loading - all ranks must participate
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                # Load on CPU first (only rank 0 has the data)
                if getattr(self.args, 'global_rank', 0) == 0:
                    state_dict = torch.load(path, map_location='cpu')
                else:
                    state_dict = None
                
                # Broadcast from rank 0 to all ranks
                if dist.is_initialized():
                    object_list = [state_dict]
                    dist.broadcast_object_list(object_list, src=0)
                    state_dict = object_list[0]
                
                if state_dict is not None:
                    self.model.load_state_dict(state_dict)
        else:
            # Standard checkpoint loading
            state_dict = torch.load(path, map_location=self.device)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
