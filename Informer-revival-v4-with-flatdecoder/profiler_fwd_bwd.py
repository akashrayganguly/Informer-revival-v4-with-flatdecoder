#!/usr/bin/env python3
"""
INFORMER FSDP Performance Profiler - FORWARD + BACKWARD PASS BREAKDOWN

Profiles YOUR ACTUAL Informer model with detailed forward AND backward pass
component timing.

FORWARD PASS BREAKDOWN:
  1. RevIN normalize
  2. Encoder Embedding (token conv + ROPE position + ROPE channel)
  3. Encoder (per-layer: attention, FFN, channel mixing, distillation conv)
  4. Decoder Embedding
  5. Decoder (per-layer: self-attn, cross-attn, FFN, channel mixing)
  6. Projection (final linear)
  7. RevIN denormalize

BACKWARD PASS BREAKDOWN:
  1. Per-component gradient propagation (autograd through each module)
     -- Projection, Decoder layers, Encoder layers, Embeddings, RevIN
  2. FSDP reduce-scatter communication overhead (estimated)
  3. Gradient clipping time
  4. Total backward = sum of all above

Usage:
    torchrun --nproc_per_node=2 profiler_fwd_bwd.py

Or via SLURM:
    sbatch run_profiler_fwd_bwd.slurm
"""

import os
import sys
import time
import functools
import numpy as np
from datetime import timedelta
from contextlib import contextmanager, nullcontext
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Add your project to path
PROJECT_DIR = 'VanillaInformerROPE-Revin-hybridfsdp'
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

# Import YOUR model and data
from models.model import Informer, InformerStack
from models.encoder import EncoderLayer, ConvLayer
from models.decoder import DecoderLayer
from models.attn import AttentionLayer, ProbAttention, FullAttention
from models.embed import (
    DataEmbedding, TokenEmbedding,
    RotaryPositionalEmbedding, RotaryPositionalEmbeddingFixed,
    RotaryChannelEmbeddingLearnable, RotaryChannelEmbeddingFixed,
)
from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from utils.tools import dotdict


# =============================================================================
# CUDA TIMER - Accurate GPU timing
# =============================================================================
class CUDATimer:
    """Accurate GPU timing using CUDA events"""

    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        self.times = []

    @contextmanager
    def __call__(self):
        if not self.enabled or not torch.cuda.is_available():
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()

        self.times.append(start.elapsed_time(end))

    def avg_ms(self):
        return sum(self.times) / len(self.times) if self.times else 0

    def total_ms(self):
        return sum(self.times)

    def min_ms(self):
        return min(self.times) if self.times else 0

    def max_ms(self):
        return max(self.times) if self.times else 0

    def reset(self):
        self.times = []


# =============================================================================
# FORWARD PASS COMPONENT PROFILER (hooks-based) -- UNCHANGED
# =============================================================================
class ForwardComponentProfiler:
    """
    Profiles forward pass sub-components by registering CUDA-event hooks
    on model submodules.

    Hooks are registered BEFORE FSDP wrapping so they persist through
    the FSDP module hierarchy.
    """

    def __init__(self):
        self.hooks = []
        self._events = defaultdict(list)
        self.results = defaultdict(list)
        self._registered = False

    def register(self, model):
        if self._registered:
            return

        if hasattr(model, 'revin') and model.revin is not None:
            self._revin_call_count = 0
            self._hook_module(model.revin, '_revin_dispatch')

        enc_emb = model.enc_embedding
        self._hook_module(enc_emb, 'enc_embedding')
        self._hook_module(enc_emb.value_embedding, 'enc_emb/token_conv')
        self._hook_module(enc_emb.rpe, 'enc_emb/rope_pos_learn')
        self._hook_module(enc_emb.rpe_fixed, 'enc_emb/rope_pos_fixed')
        self._hook_module(enc_emb.learnable_channel_embedding, 'enc_emb/rope_chan_learn')
        self._hook_module(enc_emb.fixed_channel_embedding, 'enc_emb/rope_chan_fixed')

        self._hook_module(model.encoder, 'encoder')
        for i, layer in enumerate(model.encoder.attn_layers):
            self._hook_module(layer, f'enc_layer_{i}')
            self._hook_module(layer.attention, f'enc_layer_{i}/attn_layer')
            if hasattr(layer.attention, 'inner_attention'):
                self._hook_module(layer.attention.inner_attention, f'enc_layer_{i}/inner_attn')
                self._hook_module(layer.attention.query_projection, f'enc_layer_{i}/q_proj')
                self._hook_module(layer.attention.key_projection, f'enc_layer_{i}/k_proj')
                self._hook_module(layer.attention.value_projection, f'enc_layer_{i}/v_proj')
                self._hook_module(layer.attention.out_projection, f'enc_layer_{i}/out_proj')
            self._hook_module(layer.conv1, f'enc_layer_{i}/ffn_conv1')
            self._hook_module(layer.conv2, f'enc_layer_{i}/ffn_conv2')
            self._hook_module(layer.norm1, f'enc_layer_{i}/norm1')
            self._hook_module(layer.norm2, f'enc_layer_{i}/norm2')
            if layer.W is not None:
                if hasattr(layer, 'norm3') and layer.norm3 is not None:
                    self._hook_module(layer.norm3, f'enc_layer_{i}/chan_mix_norm')

        if model.encoder.conv_layers is not None:
            for i, conv in enumerate(model.encoder.conv_layers):
                self._hook_module(conv, f'enc_distil_conv_{i}')

        if model.encoder.norm is not None:
            self._hook_module(model.encoder.norm, 'enc_final_norm')

        dec_emb = model.dec_embedding
        self._hook_module(dec_emb, 'dec_embedding')
        self._hook_module(dec_emb.value_embedding, 'dec_emb/token_conv')
        self._hook_module(dec_emb.rpe, 'dec_emb/rope_pos_learn')
        self._hook_module(dec_emb.rpe_fixed, 'dec_emb/rope_pos_fixed')
        self._hook_module(dec_emb.learnable_channel_embedding, 'dec_emb/rope_chan_learn')
        self._hook_module(dec_emb.fixed_channel_embedding, 'dec_emb/rope_chan_fixed')

        self._hook_module(model.decoder, 'decoder')
        for i, layer in enumerate(model.decoder.layers):
            self._hook_module(layer, f'dec_layer_{i}')
            self._hook_module(layer.self_attention, f'dec_layer_{i}/self_attn_layer')
            if hasattr(layer.self_attention, 'inner_attention'):
                self._hook_module(layer.self_attention.inner_attention, f'dec_layer_{i}/self_inner_attn')
            self._hook_module(layer.cross_attention, f'dec_layer_{i}/cross_attn_layer')
            if hasattr(layer.cross_attention, 'inner_attention'):
                self._hook_module(layer.cross_attention.inner_attention, f'dec_layer_{i}/cross_inner_attn')
            self._hook_module(layer.conv1, f'dec_layer_{i}/ffn_conv1')
            self._hook_module(layer.conv2, f'dec_layer_{i}/ffn_conv2')
            self._hook_module(layer.norm1, f'dec_layer_{i}/norm1')
            self._hook_module(layer.norm2, f'dec_layer_{i}/norm2')
            self._hook_module(layer.norm3, f'dec_layer_{i}/norm3')
            if layer.W is not None and hasattr(layer, 'norm4') and layer.norm4 is not None:
                self._hook_module(layer.norm4, f'dec_layer_{i}/chan_mix_norm')

        if model.decoder.norm is not None:
            self._hook_module(model.decoder.norm, 'dec_final_norm')

        self._hook_module(model.projection, 'projection')
        self._registered = True

    def _hook_module(self, module, name):
        h1 = module.register_forward_pre_hook(self._make_pre_hook(name))
        h2 = module.register_forward_hook(self._make_post_hook(name))
        self.hooks.extend([h1, h2])

    def _make_pre_hook(self, name):
        profiler = self
        def hook(module, args):
            actual_name = name
            if name == '_revin_dispatch':
                profiler._revin_call_count += 1
                if profiler._revin_call_count % 2 == 1:
                    actual_name = 'revin_normalize'
                else:
                    actual_name = 'revin_denormalize'
                module._profiler_current_name = actual_name
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            profiler._events[actual_name if name != '_revin_dispatch' else actual_name].append([start, None])
        return hook

    def _make_post_hook(self, name):
        profiler = self
        def hook(module, args, output):
            actual_name = name
            if name == '_revin_dispatch':
                actual_name = getattr(module, '_profiler_current_name', 'revin_unknown')
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if profiler._events[actual_name]:
                profiler._events[actual_name][-1][1] = end
        return hook

    def synchronize_and_collect(self):
        torch.cuda.synchronize()
        for name, pairs in self._events.items():
            total_ms = 0.0
            for start_ev, end_ev in pairs:
                if end_ev is not None:
                    total_ms += start_ev.elapsed_time(end_ev)
            self.results[name].append(total_ms)
        self._events.clear()
        self._revin_call_count = 0

    def reset(self):
        self._events.clear()
        self.results.clear()
        self._revin_call_count = 0

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def summary(self, forward_total_ms=None):
        out = OrderedDict()
        for name, times_list in sorted(self.results.items()):
            out[name] = sum(times_list) / len(times_list) if times_list else 0.0
        return out


# =============================================================================
# BACKWARD PASS COMPONENT PROFILER (hooks-based) -- NEW
# =============================================================================
class BackwardComponentProfiler:
    """
    Profiles backward pass sub-components using:
      - register_full_backward_pre_hook:  fires BEFORE module backward
        (when grad_output arrives at module output boundary)
      - register_full_backward_hook:      fires AFTER module backward
        (when grad_input computed for module inputs/params)

    Elapsed time between pre_hook -> hook = gradient computation time
    for that module (includes FSDP all-gather/reduce-scatter within
    that module's backward boundary).

    Hooks registered BEFORE FSDP wrapping persist on inner modules.

    Backward execution order is REVERSE of forward:
      Forward:  RevIN -> EncEmb -> Encoder -> DecEmb -> Decoder -> Projection
      Backward: Projection -> Decoder -> DecEmb -> Encoder -> EncEmb -> RevIN
    """

    def __init__(self):
        self.hooks = []
        self._events = defaultdict(list)
        self.results = defaultdict(list)
        self._registered = False
        self._revin_bwd_call_count = 0

    def register(self, model):
        if self._registered:
            return

        # RevIN: backward fires denorm first, then norm
        if hasattr(model, 'revin') and model.revin is not None:
            self._revin_bwd_call_count = 0
            self._hook_module(model.revin, '_revin_bwd_dispatch')

        # Encoder Embedding
        enc_emb = model.enc_embedding
        self._hook_module(enc_emb, 'enc_embedding')
        self._hook_module(enc_emb.value_embedding, 'enc_emb/token_conv')
        self._hook_module(enc_emb.rpe, 'enc_emb/rope_pos_learn')
        self._hook_module(enc_emb.rpe_fixed, 'enc_emb/rope_pos_fixed')
        self._hook_module(enc_emb.learnable_channel_embedding, 'enc_emb/rope_chan_learn')
        self._hook_module(enc_emb.fixed_channel_embedding, 'enc_emb/rope_chan_fixed')

        # Encoder
        self._hook_module(model.encoder, 'encoder')
        for i, layer in enumerate(model.encoder.attn_layers):
            self._hook_module(layer, f'enc_layer_{i}')
            self._hook_module(layer.attention, f'enc_layer_{i}/attn_layer')
            if hasattr(layer.attention, 'inner_attention'):
                self._hook_module(layer.attention.inner_attention, f'enc_layer_{i}/inner_attn')
                self._hook_module(layer.attention.query_projection, f'enc_layer_{i}/q_proj')
                self._hook_module(layer.attention.key_projection, f'enc_layer_{i}/k_proj')
                self._hook_module(layer.attention.value_projection, f'enc_layer_{i}/v_proj')
                self._hook_module(layer.attention.out_projection, f'enc_layer_{i}/out_proj')
            self._hook_module(layer.conv1, f'enc_layer_{i}/ffn_conv1')
            self._hook_module(layer.conv2, f'enc_layer_{i}/ffn_conv2')
            self._hook_module(layer.norm1, f'enc_layer_{i}/norm1')
            self._hook_module(layer.norm2, f'enc_layer_{i}/norm2')
            if layer.W is not None:
                if hasattr(layer, 'norm3') and layer.norm3 is not None:
                    self._hook_module(layer.norm3, f'enc_layer_{i}/chan_mix_norm')

        if model.encoder.conv_layers is not None:
            for i, conv in enumerate(model.encoder.conv_layers):
                self._hook_module(conv, f'enc_distil_conv_{i}')

        if model.encoder.norm is not None:
            self._hook_module(model.encoder.norm, 'enc_final_norm')

        # Decoder Embedding
        dec_emb = model.dec_embedding
        self._hook_module(dec_emb, 'dec_embedding')
        self._hook_module(dec_emb.value_embedding, 'dec_emb/token_conv')
        self._hook_module(dec_emb.rpe, 'dec_emb/rope_pos_learn')
        self._hook_module(dec_emb.rpe_fixed, 'dec_emb/rope_pos_fixed')
        self._hook_module(dec_emb.learnable_channel_embedding, 'dec_emb/rope_chan_learn')
        self._hook_module(dec_emb.fixed_channel_embedding, 'dec_emb/rope_chan_fixed')

        # Decoder
        self._hook_module(model.decoder, 'decoder')
        for i, layer in enumerate(model.decoder.layers):
            self._hook_module(layer, f'dec_layer_{i}')
            self._hook_module(layer.self_attention, f'dec_layer_{i}/self_attn_layer')
            if hasattr(layer.self_attention, 'inner_attention'):
                self._hook_module(layer.self_attention.inner_attention, f'dec_layer_{i}/self_inner_attn')
            self._hook_module(layer.cross_attention, f'dec_layer_{i}/cross_attn_layer')
            if hasattr(layer.cross_attention, 'inner_attention'):
                self._hook_module(layer.cross_attention.inner_attention, f'dec_layer_{i}/cross_inner_attn')
            self._hook_module(layer.conv1, f'dec_layer_{i}/ffn_conv1')
            self._hook_module(layer.conv2, f'dec_layer_{i}/ffn_conv2')
            self._hook_module(layer.norm1, f'dec_layer_{i}/norm1')
            self._hook_module(layer.norm2, f'dec_layer_{i}/norm2')
            self._hook_module(layer.norm3, f'dec_layer_{i}/norm3')
            if layer.W is not None and hasattr(layer, 'norm4') and layer.norm4 is not None:
                self._hook_module(layer.norm4, f'dec_layer_{i}/chan_mix_norm')

        if model.decoder.norm is not None:
            self._hook_module(model.decoder.norm, 'dec_final_norm')

        # Projection
        self._hook_module(model.projection, 'projection')
        self._registered = True

    def _hook_module(self, module, name):
        h1 = module.register_full_backward_pre_hook(self._make_pre_hook(name))
        h2 = module.register_full_backward_hook(self._make_post_hook(name))
        self.hooks.extend([h1, h2])

    def _make_pre_hook(self, name):
        """full_backward_pre_hook(module, grad_output) -- before backward"""
        profiler = self
        def hook(module, grad_output):
            actual_name = name
            if name == '_revin_bwd_dispatch':
                profiler._revin_bwd_call_count += 1
                # Backward reverses forward: call 1=denorm, call 2=norm
                if profiler._revin_bwd_call_count % 2 == 1:
                    actual_name = 'revin_denormalize'
                else:
                    actual_name = 'revin_normalize'
                module._bwd_profiler_current_name = actual_name
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            profiler._events[actual_name].append([start, None])
        return hook

    def _make_post_hook(self, name):
        """full_backward_hook(module, grad_input, grad_output) -- after backward"""
        profiler = self
        def hook(module, grad_input, grad_output):
            actual_name = name
            if name == '_revin_bwd_dispatch':
                actual_name = getattr(module, '_bwd_profiler_current_name', 'revin_bwd_unknown')
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if profiler._events[actual_name]:
                profiler._events[actual_name][-1][1] = end
        return hook

    def synchronize_and_collect(self):
        torch.cuda.synchronize()
        for name, pairs in self._events.items():
            total_ms = 0.0
            for start_ev, end_ev in pairs:
                if end_ev is not None:
                    total_ms += start_ev.elapsed_time(end_ev)
            self.results[name].append(total_ms)
        self._events.clear()
        self._revin_bwd_call_count = 0

    def reset(self):
        self._events.clear()
        self.results.clear()
        self._revin_bwd_call_count = 0

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def summary(self):
        out = OrderedDict()
        for name, times_list in sorted(self.results.items()):
            out[name] = sum(times_list) / len(times_list) if times_list else 0.0
        return out


# =============================================================================
# DATA PREFETCHER
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
            self.next_batch = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    self.next_batch.append(item.to(self.device, non_blocking=True))
                else:
                    self.next_batch.append(item)
            self.next_batch = tuple(self.next_batch)

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
# CONFIGURATION
# =============================================================================
def create_args():
    args = dotdict()
    args.model = 'informer'
    args.data = 'custom'
    args.root_path = './ETDataset/ETT-small/'
    args.data_path = 'nc_m8_tau12.csv'
    args.features = 'M'
    args.target = 'data9'
    args.freq = 'h'
    args.cols = None
    args.inverse = False
    args.seq_len = 96 * 7
    args.label_len = 48 * 7
    args.pred_len = 96 * 7
    args.enc_in = 9
    args.dec_in = 9
    args.c_out = 9
    args.factor = 5
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.s_layers = [3, 2, 1]
    args.d_ff = 2048
    args.dropout = 0.05
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.distil = True
    args.output_attention = False
    args.mix = True
    args.padding = 0
    args.use_revin = True
    args.channel_mix_size = None
    args.channel_period = 7
    args.max_len = 200000
    args.batch_size = 32
    args.learning_rate = 0.0001
    args.use_amp = True
    args.gradient_accumulation_steps = 1
    args.max_grad_norm = 1.0
    args.num_workers = 6
    args.prefetch_factor = 4
    args.use_prefetcher = True
    args.use_fsdp = True
    args.fsdp_sharding_strategy = 'HYBRID_SHARD'
    args.fsdp_activation_checkpointing = False
    args.seed = 2021
    return args


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        num_gpus = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        num_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', torch.cuda.device_count()))
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        num_gpus = torch.cuda.device_count()
    return rank, world_size, local_rank, num_gpus


def get_amp_config():
    if not torch.cuda.is_available():
        return {'supported': False, 'dtype': torch.float32, 'use_scaler': False, 'name': 'FP32'}
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return {'supported': True, 'dtype': torch.bfloat16, 'use_scaler': False, 'name': 'BF16'}
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:
        return {'supported': True, 'dtype': torch.float16, 'use_scaler': True, 'name': 'FP16'}
    return {'supported': False, 'dtype': torch.float32, 'use_scaler': False, 'name': 'FP32'}


# =============================================================================
# MODEL + FSDP
# =============================================================================
def build_model(args, device):
    model = Informer(
        enc_in=args.enc_in, dec_in=args.dec_in, c_out=args.c_out,
        seq_len=args.seq_len, label_len=args.label_len, out_len=args.pred_len,
        factor=args.factor, d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers, d_ff=args.d_ff,
        dropout=args.dropout, attn=args.attn, embed=args.embed, freq=args.freq,
        activation=args.activation, output_attention=args.output_attention,
        distil=args.distil, mix=args.mix, device=device,
        use_revin=args.use_revin, channel_mix_size=args.channel_mix_size,
        channel_period=args.channel_period, max_len=args.max_len,
    ).float()
    return model


def wrap_with_fsdp(model, args, device, amp_config):
    if args.fsdp_activation_checkpointing:
        check_fn = lambda m: isinstance(m, (EncoderLayer, DecoderLayer))
        wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)

    strategy = getattr(ShardingStrategy, args.fsdp_sharding_strategy)
    mp_policy = None
    if args.use_amp and amp_config['supported']:
        mp_policy = MixedPrecision(
            param_dtype=amp_config['dtype'],
            reduce_dtype=amp_config['dtype'],
            buffer_dtype=amp_config['dtype'],
        )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={EncoderLayer, DecoderLayer},
    )

    device_mesh = None
    is_hybrid = args.fsdp_sharding_strategy in ['HYBRID_SHARD', '_HYBRID_SHARD_ZERO2']
    if is_hybrid:
        try:
            from torch.distributed.device_mesh import init_device_mesh
            world_size = dist.get_world_size()
            num_gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE',
                                    os.environ.get('SLURM_GPUS_ON_NODE',
                                    torch.cuda.device_count())))
            num_nodes = world_size // num_gpus_per_node
            if num_nodes > 1:
                device_mesh = init_device_mesh(
                    "cuda", (num_nodes, num_gpus_per_node),
                    mesh_dim_names=("replicate", "shard"),
                )
        except Exception:
            device_mesh = None

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
    if is_hybrid and device_mesh is not None:
        fsdp_kwargs['device_mesh'] = device_mesh

    model = FSDP(model, **fsdp_kwargs)
    return model


# =============================================================================
# DATA LOADING
# =============================================================================
def get_data_loader(args, flag='train'):
    data_dict = {
        'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
        'custom': Dataset_Custom,
    }
    Data = data_dict.get(args.data, Dataset_Custom)
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle = True if flag == 'train' else False

    data_set = Data(
        root_path=args.root_path, data_path=args.data_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, inverse=args.inverse,
        timeenc=timeenc, freq=args.freq, cols=args.cols
    )
    sampler = DistributedSampler(data_set, shuffle=shuffle, drop_last=True)
    loader = DataLoader(
        data_set, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    return data_set, loader


# =============================================================================
# PROCESS ONE BATCH
# =============================================================================
def process_batch(model, batch, args, device, use_prefetcher=False):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    if not use_prefetcher:
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)
    else:
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

    if args.padding == 0:
        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]],
                              device=device, dtype=torch.float32)
    else:
        dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]],
                             device=device, dtype=torch.float32)
    dec_inp = torch.cat([batch_y[:, :args.label_len, :].to(device), dec_inp], dim=1)

    if args.output_attention:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    else:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    f_dim = -1 if args.features == 'MS' else 0
    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
    if outputs.dtype != batch_y.dtype:
        batch_y = batch_y.to(outputs.dtype)
    return outputs, batch_y


# =============================================================================
# PRINTING - FORWARD BREAKDOWN (unchanged from profiler_forward.py)
# =============================================================================
def print_forward_breakdown(fwd_profiler, forward_timer, rank):
    if rank != 0:
        return
    summary = fwd_profiler.summary()
    total_fwd_ms = forward_timer.avg_ms()

    print()
    print("=" * 90)
    print("FORWARD PASS BREAKDOWN")
    print("=" * 90)
    print()
    print(f"{'Component':<45} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'% Fwd':<8}")
    print("-" * 90)

    groups = OrderedDict()
    groups['RevIN normalize'] = 'revin_normalize'
    groups['RevIN denormalize'] = 'revin_denormalize'
    groups['Encoder Embedding (total)'] = 'enc_embedding'
    groups['  \u251c\u2500 Token Conv1d'] = 'enc_emb/token_conv'
    groups['  \u251c\u2500 ROPE pos (learnable)'] = 'enc_emb/rope_pos_learn'
    groups['  \u251c\u2500 ROPE pos (fixed)'] = 'enc_emb/rope_pos_fixed'
    groups['  \u251c\u2500 ROPE chan (learnable)'] = 'enc_emb/rope_chan_learn'
    groups['  \u2514\u2500 ROPE chan (fixed)'] = 'enc_emb/rope_chan_fixed'

    groups['Encoder (total)'] = 'encoder'
    n_enc = len([k for k in summary if k.startswith('enc_layer_') and '/' not in k])
    for i in range(n_enc):
        groups[f'  Encoder Layer {i} (total)'] = f'enc_layer_{i}'
        groups[f'    \u251c\u2500 Attention Layer'] = f'enc_layer_{i}/attn_layer'
        groups[f'      \u251c\u2500 Q projection'] = f'enc_layer_{i}/q_proj'
        groups[f'      \u251c\u2500 K projection'] = f'enc_layer_{i}/k_proj'
        groups[f'      \u251c\u2500 V projection'] = f'enc_layer_{i}/v_proj'
        groups[f'      \u251c\u2500 Inner Attention'] = f'enc_layer_{i}/inner_attn'
        groups[f'      \u2514\u2500 Out projection'] = f'enc_layer_{i}/out_proj'
        groups[f'    \u251c\u2500 LayerNorm 1'] = f'enc_layer_{i}/norm1'
        groups[f'    \u251c\u2500 FFN Conv1'] = f'enc_layer_{i}/ffn_conv1'
        groups[f'    \u251c\u2500 FFN Conv2'] = f'enc_layer_{i}/ffn_conv2'
        groups[f'    \u251c\u2500 LayerNorm 2'] = f'enc_layer_{i}/norm2'
        if f'enc_layer_{i}/chan_mix_norm' in summary:
            groups[f'    \u2514\u2500 Channel Mix Norm'] = f'enc_layer_{i}/chan_mix_norm'
        if f'enc_distil_conv_{i}' in summary:
            groups[f'  Distil Conv {i}'] = f'enc_distil_conv_{i}'
    if 'enc_final_norm' in summary:
        groups['  Encoder Final Norm'] = 'enc_final_norm'

    groups['Decoder Embedding (total)'] = 'dec_embedding'
    groups['  \u251c\u2500 Token Conv1d '] = 'dec_emb/token_conv'
    groups['  \u251c\u2500 ROPE pos (learnable) '] = 'dec_emb/rope_pos_learn'
    groups['  \u251c\u2500 ROPE pos (fixed) '] = 'dec_emb/rope_pos_fixed'
    groups['  \u251c\u2500 ROPE chan (learnable) '] = 'dec_emb/rope_chan_learn'
    groups['  \u2514\u2500 ROPE chan (fixed) '] = 'dec_emb/rope_chan_fixed'

    groups['Decoder (total)'] = 'decoder'
    n_dec = len([k for k in summary if k.startswith('dec_layer_') and '/' not in k])
    for i in range(n_dec):
        groups[f'  Decoder Layer {i} (total)'] = f'dec_layer_{i}'
        groups[f'    \u251c\u2500 Self-Attn Layer'] = f'dec_layer_{i}/self_attn_layer'
        groups[f'      \u2514\u2500 Self Inner Attn'] = f'dec_layer_{i}/self_inner_attn'
        groups[f'    \u251c\u2500 Norm 1'] = f'dec_layer_{i}/norm1'
        groups[f'    \u251c\u2500 Cross-Attn Layer'] = f'dec_layer_{i}/cross_attn_layer'
        groups[f'      \u2514\u2500 Cross Inner Attn'] = f'dec_layer_{i}/cross_inner_attn'
        groups[f'    \u251c\u2500 Norm 2'] = f'dec_layer_{i}/norm2'
        groups[f'    \u251c\u2500 FFN Conv1 '] = f'dec_layer_{i}/ffn_conv1'
        groups[f'    \u251c\u2500 FFN Conv2 '] = f'dec_layer_{i}/ffn_conv2'
        groups[f'    \u251c\u2500 Norm 3'] = f'dec_layer_{i}/norm3'
        if f'dec_layer_{i}/chan_mix_norm' in summary:
            groups[f'    \u2514\u2500 Channel Mix Norm '] = f'dec_layer_{i}/chan_mix_norm'
    if 'dec_final_norm' in summary:
        groups['  Decoder Final Norm'] = 'dec_final_norm'
    groups['Projection (Linear)'] = 'projection'

    for display_name, key in groups.items():
        if key not in summary:
            continue
        avg = summary[key]
        times = fwd_profiler.results.get(key, [])
        min_t = min(times) if times else 0
        max_t = max(times) if times else 0
        pct = (avg / total_fwd_ms * 100) if total_fwd_ms > 0 else 0
        print(f"{display_name:<45} {avg:<12.2f} {min_t:<12.2f} {max_t:<12.2f} {pct:<8.1f}")

    print("-" * 90)
    print(f"{'Forward Total (timer)':<45} {total_fwd_ms:<12.2f}")
    print()

    # Aggregate
    print("=" * 70)
    print("FORWARD PASS AGGREGATE SUMMARY")
    print("=" * 70)
    agg = OrderedDict()
    agg['RevIN (norm+denorm)'] = summary.get('revin_normalize', 0) + summary.get('revin_denormalize', 0)
    agg['Encoder Embedding'] = summary.get('enc_embedding', 0)
    agg['Encoder'] = summary.get('encoder', 0)
    agg['Decoder Embedding'] = summary.get('dec_embedding', 0)
    agg['Decoder'] = summary.get('decoder', 0)
    agg['Projection'] = summary.get('projection', 0)

    enc_attn_total = sum(summary.get(f'enc_layer_{i}/attn_layer', 0) for i in range(n_enc))
    enc_ffn_total = sum(summary.get(f'enc_layer_{i}/ffn_conv1', 0) + summary.get(f'enc_layer_{i}/ffn_conv2', 0) for i in range(n_enc))
    enc_norm_total = sum(summary.get(f'enc_layer_{i}/norm1', 0) + summary.get(f'enc_layer_{i}/norm2', 0) for i in range(n_enc))
    enc_distil_total = sum(summary.get(f'enc_distil_conv_{i}', 0) for i in range(n_enc))
    dec_self_attn_total = sum(summary.get(f'dec_layer_{i}/self_attn_layer', 0) for i in range(n_dec))
    dec_cross_attn_total = sum(summary.get(f'dec_layer_{i}/cross_attn_layer', 0) for i in range(n_dec))
    dec_ffn_total = sum(summary.get(f'dec_layer_{i}/ffn_conv1', 0) + summary.get(f'dec_layer_{i}/ffn_conv2', 0) for i in range(n_dec))

    print()
    print(f"{'Component':<35} {'Avg (ms)':<12} {'% Fwd':<8}")
    print("-" * 55)
    for name, ms in agg.items():
        pct = (ms / total_fwd_ms * 100) if total_fwd_ms > 0 else 0
        print(f"{name:<35} {ms:<12.2f} {pct:<8.1f}")
    print("-" * 55)
    print()
    print(f"{'Within Encoder:':<35}")
    for label, val in [('  Attention (all layers)', enc_attn_total), ('  FFN (all layers)', enc_ffn_total),
                       ('  LayerNorm (all layers)', enc_norm_total), ('  Distillation Conv', enc_distil_total)]:
        pct = (val / total_fwd_ms * 100) if total_fwd_ms > 0 else 0
        print(f"{label:<35} {val:<12.2f} {pct:<8.1f}")
    print()
    print(f"{'Within Decoder:':<35}")
    for label, val in [('  Self-Attention (all layers)', dec_self_attn_total),
                       ('  Cross-Attention (all layers)', dec_cross_attn_total),
                       ('  FFN (all layers)', dec_ffn_total)]:
        pct = (val / total_fwd_ms * 100) if total_fwd_ms > 0 else 0
        print(f"{label:<35} {val:<12.2f} {pct:<8.1f}")
    print()

    max_bar = 50
    print("=" * 70)
    print("FORWARD PASS TIME DISTRIBUTION (visual)")
    print("=" * 70)
    for name, ms in agg.items():
        pct = (ms / total_fwd_ms * 100) if total_fwd_ms > 0 else 0
        bar_len = int(pct / 100 * max_bar)
        print(f"  {name:<25} {pct:5.1f}% {chr(9608) * bar_len}")
    print()


# =============================================================================
# PRINTING - BACKWARD BREAKDOWN (NEW)
# =============================================================================
def print_backward_breakdown(bwd_profiler, backward_timer, grad_clip_timer, rank):
    if rank != 0:
        return
    summary = bwd_profiler.summary()
    total_bwd_ms = backward_timer.avg_ms()

    print()
    print("=" * 90)
    print("BACKWARD PASS BREAKDOWN (Gradient Propagation per Component)")
    print("=" * 90)
    print()
    print("NOTE: Backward runs in REVERSE order of forward pass.")
    print("      Each module time = grad computation for params + inputs")
    print("      + any FSDP all-gather/reduce-scatter within that boundary.")
    print()
    print(f"{'Component':<45} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'% Bwd':<8}")
    print("-" * 90)

    # Display in backward execution order (reverse of forward)
    groups = OrderedDict()

    groups['Projection (Linear)'] = 'projection'

    n_dec = len([k for k in summary if k.startswith('dec_layer_') and '/' not in k])
    if 'dec_final_norm' in summary:
        groups['  Decoder Final Norm'] = 'dec_final_norm'
    groups['Decoder (total)'] = 'decoder'
    for i in reversed(range(n_dec)):
        groups[f'  Decoder Layer {i} (total)'] = f'dec_layer_{i}'
        if f'dec_layer_{i}/chan_mix_norm' in summary:
            groups[f'    \u251c\u2500 Channel Mix Norm'] = f'dec_layer_{i}/chan_mix_norm'
        groups[f'    \u251c\u2500 Norm 3'] = f'dec_layer_{i}/norm3'
        groups[f'    \u251c\u2500 FFN Conv2'] = f'dec_layer_{i}/ffn_conv2'
        groups[f'    \u251c\u2500 FFN Conv1'] = f'dec_layer_{i}/ffn_conv1'
        groups[f'    \u251c\u2500 Norm 2'] = f'dec_layer_{i}/norm2'
        groups[f'    \u251c\u2500 Cross-Attn Layer'] = f'dec_layer_{i}/cross_attn_layer'
        groups[f'      \u2514\u2500 Cross Inner Attn'] = f'dec_layer_{i}/cross_inner_attn'
        groups[f'    \u251c\u2500 Norm 1'] = f'dec_layer_{i}/norm1'
        groups[f'    \u251c\u2500 Self-Attn Layer'] = f'dec_layer_{i}/self_attn_layer'
        groups[f'      \u2514\u2500 Self Inner Attn'] = f'dec_layer_{i}/self_inner_attn'

    groups['Decoder Embedding (total)'] = 'dec_embedding'

    n_enc = len([k for k in summary if k.startswith('enc_layer_') and '/' not in k])
    if 'enc_final_norm' in summary:
        groups['  Encoder Final Norm'] = 'enc_final_norm'
    groups['Encoder (total)'] = 'encoder'
    for i in reversed(range(n_enc)):
        if f'enc_distil_conv_{i}' in summary:
            groups[f'  Distil Conv {i}'] = f'enc_distil_conv_{i}'
        groups[f'  Encoder Layer {i} (total)'] = f'enc_layer_{i}'
        if f'enc_layer_{i}/chan_mix_norm' in summary:
            groups[f'    \u251c\u2500 Channel Mix Norm'] = f'enc_layer_{i}/chan_mix_norm'
        groups[f'    \u251c\u2500 LayerNorm 2'] = f'enc_layer_{i}/norm2'
        groups[f'    \u251c\u2500 FFN Conv2'] = f'enc_layer_{i}/ffn_conv2'
        groups[f'    \u251c\u2500 FFN Conv1'] = f'enc_layer_{i}/ffn_conv1'
        groups[f'    \u251c\u2500 LayerNorm 1'] = f'enc_layer_{i}/norm1'
        groups[f'    \u251c\u2500 Attention Layer'] = f'enc_layer_{i}/attn_layer'
        groups[f'      \u2514\u2500 Out projection'] = f'enc_layer_{i}/out_proj'
        groups[f'      \u251c\u2500 Inner Attention'] = f'enc_layer_{i}/inner_attn'
        groups[f'      \u251c\u2500 V projection'] = f'enc_layer_{i}/v_proj'
        groups[f'      \u251c\u2500 K projection'] = f'enc_layer_{i}/k_proj'
        groups[f'      \u251c\u2500 Q projection'] = f'enc_layer_{i}/q_proj'

    groups['Encoder Embedding (total)'] = 'enc_embedding'
    groups['RevIN denormalize (bwd)'] = 'revin_denormalize'
    groups['RevIN normalize (bwd)'] = 'revin_normalize'

    for display_name, key in groups.items():
        if key not in summary:
            continue
        avg = summary[key]
        times = bwd_profiler.results.get(key, [])
        min_t = min(times) if times else 0
        max_t = max(times) if times else 0
        pct = (avg / total_bwd_ms * 100) if total_bwd_ms > 0 else 0
        print(f"{display_name:<45} {avg:<12.2f} {min_t:<12.2f} {max_t:<12.2f} {pct:<8.1f}")

    print("-" * 90)
    grad_clip_ms = grad_clip_timer.avg_ms()
    pct_clip = (grad_clip_ms / total_bwd_ms * 100) if total_bwd_ms > 0 else 0
    print(f"{'Gradient Clipping':<45} {grad_clip_ms:<12.2f} {grad_clip_timer.min_ms():<12.2f} {grad_clip_timer.max_ms():<12.2f} {pct_clip:<8.1f}")
    print("-" * 90)
    print(f"{'Backward Total (timer)':<45} {total_bwd_ms:<12.2f}")
    print()

    # Aggregate
    print("=" * 70)
    print("BACKWARD PASS AGGREGATE SUMMARY")
    print("=" * 70)
    agg = OrderedDict()
    agg['Projection'] = summary.get('projection', 0)
    agg['Decoder'] = summary.get('decoder', 0)
    agg['Decoder Embedding'] = summary.get('dec_embedding', 0)
    agg['Encoder'] = summary.get('encoder', 0)
    agg['Encoder Embedding'] = summary.get('enc_embedding', 0)
    agg['RevIN (norm+denorm)'] = summary.get('revin_normalize', 0) + summary.get('revin_denormalize', 0)
    agg['Grad Clipping'] = grad_clip_ms

    hooked_sum = sum(v for k, v in agg.items() if k != 'Grad Clipping')
    autograd_overhead = max(0, total_bwd_ms - hooked_sum - grad_clip_ms)
    agg['Autograd/FSDP Overhead'] = autograd_overhead

    print()
    print(f"{'Component':<40} {'Avg (ms)':<12} {'% Bwd':<8}")
    print("-" * 60)
    for name, ms in agg.items():
        pct = (ms / total_bwd_ms * 100) if total_bwd_ms > 0 else 0
        print(f"{name:<40} {ms:<12.2f} {pct:<8.1f}")
    print("-" * 60)
    print(f"{'Backward Total':<40} {total_bwd_ms:<12.2f} {'100.0':<8}")

    # Within Encoder/Decoder backward
    enc_attn_bwd = sum(summary.get(f'enc_layer_{i}/attn_layer', 0) for i in range(n_enc))
    enc_ffn_bwd = sum(summary.get(f'enc_layer_{i}/ffn_conv1', 0) + summary.get(f'enc_layer_{i}/ffn_conv2', 0) for i in range(n_enc))
    enc_norm_bwd = sum(summary.get(f'enc_layer_{i}/norm1', 0) + summary.get(f'enc_layer_{i}/norm2', 0) for i in range(n_enc))
    dec_self_attn_bwd = sum(summary.get(f'dec_layer_{i}/self_attn_layer', 0) for i in range(n_dec))
    dec_cross_attn_bwd = sum(summary.get(f'dec_layer_{i}/cross_attn_layer', 0) for i in range(n_dec))
    dec_ffn_bwd = sum(summary.get(f'dec_layer_{i}/ffn_conv1', 0) + summary.get(f'dec_layer_{i}/ffn_conv2', 0) for i in range(n_dec))

    print()
    print(f"{'Within Encoder Backward:':<40}")
    for label, val in [('  Attention (all layers)', enc_attn_bwd), ('  FFN (all layers)', enc_ffn_bwd),
                       ('  LayerNorm (all layers)', enc_norm_bwd)]:
        pct = (val / total_bwd_ms * 100) if total_bwd_ms > 0 else 0
        print(f"{label:<40} {val:<12.2f} {pct:<8.1f}")
    print()
    print(f"{'Within Decoder Backward:':<40}")
    for label, val in [('  Self-Attention (all layers)', dec_self_attn_bwd),
                       ('  Cross-Attention (all layers)', dec_cross_attn_bwd),
                       ('  FFN (all layers)', dec_ffn_bwd)]:
        pct = (val / total_bwd_ms * 100) if total_bwd_ms > 0 else 0
        print(f"{label:<40} {val:<12.2f} {pct:<8.1f}")
    print()

    max_bar = 50
    print("=" * 70)
    print("BACKWARD TIME DISTRIBUTION (visual)")
    print("=" * 70)
    for name, ms in agg.items():
        pct = (ms / total_bwd_ms * 100) if total_bwd_ms > 0 else 0
        bar_len = int(pct / 100 * max_bar)
        print(f"  {name:<35} {pct:5.1f}% {chr(9608) * bar_len}")
    print()


# =============================================================================
# MAIN PROFILING FUNCTION
# =============================================================================
def run_profiling():
    rank, world_size, local_rank, num_gpus = setup_distributed()

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl', init_method='env://',
                world_size=world_size, rank=rank,
                timeout=timedelta(minutes=10)
            )

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    args = create_args()
    args.global_rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.device = device

    amp_config = get_amp_config()

    if rank == 0:
        print("=" * 70)
        print("INFORMER FSDP PERFORMANCE PROFILER (FORWARD + BACKWARD BREAKDOWN)")
        print("=" * 70)
        print(f"World size: {world_size} ({world_size // num_gpus} nodes x {num_gpus} GPUs)")
        print(f"Device: {device}")
        print(f"AMP: {amp_config['name']}")
        print()
        print("Model Configuration:")
        print(f"  seq_len: {args.seq_len}, label_len: {args.label_len}, pred_len: {args.pred_len}")
        print(f"  d_model: {args.d_model}, n_heads: {args.n_heads}")
        print(f"  e_layers: {args.e_layers}, d_layers: {args.d_layers}")
        print(f"  batch_size: {args.batch_size}, grad_accum: {args.gradient_accumulation_steps}")
        print(f"  use_revin: {args.use_revin}, channel_period: {args.channel_period}")
        print(f"  num_workers: {args.num_workers}, prefetch_factor: {args.prefetch_factor}")
        print("=" * 70)
        print()

    # Build model
    if rank == 0:
        print("Building Informer model...")
    model = build_model(args, device)
    model = model.to(device)

    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

    # Register BOTH forward + backward hooks BEFORE FSDP wrapping
    fwd_profiler = ForwardComponentProfiler()
    fwd_profiler.register(model)

    bwd_profiler = BackwardComponentProfiler()
    bwd_profiler.register(model)

    if rank == 0:
        print(f"  Forward profiler: {len(fwd_profiler.hooks)} hooks registered")
        print(f"  Backward profiler: {len(bwd_profiler.hooks)} hooks registered")

    # Wrap with FSDP
    if world_size > 1:
        if rank == 0:
            print("Wrapping with FSDP...")
        model = wrap_with_fsdp(model, args, device, amp_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=getattr(args, 'weight_decay', 0.01))
    criterion = nn.MSELoss()

    use_amp = args.use_amp and amp_config['supported']
    use_scaler = use_amp and amp_config['use_scaler']
    scaler = GradScaler() if use_scaler else None

    # Data
    if rank == 0:
        print("Loading data...")
    try:
        train_data, train_loader = get_data_loader(args, flag='train')
        if rank == 0:
            print(f"  Train samples: {len(train_data)}, Batches: {len(train_loader)}")
    except Exception as e:
        if rank == 0:
            print(f"  Warning: Could not load real data: {e}")
            print("  Using synthetic data for profiling...")
        train_loader = None

    # Timers
    timers = {
        'data_transfer': CUDATimer('Data Transfer'),
        'forward': CUDATimer('Forward Pass'),
        'backward_total': CUDATimer('Backward Total'),
        'backward_compute': CUDATimer('Backward Compute'),
        'backward_comm': CUDATimer('Backward Communication'),
        'grad_clip': CUDATimer('Gradient Clipping'),
        'optimizer': CUDATimer('Optimizer Step'),
        'total': CUDATimer('Total Step'),
    }

    # Warmup
    if rank == 0:
        print("\nRunning warmup iterations...")

    model.train()
    grad_accum = args.gradient_accumulation_steps

    dummy_x = torch.randn(args.batch_size, args.seq_len, args.enc_in)
    dummy_y = torch.randn(args.batch_size, args.label_len + args.pred_len, args.dec_in)
    dummy_x_mark = torch.randn(args.batch_size, args.seq_len, 4)
    dummy_y_mark = torch.randn(args.batch_size, args.label_len + args.pred_len, 4)

    for _ in range(2):
        batch = (dummy_x.clone(), dummy_y.clone(), dummy_x_mark.clone(), dummy_y_mark.clone())
        outputs, targets = process_batch(model, batch, args, device, use_prefetcher=False)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Discard warmup hook data
        fwd_profiler._events.clear()
        fwd_profiler._revin_call_count = 0
        bwd_profiler._events.clear()
        bwd_profiler._revin_bwd_call_count = 0

    fwd_profiler.reset()
    bwd_profiler.reset()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # Profile iterations
    n_profile_iters = 10
    if rank == 0:
        print(f"\nRunning {n_profile_iters} profiling iterations...")
        print()

    use_real_data = train_loader is not None and len(train_loader) >= n_profile_iters
    if use_real_data:
        data_iter = iter(DataPrefetcher(train_loader, device)) if args.use_prefetcher else iter(train_loader)
    else:
        data_iter = None

    for i in range(n_profile_iters):
        with timers['total']():
            # Data
            with timers['data_transfer']():
                if data_iter is not None:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(DataPrefetcher(train_loader, device)) if args.use_prefetcher else iter(train_loader)
                        batch = next(data_iter)
                else:
                    batch = (dummy_x.clone().to(device, non_blocking=True), dummy_y.clone(),
                             dummy_x_mark.clone().to(device, non_blocking=True),
                             dummy_y_mark.clone().to(device, non_blocking=True))
                torch.cuda.synchronize()

            is_accum_step = (i + 1) % grad_accum != 0
            should_sync = not is_accum_step

            if args.use_fsdp and world_size > 1 and not should_sync:
                sync_ctx = model.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                # ============= FORWARD =============
                with timers['forward']():
                    outputs, targets = process_batch(
                        model, batch, args, device,
                        use_prefetcher=(args.use_prefetcher and use_real_data)
                    )
                    loss = criterion(outputs, targets)
                    loss_scaled = loss / grad_accum

                fwd_profiler.synchronize_and_collect()

                # ============= BACKWARD =============
                with timers['backward_total']():
                    with timers['backward_compute']():
                        if use_scaler:
                            scaler.scale(loss_scaled).backward()
                        else:
                            loss_scaled.backward()

                    with timers['backward_comm']():
                        torch.cuda.synchronize()

                # Collect backward component timings
                bwd_profiler.synchronize_and_collect()

        # Optimizer step
        if should_sync:
            with timers['optimizer']():
                if use_scaler:
                    scaler.unscale_(optimizer)

                with timers['grad_clip']():
                    if hasattr(model, 'clip_grad_norm_'):
                        model.clip_grad_norm_(args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        if rank == 0 and (i + 1) % 5 == 0:
            print(f"  Iteration {i + 1}/{n_profile_iters} - Loss: {loss.item():.6f}")

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # =====================================================================
    # RESULTS
    # =====================================================================
    if rank == 0:
        print()
        print("=" * 90)
        print("TOP-LEVEL PROFILING RESULTS")
        print("=" * 90)
        print()

        total_time = timers['total'].avg_ms()
        print(f"{'Phase':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'% Total':<10} {'Status'}")
        print("-" * 85)

        for phase in ['data_transfer', 'forward', 'backward_total', 'backward_compute',
                      'backward_comm', 'grad_clip', 'optimizer']:
            avg = timers[phase].avg_ms()
            min_t = timers[phase].min_ms()
            max_t = timers[phase].max_ms()
            pct = (avg / total_time * 100) if total_time > 0 else 0
            if phase == 'data_transfer' and pct > 20:
                status = "!! BOTTLENECK"
            elif phase == 'backward_total' and pct > 65:
                status = "!! High"
            elif phase == 'optimizer' and pct > 15:
                status = "!! High"
            else:
                status = "OK"
            print(f"{phase:<25} {avg:<12.2f} {min_t:<12.2f} {max_t:<12.2f} {pct:<10.1f} {status}")

        print("-" * 85)
        print(f"{'TOTAL':<25} {total_time:<12.2f} {'':<12} {'':<12} {'100.0':<10}")
        print()

        samples_per_sec = args.batch_size * 1000 / total_time if total_time > 0 else 0
        effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
        print(f"Throughput: {samples_per_sec:.1f} samples/sec/GPU")
        print(f"Effective batch size: {effective_batch}")

        # Forward breakdown
        print_forward_breakdown(fwd_profiler, timers['forward'], rank)

        # Backward breakdown
        print_backward_breakdown(bwd_profiler, timers['backward_total'], timers['grad_clip'], rank)

        # ---- Forward vs Backward comparison ----
        print("=" * 90)
        print("FORWARD vs BACKWARD COMPARISON")
        print("=" * 90)
        print()

        fwd_summary = fwd_profiler.summary()
        bwd_summary = bwd_profiler.summary()
        fwd_total = timers['forward'].avg_ms()
        bwd_total = timers['backward_total'].avg_ms()

        print(f"{'Component':<30} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Bwd/Fwd':<10} {'Note'}")
        print("-" * 80)

        for label, key in [('Encoder Embedding', 'enc_embedding'), ('Encoder', 'encoder'),
                           ('Decoder Embedding', 'dec_embedding'), ('Decoder', 'decoder'),
                           ('Projection', 'projection')]:
            fwd_ms = fwd_summary.get(key, 0)
            bwd_ms = bwd_summary.get(key, 0)
            ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0
            if ratio > 3.0:
                note = "!! Unexpectedly high"
            elif ratio < 0.5 and fwd_ms > 1:
                note = "!! Unexpectedly low"
            else:
                note = "Normal (~2x expected)"
            print(f"{label:<30} {fwd_ms:<12.2f} {bwd_ms:<12.2f} {ratio:<10.2f} {note}")

        print("-" * 80)
        ratio_total = bwd_total / fwd_total if fwd_total > 0 else 0
        print(f"{'TOTAL':<30} {fwd_total:<12.2f} {bwd_total:<12.2f} {ratio_total:<10.2f}")
        print()

        n_enc = len([k for k in fwd_summary if k.startswith('enc_layer_') and '/' not in k])
        n_dec = len([k for k in fwd_summary if k.startswith('dec_layer_') and '/' not in k])

        print(f"{'Attention Detail':<30} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Bwd/Fwd':<10}")
        print("-" * 65)
        for i in range(n_enc):
            fwd_ms = fwd_summary.get(f'enc_layer_{i}/attn_layer', 0)
            bwd_ms = bwd_summary.get(f'enc_layer_{i}/attn_layer', 0)
            ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0
            print(f"{'  Enc Layer ' + str(i) + ' Attention':<30} {fwd_ms:<12.2f} {bwd_ms:<12.2f} {ratio:<10.2f}")
        for i in range(n_dec):
            for attn_type, lbl in [('self_attn_layer', 'Self-Attn'), ('cross_attn_layer', 'Cross-Attn')]:
                fwd_ms = fwd_summary.get(f'dec_layer_{i}/{attn_type}', 0)
                bwd_ms = bwd_summary.get(f'dec_layer_{i}/{attn_type}', 0)
                ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0
                print(f"{'  Dec Layer ' + str(i) + ' ' + lbl:<30} {fwd_ms:<12.2f} {bwd_ms:<12.2f} {ratio:<10.2f}")
        print()

        # Overall distribution
        backward_total_pct = timers['backward_total'].avg_ms() / total_time * 100 if total_time > 0 else 0
        backward_compute_pct = timers['backward_compute'].avg_ms() / total_time * 100 if total_time > 0 else 0
        backward_comm_pct = timers['backward_comm'].avg_ms() / total_time * 100 if total_time > 0 else 0
        data_pct = timers['data_transfer'].avg_ms() / total_time * 100 if total_time > 0 else 0
        forward_pct = timers['forward'].avg_ms() / total_time * 100 if total_time > 0 else 0
        optimizer_pct = timers['optimizer'].avg_ms() / total_time * 100 if total_time > 0 else 0
        compute_pct = forward_pct + backward_compute_pct

        print("=" * 70)
        print("OVERALL TIME DISTRIBUTION")
        print("=" * 70)
        print(f"  Data loading:    {data_pct:5.1f}% {chr(9608) * int(data_pct / 2)}")
        print(f"  Forward:         {forward_pct:5.1f}% {chr(9608) * int(forward_pct / 2)}")
        print(f"  Backward total:  {backward_total_pct:5.1f}% {chr(9608) * int(backward_total_pct / 2)}")
        print(f"    |- compute:    {backward_compute_pct:5.1f}% {chr(9608) * int(backward_compute_pct / 2)}")
        print(f"    \\- comm:       {backward_comm_pct:5.1f}% {chr(9608) * int(backward_comm_pct / 2)}")
        print(f"  Optimizer:       {optimizer_pct:5.1f}% {chr(9608) * int(optimizer_pct / 2)}")
        print()

        # Recommendations
        print("=" * 70)
        print("ANALYSIS & RECOMMENDATIONS")
        print("=" * 70)

        if data_pct > 20:
            print(f"\n!! DATA LOADING BOTTLENECK ({data_pct:.0f}%)")
            print(f"   Increase num_workers (curr {args.num_workers}) or prefetch_factor (curr {args.prefetch_factor})")
        if backward_total_pct > 60:
            print(f"\n!! BACKWARD PASS IS {backward_total_pct:.0f}% OF TOTAL")
            print("   Check inter-node bandwidth; try HYBRID_SHARD_ZERO2")
        if compute_pct > 80:
            print(f"\n+ GOOD: {compute_pct:.0f}% time spent on compute")

        if fwd_total > 0 and bwd_total > 0:
            bwd_fwd_ratio = bwd_total / fwd_total
            if bwd_fwd_ratio > 3.0:
                print(f"\n!! BACKWARD/FORWARD RATIO = {bwd_fwd_ratio:.1f}x (expected ~2x)")
                print("   Possible causes: FSDP comm overhead, memory pressure, fragmentation")
            elif bwd_fwd_ratio < 1.5:
                print(f"\n  BACKWARD/FORWARD RATIO = {bwd_fwd_ratio:.1f}x (lower than typical 2x)")
                print("   Possible: element-wise heavy model or frozen parameters")

        # Forward-specific recommendations
        if fwd_total > 0:
            enc_emb_pct = fwd_summary.get('enc_embedding', 0) / fwd_total * 100
            dec_emb_pct = fwd_summary.get('dec_embedding', 0) / fwd_total * 100
            enc_pct = fwd_summary.get('encoder', 0) / fwd_total * 100
            dec_pct = fwd_summary.get('decoder', 0) / fwd_total * 100

            if enc_emb_pct + dec_emb_pct > 40:
                print(f"\n!! EMBEDDING IS {enc_emb_pct + dec_emb_pct:.0f}% OF FORWARD")
                print("   Consider fixed ROPE only, reducing max_len, or pre-computing embeddings")
            if enc_pct > 60:
                print(f"\n  Encoder dominates forward ({enc_pct:.0f}%)")
                for li in range(n_enc):
                    attn_ms = fwd_summary.get(f'enc_layer_{li}/attn_layer', 0)
                    ffn_ms = fwd_summary.get(f'enc_layer_{li}/ffn_conv1', 0) + fwd_summary.get(f'enc_layer_{li}/ffn_conv2', 0)
                    if attn_ms > ffn_ms * 2:
                        print(f"   Enc layer {li}: Attention-bound (attn={attn_ms:.0f}ms vs FFN={ffn_ms:.0f}ms)")
                    elif ffn_ms > attn_ms * 2:
                        print(f"   Enc layer {li}: FFN-bound (FFN={ffn_ms:.0f}ms vs attn={attn_ms:.0f}ms)")

        # GPU memory
        print()
        print("=" * 70)
        print("GPU MEMORY USAGE")
        print("=" * 70)
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem_pct = (max_mem / total_mem) * 100
        print(f"  Usage: {mem_pct:.0f}% of {total_mem:.0f} GB total")
        if mem_pct > 90:
            print("\n  !! Memory pressure! Reduce batch_size, enable CPU offload, or use activation checkpointing")
        print()
        print("=" * 70)

    # Cleanup
    fwd_profiler.remove_hooks()
    bwd_profiler.remove_hooks()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    run_profiling()
