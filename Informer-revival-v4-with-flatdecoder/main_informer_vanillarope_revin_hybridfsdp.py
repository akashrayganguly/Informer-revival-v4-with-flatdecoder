#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAXIMUM PERFORMANCE Main Script for VanillaInformerROPE-Revin-HybridFSDP

DELTAS vs your previous main_informer_vanillarope_revin_hybridfsdp.py:

  1. args.label_len = 0
        The decoder is gone; only the encoder + shared per-channel
        flatten head remain. With label_len=0 the original dataloader
        already returns seq_y of shape (pred_len, F) — see
        data/data_loader.py.__getitem__:
            r_begin = s_end - label_len      # = s_end
            r_end   = r_begin + label_len + pred_len   # = s_end + pred_len
        So data/data_loader.py and exp/exp_informer.py need NO changes.

  2. args.des = 'flatten_head'
        Cosmetic — keeps test/checkpoint folders distinct from the
        decoder-based runs.

  3. args.max_len uses max(seq_len, pred_len) since label_len = 0.
        (Original used max(seq_len, label_len + pred_len) which now
        reduces to the same thing.)

Everything else is byte-identical to your previous version, including:
  - PROJECT_DIR setup
  - setup_distributed / init_distributed_mode
  - SLURM / torchrun rendezvous handling
  - FSDP HYBRID_SHARD configuration
  - AMP, scheduler, optimizer settings
  - print_config formatting
  - dual test (vali-best + test-best) with _broadcast_flag
"""

import sys
import os
import gc
import torch
import torch.distributed as dist
from datetime import timedelta

# ============================================================================
# CRITICAL: Point to the correct repo directory
# ============================================================================
PROJECT_DIR = 'VanillaInformerROPE-Revin-hybridfsdp'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.tools import dotdict
from exp.exp_informer import Exp_Informer


def get_available_gpus():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        num_gpus = int(os.environ.get('LOCAL_WORLD_SIZE', get_available_gpus()))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        num_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', get_available_gpus()))
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['LOCAL_WORLD_SIZE'] = str(num_gpus)
    else:
        rank, world_size, local_rank = 0, 1, 0
        num_gpus = max(1, get_available_gpus())
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['LOCAL_WORLD_SIZE'] = str(num_gpus)

    return rank, world_size, local_rank, num_gpus


def init_distributed_mode(args):
    if args.use_fsdp:
        rank, world_size, local_rank, num_gpus = setup_distributed()

        args.global_rank = rank
        args.world_size = world_size
        args.local_rank = local_rank
        args.num_gpus_per_node = num_gpus

        if not dist.is_initialized() and world_size > 1:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method='env://',
                                    world_size=world_size, rank=rank,
                                    timeout=timedelta(minutes=30))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = torch.device(f'cuda:{local_rank}')
            args.gpu = local_rank
        else:
            args.device = torch.device('cpu')
            args.gpu = None

        if rank == 0:
            num_nodes = world_size // num_gpus
            print(f"\n{'='*60}")
            print(f"DISTRIBUTED: {num_nodes} nodes x {num_gpus} GPUs = {world_size} total")
            print(f"{'='*60}\n")
    else:
        args.global_rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.num_gpus_per_node = get_available_gpus()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_args():
    args = dotdict()

    # Model
    args.model = 'informer'

    # Data
    args.data = 'custom'
    args.root_path = './ETDataset/ETT-small/'
    args.data_path = 'nc_by_meff_Etth1.csv'
    args.features = 'M'
    args.target = 'data9'
    args.freq = 'h'
    args.checkpoints = './checkpoints'
    args.cols = None
    args.inverse = False

    # Sequences
    args.channel_period = 7
    args.seq_len = 336 * args.channel_period
    # ── CHANGE 1: label_len = 0. With this, the original dataloader's
    #    seq_y slice [s_end - label_len : s_end + pred_len] reduces to
    #    [s_end : s_end + pred_len] — exactly the (pred_len, F) target
    #    the flatten-head model expects. No dataloader edits needed.
    args.label_len = 0
    args.pred_len =720 * args.channel_period

    # Model params
    args.enc_in = 9
    args.dec_in = 9
    args.c_out = 1
    args.factor = 5
    args.d_model = 16
    args.n_heads = 4
    args.e_layers = 3
    args.d_layers = 1
    args.s_layers = [3, 2, 1]
    args.d_ff = 128
    args.dropout = 0.2
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.distil = True
    args.output_attention = False
    args.mix = True
    args.padding = 0
    args.stride = 1

    # RevIN and Channel Mixing (unchanged)
    args.use_revin = True
    args.channel_mix_size = None
    #args.channel_mix_size = None

    # ── max_len: with label_len=0 this is just max(seq_len, pred_len) ────
    _max_seq = max(args.seq_len, args.label_len + args.pred_len)
    args.max_len = int(_max_seq * 1.5)

    # Training
    args.batch_size = 4
    args.learning_rate = 0.0004
    args.weight_decay = 0.02
    args.loss = 'mse'
    args.lradj = 'cosine'
    args.min_lr = 1e-6
    args.warmup_ratio = 0.05
    args.use_amp = True
    args.train_epochs = 25
    args.patience = 5

    # Gradient accumulation
    args.gradient_accumulation_steps = 1
    args.max_grad_norm = 1.0

    # Data loading
    args.num_workers = 6
    args.prefetch_factor = 4
    args.use_prefetcher = True

    # Experiment
    args.itr = 1
    # ── CHANGE 2: cosmetic — keep this run's artifacts separate from
    #    the previous decoder-based runs.
    args.des = 'flatten_head'
    args.seed = 2021

    # GPU
    args.use_gpu = torch.cuda.is_available()
    args.use_fsdp = True
    args.use_multi_gpu = False
    args.gpu = 0
    args.devices = 'auto'
    args.device_ids = None

    # FSDP
    args.fsdp_sharding_strategy = 'HYBRID_SHARD'
    args.fsdp_auto_wrap_min_params = 1e6
    args.fsdp_backward_prefetch = 'BACKWARD_PRE'
    args.fsdp_cpu_offload = False
    args.fsdp_activation_checkpointing = False

    return args


def setup_data_parser(args):
    data_parser = {
        'custom': {'data': 'nc_by_meff_Etth1.csv', 'T': 'data9',
                   'M': [9, 9, 1], 'S': [1, 1, 1], 'MS': [9, 9, 1]},
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    }
    if args.data in data_parser:
        info = data_parser[args.data]
        args.data_path = info['data']
        args.target = info['T']
        args.enc_in, args.dec_in, args.c_out = info[args.features]


def print_config(args):
    if args.use_fsdp and args.global_rank != 0:
        return

    num_nodes = args.world_size // args.num_gpus_per_node
    effective_batch = args.batch_size * args.gradient_accumulation_steps * args.world_size
    _max_seq = max(args.seq_len, args.label_len + args.pred_len)

    print(f"\n{'='*60}")
    print(f"CONFIGURATION  (SHARED PER-CHANNEL FLATTEN HEAD — NO DECODER)")
    print(f"{'='*60}")
    print(f"Distributed: {num_nodes} nodes x {args.num_gpus_per_node} GPUs")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} accum x {args.world_size} = {effective_batch}")
    print(f"Strategy: {args.fsdp_sharding_strategy}")
    print(f"Data loading: {args.num_workers} workers, prefetch={args.prefetch_factor}, CUDA prefetch={args.use_prefetcher}")
    print(f"Sequences: enc={args.seq_len}, label_len={args.label_len}, pred={args.pred_len}")
    print(f"max_len: {args.max_len}  (1.5 x max_seq={_max_seq})")
    print(f"c_out: {args.c_out}  enc_in: {args.enc_in}")
    print(f"RevIN: {args.use_revin}")
    print(f"Channel Period: {args.channel_period}  Channel Mix Size: {args.channel_mix_size}")
    print(f"Optimizer: AdamW (lr={args.learning_rate}, weight_decay={args.weight_decay})")
    if args.lradj == 'plateau':
        print(f"LR Schedule: plateau  "
              f"(factor={args.plateau_factor}, patience={args.plateau_patience}, "
              f"min_lr={args.min_lr:.2e})")
    else:
        print(f"LR Schedule: {args.lradj} (warmup_ratio={args.warmup_ratio})")
    print(f"Dropout: {args.dropout}  Patience: {args.patience}  Epochs: {args.train_epochs}")
    print(f"{'='*60}\n")


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _broadcast_flag(flag_bool, args):
    """
    Broadcast a boolean decision from rank 0 to all ranks.
    Required before any conditional that gates a collective operation
    (e.g. _load_checkpoint in FSDP mode) to prevent deadlock.
    """
    if args.use_fsdp and dist.is_initialized():
        flag_tensor = torch.tensor(
            int(flag_bool),
            device=torch.device(f'cuda:{args.local_rank}')
        )
        dist.broadcast(flag_tensor, src=0)
        return bool(flag_tensor.item())
    return flag_bool


def main():
    try:
        args = create_args()

        if args.use_fsdp:
            init_distributed_mode(args)
        else:
            args.global_rank = 0
            args.world_size = 1
            args.local_rank = 0
            args.num_gpus_per_node = get_available_gpus()

        setup_data_parser(args)
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        set_seed(args.seed + args.global_rank)
        print_config(args)

        for ii in range(args.itr):
            setting = f'{args.model}_{args.data}_{args.des}_{ii}'

            if args.global_rank == 0:
                print(f"\n>>> Starting: {setting}")

            exp = Exp_Informer(args)
            exp.train(setting)

            # ── Test vali-best checkpoint (standard protocol) ──────────────
            if args.global_rank == 0:
                print(f"\n>>> Testing vali-best: {setting}")
            exp.test(setting)

            torch.cuda.empty_cache()
            gc.collect()

            # ── Test test-best checkpoint (oracle upper bound) ─────────────
            if args.use_fsdp and dist.is_initialized():
                dist.barrier()

            test_best_ckpt = os.path.join(
                args.checkpoints, setting + '_testbest', 'checkpoint.pth'
            )
            test_best_found = _broadcast_flag(
                os.path.exists(test_best_ckpt) if args.global_rank == 0 else False,
                args
            )

            if test_best_found:
                if args.global_rank == 0:
                    print(f"\n>>> Testing test-best: {setting}_testbest")
                exp.test(setting + '_testbest', load=True)
            else:
                if args.global_rank == 0:
                    print(f"\n>>> Test-best checkpoint not found, skipping.")

            torch.cuda.empty_cache()
            if args.use_fsdp and dist.is_initialized():
                dist.barrier()

        if args.global_rank == 0:
            print(f"\n{'='*60}")
            print("COMPLETED")
            print(f"{'='*60}\n")

        cleanup()

    except Exception as e:
        if 'args' in locals() and getattr(args, 'global_rank', 0) == 0:
            import traceback
            print(f"\nERROR: {e}")
            traceback.print_exc()
        cleanup()
        raise


if __name__ == "__main__":
    main()
