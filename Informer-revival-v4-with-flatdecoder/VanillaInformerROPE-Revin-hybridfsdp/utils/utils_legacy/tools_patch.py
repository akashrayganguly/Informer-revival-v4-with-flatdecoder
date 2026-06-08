"""
utils/tools.py - Utility Functions with FSDP Support

CRITICAL FIX: 
The original code had a bug where only rank 0 called model.state_dict() during 
checkpoint saving. In FSDP, model.state_dict() is a COLLECTIVE operation - 
ALL ranks must call it, even though only rank 0 saves the result.

The rank0_only=True in FullStateDictConfig means:
- All ranks participate in the all_gather operation
- Only rank 0 receives the complete state dict (others get empty dict)
- Only rank 0 should save to disk

This was causing the NCCL timeout because:
- Rank 0 was stuck in all_gather (waiting for other ranks)
- Ranks 1-15 had already moved to the all_reduce for early stopping sync
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
    """Standard Scaler for data normalization"""

    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class EarlyStopping:
    """
    Early stopping with FSDP support

    CRITICAL: In FSDP mode, checkpoint saving requires ALL ranks to participate
    in the state_dict gathering, even though only rank 0 saves the file.
    """

    def __init__(self, patience=7, verbose=False, delta=0, use_fsdp=False, global_rank=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            use_fsdp (bool): Whether FSDP is being used
            global_rank (int): Global rank of current process (for FSDP)
        """
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

        # Synchronize early_stop decision across all processes in FSDP
        if self.use_fsdp and dist.is_initialized():
            device = next(model.parameters()).device
            early_stop_tensor = torch.tensor(int(self.early_stop), device=device)
            dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
            self.early_stop = bool(early_stop_tensor.item())

    def save_checkpoint(self, val_loss, model, path):
        """
        Saves model when validation loss decreases.
        
        CRITICAL FIX FOR FSDP:
        ======================
        In FSDP, model.state_dict() triggers collective operations (all_gather).
        ALL ranks must call model.state_dict() for the collective to complete!
        
        The rank0_only=True in FullStateDictConfig means:
        - All ranks participate in gathering the sharded parameters
        - Only rank 0 receives the complete state dict
        - Only rank 0 should save to disk
        
        WRONG (causes deadlock):
            if rank == 0:
                state_dict = model.state_dict()  # Other ranks don't call this!
                
        CORRECT:
            state_dict = model.state_dict()  # All ranks call this
            if rank == 0:
                torch.save(state_dict, path)  # Only rank 0 saves
        """
        if self.verbose and self._should_print():
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.use_fsdp and isinstance(model, FSDP):
            # ================================================================
            # CRITICAL FIX: ALL ranks must participate in state_dict gathering!
            # ================================================================
            with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                # ALL ranks must call this - it's a collective operation!
                state_dict = model.state_dict()
                
                # Only rank 0 actually saves the file
                if self._should_print():
                    torch.save(state_dict, path + '/' + 'checkpoint.pth')
        else:
            # Standard checkpoint saving (non-FSDP)
            if self._should_print():
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), path + '/' + 'checkpoint.pth')
                else:
                    torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')

        self.val_loss_min = val_loss

    def _should_print(self):
        """Check if current process should print (only rank 0 in FSDP mode)"""
        if self.use_fsdp:
            return self.global_rank == 0
        return True


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate based on epoch"""
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
        lr = min_lr + (args.learning_rate - min_lr) * (1 + math.cos(math.pi * epoch / args.train_epochs)) / 2
        lr_adjust = {epoch: lr}
    else:
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        should_print = not hasattr(args, 'use_fsdp') or not args.use_fsdp or args.global_rank == 0
        if should_print:
            print('Updating learning rate to {}'.format(lr))


def visual(true, preds=None, name='./pic/test.pdf'):
    """Results visualization"""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def save_checkpoint_fsdp(model, optimizer, epoch, path, args):
    """
    Enhanced checkpoint saving with optimizer state
    
    CRITICAL: All ranks must participate in model.state_dict() for FSDP!
    """
    if hasattr(args, 'use_fsdp') and args.use_fsdp and isinstance(model, FSDP):
        # FSDP mode - all ranks must participate in state_dict gathering
        with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            # All ranks call this
            model_state_dict = model.state_dict()
            
            # Only rank 0 saves
            if args.global_rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, path)
    else:
        # Non-FSDP mode
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
    """Enhanced checkpoint loading with optimizer state"""
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
    """Gradient clipping with FSDP support"""
    if max_norm <= 0:
        return
        
    if hasattr(args, 'use_fsdp') and args.use_fsdp and isinstance(model, FSDP):
        model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_parameter_count(model, args):
    """Get total parameter count"""
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model, args):
    """Print model summary (only from rank 0 in FSDP mode)"""
    should_print = not hasattr(args, 'use_fsdp') or not args.use_fsdp or args.global_rank == 0

    if not should_print:
        return

    total_params = get_parameter_count(model, args)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('=' * 50)
    print('Model Summary')
    print('=' * 50)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print('=' * 50)
