"""
exp_basic.py - Base Experiment Class with FSDP Support

FIXES:
1. FSDP-aware device acquisition (uses local_rank, not args.gpu)
2. Does NOT set CUDA_VISIBLE_DEVICES in FSDP mode (breaks multi-GPU)
3. Only rank 0 prints device info
"""

import os
import torch
import numpy as np


class Exp_Basic(object):
    """
    Base experiment class with FSDP-aware device management
    """
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """Build model - to be implemented by subclass"""
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        """
        Acquire the appropriate device for training
        
        CRITICAL FSDP FIX:
        - In FSDP mode, each process uses its local_rank to determine GPU
        - Do NOT set CUDA_VISIBLE_DEVICES in FSDP mode (breaks multi-GPU visibility)
        - Only rank 0 prints device information
        """
        # =====================================================================
        # FSDP MODE: Use device assigned by local_rank
        # =====================================================================
        if hasattr(self.args, 'use_fsdp') and self.args.use_fsdp:
            # Device should already be set by init_distributed_mode in main script
            if hasattr(self.args, 'device') and self.args.device is not None:
                device = self.args.device
            elif hasattr(self.args, 'local_rank'):
                # Fallback: construct device from local_rank
                local_rank = self.args.local_rank
                device = torch.device(f'cuda:{local_rank}')
            else:
                # Last resort fallback
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # Only rank 0 prints
            if self._should_print():
                print(f'FSDP Mode - Using device: {device}')
            
            return device
        
        # =====================================================================
        # STANDARD MODE: Single GPU or DataParallel
        # =====================================================================
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                # Multi-GPU with DataParallel
                # Set CUDA_VISIBLE_DEVICES only in non-FSDP mode
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Use Multi-GPU: {self.args.devices}')
            else:
                # Single GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        
        return device

    def _should_print(self):
        """Check if current process should print (only rank 0 in FSDP mode)"""
        if hasattr(self.args, 'use_fsdp') and self.args.use_fsdp:
            return getattr(self.args, 'global_rank', 0) == 0
        return True

    def _get_data(self):
        """Get data - to be implemented by subclass"""
        pass

    def vali(self):
        """Validation - to be implemented by subclass"""
        pass

    def train(self):
        """Training - to be implemented by subclass"""
        pass

    def test(self):
        """Testing - to be implemented by subclass"""
        pass
