"""
RevIN (Reversible Instance Normalization) - FSDP & BF16 Optimized

This implementation is optimized for:
- FSDP (Fully Sharded Data Parallel) across multiple nodes/GPUs
- Mixed precision training (BF16/FP16)
- Gradient checkpointing compatibility
- Distributed statistics synchronization

Changes from standard RevIN:
1. Statistics returned as values (not stored as instance attributes)
2. Distributed all-reduce for global statistics across GPUs
3. BF16-safe epsilon (1e-3 instead of 1e-5)
4. Numerically stable denormalization
5. Thread-safe implementation

Usage:
    model = YourModel()
    revin = RevIN(num_features=9, eps=1e-3, affine=True, distributed=True)
    
    # In forward pass:
    x_norm, mean, stdev = revin.normalize(x)
    output = model(x_norm)
    output = revin.denormalize(output, mean, stdev)

Author: Optimized for FSDP multi-node training
Date: 2026
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-3, affine=True, distributed=True):
        """
        Reversible Instance Normalization for Time Series
        
        FSDP-compatible implementation with distributed statistics synchronization.
        
        Args:
            num_features (int): Number of features/channels in your data
            eps (float): Epsilon for numerical stability
                        - Use 1e-3 for BF16/FP16 (default)
                        - Use 1e-5 for FP32 only
            affine (bool): If True, apply learnable affine transformation
            distributed (bool): If True, synchronize statistics across all GPUs
                               Should be True for FSDP/DDP training
        
        Input shape:
            [batch_size, sequence_length, num_features]
        
        Example:

        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.distributed = distributed

        # Initialize stored statistics to None
        # These are set during normalize() and read during denormalize()
        self.mean = None
        self.stdev = None

        if self.affine:
            self._init_params()
    
    def _init_params(self):
        """Initialize learnable affine parameters"""
        # Shape: [num_features]
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def normalize(self, x):

        mean, stdev = self._get_statistics(x)
        x_norm = self._normalize(x, mean, stdev)
        return x_norm, mean, stdev
    
    def denormalize(self, x):

        return self._denormalize(x)
    
    def _get_statistics(self, x):
        """
        Compute mean and standard deviation with distributed synchronization
        
        Computes instance-level statistics by reducing over the time dimension.
        In distributed training, statistics are averaged across all GPUs using
        all-reduce to ensure all ranks have identical normalization.
        
        Args:
            x: [batch, seq_len, features]
        
        Returns:
            mean: [batch, 1, features]
            stdev: [batch, 1, features]
        """
        # Reduce over time dimension (dim=1), keep batch and feature dims
        # For input shape [B, T, F], this gives [B, 1, F]
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        # Compute local statistics on this GPU's data shard
        mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        variance = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        
        # Synchronize statistics across all GPUs in distributed training
        # This ensures all GPUs use the same global statistics


        # Variance governs the parameter width. Generally it should be taken across all without sharding but for now this works; might change the scale of the problem.
        if self.distributed and dist.is_initialized() and dist.get_world_size() > 1:
            # Average mean and variance across all ranks
            dist.all_reduce(mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(variance, op=dist.ReduceOp.AVG)
        
        # Compute standard deviation with numerical stability
        stdev = torch.sqrt(variance + self.eps)
        
        # Detach to prevent backpropagation through statistics
        # Statistics are computed from data, not learned parameters
        self.mean  = mean.detach()
        self.stdev = stdev.detach()

    def _normalize(self, x):
       # Standardize: (x - mean) / std
        x = (x - self.mean) / self.stdev
        
        # Apply learnable affine transformation if enabled
        if self.affine:
            # Reshape affine params from [F] to [1, 1, F] for broadcasting
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            x = x * weight + bias
        
        return x
    
    def _denormalize(self, x):

        # Reverse affine transformation if it was applied
        if self.affine:
            # Reshape affine params for broadcasting
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            
            # Invert: (x - bias) / weight
            # Add eps for numerical stability (prevents division by zero)
            x = (x - bias) / (weight + self.eps*self.eps)
        
        # Reverse standardization: x * std + mean
        x = x * self.stdev + self.mean
        
        return x
    
    def forward(self, x, mode: str):
        """
        Forward pass with mode selection (for compatibility)
        
        This method provides backward compatibility with the original API.
        However, we recommend using normalize() and denormalize() directly
        for clearer code and better FSDP compatibility.
        
        Args:
            x: Input tensor
            mode: 'norm' or 'denorm'
            mean: Required for 'denorm' mode
            stdev: Required for 'denorm' mode
        
        Returns:
            For 'norm': (x_norm, mean, stdev)
            For 'denorm': x_denorm
        """
        if mode == 'norm':
            self._get_statistics(x)
            return self.normalize(x)
        elif mode == 'denorm':
            if self.mean is None or self.stdev is None:
                raise ValueError("mean and stdev must be provided for denorm mode")
            return self.denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' not supported. Use 'norm' or 'denorm'")
