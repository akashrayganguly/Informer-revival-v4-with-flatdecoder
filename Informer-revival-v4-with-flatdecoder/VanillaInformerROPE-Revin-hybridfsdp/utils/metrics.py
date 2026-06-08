import numpy as np


def RSE(pred, true):
    """
    Root Squared Error

    This metric works the same for both standard and FSDP training
    as it operates on numpy arrays after data is gathered from GPUs
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Correlation coefficient

    This metric works the same for both standard and FSDP training
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Mean Absolute Error

    This metric works the same for both standard and FSDP training
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Mean Squared Error

    This metric works the same for both standard and FSDP training
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Root Mean Squared Error

    This metric works the same for both standard and FSDP training
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Mean Absolute Percentage Error

    This metric works the same for both standard and FSDP training
    """
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """
    Mean Squared Percentage Error

    This metric works the same for both standard and FSDP training
    """
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    Calculate all metrics at once

    Args:
        pred: predictions (numpy array)
        true: ground truth (numpy array)

    Returns:
        mae, mse, rmse, mape, mspe

    Note: In FSDP mode, this should be called after gathering predictions
    from all GPUs, typically only on rank 0 or after all_gather operation
    
    ✅ This function doesn't print - safe to call from any rank
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def print_metrics(pred, true, prefix="", args=None):
    """
    Print all metrics in a formatted way

    Args:
        pred: predictions
        true: ground truth
        prefix: string prefix for printing (e.g., "Train", "Val", "Test")
        args: arguments object (to check if using FSDP and rank)

    ✅ FIXED: Only prints from rank 0 in FSDP mode
    """
    # Check if we should print (only rank 0 in FSDP mode)
    should_print = True
    if args is not None:
        if hasattr(args, 'use_fsdp') and args.use_fsdp:
            should_print = hasattr(args, 'global_rank') and args.global_rank == 0

    if not should_print:
        return

    mae, mse, rmse, mape, mspe = metric(pred, true)

    print('=' * 60)
    if prefix:
        print(f'{prefix} Metrics:')
    else:
        print('Metrics:')
    print('=' * 60)
    print(f'MAE:  {mae:.6f}')
    print(f'MSE:  {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'MAPE: {mape:.6f}')
    print(f'MSPE: {mspe:.6f}')
    print('=' * 60)

    return mae, mse, rmse, mape, mspe
