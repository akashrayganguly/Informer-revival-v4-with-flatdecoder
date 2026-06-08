#!/bin/bash
#===============================================================================
# SLURM SCRIPT FOR INFORMER FSDP PROFILER (FORWARD + BACKWARD BREAKDOWN)
#
# Profiles your actual Informer model with detailed forward AND backward pass
# component timing: per-module gradient propagation, FSDP comm overhead,
# gradient clipping, and forward vs backward comparison.
#
# USAGE:
#   sbatch run_profiler_fwd_bwd.slurm                # Default (2 nodes, 2 GPUs)
#   sbatch --nodes=4 run_profiler_fwd_bwd.slurm      # 4 nodes
#===============================================================================

#SBATCH --job-name=informer-fwd-bwd-profile
#SBATCH --mail-user=shivansh@prl.res.in
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:2                      # GPUs per node
#SBATCH --nodes=2                         # Default: 2 nodes
#SBATCH --ntasks-per-node=1               # 1 task per node
#SBATCH --cpus-per-task=8                 # CPUs for DataLoader workers
#SBATCH --mem=32G                         # Memory per node
#SBATCH --time=01:30:00                   # 1.5 hours (backward profiling adds overhead)
#SBATCH --output=logs/profile_fwd_bwd_%j.log
#SBATCH --error=logs/profile_fwd_bwd_%j.error

#===============================================================================
# HEADER
#===============================================================================
echo "=========================================================================="
echo "    INFORMER FSDP PERFORMANCE PROFILER (FORWARD + BACKWARD BREAKDOWN)"
echo "    Forward: per-module timing (attention, FFN, embedding, etc.)"
echo "    Backward: per-module gradient propagation, FSDP comm, grad clipping"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================================================="

#===============================================================================
# AUTOMATIC GPU AND NODE DETECTION
#===============================================================================
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS=${NUM_GPUS:-2}
fi

NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
WORLD_SIZE=$((NUM_GPUS * NUM_NODES))

echo ""
echo "Profiler Configuration:"
echo "  - Nodes: $NUM_NODES"
echo "  - GPUs per node: $NUM_GPUS"
echo "  - Total GPUs (world_size): $WORLD_SIZE"
echo "  - CPUs per task: $SLURM_CPUS_PER_TASK"
echo "=========================================================================="

#===============================================================================
# ENVIRONMENT VARIABLES - Same as training script
#===============================================================================
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29502
export WORLD_SIZE=$WORLD_SIZE
export NNODES=$NUM_NODES

# NCCL Configuration for A100 GPUs
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_NET_GDR_LEVEL=5

# Fast intra-node communication
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=4

# PyTorch distributed
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# CUDA Memory Management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# CPU Threading
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Timeouts
export NCCL_TIMEOUT=600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

echo ""
echo "Environment Variables:"
echo "  - MASTER_ADDR: $MASTER_ADDR"
echo "  - MASTER_PORT: $MASTER_PORT"
echo "  - WORLD_SIZE: $WORLD_SIZE"
echo "  - OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "=========================================================================="

#===============================================================================
# SETUP
#===============================================================================
mkdir -p logs

WORK_DIR="/home/akashg/time-series/python-files/Informer-RPE-Revin-Sharding-Etth-dropout-cout1"
cd $WORK_DIR || {
    echo "ERROR: Cannot cd to $WORK_DIR"
    exit 1
}
echo "Working directory: $(pwd)"

#===============================================================================
# Conda Environment Setup
#===============================================================================
echo "Setting up Python environment..."
source /home/akashg/miniconda3/etc/profile.d/conda.sh
conda activate informerworking

echo "Python Configuration:"
echo "  - Conda environment: $CONDA_DEFAULT_ENV"
echo "  - Python: $(which python)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================================================="

#===============================================================================
# GPU Information
#===============================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================================================="

#===============================================================================
# VERIFY FILES
#===============================================================================
echo "Verifying required files..."

REQUIRED_FILES=(
    "profiler_fwd_bwd.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/model.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/encoder.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/decoder.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/attn.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/embed.py"
    "VanillaInformerROPE-Revin-hybridfsdp/data/data_loader.py"
    "VanillaInformerROPE-Revin-hybridfsdp/utils/tools.py"
    "VanillaInformerROPE-Revin-hybridfsdp/utils/RevIN.py"
)

ALL_OK=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  OK $file"
    else
        echo "  MISSING: $file"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Required files missing!"
    echo "Make sure profiler_fwd_bwd.py is in $WORK_DIR"
    echo "and VanillaInformerROPE-Revin-hybridfsdp/ exists."
    exit 1
fi
echo "=========================================================================="

#===============================================================================
# LAUNCH PROFILER
#===============================================================================
echo ""
echo "Launching Informer FSDP Forward + Backward Profiler..."
echo "  - Nodes: $NUM_NODES"
echo "  - GPUs per node: $NUM_GPUS"
echo "  - Total GPUs: $WORLD_SIZE"
echo "  - FSDP Strategy: HYBRID_SHARD"
echo "=========================================================================="
echo "Profiling start: $(date)"
echo "=========================================================================="

srun --nodes=$NUM_NODES \
     --ntasks=$NUM_NODES \
     --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     --kill-on-bad-exit=1 \
     bash -c "
         export NODE_RANK=\$SLURM_NODEID
         
         echo \"[Node \$(hostname), rank \$NODE_RANK] Starting fwd+bwd profiler with $NUM_GPUS GPUs\"
         
         torchrun \
             --nnodes=$NUM_NODES \
             --nproc_per_node=$NUM_GPUS \
             --node_rank=\$NODE_RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             --rdzv_backend=c10d \
             --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
             profiler_fwd_bwd.py
     "

EXIT_CODE=$?

#===============================================================================
# SUMMARY
#===============================================================================
echo ""
echo "=========================================================================="
echo "PROFILING COMPLETE"
echo "=========================================================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "PROFILING COMPLETED SUCCESSFULLY"
    echo ""
    echo "Review the output above for:"
    echo ""
    echo "  FORWARD PASS BREAKDOWN:"
    echo "    - RevIN, Embeddings (ROPE pos/chan), Encoder/Decoder layers"
    echo "    - Per-layer: Attention vs FFN vs LayerNorm"
    echo "    - Q/K/V projections, Inner attention (FlashAttention)"
    echo ""
    echo "  BACKWARD PASS BREAKDOWN:"
    echo "    - Per-module gradient propagation (reverse order)"
    echo "    - Projection -> Decoder -> DecEmb -> Encoder -> EncEmb -> RevIN"
    echo "    - Attention backward vs FFN backward per layer"
    echo "    - Gradient clipping time"
    echo "    - Autograd/FSDP overhead (untracked time)"
    echo ""
    echo "  FORWARD vs BACKWARD COMPARISON:"
    echo "    - Per-component backward/forward ratio (~2x expected for matmuls)"
    echo "    - Flags anomalies (>3x or <0.5x ratios)"
    echo ""
    echo "  OVERALL: Data vs compute vs communication, memory, recommendations"
else
    echo ""
    echo "PROFILING FAILED"
    echo ""
    echo "Check logs:"
    echo "  - logs/profile_fwd_bwd_${SLURM_JOB_ID}.log"
    echo "  - logs/profile_fwd_bwd_${SLURM_JOB_ID}.error"
    echo ""
    echo "Common issues:"
    echo "  1. OOM: Backward hooks add slight memory overhead for CUDA events"
    echo "  2. Data not found: Check root_path in profiler_fwd_bwd.py"
    echo "  3. Hook errors: Ensure PyTorch >= 2.0 for register_full_backward_pre_hook"
fi

echo "=========================================================================="
exit $EXIT_CODE
