#!/bin/bash
#===============================================================================
# SLURM SCRIPT FOR INFORMER FSDP HYBRID_SHARD TRAINING
#
# Target: 12 nodes × 2 A100 GPUs = 24 GPUs total
# Repo:   VanillaInformerROPE-Revin-HybridFSDP
#
# USAGE:
#   sbatch run_vanillarope_revin_hybridfsdp.slurm                # 12 nodes (default)
#   sbatch --nodes=4 run_vanillarope_revin_hybridfsdp.slurm      # 4 nodes
#===============================================================================

#SBATCH --job-name=informer-rope-revin
#SBATCH --mail-user=shivansh@prl.res.in
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:2                      # 2 A100 GPUs per node
#SBATCH --nodes=2                        # 4 nodes
#SBATCH --ntasks-per-node=1               # 1 task per node (torchrun handles GPU processes)
#SBATCH --cpus-per-task=8                # 12 CPUs per GPU × 2 GPUs (headroom for workers)
#SBATCH --mem=64G                         # 64GB per node — required for persistent
                                          # DataLoader workers + dual test() calls
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/informer_rope_revin_hybrid_%j.log
#SBATCH --error=logs/informer_rope_revin_hybrid_%j.error

#===============================================================================
# HEADER
#===============================================================================
echo "=========================================================================="
echo "    INFORMER ROPE-RevIN FSDP TRAINING WITH HYBRID_SHARD"
echo "    Automatically scales with number of nodes"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================================================="

#==============================nc_by_meff_Etth1.csv=================================================
# AUTOMATIC GPU AND NODE DETECTION
#===============================================================================
# Detect GPUs per node
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS=${NUM_GPUS:-2}  # Default to 2 if detection fails
fi

# Get node count
NUM_NODES=${SLURM_JOB_NUM_NODES:-12}

# Calculate world size
WORLD_SIZE=$((NUM_GPUS * NUM_NODES))

echo ""
echo "Cluster Configuration (Auto-Detected):"
echo "  - Nodes: $NUM_NODES"
echo "  - GPUs per node: $NUM_GPUS"
echo "  - Total GPUs (world_size): $WORLD_SIZE"
echo "  - CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  - Memory per node: ${SLURM_MEM_PER_NODE:-64G}"
echo ""
echo "HYBRID_SHARD Configuration:"
echo "  - Sharding: Within each node ($NUM_GPUS GPUs, fast NVLink)"
echo "  - Replication: Across $NUM_NODES nodes (gradient sync only)"
echo "  - Expected benefit: ~${NUM_NODES}x less inter-node param communication"
echo "=========================================================================="

echo ""
echo "Allocated nodes:"
scontrol show hostname $SLURM_NODELIST | nl
echo "=========================================================================="

#===============================================================================
# ENVIRONMENT VARIABLES
#===============================================================================
# Master node (first in allocation)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29504
export WORLD_SIZE=$WORLD_SIZE
export NNODES=$NUM_NODES

# NCCL Configuration for A100 GPUs
export NCCL_DEBUG=WARN                    # WARN for production, INFO for debugging
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_IB_GID_INDEX=3                # InfiniBand GID
export NCCL_SOCKET_IFNAME=^docker0,lo     # Exclude docker and loopback
export NCCL_NET_GDR_LEVEL=5               # GPU Direct RDMA

# CRITICAL for HYBRID_SHARD: Fast intra-node communication
export NCCL_P2P_LEVEL=NVL                 # Use NVLink for peer-to-peer
export NCCL_P2P_DISABLE=0                 # Enable P2P
export NCCL_SHM_DISABLE=0                 # Enable shared memory

# NCCL algorithm
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=4

# PyTorch distributed
export TORCH_DISTRIBUTED_DEBUG=OFF        # OFF for production
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# CUDA Memory Management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# CPU Threading (for DataLoader workers)
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Timeouts (generous for 12-node setup)
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

echo ""
echo "Environment Variables:"
echo "  - MASTER_ADDR: $MASTER_ADDR"
echo "  - MASTER_PORT: $MASTER_PORT"
echo "  - WORLD_SIZE: $WORLD_SIZE"
echo "  - OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  - NCCL_P2P_LEVEL: $NCCL_P2P_LEVEL"
echo "=========================================================================="

#===============================================================================
# SETUP
#===============================================================================
# Create log directory
mkdir -p logs

# Navigate to working directory
WORK_DIR="/home/akashg/time-series/python-files/Informer-revival-v4-with-flatdecoder"
cd $WORK_DIR || {
    echo "ERROR: Cannot cd to $WORK_DIR"
    exit 1
}
echo "Working directory: $(pwd)"

# ============================================================================
# Conda Environment Setup
# ============================================================================
echo "Setting up Python environment..."
source /home/akashg/miniconda3/etc/profile.d/conda.sh

echo "Activating informerworking environment..."
conda activate informerworking

# Verify Python environment
echo "Python Configuration:"
echo "  - Conda environment: $CONDA_DEFAULT_ENV"
echo "  - Python: $(which python)"
echo "  - Python version: $(python --version 2>&1)"
echo "  - PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  - CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "  - cuDNN version: $(python -c 'import torch; print(torch.backends.cudnn.version())')"
echo "  - GPUs visible: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  - BF16 supported: $(python -c 'import torch; print(torch.cuda.is_bf16_supported())')"
echo "  - torchrun: $(which torchrun)"
echo "=========================================================================="

# ============================================================================
# GPU Information (on master node)
# ============================================================================
echo "GPU Hardware Information on $(hostname):"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap --format=csv
echo "=========================================================================="

# Display GPU topology (helpful for debugging HYBRID_SHARD)
echo "GPU Topology on $(hostname):"
nvidia-smi topo -m
echo "=========================================================================="

# ============================================================================
# Test Inter-Node Connectivity
# ============================================================================
echo "Testing inter-node connectivity..."
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 \
     bash -c "echo \"Node \$(hostname) reachable (Node rank: \$SLURM_NODEID)\""
echo "=========================================================================="

#===============================================================================
# VERIFY FILES
#===============================================================================
echo ""
echo "Verifying required files..."
REQUIRED_FILES=(
    "main_informer_vanillarope_revin_hybridfsdp.py"
    "VanillaInformerROPE-Revin-hybridfsdp/exp/exp_informer.py"
    "VanillaInformerROPE-Revin-hybridfsdp/exp/exp_basic.py"
    "VanillaInformerROPE-Revin-hybridfsdp/data/data_loader.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/model.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/attn.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/encoder.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/decoder.py"
    "VanillaInformerROPE-Revin-hybridfsdp/models/embed.py"
    "VanillaInformerROPE-Revin-hybridfsdp/utils/tools.py"
    "VanillaInformerROPE-Revin-hybridfsdp/utils/RevIN.py"
    "VanillaInformerROPE-Revin-hybridfsdp/utils/masking.py"
    "VanillaInformerROPE-Revin-hybridfsdp/utils/metrics.py"
    "VanillaInformerROPE-Revin-hybridfsdp/utils/timefeatures.py"
)

ALL_OK=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  OK: $file"
    else
        echo "  MISSING: $file"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Required files missing!"
    exit 1
fi
echo "=========================================================================="

# ============================================================================
# Launch FSDP Distributed Training with torchrun
# ============================================================================
echo "Launching FSDP Training with torchrun..."
echo "Configuration Summary:"
echo "  - Nodes: $NUM_NODES"
echo "  - GPUs per node: $NUM_GPUS"
echo "  - Total GPUs: $WORLD_SIZE"
echo "  - Launcher: torchrun --nproc_per_node=$NUM_GPUS"
echo "  - Master: $MASTER_ADDR:$MASTER_PORT"
echo "  - Training script: main_informer_vanillarope_revin_hybridfsdp.py"
echo "  - FSDP Strategy: HYBRID_SHARD"
echo "  - Model: Informer + ROPE + RevIN"
echo "=========================================================================="
echo "Training start: $(date)"
echo "=========================================================================="

# Use srun to launch torchrun on each node
# Each node runs one torchrun instance, which spawns NUM_GPUS processes (1 per GPU)
srun --nodes=$NUM_NODES \
     --ntasks=$NUM_NODES \
     --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     --kill-on-bad-exit=1 \
     bash -c "
         # Set node rank
         export NODE_RANK=\$SLURM_NODEID

         echo \"[Node \$(hostname), rank \$NODE_RANK] Starting torchrun with $NUM_GPUS GPUs\"

         # Launch torchrun
         torchrun \
             --nnodes=$NUM_NODES \
             --nproc_per_node=$NUM_GPUS \
             --node_rank=\$NODE_RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             --rdzv_backend=c10d \
             --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
              main_informer_vanillarope_revin_hybridfsdp.py
     "

EXIT_CODE=$?

#===============================================================================
# SUMMARY
#===============================================================================
echo ""
echo "=========================================================================="
echo "JOB SUMMARY"
echo "=========================================================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo ""
    echo "Configuration used:"
    echo "  - Nodes: $NUM_NODES"
    echo "  - GPUs: $WORLD_SIZE"
    echo "  - Strategy: HYBRID_SHARD"
    echo "  - Model: Informer + ROPE + RevIN"
    echo ""
    echo "Results: ./results/"
    echo "Checkpoints: ./checkpoints/"
else
    echo ""
    echo "TRAINING FAILED"
    echo ""
    echo "Check logs:"
    echo "  - logs/informer_rope_revin_${SLURM_JOB_ID}.log"
    echo "  - logs/informer_rope_revin_${SLURM_JOB_ID}.error"
    echo ""
    echo "Common issues:"
    echo "  1. OOM: Reduce batch_size or enable CPU offload"
    echo "  2. NCCL timeout: Check network, increase NCCL_TIMEOUT"
    echo "  3. Import error: Check conda environment"
    echo "  4. RevIN error: Ensure utils/RevIN.py is the corrected version"
fi

echo "=========================================================================="

exit $EXIT_CODE
