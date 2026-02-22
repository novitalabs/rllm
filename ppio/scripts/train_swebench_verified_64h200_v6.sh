#!/bin/bash
set -x

# =============================================================================
# DeepSWE Training Script v6 for 64x H200 GPUs (8 nodes x 8 GPUs)
# Dataset: SWE-Bench Verified (500 samples)
# Model: Qwen3-32B
#
# v6 changes from v5 (bs=32, 2 nodes):
#   - Scaled to 8 nodes (64 GPUs)
#   - train_batch_size: 32 -> 8
#   - ppo_mini_batch_size: 8 (unchanged, but now = batch_size)
#   - FSDP param_offload: True -> False (8x more parameter sharding with
#     DP=8, memory per GPU much lower, offloading unnecessary)
#   - FSDP optimizer_offload: True -> False (same reason)
#   - ppo_max_token_len_per_gpu: 128000 -> 32000 (match v1 for bs=8)
#
# v6.1 changes:
#   - val_batch_size: 64 -> 512 (all 500 samples in 1 batch, avoid 8-batch
#     sequential validation that took ~13h at test_freq=5)
#   - test_freq: 5 -> 10 (validate less frequently)
#   - No resume checkpoint (train from base Qwen3-32B)
#
# v6.2 changes:
#   - train_batch_size: 8 -> 64 (8x increase for better gradient estimates
#     and more policy updates per step; pg_clipfrac=0 in v6.1 steps 1-3)
#   - ppo_mini_batch_size: 8 -> 16 (64/8=8 mini-batches/epoch × 4 = 32 opt steps)
#   - ppo_max_token_len_per_gpu: 32000 -> 64000 (accommodate 2 samples per
#     mini-batch per GPU: 16/8=2)
#   - SANDBOX_POOL_SIZE: 512 (matches 64×8=512 trajectories)
#
# With 8 nodes (DP=8) and bs=64:
#   - 64 samples × 8 rollouts = 512 trajectories per step
#   - ppo_mini_batch_size=16 → 64/16 = 4 mini-batches per epoch
#   - 4 epochs × 4 mini-batches = 16 optimizer steps per step
#   - Each DP rank processes 16/8 = 2 samples per mini-batch
#
# Training from base model (no resume).
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"

echo "=============================================="
echo "DeepSWE Training v6 - SWE-Bench Verified - 64x H200 (8 nodes x 8 GPUs)"
echo "Key change: 8 nodes, batch_size=64, no offloading"
echo "RLLM_DIR: $RLLM_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Proxy: disable for PPIO direct connections
if [ "${USE_PROXY:-0}" = "1" ]; then
    echo "Using proxy: ${https_proxy:-$HTTPS_PROXY}"
else
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
fi

# HuggingFace offline mode
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# WandB - determine logger based on API key availability
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -n "$WANDB_API_KEY" ]; then
    TRAINER_LOGGER="['console','wandb']"
else
    TRAINER_LOGGER="['console']"
    echo "WARNING: WANDB_API_KEY not set, logging to console only."
fi

# vLLM settings
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# NCCL settings for multi-node communication
# Each node has 8x ConnectX-7 400Gbps RoCE NICs (one per GPU: GPU0-GPU7)
# with /dev/infiniband mounted. Verified: cross-node RDMA at 23 GB/s per NIC.
# Key: NCCL_IB_GID_INDEX=3 selects the RoCE v2 GID with routable IPv4 address.
export NCCL_DEBUG=INFO  # Force INFO (Docker image may set WARN)
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-b_manage0}
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=0

# PPIO API key
if [ -z "$PPIO_API_KEY" ]; then
    if [ -f "$SCRIPT_DIR/.env" ]; then
        export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    elif [ -f "$RLLM_DIR/.env" ]; then
        export $(grep -v '^#' "$RLLM_DIR/.env" | xargs)
    fi
fi

if [ -z "$PPIO_API_KEY" ]; then
    echo "Error: PPIO_API_KEY not set. Please set it in environment or .env file."
    exit 1
fi

echo "PPIO_API_KEY: ${PPIO_API_KEY:0:10}...${PPIO_API_KEY: -4}"

# -----------------------------------------------------------------------------
# Verify Ray Cluster
# -----------------------------------------------------------------------------

echo ""
echo "Checking Ray cluster..."
if ! python3 -c "import ray; ray.init(address='auto'); print(f'Ray cluster: {ray.cluster_resources()}'); ray.shutdown()" 2>/dev/null; then
    echo "ERROR: Cannot connect to Ray cluster."
    echo "Please start the Ray cluster first."
    exit 1
fi
echo "Ray cluster OK."

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------

TRAIN_DATA="${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet"
VAL_DATA="${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found: $TRAIN_DATA"
    exit 1
fi

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

MODEL="${MODEL_PATH:-/data/models/Qwen3-32B}"

if [ ! -d "$MODEL" ]; then
    echo "WARNING: Model directory not found: $MODEL"
fi

# -----------------------------------------------------------------------------
# No Resume — training from base model
# -----------------------------------------------------------------------------

echo "Training from base model (no checkpoint resume)"

# -----------------------------------------------------------------------------
# 64x H200 GPU Configuration (8 nodes x 8 GPUs)
# -----------------------------------------------------------------------------

# Cluster topology
NNODES=8
N_GPUS_PER_NODE=8

# Parallelism: TP=8 within each node, FSDP across 8 nodes (DP=8)
TENSOR_PARALLEL=8
SEQUENCE_PARALLEL=8

# v6.2: large batch for better gradient estimates and more policy updates
# 64 samples × 8 rollouts = 512 trajectories per step
# 64 / 16 = 4 mini-batches per epoch × 4 epochs = 16 optimizer steps
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-8}

# Sequence lengths
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=32768

# Memory settings
# With DP=8, parameters sharded 8 ways → much less memory per GPU.
# No offloading needed. ppo_max_token_len_per_gpu=64000 (2 samples/GPU per mini-batch).
GPU_MEMORY_UTILIZATION=0.7
PPO_MAX_TOKEN_LEN_PER_GPU=64000

# Sandbox pool size — must be >= batch_size * rollout_n for full concurrency
SANDBOX_POOL_SIZE=512

# Checkpoint and logging
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-swebench-64h200-v6.2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/checkpoints/deepswe-swebench-64h200/${EXPERIMENT_NAME}}"
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-10}

echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: SWE-Bench Verified (500 samples)"
echo "Nodes: $NNODES x ${N_GPUS_PER_NODE} GPUs = $((NNODES * N_GPUS_PER_NODE)) total"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Sequence Parallel: $SEQUENCE_PARALLEL"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "Rollout N: $ROLLOUT_N"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "PPO Max Token Len Per GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
echo "Sandbox Pool Size: $SANDBOX_POOL_SIZE"
echo "FSDP param_offload: False (DP=8 sharding sufficient)"
echo "FSDP optimizer_offload: False (DP=8 sharding sufficient)"
echo "Learning Rate: 1e-6"
echo "Resume From: None (base model)"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "=============================================="

# -----------------------------------------------------------------------------
# Training Launch
# -----------------------------------------------------------------------------

python3 -m rllm.trainer.verl.train_agent_ppo \
    +ray_init.address="auto" \
    algorithm.adv_estimator=rloo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=512 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SEQUENCE_PARALLEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.nccl_timeout=3600 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=$TRAINER_LOGGER \
    trainer.project_name='deepswe-swebench-64h200' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    rllm.env.name=swe_ppio_multistep \
    +rllm.env.env_args.sandbox_pause=False \
    +rllm.env.env_args.pool_size=$SANDBOX_POOL_SIZE \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=30 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=3600 \
    trainer.total_epochs=200
