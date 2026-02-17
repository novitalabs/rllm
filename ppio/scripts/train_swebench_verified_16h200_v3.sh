#!/bin/bash
set -x

# =============================================================================
# DeepSWE Training Script v3 for 16x H200 GPUs (2 nodes x 8 GPUs)
# Dataset: SWE-Bench Verified (500 samples, 100% PPIO template coverage)
# Model: Qwen3-32B
#
# v3 changes from v2:
#   - Batch size: 16 -> 32 (4x original; more gradient signal, higher sandbox concurrency)
#   - Learning rate: 5e-6 -> 1e-6 (revert; 5x increase didn't fix pg_clipfrac=0)
#   - ppo_max_token_len_per_gpu: 64000 -> 128000 (accommodate 4x batch)
#   - pool_size: 32 -> 512 (fix sandbox concurrency bottleneck)
#   - Resume from v1 step 15 checkpoint (same as v2)
#
# Expected concurrency: batch_size(32) x rollout_n(8) = 256 trajectories
# With sandbox pool fix, all 256 sandboxes can be created concurrently.
#
# Prerequisites:
#   - Ray cluster running across 2 nodes
#   - v1 checkpoint at /data/checkpoints/deepswe-swebench-16h200/qwen3-32b-swebench-16h200-v1
#   - Run this script from the head node inside Docker container
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"

echo "=============================================="
echo "DeepSWE Training v3 - SWE-Bench Verified - 16x H200 (2 nodes x 8 GPUs)"
echo "Resuming from v1 step 15 checkpoint"
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
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-b_manage0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
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
# v1 Checkpoint for Resume
# -----------------------------------------------------------------------------

RESUME_CHECKPOINT="/data/checkpoints/deepswe-swebench-16h200/qwen3-32b-swebench-16h200-v1"

if [ ! -d "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: v1 checkpoint not found: $RESUME_CHECKPOINT"
    exit 1
fi

ITER_FILE="$RESUME_CHECKPOINT/latest_checkpointed_iteration.txt"
CURRENT_ITER=$(cat "$ITER_FILE" 2>/dev/null)
if [ "$CURRENT_ITER" != "15" ]; then
    echo "WARNING: latest_checkpointed_iteration.txt says $CURRENT_ITER, expected 15."
    echo "Setting to 15 to resume from best validation checkpoint."
    echo 15 > "$ITER_FILE"
fi

echo "Resuming from checkpoint: $RESUME_CHECKPOINT (step 15)"

# -----------------------------------------------------------------------------
# 16x H200 GPU Configuration (2 nodes x 8 GPUs)
# -----------------------------------------------------------------------------

# Cluster topology
NNODES=2
N_GPUS_PER_NODE=8

# Parallelism
TENSOR_PARALLEL=8
SEQUENCE_PARALLEL=8

# Batch sizes — v3: 4x original for maximum gradient signal
# With 500 samples and batch_size=32, ~15 steps per epoch
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}

# Sequence lengths
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=32768

# Memory settings
GPU_MEMORY_UTILIZATION=0.7
PPO_MAX_TOKEN_LEN_PER_GPU=128000

# Sandbox pool size — must be >= batch_size * rollout_n for full concurrency
SANDBOX_POOL_SIZE=512

# Checkpoint and logging
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-32b-swebench-16h200-v3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/checkpoints/deepswe-swebench-16h200/${EXPERIMENT_NAME}}"
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-5}

echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: SWE-Bench Verified (500 samples)"
echo "Nodes: $NNODES x ${N_GPUS_PER_NODE} GPUs = $((NNODES * N_GPUS_PER_NODE)) total"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Sequence Parallel: $SEQUENCE_PARALLEL"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Rollout N: $ROLLOUT_N"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "PPO Max Token Len Per GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
echo "Sandbox Pool Size: $SANDBOX_POOL_SIZE"
echo "Learning Rate: 1e-6"
echo "Resume From: $RESUME_CHECKPOINT (step 15)"
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
    data.val_batch_size=64 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=$TRAINER_LOGGER \
    trainer.project_name='deepswe-swebench-16h200' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.resume_from_path=$RESUME_CHECKPOINT \
    rllm.env.name=swe_ppio_multistep \
    +rllm.env.env_args.sandbox_pause=False \
    +rllm.env.env_args.pool_size=$SANDBOX_POOL_SIZE \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=30 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=3600 \
    trainer.total_epochs=50
