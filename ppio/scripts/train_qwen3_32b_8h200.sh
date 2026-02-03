#!/bin/bash
set -x

# =============================================================================
# DeepSWE Training Script for 8x H200 GPUs with PPIO Sandbox
# Model: Qwen3-32B
# Target: Reproduce DeepSWE 42.2% Pass@1 on SWE-Bench-Verified
# =============================================================================

# Find rllm directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"

echo "=============================================="
echo "DeepSWE Training - 8x H200 Configuration"
echo "RLLM_DIR: $RLLM_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Clear proxy for PPIO connections (PPIO SDK needs direct SSL connection)
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# Use offline mode for HuggingFace (model should be cached locally)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# WandB configuration
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# vLLM settings
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

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
# Data Preparation
# -----------------------------------------------------------------------------

TRAIN_DATA="${RLLM_DIR}/data/swe/R2E_Gym_Subset.parquet"
VAL_DATA="${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found: $TRAIN_DATA"
    echo "Please run: python3 examples/swe/prepare_swe_data.py"
    exit 1
fi

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

MODEL="Qwen/Qwen3-32B"

# -----------------------------------------------------------------------------
# 8x H200 GPU Configuration
# Total VRAM: 8 x 80GB = 640GB
# Qwen3-32B BF16: ~64GB, fits comfortably with TP=8
# -----------------------------------------------------------------------------

# GPU parallelism
N_GPUS=8
TENSOR_PARALLEL=8
SEQUENCE_PARALLEL=8

# Batch sizes (matching original DeepSWE)
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=8
ROLLOUT_N=8

# Sequence lengths
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=32768

# Memory settings (H200 has 80GB HBM3)
GPU_MEMORY_UTILIZATION=0.7
PPO_MAX_TOKEN_LEN_PER_GPU=32000

echo "=============================================="
echo "Model: $MODEL"
echo "GPUs: ${N_GPUS}x H200"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Sequence Parallel: $SEQUENCE_PARALLEL"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "=============================================="

# -----------------------------------------------------------------------------
# Training Launch
# -----------------------------------------------------------------------------

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=256 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
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
    trainer.logger=['console'] \
    trainer.project_name='deepswe-8h200' \
    trainer.experiment_name='qwen3-32b-r2e-gym' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe_ppio \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=50 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=5400 \
    trainer.total_epochs=200
