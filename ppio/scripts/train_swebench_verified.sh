#!/bin/bash
set -x

# =============================================================================
# DeepSWE Training Script using SWE-Bench Verified Dataset
# Uses pre-built templates with 100% coverage for verifying training fixes
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"

echo "=============================================="
echo "DeepSWE Training - SWE-Bench Verified"
echo "RLLM_DIR: $RLLM_DIR"
echo "=============================================="

# Environment Setup
if [ "${USE_PROXY:-0}" = "1" ]; then
    echo "Using proxy: ${https_proxy:-$HTTPS_PROXY}"
else
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# Load PPIO API key
if [ -z "$PPIO_API_KEY" ]; then
    if [ -f "$SCRIPT_DIR/.env" ]; then
        export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    fi
fi

if [ -z "$PPIO_API_KEY" ]; then
    echo "Error: PPIO_API_KEY not set"
    exit 1
fi
echo "PPIO_API_KEY: ${PPIO_API_KEY:0:10}..."

# Use SWE-Bench Verified for both training and validation
# This has 100% template coverage with pre-cloned repos
TRAIN_DATA="${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet"
VAL_DATA="${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found: $TRAIN_DATA"
    exit 1
fi

MODEL="/models/models/Qwen3-32B"
N_GPUS=8
TENSOR_PARALLEL=8
SEQUENCE_PARALLEL=8

# Smaller batch for faster iteration during testing
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=4
ROLLOUT_N=4

MAX_PROMPT_LENGTH=8192     # Increased: some SWE-Bench samples have prompts > 4096 tokens
MAX_RESPONSE_LENGTH=32768  # Increased to match DeepSWE paper settings
GPU_MEMORY_UTILIZATION=0.7
PPO_MAX_TOKEN_LEN_PER_GPU=32000  # Match max_response_length

echo "=============================================="
echo "Training on SWE-Bench Verified (500 samples)"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "=============================================="

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=64 \
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
    trainer.project_name='deepswe-swebench' \
    trainer.experiment_name='qwen3-32b-verified-v6' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/3fsdata/data0/tengwan/checkpoints/deepswe-swebench/qwen3-32b-verified-v6 \
    rllm.env.name=swe_ppio_multistep \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=30 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=3600 \
    trainer.total_epochs=50
