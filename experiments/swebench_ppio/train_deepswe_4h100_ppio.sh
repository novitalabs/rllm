#!/bin/bash
set -x

# Training script for DeepSWE reproduction on 4x H100 GPUs with PPIO sandbox
# Based on train_deepswe_32b.sh, adjusted for 4 GPUs

# Set PYTHONPATH for rllm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"

# Activate rllm venv
source "$RLLM_DIR/venv/bin/activate"

# CUDA environment
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=$CUDA_HOME/bin:$PATH

# vLLM settings
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# PPIO API key (load from .env or environment)
if [ -z "$PPIO_API_KEY" ]; then
    if [ -f "$(dirname "$0")/.env" ]; then
        export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
    fi
fi

# Verify PPIO key is set
if [ -z "$PPIO_API_KEY" ]; then
    echo "Error: PPIO_API_KEY not set. Please set it in environment or .env file."
    exit 1
fi

echo "RLLM_DIR: $RLLM_DIR"
echo "PPIO_API_KEY: ${PPIO_API_KEY:0:10}...${PPIO_API_KEY: -4}"

# Prepare data if not exists
if [ ! -f "${RLLM_DIR}/data/swe/R2E_Gym_Subset.parquet" ]; then
    echo "Preparing R2E-Gym data..."
    python3 ${SCRIPT_DIR}/prepare_r2e_gym_data.py
fi

# Model selection (choose one)
MODEL="Qwen/Qwen3-8B"     # Recommended for initial validation
# MODEL="Qwen/Qwen3-14B"  # Medium
# MODEL="Qwen/Qwen3-32B"  # Full DeepSWE (may need more memory optimization)

echo "Using model: $MODEL"

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=${RLLM_DIR}/data/swe/R2E_Gym_Subset.parquet \
    data.val_files=${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepswe-ppio' \
    trainer.experiment_name='4h100-r2e-gym' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
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
