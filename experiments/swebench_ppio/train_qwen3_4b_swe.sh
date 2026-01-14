#!/bin/bash
set -x

# Training script for Qwen3-4B on SWE-bench Lite with PPIO sandbox
# Adapted from train_deepswe_32b.sh for smaller model and local 8xRTX4090

# Set PYTHONPATH for rllm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLLM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"

# Use SDPA instead of Flash Attention due to GLIBC compatibility
export VLLM_ATTENTION_BACKEND=XFORMERS
# Disable flash attention in transformers
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

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

# Prepare data if not exists
if [ ! -f "${RLLM_DIR}/data/swe/SWE_Bench_Lite.parquet" ]; then
    echo "Preparing SWE-bench data..."
    python3.10 ${RLLM_DIR}/examples/swe/prepare_swe_data.py
fi

python3.10 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=${RLLM_DIR}/data/swe/SWE_Bench_Lite.parquet \
    data.val_files=${RLLM_DIR}/data/swe/SWE_Bench_Lite.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='qwen3-swe-ppio' \
    trainer.experiment_name='4b-swe-bench-lite' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=30 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=3600 \
    trainer.total_epochs=100
