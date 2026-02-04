#\!/bin/bash
# SWE-bench RL Training with PPIO Sandbox and verl backend
#
# This script trains SWE agent using:
# - PPIO sandbox for code execution
# - verl backend for distributed PPO
# - Qwen3-32B as base model
#
# Prerequisites:
# - PPIO_API_KEY environment variable set
# - Ray cluster available
# - vLLM installed

set -x

# vLLM configuration
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# PPIO configuration (load from .env if not set)
if [ -z "$PPIO_API_KEY" ]; then
    if [ -f "/home/claude/work/rft-tinker/.env" ]; then
        export $(grep PPIO_API_KEY /home/claude/work/rft-tinker/.env | xargs)
    fi
fi

# Find rllm data directory
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# Configuration options:
# - Single node (8 GPUs): nnodes=1
# - Multi node (e.g., 8 nodes): nnodes=8
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
BATCH_SIZE=${BATCH_SIZE:-8}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-50}
LR=${LR:-1e-6}

python3 -m rllm.trainer.verl.train_agent_ppo     algorithm.adv_estimator=rloo     data.train_files=${RLLM_DIR}/data/swe/R2E_Gym_Subset.parquet     data.val_files=${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet     data.train_batch_size=${BATCH_SIZE}     data.val_batch_size=512     data.max_prompt_length=4096     data.max_response_length=32768     data.filter_overlong_prompts=True     data.filter_overlong_prompts_workers=32     actor_rollout_ref.model.path=${MODEL_PATH}     actor_rollout_ref.hybrid_engine=True     actor_rollout_ref.actor.optim.lr=${LR}     actor_rollout_ref.model.use_remove_padding=True     actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum     actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE}     actor_rollout_ref.actor.use_dynamic_bsz=False     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1     actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000     actor_rollout_ref.actor.use_kl_loss=False     actor_rollout_ref.actor.clip_ratio_high=0.28     actor_rollout_ref.actor.kl_loss_coef=0.001     actor_rollout_ref.actor.kl_loss_type=low_var_kl     actor_rollout_ref.actor.ulysses_sequence_parallel_size=${GPUS_PER_NODE}     actor_rollout_ref.model.enable_gradient_checkpointing=True     actor_rollout_ref.actor.fsdp_config.param_offload=True     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True     actor_rollout_ref.rollout.tensor_model_parallel_size=${GPUS_PER_NODE}     actor_rollout_ref.rollout.name=vllm     actor_rollout_ref.rollout.mode="async"     actor_rollout_ref.rollout.enforce_eager=False     actor_rollout_ref.rollout.temperature=1.0     actor_rollout_ref.rollout.gpu_memory_utilization=0.6     actor_rollout_ref.rollout.n=${GROUP_SIZE}     actor_rollout_ref.rollout.val_kwargs.n=1     actor_rollout_ref.rollout.val_kwargs.temperature=0     actor_rollout_ref.ref.fsdp_config.param_offload=True     actor_rollout_ref.actor.entropy_coeff=0.0     algorithm.kl_ctrl.kl_coef=0.001     rllm.mask_truncated_samples=False     trainer.critic_warmup=0     trainer.logger=['console','wandb']     trainer.project_name='swe-ppio-rl'     trainer.experiment_name='swe-agent-ppio'     trainer.val_before_train=False     trainer.n_gpus_per_node=${GPUS_PER_NODE}     trainer.nnodes=${NNODES}     trainer.save_freq=10     trainer.test_freq=10     trainer.default_hdfs_dir=null     rllm.env.name=swe_ppio     rllm.env.env_args.timeout=3600     rllm.env.env_args.workdir=/testbed     rllm.env.env_args.use_pool=True     rllm.env.env_args.pool_size=32     rllm.agent.name=sweagent     rllm.agent.max_steps=${MAX_STEPS}     rllm.agent.overlong_filter=True     rllm.agent.trajectory_timeout=5400     rllm.agent.agent_args.scaffold=r2egym     rllm.agent.agent_args.use_fn_calling=False     trainer.total_epochs=1000
