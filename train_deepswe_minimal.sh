#!/bin/bash
set -x

# DeepSWE Minimal Test - Option B: TP=16 (rollout), Ulysses=8 (FSDP)
# Single-node FSDP for fast PPO update (~50s instead of 2+ hours)

cd /home/claude/work/rllm-origin

export RLLM_DIR=/home/claude/work/rllm-origin
export R2EGYM_DIR=/home/claude/work/R2E-Gym/src
export PYTHONPATH="$RLLM_DIR:$R2EGYM_DIR:$PYTHONPATH"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_MODE=disabled
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_IFNAME=b_manage0

source $RLLM_DIR/.venv/bin/activate

MODEL="/home/claude/work/rllm/models/Qwen3-32B"

echo "=== DeepSWE Minimal Test (Option B) ==="
echo "Model: $MODEL"
echo "Config: TP=16 (rollout), Ulysses=8 (FSDP)"
echo "Dataset: R2E_Gym_Mini (2 samples)"
echo "Steps: 1"

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${RLLM_DIR}/rllm/data/datasets/R2E_Gym_Mini/train_mini.parquet \
    data.val_files=${RLLM_DIR}/rllm/data/datasets/SWE_Bench_Verified/test_verl.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    rllm.filter_token_mismatch=False \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=deepswe-minimal-optionb \
    trainer.experiment_name=tp16-ulysses8-test \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe \
    +rllm.env.env_args.backend=docker \
    +rllm.env.env_args.scaffold=r2egym \
    rllm.agent.name=r2egym \
    rllm.agent.max_steps=50 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=1800 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    2>&1 | tee ~/work/logs/deepswe_minimal_optionb.log
