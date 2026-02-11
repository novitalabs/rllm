#!/bin/bash
set -x

# Disable NVLS transport to avoid NCCL error
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

# DeepSWE reproduction with Docker backend
export RLLM_DIR=/home/claude/work/rllm-origin
export R2EGYM_DIR=/home/claude/work/R2E-Gym/src
export PYTHONPATH="$RLLM_DIR:$R2EGYM_DIR:$PYTHONPATH"

export HTTP_PROXY=http://127.0.0.1:1081
export HTTPS_PROXY=http://127.0.0.1:1081

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_MODE=disabled
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

source $RLLM_DIR/.venv/bin/activate

MODEL="/home/claude/work/rllm/models/Qwen3-32B"
TRAIN_DATA="/home/claude/work/rllm-origin/rllm/data/datasets/R2E_Gym_Subset/train_verl.parquet"
VAL_DATA="/home/claude/work/rllm-origin/rllm/data/datasets/SWE_Bench_Verified/test_verl.parquet"

echo "=== DeepSWE Reproduction ==="
echo "Model: $MODEL"
echo "Train: $TRAIN_DATA"
echo "Val: $VAL_DATA"

ray stop --force 2>/dev/null || true
pkill -9 -f ray 2>/dev/null || true
sleep 2

python3 -m rllm.trainer.verl.train_agent_ppo     algorithm.adv_estimator=rloo     data.train_files=$TRAIN_DATA     data.val_files=$VAL_DATA     data.train_batch_size=8     data.val_batch_size=16     data.max_prompt_length=32768     data.max_response_length=16384     data.filter_overlong_prompts=False     actor_rollout_ref.model.path=$MODEL     actor_rollout_ref.hybrid_engine=True     actor_rollout_ref.actor.optim.lr=1e-6     actor_rollout_ref.model.use_remove_padding=True     actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum     actor_rollout_ref.actor.ppo_mini_batch_size=8     actor_rollout_ref.actor.use_dynamic_bsz=False     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1     actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000     actor_rollout_ref.actor.use_torch_compile=False     actor_rollout_ref.actor.use_kl_loss=False     actor_rollout_ref.actor.clip_ratio_high=0.28     actor_rollout_ref.actor.ulysses_sequence_parallel_size=8     actor_rollout_ref.model.enable_gradient_checkpointing=True     actor_rollout_ref.actor.fsdp_config.param_offload=True     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True     actor_rollout_ref.rollout.tensor_model_parallel_size=8     actor_rollout_ref.rollout.name=vllm     actor_rollout_ref.rollout.mode="async"     actor_rollout_ref.rollout.enforce_eager=False     actor_rollout_ref.rollout.temperature=0.6     actor_rollout_ref.rollout.top_p=0.95     actor_rollout_ref.rollout.gpu_memory_utilization=0.6     actor_rollout_ref.rollout.n=4     actor_rollout_ref.rollout.val_kwargs.n=1     actor_rollout_ref.rollout.val_kwargs.temperature=0     actor_rollout_ref.ref.fsdp_config.param_offload=True     actor_rollout_ref.actor.entropy_coeff=0.0     algorithm.kl_ctrl.kl_coef=0.001     rllm.mask_truncated_samples=False     trainer.critic_warmup=0     trainer.logger=["console"]     trainer.project_name="deepswe-reproduction"     trainer.experiment_name="youyun-37-8h200"     trainer.val_before_train=False     trainer.n_gpus_per_node=8     trainer.nnodes=1     trainer.save_freq=10     trainer.test_freq=10     trainer.default_hdfs_dir=null     rllm.env.name=swe     +rllm.env.env_args.backend=docker     +rllm.env.env_args.scaffold=sweagent     rllm.agent.name=sweagent     rllm.agent.max_steps=100     rllm.agent.overlong_filter=True     rllm.agent.trajectory_timeout=1800     trainer.total_epochs=1     trainer.total_training_steps=100
