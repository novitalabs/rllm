set -x

# DeepSWE training with Qwen3-30B-A3B (MoE: 30B total, 3B active)
# Runs on 2 nodes (16x H200): 10.83.115.18, 10.83.115.21
#
# Uses Megatron backend with Expert Parallel (EP) for efficient MoE training.
# EP=8: 128 experts sharded across 8 GPUs, each holds 16 experts.
# TP=1: no tensor parallel (attention/shared layers replicated).
# DP=2: data parallel across 2 groups (TP*EP=8 GPUs per group = 1 node).
#
# Parallelism layout per node (8 GPUs):
#   [EP0] [EP1] [EP2] [EP3] [EP4] [EP5] [EP6] [EP7]
# 2 nodes = DP=2

python3 -m rllm.trainer.verl.train_agent_ppo \
    --config-name=agent_ppo_trainer_megatron \
    algorithm.adv_estimator=rloo \
    data.train_files=/workspace/rllm/data/swe/R2E_Gym_Subset.parquet \
    data.val_files=/workspace/rllm/data/swe/SWE_Bench_Verified.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=/data/models/Qwen3-30B-A3B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.megatron.use_mbridge=true \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.sequence_parallel=false \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=true \
    actor_rollout_ref.actor.megatron.param_offload=true \
    actor_rollout_ref.actor.megatron.optimizer_offload=true \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=48 \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
    actor_rollout_ref.ref.megatron.use_mbridge=true \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.ref.megatron.sequence_parallel=false \
    actor_rollout_ref.ref.megatron.param_offload=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='deepscaler-agent' \
    trainer.experiment_name='swe-agent-rl-qwen3-30b-moe-megatron' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe \
    +rllm.env.env_args.backend=docker \
    +rllm.env.env_args.delete_image=True \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=30 \
    rllm.agent.overlong_filter=True \
    rllm.agent.trajectory_timeout=3600 \
    trainer.total_epochs=1000
