set -x

# DeepSWE training with Qwen3-235B-A22B (MoE: 235B total, 22B active)
# Runs on 2 nodes (16x H200): 10.83.115.22, 10.83.115.23

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=/workspace/rllm/data/swe/R2E_Gym_Subset.parquet \
    data.val_files=/workspace/rllm/data/swe/SWE_Bench_Verified.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=/data/models/Qwen3-235B-A22B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
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
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='deepscaler-agent' \
    trainer.experiment_name='swe-agent-rl-qwen3-235b-moe' \
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
