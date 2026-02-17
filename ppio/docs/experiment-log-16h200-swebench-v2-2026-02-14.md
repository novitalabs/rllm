# 16xH200 SWE-Bench Verified v2 Training Experiment Log

Date: 2026-02-13 ~ 2026-02-14
Cluster: 2 nodes x 8x NVIDIA H200 (141GB HBM3e each)
Model: Qwen3-32B (resumed from v1 step 15 checkpoint)
Script: `ppio/scripts/train_swebench_verified_16h200_v2.sh`
Dataset: SWE_Bench_Verified (500 samples)

## Changes from v1

| Parameter | v1 Value | v2 Value | Rationale |
|-----------|----------|----------|-----------|
| optim.lr | 1e-6 | **5e-6** | 5x increase; v1 pg_clipfrac=0 indicates policy barely updates |
| train_batch_size | 8 | **16** | 2x batch for better gradient estimates |
| ppo_mini_batch_size | 8 | **16** | Match train_batch_size |
| ppo_max_token_len_per_gpu | 32000 | **64000** | Accommodate doubled batch |
| resume_from | — | v1 step 15 | Best v1 validation (test_score=0.121) |

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| nnodes | 2 |
| n_gpus_per_node | 8 |
| train_batch_size | **16** |
| ppo_mini_batch_size | **16** |
| rollout_n | 8 |
| tensor_parallel | 8 |
| sequence_parallel | 8 |
| max_prompt_length | 8192 |
| max_response_length | 32768 |
| gpu_memory_utilization | 0.7 |
| ppo_max_token_len_per_gpu | **64000** |
| optim.lr | **5e-6** |
| clip_ratio_high | 0.28 |
| use_kl_loss | False |
| entropy_coeff | 0.0 |
| adv_estimator | rloo |
| loss_agg_mode | seq-mean-token-sum |
| total_epochs | 50 |
| save_freq | 5 |
| test_freq | 5 |
| agent.max_steps | 30 |
| trajectory_timeout | 3600s |
| sandbox_pause | False |
| fsdp param_offload | True |
| fsdp optimizer_offload | True |
| val_before_train | False |

## Training Progress

Ran for **11 steps** before manual stop for v3 parameter tuning.
Total trajectories: ~1,583 (255 reward=1.0, 1,328 reward=0.0 — **16.1% solve rate**, up from v1's 10.4%).

### Step-by-step Metrics

| Step | score/mean | pg_loss | grad_norm | entropy | pg_clipfrac | token_mismatch | resp_len/mean | solve (n/p/a) | step_time (s) |
|------|-----------|---------|-----------|---------|-------------|----------------|---------------|---------------|---------------|
| 1 | 0.057 | -2.05 | 315 | 6,657 | 0.0 | 0.25 | 16,366 | 12/4/0 | 7,079 |
| 2 | 0.057 | -0.36 | 335 | 6,597 | 0.0 | 0.30 | 18,608 | 12/4/0 | 7,841 |
| 3 | 0.051 | 0.42 | 238 | 6,601 | 0.0 | 0.28 | 14,785 | 14/2/0 | 7,603 |
| 4 | 0.102 | 2.21 | 372 | 6,151 | 0.0 | 0.34 | 15,864 | 11/5/0 | 8,057 |
| **5** | **0.160** | 4.97 | 370 | 6,211 | 0.0 | 0.27 | 14,738 | **7/9/0** | 19,232* |
| 6 | 0.116 | 0.36 | 390 | 6,067 | 0.0 | 0.28 | 16,867 | 11/5/0 | 8,003 |
| 7 | 0.184 | 1.41 | 455 | 6,159 | 0.0 | 0.35 | 17,229 | 9/7/0 | 9,517 |
| 8 | 0.189 | 6.77 | 439 | 7,608 | 0.0 | 0.34 | 17,073 | 9/7/0 | 8,261 |
| **9** | **0.191** | 3.82 | 371 | 5,760 | 0.0 | 0.19 | 17,872 | **10/5/1** | 8,002 |
| **10** | 0.109 | 0.13 | 418 | 6,116 | 0.0 | 0.30 | 18,781 | 11/5/0 | 17,400* |
| 11 | 0.148 | 1.05 | 428 | 6,381 | 0.0 | 0.26 | 18,171 | **7/9/0** | 7,867 |

*Steps 5, 10 include validation + checkpoint save.

### Timing Breakdown (per step)

| Phase | Average (s) | % of step |
|-------|------------|-----------|
| collect_trajectory | 1,094 | 14% |
| old_log_prob | 115 | 1% |
| update_actor | 6,603 | 83% |
| testing (val steps) | ~10,264 | — |
| save_checkpoint | ~80 | — |
| **Total (non-val step)** | **~8,100** | 100% |
| **Total (val step)** | **~18,300** | — |

### Validation Scores

| Step | test_score | pass@k | Notes |
|------|-----------|--------|-------|
| 5 | 0.096 | 0.086 | Below v1 peak |
| 10 | **0.088** | **0.08** | Declining |

### Key Observations

1. **pg_clipfrac=0 and ppo_kl=0 persist despite 5x lr increase (1e-6 → 5e-6)**
   - Policy updates remain too small to trigger PPO clipping
   - Suggests the issue is deeper than lr alone — FSDP param/optimizer offload may dilute gradients or reduce effective precision

2. **Solve rate improved: 16.1% vs v1's 10.4%**
   - Step 9: first batch with `solve_all=1` (all 8 rollouts of one problem solved)
   - More batches with high solve_partial counts (7/9, 9/7)
   - Possible explanation: larger batch (16 vs 8) provides more diverse problems, including easier ones

3. **Validation declining: 0.096 → 0.088**
   - Same pattern as v1 (val peaked at step 15 then declined)
   - Model overfits to training distribution while losing generalization
   - With pg_clipfrac=0, the training signal is effectively noise — random walks in parameter space

4. **update_actor dominates step time (83%)**
   - ~6,600s per step for FSDP forward/backward + all-reduce
   - Doubled from v1 (~3,200s) due to 2x batch size
   - collect_trajectory is fast (~1,100s) relative to training

5. **Step 9 had lowest token_mismatch (0.19)** — occasional good batches exist
   - Average token_mismatch: 0.29 (slightly higher than v1's 0.25)

## Conclusions

1. **5x lr increase did not fix pg_clipfrac=0**: The fundamental issue persists. With FSDP param_offload=True and optimizer_offload=True, the optimizer state may lose precision during offload/reload cycles, effectively zeroing out the policy KL divergence.

2. **Larger batch size helps solve rate but hurts step time**: 16.1% vs 10.4% solve rate, but each step takes ~2x longer.

3. **Validation confirms overfitting**: Both v1 and v2 show declining val scores after initial improvement, despite pg_clipfrac=0.

4. **v3 direction**: Increase batch_size to 32 (more gradient signal per step), revert lr to 1e-6 (higher lr didn't help pg_clipfrac). Focus on sandbox concurrency to reduce collect_trajectory time.

## v3 Hyperparameter Changes

| Parameter | v2 Value | v3 Value | Rationale |
|-----------|----------|----------|-----------|
| train_batch_size | 16 | **32** | 4x original; more diverse gradient signal |
| ppo_mini_batch_size | 16 | **32** | Match train_batch_size |
| optim.lr | 5e-6 | **1e-6** | Revert: higher lr didn't fix pg_clipfrac=0 |
| ppo_max_token_len_per_gpu | 64000 | **128000** | Accommodate 4x batch |
| pool_size | 32 (default) | **512** | Fix sandbox concurrency bottleneck |

See: `ppio/scripts/train_swebench_verified_16h200_v3.sh`
