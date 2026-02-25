# Experiment Log: SWE-Bench Verified v6.4 — 64x H200 (8 nodes x 8 GPUs)

**Date:** 2026-02-23 ~ 2026-02-24
**Status:** Running (step 25 completed, step 26 in progress)
**Previous:** v6.3 (stopped after step 2, migrated to v6.4)

## v6.4 Configuration

| Parameter | v6.3 Value | v6.4 Value | Rationale |
|-----------|-----------|------------|-----------|
| ppo_mini_batch_size | 8 | **64** | = train_batch_size, 1 mini-batch/epoch × 4 = 4 optimizer steps |
| ppo_max_token_len_per_gpu | 64000 | **256000** | 8 samples/GPU per mini-batch (64/8=8) |
| All other params | — | unchanged | Same as v6.3 |

### Full Config
- 8 nodes × 8 GPUs = 64 H200, TP=8, SP=8, DP=8
- train_batch_size=64, rollout_n=8, pool_size=512 → 512 trajectories/step
- ppo_mini_batch_size=64 → 1 mini-batch/epoch × 4 epochs = **4 optimizer steps/step**
- ppo_max_token_len_per_gpu=256000
- lr=1e-6, max_steps=30, max_response=32768, trajectory_timeout=3600
- Model: Qwen3-32B (base, no resume)
- Dataset: SWE-Bench Verified (500 samples)
- RLOO advantage estimator, clip_ratio_high=0.28
- distribute_rollouts=True (8 DistributedTrajectoryWorker actors)

---

## v6.4 Step Metrics (Steps 1-25)

| Step | Score | Entropy | pg_clipfrac | Steps_mean | Token_mismatch | solve_none/all/partial | Grad_norm | Rollout(s) | Training(s) | Total(s) |
|------|-------|---------|-------------|------------|----------------|------------------------|-----------|------------|-------------|----------|
| 1 | 0.083 | 6860 | 0.0 | 17.9 | 0.297 | 47/1/16 | 156 | 5844 | 669 | 6576 |
| 2 | 0.125 | 7073 | 0.0 | 17.8 | 0.313 | 43/2/19 | 190 | 3989 | 642 | 4689 |
| 3 | 0.124 | 6800 | 0.0 | 18.2 | 0.270 | 43/1/20 | 169 | 4048 | 634 | 4747 |
| 4 | 0.129 | 6720 | 0.0 | 17.5 | 0.334 | 39/1/24 | 202 | 3996 | 633 | 4684 |
| 5 | **0.170** | 6659 | 0.0 | 18.0 | 0.262 | 40/0/24 | 212 | 3772 | 638 | 4499 |
| 6 | 0.116 | 6660 | 0.0 | 18.4 | 0.289 | 42/0/22 | 195 | 3946 | 638 | 4638 |
| 7 | 0.147 | 6919 | 0.0 | 16.1 | 0.371 | 40/0/24 | 238 | 3759 | 634 | 4451 |
| 8 | 0.149 | 6764 | 0.0 | 17.0 | 0.279 | 39/0/25 | 224 | 3964 | 635 | 4654 |
| 9 | **0.187** | 6849 | 0.0 | 17.7 | 0.273 | 35/0/29 | 255 | 4381 | 646 | 5081 |
| 10 | 0.118 | 6385 | 0.0 | 17.3 | 0.232 | 45/0/19 | 188 | 3958 | 635 | 10667* |
| 11 | **0.194** | 6517 | 0.0 | 17.5 | 0.283 | 38/1/25 | 206 | 5772 | 629 | 6454 |
| 12 | 0.081 | 6603 | 0.0 | 16.3 | 0.340 | 47/0/17 | 181 | 3790 | 635 | 4483 |
| 13 | 0.131 | 6636 | 0.0 | 18.3 | 0.305 | 46/2/16 | 179 | 4005 | 641 | 4702 |
| 14 | 0.106 | 6680 | 0.0 | 18.5 | 0.297 | 43/0/21 | 169 | 3919 | 632 | 4604 |
| 15 | 0.077 | 6602 | 0.0 | 18.2 | 0.266 | 48/0/16 | 168 | 3875 | 635 | 4596 |
| 16 | 0.173 | 6859 | 0.0 | 18.5 | 0.334 | 32/2/30 | 253 | 5309 | 635 | 5998 |
| 17 | 0.138 | 6463 | 0.0 | 17.7 | 0.352 | 43/0/21 | 214 | 3887 | 627 | 4569 |
| 18 | 0.132 | 6785 | 0.0 | 18.2 | 0.240 | 41/0/23 | 227 | 3905 | 627 | 4584 |
| 19 | 0.116 | 6568 | 0.0 | 17.6 | 0.320 | 46/1/17 | 171 | 3788 | 631 | 4472 |
| 20 | 0.153 | 6422 | 0.0 | 18.5 | 0.287 | 41/0/23 | 206 | 4133 | 668 | 11385* |
| 21 | 0.159 | 6672 | 0.0 | 17.1 | 0.307 | 37/0/27 | 211 | 4149 | 626 | 4826 |
| 22 | 0.098 | 6739 | 0.0 | 18.4 | 0.285 | 38/0/26 | 190 | 4009 | 623 | 4689 |
| 23 | 0.133 | 6748 | 0.0 | 17.8 | 0.275 | 42/0/22 | 211 | 3971 | 652 | 4677 |
| 24 | 0.170 | 6319 | 0.0 | 18.8 | 0.289 | 40/1/23 | 253 | 3785 | 627 | 4465 |
| 25 | 0.096 | 7037 | 0.0 | 18.2 | 0.262 | 48/1/15 | 172 | 4096 | 634 | 4814 |

\* Step 10 total includes validation (5995s) + checkpoint (25s). Step 20 total includes validation (6584s) + checkpoint.

### Validation Results
- **Step 10:** val/test_score=0.114, pass@k=0.102
- **Step 20:** val/test_score=0.099, pass@k=0.090

---

## Key Observations

### 1. pg_clipfrac = 0.0 — Policy Not Being Clipped (Critical)

**All 25 steps have pg_clipfrac=0.0.** This is the most significant finding.

With `ppo_mini_batch_size=64 = train_batch_size`:
- 1 mini-batch per epoch × 4 epochs = 4 optimizer steps
- Epoch 1: policy ratio ≈ 1.0 (just computed old_log_probs from same policy)
- Epochs 2-4: policy has been updated, but lr=1e-6 is so small that ratios never exceed clip threshold (0.28)

Compare across versions:
| Version | mini_batch | opt steps/step | pg_clipfrac range |
|---------|-----------|----------------|-------------------|
| v6.2 | 16 | 16 | 6.8e-5 → 1.8e-4 |
| v6.3 | 8 | 32 | 1.3e-4 → 2.3e-4 |
| **v6.4** | **64** | **4** | **0.0 (all steps)** |

Smaller mini-batches create more intra-epoch gradient diversity, which causes the policy to move enough between mini-batches to trigger some clipping. Full-batch updates with lr=1e-6 simply don't move the policy enough.

### 2. Score Plateaued — No Clear Learning Signal

- Score range: 0.077 — 0.194 (high variance, no upward trend)
- Mean score over steps 1-25: **0.132**
- v6.2 steps 1-10 mean: **0.126** (comparable)
- **But v6.2 showed clear upward trend** (0.060→0.155), while v6.4 oscillates

Score trajectory shows no monotonic improvement:
```
Steps  1-5:  0.083, 0.125, 0.124, 0.129, 0.170  (improving)
Steps  6-10: 0.116, 0.147, 0.149, 0.187, 0.118  (oscillating)
Steps 11-15: 0.194, 0.081, 0.131, 0.106, 0.077  (declining)
Steps 16-20: 0.173, 0.138, 0.132, 0.116, 0.153  (flat)
Steps 21-25: 0.159, 0.098, 0.133, 0.170, 0.096  (oscillating)
```

### 3. Validation Scores Declining

- Step 10 val: 0.114, pass@k: 0.102
- Step 20 val: **0.099, pass@k: 0.090** (worse!)
- v6.2 step 10 val: 0.131, pass@k: 0.116 (better than both v6.4 vals)

The model is not generalizing. 4 optimizer steps per rollout is insufficient to learn meaningful policy updates.

### 4. Entropy Oscillating (Not Monotonically Declining)

- Range: 6319 — 7073 over 25 steps
- No clear downward trend (v6.2 had 7044→5884 monotone decline in 10 steps)
- Entropy fluctuation suggests the policy is not consistently specializing

### 5. Grad Norm Lower Than v6.3

- v6.4 grad_norm range: 156 — 255 (mean ~200)
- v6.3 grad_norm: 485, 541 (steps 1-2)
- Lower grad norm with full-batch updates is expected (less noise), but combined with zero clipping, it means updates are too conservative

### 6. Training Time Consistent

- Mean training time: ~636s (same as v6.2 and v6.3)
- Full-batch 4 optimizer steps ≈ same total computation as 16 or 32 smaller-batch steps
- No training speed advantage from fewer optimizer steps

### 7. Rollout Time Stable

- Mean rollout: ~4100s, range 3759-5844s
- Comparable to v6.2/v6.3
- Long-tail trajectories still dominate (llm_time_max: 3643-5745s)

---

## Comparison: v6.2 vs v6.3 vs v6.4

| Metric | v6.2 (10 steps) | v6.3 (2 steps) | v6.4 (25 steps) |
|--------|-----------------|-----------------|------------------|
| mini_batch_size | 16 | 8 | 64 |
| opt steps/step | 16 | 32 | 4 |
| pg_clipfrac | 6.8e-5 → 1.8e-4 | 1.3e-4 → 2.3e-4 | **0.0** |
| Score trend | Upward (0.060→0.155) | Upward (0.085→0.131) | **Flat/oscillating** |
| val score (step 10) | **0.131** | N/A | 0.114 |
| val score (step 20) | N/A | N/A | 0.099 (declining) |
| Entropy trend | Monotone decline | Slow decline | **Oscillating** |
| Grad norm | 200-500 | 485-541 | 156-255 |
| Mean step time | ~5500s | ~4730s | ~5000s |

### Clear Conclusion

**More optimizer steps per rollout is better.** The ranking is:
1. **v6.3 (32 opt steps, mini_batch=8):** Highest pg_clipfrac, fastest score improvement per step
2. **v6.2 (16 opt steps, mini_batch=16):** Moderate clipfrac, clear upward trend, best val score
3. **v6.4 (4 opt steps, mini_batch=64):** Zero clipfrac, no learning trend, declining val score

Full-batch PPO updates (mini_batch=batch_size) with lr=1e-6 produce policy changes too small to be effective. The gradient signal exists (grad_norm > 0, pg_loss fluctuates) but is not sufficient to move the policy meaningfully within 4 steps.

---

## Training Pipeline Timing & Resource Analysis

### Step-Level Timing Breakdown (mean over 25 steps)

Each step averages **4855s (~81 min)**, excluding validation/checkpoint:

| Phase | Time (s) | % of Step | Description |
|-------|----------|-----------|-------------|
| **collect_trajectory (rollout)** | **4162** | **85.7%** | vLLM wake_up → distributed agent loop → vLLM sleep |
| update_actor (training) | 637 | 13.1% | 4 epochs × 1 full-batch PPO update via FSDP |
| old_log_prob | 55 | 1.1% | Recompute reference log probs from current policy |
| transform_trajectory | 1 | 0.0% | Convert raw trajectories to DataProto tensors |
| advantage computation | 0.1 | 0.0% | RLOO advantage estimation |

### Rollout Internal Breakdown (collect_trajectory = 4162s)

```
collect_trajectory (4162s)
│
├─ 1. vLLM wake_up (weight sync)                      ~30-60s est.
│     FSDP actor weights → 8 vLLM TP replicas
│     trainer.py:735-738, verl_engine.py:108-110
│
├─ 2. Dispatch to 8 DistributedTrajectoryWorker        ~1-2s
│     ray.remote(): chunk 512 trajectories → 8 workers × 64
│     trainer.py:744-760
│
├─ 3. Parallel trajectory execution (BOTTLENECK)       ~3700-5700s
│     512 trajectories across 8 nodes, bounded by slowest
│     │
│     ├─ 3a. env.reset()                              ~10-30s/traj
│     │     PPIO API: create_sandbox → clone_repo → install_deps
│     │     swe_ppio_multistep.py:378-442
│     │
│     ├─ 3b. Agent loop × 17.5 steps (mean)           ~1370s mean
│     │     Per agent step:
│     │     ├─ LLM inference: ~78s/step (97.9%)
│     │     │   verl_engine.py:80, 512 concurrent requests
│     │     │   across 8 vLLM replicas (load balanced)
│     │     ├─ Response parse + action extract: <0.1s
│     │     └─ env.step(action): ~1.7s/step (2.1%)
│     │         Sandbox HTTP: bash/edit/submit
│     │
│     └─ 3c. compute_final_reward()                   <0.3s
│           Cached from submit, negligible
│
├─ 4. ray.get() wait for all workers                   (included in 3)
│     trainer.py:764
│
└─ 5. vLLM sleep (release GPU for training)            ~10-30s est.
      trainer.py:777-780, verl_engine.py:112-114
```

### Per-Trajectory Timing

| Metric | Mean | Max | Ratio |
|--------|------|-----|-------|
| LLM inference | 1370s (23 min) | 3960s (66 min) | 2.89x |
| Env interaction | 29s | 531s | 18.3x |
| Total trajectory | 1399s (23 min) | 3973s (66 min) | 2.84x |

LLM accounts for **97.9%** of trajectory time. The per-step LLM time (~78s) is high because 512 concurrent trajectories share 8 vLLM replicas, creating inference queueing.

### Long-Tail Analysis

```
Trajectory time distribution:
  Mean:  1399s (23 min)
  Max:   3973s (66 min)  ← step completion bounded by this

  Idle wait = max - mean = 2574s (43 min)
  Idle wait = 62% of rollout time

  Estimated wasted GPU-hours (25 steps):
    2574s × 64 GPUs × 25 steps / 3600 = ~1144 GPU-hours
```

### Training Phase Breakdown (update_actor = 637s)

| Sub-phase | Time (s) | Description |
|-----------|----------|-------------|
| old_log_prob | 55 | Forward pass to recompute log probs |
| advantage (RLOO) | 0.1 | Advantage estimation (trivial for RLOO) |
| update_actor | 637 | 4 epochs × 1 mini-batch PPO update |
| **Total training** | **692** | All non-rollout GPU computation |

- MFU: **7.63%** (low, due to SP=8 communication overhead on long sequences)
- GPU memory allocated: 158.3 GB per GPU (reserved: 176.2 GB)
- CPU memory: 72.3 → 82.8 GB (slow growth, possible leak)
- Per-token timing: 0.070 ms/token (update_actor), 0.006 ms/token (advantage)

### Validation Phase Breakdown

Validation triggers at step 10 and 20 (test_freq=10):

| Component | Step 10 | Step 20 | Description |
|-----------|---------|---------|-------------|
| Total validation | **5995s** (100 min) | **6507s** (108 min) | `timing_s/testing` |
| Checkpoint save | 25s | 21s | `timing_s/save_checkpoint` |

Validation internal flow:
```
_validate_agent() (5995-6507s)
│
├─ 1. Load val data                                    <1s
│     500 samples, val_batch_size=512 → 1 batch
│     val_kwargs: n=1, temperature=0 (greedy)
│
├─ 2. init_envs_and_agents()                           ~5-10s
│     ThreadPoolExecutor(64): create 500 envs + agents
│
├─ 3. generate_agent_trajectory() ← DOMINATES          ~5900-6400s
│     Same distributed rollout pipeline:
│     wake_up → 8 workers × ~63 traj → ray.get → sleep
│
└─ 4. Aggregate rewards + compute metrics              <1s
      test_score, pass@k per data_source
```

**Why validation takes ~50% longer than training rollout:**

| Factor | Training Rollout | Validation |
|--------|-----------------|------------|
| Total trajectories | 512 | 500 |
| Unique problems | **64** (× 8 rollouts each) | **500** (× 1 each) |
| Temperature | 1.0 (random sampling) | 0 (greedy) |
| Problem diversity | Low (8 repeats per problem) | **High (all unique)** |
| Long-tail severity | Moderate | **Severe** |
| Average time | ~4100s | ~6250s |

The root cause is **problem diversity**: training samples 64 unique problems (each with 8 rollouts), so the 8 copies of the same problem have similar runtimes. Validation runs 500 unique problems, dramatically increasing the probability of extreme outliers that dominate wall-clock time.

### GPU Resource Consumption (25 steps)

| Phase | Per-step GPU-hrs | 25-step GPU-hrs | % |
|-------|-----------------|-----------------|---|
| Rollout (inference) | 74.0 | 1850 | **86%** |
| Training (FSDP) | 12.3 | 307 | 14% |
| **Total (excl val)** | **86.3** | **2158** | 100% |
| Validation (2 runs) | — | ~222 | extra |

### Waste & Inefficiency Summary

| Source | Impact | 25-step GPU-hours wasted |
|--------|--------|-------------------------|
| **Long-tail idle wait** | 62% of rollout time, GPUs idle | ~1144 |
| **Token mismatch (~30%)** | 30% trajectories lose gradient signal | ~555 (proportional) |
| **Low MFU (7.63%)** | Training phase underutilizes compute | ~230 (vs 30% target) |
| **Validation overhead** | ~100 min per validation, all 500 samples | ~222 (2 validations) |

---

## Issues

### 1. Zero Policy Clipping (Root Cause: Too Few Optimizer Steps)
pg_clipfrac=0 means the PPO trust region constraint is never active. The policy effectively does vanilla policy gradient with 4 full-batch steps, which at lr=1e-6 produces negligible updates.

### 2. Token Mismatch Still ~30%
Mean token_mismatch: 0.295 over 25 steps. ~30% of trajectories lose gradient signal. Consistent across all v6.x versions.

### 3. Long-Tail Trajectories
llm_time_max ranges 3643-5745s. Step time dominated by the slowest trajectory.

---

## Recommendations

### PPO Hyperparameters
1. **Return to small mini_batch_size.** mini_batch=8 (v6.3) or mini_batch=16 (v6.2) both show actual policy updates. v6.3's 32 optimizer steps showed the most promising per-step improvement.

2. **Consider increasing learning rate** instead of reducing mini_batch size. With mini_batch=64 and lr=5e-6 or 1e-5, the policy might move enough to trigger clipping while retaining the benefit of low-variance full-batch gradients.

### Rollout Efficiency
3. **Reduce trajectory_timeout** from 3600s to 2400s. Long-tail trajectories (max ~4000s) cause 62% of rollout time to be idle waiting. Capping at 2400s would lose ~5% of trajectories but save ~30% rollout time.

4. **Implement partial rollout** — start training when 80-90% of trajectories are complete instead of waiting for all 512. The streaming generator (`asyncio.as_completed`) already exists; only the batch collection logic needs modification.

### Validation Efficiency
5. **Reduce validation samples.** Currently validates all 500 problems (6000-6500s per validation). Sampling 100-200 problems would give stable estimates while cutting validation time by 60-70%.

6. **Increase test_freq** from 10 to 20. Two validations in 25 steps cost ~12500s total (222 GPU-hours). Less frequent validation saves significant wall time.

7. **Apply shorter timeout for validation.** Validation trajectories don't need gradient signal — a 2400s timeout would reduce long-tail impact with minimal accuracy loss.

### Token Mismatch
8. **Investigate retokenization mismatch** at the vLLM level. 30% of trajectories lose gradient signal due to response_masks being zeroed. This is the single largest source of wasted compute per trajectory.
