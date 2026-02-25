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

## Issues

### 1. Zero Policy Clipping (Root Cause: Too Few Optimizer Steps)
pg_clipfrac=0 means the PPO trust region constraint is never active. The policy effectively does vanilla policy gradient with 4 full-batch steps, which at lr=1e-6 produces negligible updates.

### 2. Token Mismatch Still ~30%
Mean token_mismatch: 0.295 over 25 steps. ~30% of trajectories lose gradient signal. Consistent across all v6.x versions.

### 3. Long-Tail Trajectories
llm_time_max ranges 3643-5745s. Step time dominated by the slowest trajectory.

---

## Recommendations

1. **Return to small mini_batch_size.** mini_batch=8 (v6.3) or mini_batch=16 (v6.2) both show actual policy updates. v6.3's 32 optimizer steps showed the most promising per-step improvement.

2. **Consider increasing learning rate** instead of reducing mini_batch size. With mini_batch=64 and lr=5e-6 or 1e-5, the policy might move enough to trigger clipping while retaining the benefit of low-variance full-batch gradients.

3. **Reduce trajectory timeout** to 2400s to mitigate long-tail rollout time.

4. **Investigate partial rollout** — start training when 80-90% of trajectories are complete instead of waiting for all 512.
