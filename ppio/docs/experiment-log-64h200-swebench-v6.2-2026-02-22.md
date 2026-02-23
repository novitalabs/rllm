# Experiment Log: SWE-Bench Verified v6.2 — 64x H200 (8 nodes x 8 GPUs)

**Date:** 2026-02-22
**Status:** Stopped after step 10 (migrating to distributed rollout v6.3)

## v6.2 Configuration

| Parameter | v6 Value | v6.2 Value | Rationale |
|-----------|----------|------------|-----------|
| train_batch_size | 8 | **64** | 8x increase for better gradient estimates and more policy updates; pg_clipfrac=0 in v6 |
| ppo_mini_batch_size | 8 | **16** | 64/16 = 4 mini-batches per epoch |
| ppo_max_token_len_per_gpu | 32000 | **64000** | 2 samples per GPU per mini-batch (16/8=2) |
| SANDBOX_POOL_SIZE | 64 | **512** | Matches 64 x 8 = 512 trajectories |

### Full Config
- 8 nodes x 8 GPUs = 64 H200, TP=8, SP=8, DP=8
- train_batch_size=64, rollout_n=8, pool_size=512 -> 512 trajectories/step
- ppo_mini_batch_size=16 -> 4 mini-batches/epoch x 4 epochs = 16 optimizer steps/step
- ppo_max_token_len_per_gpu=64000
- lr=1e-6, max_steps=30, max_response=32768, trajectory_timeout=3600
- Model: Qwen3-32B (base, no resume)
- Dataset: SWE-Bench Verified (500 samples)
- RLOO advantage estimator, clip_ratio_high=0.28

---

## v6.2 Step Metrics (Steps 1-10)

| Step | Score | Entropy | pg_clipfrac | Steps_mean | Token_mismatch | Rollout(s) | Training(s) | Total(s) |
|------|-------|---------|-------------|------------|----------------|------------|-------------|----------|
| 1 | 0.060 | 7044 | 6.8e-5 | 14.6 | 0.441 | 4314 | 657 | 5026 |
| 2 | 0.127 | 6475 | 1.2e-4 | 16.6 | 0.311 | 4176 | 636 | 4871 |
| 3 | 0.126 | 6349 | 1.3e-4 | 17.5 | 0.313 | 4249 | 639 | 4952 |
| 4 | 0.117 | 6267 | 1.3e-4 | 16.6 | 0.344 | 4270 | 632 | 4959 |
| 5 | 0.140 | 5990 | 1.7e-4 | 16.6 | 0.307 | 5126 | 643 | 5854 |
| 6 | 0.104 | 6102 | 9.6e-5 | 17.3 | 0.273 | 5827 | 666 | 6549 |
| 7 | 0.155 | 6141 | 1.8e-4 | 16.6 | 0.264 | 5164 | 642 | 5860 |
| 8 | 0.134 | 6153 | 1.8e-4 | 16.7 | 0.295 | 4478 | 647 | 5176 |
| 9 | 0.152 | 5932 | 1.8e-4 | 16.0 | 0.225 | 5567 | 638 | 6258 |
| 10 | 0.143 | 5884 | 1.1e-4 | 16.8 | 0.264 | 4536 | 653 | 18143* |

\* Step 10 total includes validation (12871s) + checkpoint save (28s). Rollout+training alone = 5189s.

### Step 10 Validation Results
- **val/test_score**: 0.131 (13.1% on SWE-Bench Verified)
- **val/pass@k**: 0.116 (11.6%)
- solve_all: 2, solve_partial: 19, solve_none: 43 (out of 64 batches)

---

## Key Observations

### Score Improving
- Step 1: 0.060 -> Steps 7-10: 0.134-0.155 range
- Significant improvement vs v6 baseline (0.021 avg in v6 steps 1-4)
- Larger batch (64 vs 8) gives much more consistent gradient signal

### pg_clipfrac > 0 (Policy Updating)
- Unlike v6 which had pg_clipfrac=0 throughout, v6.2 shows non-zero clipping from step 1
- Increases from 6.8e-5 to 1.8e-4 over 9 steps
- Still very small — lr=1e-6 is conservative, but at least the policy is moving

### Entropy Declining
- 7044 (step 1) -> 5884 (step 10): policy becoming more deterministic
- Healthy sign of learning — model concentrating probability on better actions

### Token Mismatch Declining
- 0.441 (step 1) -> 0.225 (step 9): improving over time
- Still ~22% of trajectories wasted at step 9 (retokenization issue)
- v6 had ~25% average, v6.2 trending better

### Solve Rates
- solve_partial improving: 16 -> 25 out of 64 batches (some prompts with mixed success)
- Larger batch means more diverse problems, more gradient signal per step

### Rollout Dominates Step Time
- Rollout: 4176-5827s (85-89% of step time)
- Training: 632-666s (11-13% of step time)
- Average step: ~5500s (~92 min) excluding validation
- Total 10 steps: ~15.6 hours (excluding validation/checkpoint at step 10)

### Idle Nodes During Rollout (Critical Bottleneck)
- Only 2-3 of 8 nodes are actively used during rollout
- The other 5-6 nodes sit idle waiting
- Root cause: single AsyncAgentExecutionEngine on head node runs all 512 trajectories
- **This is the problem that distributed rollout (v6.3) aims to fix**

---

## Comparison with v6

| Metric | v6 (steps 1-4, bs=8) | v6.2 (steps 1-10, bs=64) |
|--------|----------------------|--------------------------|
| Score range | 0.000-0.109 | 0.060-0.155 |
| pg_clipfrac | 0.0 (all steps) | 6.8e-5 to 1.8e-4 |
| Gradient signal | Zero on step 2 | Non-zero every step |
| Trajectories/step | 64 | 512 |
| Rollout time | ~55 min | ~72-97 min |
| Training time | ~1.5 min | ~10.5 min |
| Total step time | ~57 min | ~82-109 min |

v6.2's 8x larger batch produces dramatically better learning signal at the cost of ~1.5-2x longer steps. The training time increase (1.5 min -> 10.5 min) is well-justified by the improvement in gradient quality.

---

## Issues

### 1. Single-Node Rollout Bottleneck
All 512 trajectories run in one asyncio event loop on the head node. 5-6 of 8 nodes completely idle during rollout (85%+ of step time). This is the primary optimization target.

### 2. Token Mismatch Still Significant
22-44% of trajectories lose gradient signal due to retokenization. Declining trend is encouraging but still wasteful.

### 3. Long-Tail Trajectories
Rollout time varies 4176-5827s (70-97 min) depending on the slowest trajectory in the batch. With 512 trajectories, the probability of at least one very slow trajectory is high.

---

## Next Steps: Distributed Rollout (v6.3)

Distribute trajectory execution across all 8 nodes using `DistributedTrajectoryWorker` Ray actors:
- One worker per node, each running a local AsyncAgentExecutionEngine
- Head node chunks 512 trajectories across 8 workers (~64 per node)
- Expected rollout speedup: ~4-8x (from 8 nodes being active vs 2-3)
- Target rollout time: ~500-1000s (vs current 4176-5827s)
