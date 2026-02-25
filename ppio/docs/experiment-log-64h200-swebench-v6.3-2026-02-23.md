# Experiment Log: SWE-Bench Verified v6.3 — 64x H200 (8 nodes x 8 GPUs)

**Date:** 2026-02-23
**Status:** Running (step 2 completed)
**Previous:** v6.2 (stopped after step 10, score 0.060→0.155)

## v6.3 Configuration

| Parameter | v6.2 Value | v6.3 Value | Rationale |
|-----------|-----------|------------|-----------|
| ppo_mini_batch_size | 16 | **8** | 64/8 = 8 mini-batches/epoch × 4 = 32 optimizer steps (2x more updates) |
| distribute_rollouts | False | **True** | Distribute trajectory execution across all 8 nodes |
| All other params | — | unchanged | Same as v6.2 |

### Full Config
- 8 nodes × 8 GPUs = 64 H200, TP=8, SP=8, DP=8
- train_batch_size=64, rollout_n=8, pool_size=512 → 512 trajectories/step
- ppo_mini_batch_size=8 → 8 mini-batches/epoch × 4 epochs = 32 optimizer steps/step
- ppo_max_token_len_per_gpu=64000 (1 sample/GPU/mini-batch: 8/8=1)
- lr=1e-6, max_steps=30, max_response=32768, trajectory_timeout=3600
- Model: Qwen3-32B (base, no resume from v6.2)
- Dataset: SWE-Bench Verified (500 samples)
- RLOO advantage estimator, clip_ratio_high=0.28

### Distributed Rollout Architecture
- `DistributedTrajectoryWorker` Ray actors: 1 per node × 8 nodes
- Each worker runs a local `AsyncAgentExecutionEngine` with 64 parallel agents
- Head node chunks 512 trajectories across 8 workers (64 per node)
- Head node manages vLLM lifecycle (wake_up/sleep); workers use `StubRolloutManager`
- vLLM inference remains distributed across all 8 replicas (unchanged from v6.2)

---

## v6.3 Step Metrics (Steps 1-2)

| Step | Score | Entropy | pg_clipfrac | Steps_mean | Token_mismatch | Rollout(s) | Training(s) | Total(s) |
|------|-------|---------|-------------|------------|----------------|------------|-------------|----------|
| 1 | 0.085 | 6901 | 1.3e-4 | 17.7 | 0.305 | 4162 | 634 | 4855 |
| 2 | 0.131 | 6864 | 2.3e-4 | 17.6 | 0.270 | 3891 | 650 | 4604 |

### Detailed Step 1 Metrics
- batch/solve_none: 48, solve_all: 0, solve_partial: 16
- actor/grad_norm: 484.7
- actor/pg_loss: 12.9
- traj/llm_time_mean: 1447s, traj/llm_time_max: 3803s
- traj/env_time_mean: 24.7s, traj/env_time_max: 428s
- traj/total_time_max: 3813s (long-tail bottleneck)
- response_length/mean: 15545, response_length/max: 32768

### Detailed Step 2 Metrics
- batch/solve_none: 40, solve_all: 0, solve_partial: 24
- actor/grad_norm: 540.8
- actor/pg_loss: 15.4
- traj/llm_time_mean: 1449s, traj/llm_time_max: 3693s
- traj/env_time_mean: 31.9s, traj/env_time_max: 1802s
- traj/total_time_max: 3699s
- response_length/mean: 16151, response_length/max: 32768

---

## Key Observations

### Score Improving Quickly
- Step 1: 0.085 → Step 2: 0.131
- v6.2 had 0.060 → 0.127 for same steps (comparable trajectory)
- 32 optimizer steps per step (vs 16 in v6.2) → stronger per-step policy updates

### pg_clipfrac Higher Than v6.2
- Step 1: 1.3e-4 (vs v6.2 step 1: 6.8e-5 — nearly 2x)
- Step 2: 2.3e-4 (vs v6.2 step 2: 1.2e-4 — nearly 2x)
- More mini-batches (8 vs 4 per epoch) → more gradient updates → policy moves faster
- Still conservative — lr=1e-6 keeps updates small

### solve_partial Improving
- Step 1: 16/64 → Step 2: 24/64 (significant jump)
- v6.2 had 16 → 21 for same steps
- More optimizer steps per rollout seem to help

### Rollout Time: Marginal Improvement Over v6.2
- v6.3 step 1: 4162s, step 2: 3891s
- v6.2 step 1: 4314s, step 2: 4176s
- **Only ~4-7% faster** — far from the expected 4-8x speedup

### Why Distributed Rollout Didn't Help More

The expected 4-8x rollout speedup did not materialize. Root cause analysis:

1. **vLLM inference was already distributed in v6.2.** The `AsyncLLMServerManager` load-balances LLM requests across all 8 vLLM replicas on all nodes. Even in v6.2's single-node agent loop, all 8 nodes' GPUs were active for inference.

2. **The bottleneck is LLM inference time, not agent orchestration.** Per-trajectory breakdown:
   - LLM time: 1447-1449s mean (98% of trajectory time)
   - Env time: 25-32s mean (2% of trajectory time)
   - The agent loop overhead (env.step, sandbox HTTP calls) is negligible

3. **Long-tail trajectories dominate.** The slowest trajectory in each step takes ~3700-3800s, and the step cannot complete until ALL trajectories finish. With 64 trajectories per worker (vs 512 on one node in v6.2), the P(max > 3600s) is only slightly lower.

4. **What distributed rollout DID fix:** Agent loop orchestration is now truly parallel — each node runs its own asyncio event loop for 64 trajectories. But since the agent loop is only 2% of trajectory time, distributing it saves very little wall-clock time.

### Training Time Consistent
- 634-650s for 32 optimizer steps (vs 632-666s for 16 steps in v6.2)
- Training time roughly equal despite 2x more optimizer steps — this is because mini_batch_size=8 (vs 16) means each mini-batch has half the tokens, so each step is ~2x faster

### Entropy Declining Slowly
- 6901 → 6864 over 2 steps (37-point drop)
- v6.2 had 7044 → 6475 (569-point drop in same span)
- This may be noise from different batch sampling; need more steps to see trend

---

## Comparison: v6.2 vs v6.3

| Metric | v6.2 (step 1-2) | v6.3 (step 1-2) | Diff |
|--------|-----------------|-----------------|------|
| Score | 0.060, 0.127 | 0.085, 0.131 | Similar |
| pg_clipfrac | 6.8e-5, 1.2e-4 | 1.3e-4, 2.3e-4 | **~2x higher** |
| Token mismatch | 0.441, 0.311 | 0.305, 0.270 | Better |
| solve_partial | 16, 21 | 16, 24 | Better |
| Rollout(s) | 4314, 4176 | 4162, 3891 | ~4-7% faster |
| Training(s) | 657, 636 | 634, 650 | Same |
| Total(s) | 5026, 4871 | 4855, 4604 | ~3-5% faster |
| Optimizer steps/step | 16 | 32 | **2x more** |
| Mini-batch size | 16 | 8 | Halved |

### Key Takeaway
The main benefit of v6.3 is **2x more optimizer steps per rollout** (from mini_batch_size=8 vs 16), not the distributed rollout. The distributed rollout provides minimal speedup because the bottleneck is LLM inference (already distributed via vLLM) and long-tail trajectories, not agent loop orchestration.

---

## Issues

### 1. Long-Tail Trajectories (Primary Bottleneck)
The slowest trajectory takes 3700-3800s (~63 min), while the mean is ~1470s (~25 min). This 2.5x ratio means most workers sit idle for ~40 min waiting for stragglers. Possible mitigations:
- Reduce trajectory_timeout from 3600s to 2400s (cap long-tail at cost of losing some signal)
- Implement trajectory-level early stopping (if model is clearly stuck)
- Use async step submission (don't wait for all trajectories)

### 2. Token Mismatch Still 27-30%
~30% of trajectories have retokenization mismatches (response_masks set to 0). Declining but still wastes significant gradient signal.

### 3. Distributed Rollout Overhead Minimal but Present
The distributed workers add Ray actor communication overhead (~1-2s per step for chunking/gathering results). Negligible for ~4000s steps.

---

## Next Steps

1. **Monitor v6.3 training** — let it run to step 10+ and compare learning curves with v6.2
2. **Consider trajectory timeout reduction** — 3600s → 2400s to reduce long-tail impact
3. **Investigate token mismatch root cause** — can retokenization be fixed at the vLLM level?
4. **Evaluate whether 2x optimizer steps (32 vs 16) produces better converged score** — the pg_clipfrac increase suggests faster policy movement
