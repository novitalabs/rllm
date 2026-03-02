# Long-Tail Trajectory Analysis

Deep analysis of the longest-running trajectories that bottleneck each training step.

**Data source**: K8s run, steps 4-9, 64 trajectories/step (8 prompts × 8 rollouts), 4 nodes × 8 H200.

---

## 1. The Core Problem

Every training step is blocked by the single slowest trajectory out of 64:

| Metric | Value |
|--------|-------|
| Mean trajectory time | 17.2 min |
| Max trajectory time (the tail) | 65.9 min |
| Tail ratio (max / mean) | **3.85×** |
| Batch wall-clock (collect_trajectory) | 70.1 min |
| Bubble (collect - max_traj) | 4.2 min |

The other 63 trajectories sit idle for **~49 min** on average after they finish, waiting for the tail to complete. Over 6 steps this amounts to **169 GPU-hours wasted**.

---

## 2. Per-Step Tail Breakdown

### 2.1 Timing Summary

| Step | Collect (min) | Tail Traj (min) | Tail Ratio | LLM (min) | Env (min) | Agent Steps | Resp Tokens |
|------|---------------|-----------------|------------|-----------|-----------|-------------|-------------|
| 4 | 70.4 | 65.6 | 3.71× | 65.6 (99.9%) | 7.6 (11.6%) | 50 | 32,593 |
| 5 | 68.3 | 63.9 | 4.07× | 63.8 (99.9%) | 4.6 (7.2%) | 50 | 32,572 |
| 6 | 62.5 | 60.0 | 3.48× | 58.2 (97.0%) | 6.1 (10.1%) | 50 | 32,768 |
| 7 | 77.5 | 72.6 | 4.46× | 69.5 (95.7%) | 7.6 (10.4%) | 50 | 32,768 |
| 8 | 67.9 | 63.8 | 3.60× | 62.1 (97.3%) | 5.2 (8.2%) | 50 | 32,447 |
| 9 | 73.8 | 69.2 | 3.76× | 58.9 (85.1%) | 14.0 (20.2%) | 50 | 32,768 |
| **AVG** | **70.1** | **65.9** | **3.85×** | **63.0** | **7.5** | **50** | **32,686** |

### 2.2 Key Observations

1. **The tail ALWAYS hits max_steps=50**. In all 6 steps, `traj/steps_max = 50`. The slowest trajectory exhausts the full step budget every time.

2. **The tail ALWAYS fills the response token budget**. `response_length/max ≈ 32,768` (the limit). These trajectories generate the maximum amount of text.

3. **LLM inference is 85-100% of tail time**. Environment execution is a secondary factor (7.5 min avg), though step 9 shows env_time_max = 14 min (837s) — indicating some SWE problems have unusually expensive Docker operations.

4. **Reward computation can spike**: Step 4 shows `reward_time_max = 300s` (5 min), equal to the `reward_timeout` limit. This was likely a test suite timeout in the final evaluation.

---

## 3. Why the Tail Trajectory Takes 2× Longer Per Step

The slowest trajectory uses ~76s per agent step vs ~39s average — a consistent **1.9× slowdown**:

| Step | Tail LLM/step | Avg LLM/step | Ratio |
|------|---------------|--------------|-------|
| 4 | 78.7s | 39.3s | 2.00× |
| 5 | 76.6s | 41.3s | 1.85× |
| 6 | 69.9s | 37.6s | 1.86× |
| 7 | 83.4s | 39.4s | 2.12× |
| 8 | 74.5s | 40.4s | 1.84× |
| 9 | 70.6s | 37.7s | 1.88× |
| **AVG** | **75.6s** | **39.3s** | **1.93×** |

Three factors combine to produce this:

### 3.1 Quadratic Attention Scaling

The tail trajectory accumulates ~32K response tokens. With a 4K prompt, the total context reaches ~37K tokens. vLLM attention computation scales roughly O(n²) for prefill and O(n) per decode step, so a 32K context is significantly more expensive per token than the 15K average.

### 3.2 vLLM Request Queuing

At the start of collect_trajectory, 64 trajectories fire requests concurrently to a single vLLM instance (TP=8, 2 replicas). With 64 concurrent multi-turn conversations, each request waits in the vLLM scheduling queue. The model can only process a limited batch size per iteration due to KV cache constraints (gpu_memory_utilization=0.6).

In early steps, all 64 trajectories compete for vLLM tokens. As trajectories complete, contention drops, but the remaining tail trajectories now have longer contexts — partly offsetting the reduced queuing.

### 3.3 Multi-Step Context Growth

Each agent step appends new tokens to the conversation. At step 50, the context is ~5× larger than at step 1. The per-step inference cost grows throughout the trajectory, making the later steps (25-50) disproportionately expensive.

Estimated context growth for the tail trajectory:
```
Step  1: ~4,000 tokens (prompt only)
Step 10: ~10,500 tokens (prompt + ~6.5K response)
Step 25: ~20,000 tokens
Step 40: ~30,000 tokens
Step 50: ~37,000 tokens (prompt + max response)
```

---

## 4. Trajectory Completion Order Analysis

From the log of the last training step (step 9, train_run35.log), the completion order of all 64 trajectories reveals:

### 4.1 Termination Reason Distribution

| Reason | Count | Completion Position | Notes |
|--------|-------|-------------------|-------|
| ENV_DONE | 50 (82%) | Spread across 1-61 | Normal termination |
| TRUNCATION | 10 (16%) | Positions 3-52 | Response token limit hit, all masked |
| MAX_STEPS | 1 (2%) | Position 59 (near-last) | Hit max_steps=50, masked |

### 4.2 Tail Trajectories (Last 10 to Complete)

| Position | Traj ID | Reason | Reward | Masked |
|----------|---------|--------|--------|--------|
| 52/61 | 28 | TRUNCATION | 0.0 | YES |
| 53/61 | 1 | ENV_DONE | 0.0 | no |
| 54/61 | 3 | ENV_DONE | 0.0 | no |
| 55/61 | 21 | ENV_DONE | 0.0 | no |
| 56/61 | 50 | ENV_DONE | 0.0 | no |
| 57/61 | 44 | ENV_DONE | 0.0 | no |
| 58/61 | 10 | ENV_DONE | 0.0 | no |
| 59/61 | 41 | MAX_STEPS | 0.0 | YES |
| 60/61 | 43 | ENV_DONE | 0.0 | no |
| 61/61 | 51 | ENV_DONE | 0.0 | no |

**Critically**: The last few to finish are mostly ENV_DONE with reward=0. These are trajectories that took many steps but ultimately failed. The MAX_STEPS trajectory (traj 41) is near-last but not the absolute last — other ENV_DONE trajectories with many steps can take even longer.

### 4.3 Head Trajectories (First 10 to Complete)

| Position | Traj ID | Reason | Reward | Masked |
|----------|---------|--------|--------|--------|
| 1/61 | 60 | ENV_DONE | 0.0 | no |
| 2/61 | 62 | ENV_DONE | 0.0 | no |
| 3/61 | 7 | TRUNCATION | 0.0 | YES |
| 4/61 | 23 | ENV_DONE | 0.0 | no |
| 5/61 | 34 | TRUNCATION | 0.0 | YES |
| 6/61 | 32 | TRUNCATION | 0.0 | YES |

TRUNCATION trajectories appear early in the completion order — these hit the 32K response token limit quickly (high token-per-step rate) rather than taking many steps.

### 4.4 Missing Trajectories

Only 61 of 64 trajectories completed before the run crashed (CUBLAS error → NCCL disconnect). Trajectories 11, 27, 52 never completed — these would have been the true tail. The run crashed at 18:15, about 60 minutes after step 9 started at ~17:14.

---

## 5. Environment Time Outliers

While LLM dominates the tail, environment time has significant variance:

| Step | Env Mean | Env Max | Max/Mean Ratio |
|------|----------|---------|----------------|
| 4 | 41.6s | 456.6s | 11.0× |
| 5 | 16.4s | 276.2s | 16.8× |
| 6 | 23.3s | 363.4s | 15.6× |
| 7 | 68.9s | 454.5s | 6.6× |
| 8 | 37.5s | 313.7s | 8.4× |
| 9 | 94.0s | 837.3s | 8.9× |

The `env_time_max` spike in step 9 (837s = 14 min) suggests a particularly expensive Docker operation, likely one of:

- **Large test suite execution**: Some repos (pandas, numpy) have test suites that take minutes to run.
- **Container cold start**: First invocation of a repo's Docker image requires extracting layers.
- **Container resource contention**: 64 Docker containers share CPU, memory, and disk I/O.

The 10 repos in the training set vary dramatically in container cost:

| Repo | Tasks | Estimated Test Cost |
|------|-------|-------------------|
| pandas | 1,444 | Heavy (large test suite, compilation) |
| numpy | 781 | Heavy (C extension compilation, large tests) |
| pillow | 620 | Medium (image processing tests) |
| orange3 | 482 | Medium |
| aiohttp | 299 | Light (async I/O tests) |
| tornado | 261 | Light |
| scrapy | 215 | Light |
| pyramid | 189 | Light |
| datalad | 179 | Medium (git operations) |
| coveragepy | 108 | Light |

---

## 6. Reward Time Outlier (Step 4)

Step 4 shows `reward_time_max = 300.0s` — exactly equal to `reward_timeout=300`. This means one trajectory's final reward computation (running the test suite to evaluate the patch) hit the timeout limit. This happens when:

1. The agent produced a complex patch that triggers many tests
2. The repository's test runner is inherently slow (e.g., pandas with thousands of tests)
3. The Docker container was under resource pressure from 63 other concurrent containers

The 300s reward timeout adds directly to the tail trajectory's total time.

---

## 7. Quantification of Waste

### 7.1 Per-Step GPU Waste

```
Average step:
├── Useful time (training):         145s  ( 3.3%)
├── Useful time (mean trajectory):  1030s (23.6%)
└── Waste (idle GPUs waiting):     3191s (73.1%)  ← 63 trajectories done, 1 still running
```

### 7.2 Aggregate Waste (6 Steps)

| Metric | Value |
|--------|-------|
| Total wall-clock | 7.3 h |
| Total GPU-hours allocated | 233 h |
| GPU-hours used for training | 7.7 h (3.3%) |
| GPU-hours idle (waiting for tail) | **169 h** (72.5%) |

### 7.3 Projected Daily Waste (~20 steps/day)

| Metric | Value |
|--------|-------|
| Wall-clock per day | 24 h (limited by step time) |
| GPU-hours wasted on tail | **563 h/day** |
| Equivalent cost (at $3/GPU-h) | $1,689/day |

---

## 8. Root Cause Summary

The tail trajectory is slow because of a **compound effect**:

```
                    ┌─────────────────────────────┐
                    │  Hard SWE problem            │
                    │  (agent can't solve it)      │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  Agent takes all 50 steps    │
                    │  (max_steps exhausted)        │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼───────┐ ┌─────▼──────┐ ┌───────▼──────┐
   │  Context grows    │ │ 50 vLLM    │ │ 50 Docker    │
   │  to ~37K tokens   │ │ round-trips│ │ round-trips  │
   │  (quadratic attn) │ │ (queuing)  │ │ (env_time)   │
   └──────────┬───────┘ └─────┬──────┘ └───────┬──────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  ~66 min total               │
                    │  (3.85× mean = 17 min)       │
                    │  Reward = 0.0 (wasted)       │
                    └─────────────────────────────┘
```

**The irony**: the slowest trajectory almost always has reward=0. It consumed the most resources but produced zero learning signal (and is often additionally masked by the overlong filter).

---

## 9. Mitigation Strategies

### 9.1 Reduce max_steps (Immediate, High Impact)

The tail always hits max_steps=50. Reducing to 30 would cut tail LLM time by ~40%:

| max_steps | Est. Tail Time | Est. Step Time | Savings/step |
|-----------|---------------|----------------|--------------|
| 50 (current) | 66 min | 70 min | — |
| 30 | ~40 min | ~44 min | ~26 min |
| 25 | ~33 min | ~37 min | ~33 min |

**Tradeoff**: Some problems may need >30 steps to solve. Analyze the step distribution of reward=1 trajectories to find the optimal cutoff. If 95% of successful trajectories complete in ≤30 steps, the cost of losing 5% of solutions is far outweighed by the 40% speedup.

### 9.2 Early Termination for Stalled Trajectories

Add a heuristic: if the agent has made no progress (e.g., same error message for N consecutive steps, or running `submit` with empty diff), terminate early. This targets the "hard problem, agent is flailing" pattern.

### 9.3 Tighten trajectory_timeout

Current: 5400s (90 min). Since the worst tail is 73 min, reducing to 3600s (60 min) would cap only the most extreme outliers while leaving >95% of trajectories unaffected.

### 9.4 Reduce vLLM Contention at Tail

When most trajectories have finished, the remaining tail trajectories still share vLLM with the infrastructure overhead of supporting 64 potential requests. Options:
- **Dynamic batching priority**: Give remaining active trajectories higher priority in the vLLM scheduler
- **Reduce max_model_len for tail**: After 40 steps, truncate early conversation turns from the context

### 9.5 Prefix Caching (Enabled by Token Accumulation Fix)

The token accumulation fix in `agent_execution_engine.py` now sends exact token prefixes to vLLM. This enables vLLM's automatic prefix caching — subsequent steps reuse the KV cache from previous steps, avoiding redundant prefill computation. This directly reduces the per-step LLM time for all trajectories, especially the tail (which has the longest prefixes).

---

## 10. Observability Gap

**Critical missing data**: The current metrics only log aggregate statistics (mean/min/max) per step. There is no way to trace a specific long-tail trajectory back to its SWE-bench problem instance. To identify which repos/problems consistently produce tails, the following should be added:

1. **Per-trajectory logging**: Include `instance_id`, `repo_name`, `steps`, `total_time`, `termination_reason` in a per-trajectory JSONL log
2. **Tail identification**: Log the trajectory index, problem ID, and timing breakdown for the top-5 slowest trajectories per step
3. **Repo-level aggregation**: Track mean/max trajectory time per repository to identify consistently expensive repos

This data would enable targeted optimizations like repo-specific timeout tuning or Docker image prewarming for expensive repos.
