# Experiment Log: SWE-Bench Verified v6.4 — 64x H200 (8 nodes x 8 GPUs)

**Date:** 2026-02-23 ~ 2026-02-25
**Status:** Crashed after step 28 (during step 29 rollout). Ray worker OOM/SYSTEM_ERROR.
**Previous:** v6.3 (stopped after step 2, migrated to v6.4)
**Last checkpoint:** global_step_25

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

## v6.4 Step Metrics (Steps 1-28)

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
| 26 | **0.192** | 6573 | 0.0 | 17.8 | 0.270 | 34/3/27 | 213 | 4716 | 631 | 5402 |
| 27 | 0.124 | 6428 | 0.0 | 18.5 | 0.254 | 43/0/21 | 211 | 3779 | 632 | 4468 |
| 28 | 0.150 | 6868 | 0.0 | 18.5 | 0.324 | 41/1/22 | 201 | 5453 | 626 | 6132 |

\* Step 10 total includes validation (5995s) + checkpoint (25s). Step 20 total includes validation (6584s) + checkpoint.

**Step 29:** Crashed during rollout — Ray worker died (SYSTEM_ERROR). See [Crash Analysis](#crash-analysis-step-29) below.

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
Steps 26-28: 0.192, 0.124, 0.150                 (oscillating)
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

| Metric | v6.2 (10 steps) | v6.3 (2 steps) | v6.4 (28 steps) |
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
| Outcome | Stopped (manual) | Stopped (migrated) | **Crashed (OOM)** |

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

### 4. Crash Analysis (Step 29) {#crash-analysis-step-29}

**Crash point:** During step 29 rollout (trajectory execution phase), after step 28 completed successfully.

**Error:**
```
The actor is dead because its worker process has died.
Worker exit type: SYSTEM_ERROR
Worker exit detail: Worker unexpectedly exits with a connection error code 2.
End of file.
Potential root causes: (1) SIGKILL by OOM killer (2) ray stop --force (3) SIGSEGV
```

**Cascading failure:** After the Ray worker died, all NCCL ProcessGroup heartbeat monitors across the cluster reported `Broken pipe` errors to the TCPStore on host-10-83-115-14:34403 (head node). This means one worker death brought down the entire distributed training job.

**Probable cause: CPU OOM.** CPU memory trajectory:
```
Step 1:  72.3 GB
Step 10: 80.2 GB
Step 20: 82.2 GB
Step 25: 82.8 GB
Step 26: 82.9 GB
Step 27: 83.1 GB
Step 28: 84.2 GB  ← 1.1 GB jump (largest since step 5→10)
```

The step 27→28 jump of +1.1 GB (vs typical +0.1-0.6 GB) suggests accelerating memory pressure. The step 28 rollout was also unusually long (5453s, `llm_time_max=5378s` — one of the longest trajectories across all steps), which would have required more concurrent state in memory. During step 29 rollout, with ~84+ GB CPU memory and 512 concurrent trajectories each holding environment state, the worker likely exceeded the node's available RAM.

**Impact:** Steps 26-28 ran without checkpointing (save_freq=5, last checkpoint at step 25). These 3 steps of training (~15,000s = 4.2 hours) are lost. The model state reverted to step 25.

**Saved checkpoints:** global_step_5, 10, 15, 20, 25

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

---

## Detailed Analysis (28 Steps)

### 1. Reward Distribution

Over 28 steps × 512 trajectories = 14,336 total trajectories (10,717+ logged with completion reasons, excluding Ray log deduplication):

| Reward | Count | % | Description |
|--------|-------|---|-------------|
| 0.0 (fail) | 9,215 | **86.3%** | No tests passed |
| 1.0 (full solve) | 1,277 | **12.0%** | All tests passed |
| 0 < r < 1 (partial) | 191 | **1.8%** | Some tests passed |

**Key observations:**
- Reward is overwhelmingly binary: 98.3% of trajectories get exactly 0.0 or 1.0
- Only 1.8% achieve partial credit — the reward landscape is essentially pass/fail
- With RLOO advantage estimation, the advantage for each trajectory is computed relative to the mean reward of its 8 rollouts for the same problem
- Since most problems are either always-fail (64.7%) or mixed (34.3%), the advantage signal is sparse

**Partial reward breakdown:**
- 0.3333 (88 trajectories), 0.5 (88), 0.6667 (24), 0.8077 (22), 0.7055 (14) — a few distinctive partial scores from multi-test problems

### 2. Termination Reasons

| Reason | Count | % | Description |
|--------|-------|---|-------------|
| ENV_DONE | 9,299 | **86.8%** | Agent submitted solution and env terminated normally |
| TRUNCATION | 621 | **5.8%** | Response exceeded max_response=32768 tokens |
| MAX_STEPS | 412 | **3.8%** | Agent reached max_steps=30 without submitting |
| ENV_TIMEOUT | 351 | **3.3%** | Trajectory exceeded trajectory_timeout=3600s |
| PROMPT_OVERLONG | 34 | **0.3%** | Prompt exceeded max_prompt_length=8192 tokens |

**Analysis:**
- 86.8% terminate normally (ENV_DONE) — the agent learns to submit solutions
- TRUNCATION (5.8%): These trajectories hit the 32K token response limit. The response_length/clip_ratio (1.08% mean) measures a different thing — it counts samples at exactly max length, which is lower because truncated trajectories often get shorter assembled responses
- MAX_STEPS (3.8%): Agent uses all 30 steps without submitting. These trajectories consume maximum LLM time (~78s × 30 = 2340s) and contribute disproportionately to long-tail
- ENV_TIMEOUT (3.3%): Trajectories hitting the 3600s wall-clock limit. These are the primary long-tail offenders
- PROMPT_OVERLONG (0.3%): 34 trajectories filtered due to prompts exceeding 8192 tokens. These generate zero gradient signal (immediate discard)

**Recommendation:** MAX_STEPS + ENV_TIMEOUT account for 7.1% of trajectories but dominate wall-clock time. Reducing max_steps from 30 to 25 and trajectory_timeout from 3600s to 2400s would reduce long-tail with minimal reward impact (these trajectories almost always score 0.0).

### 3. Per-Problem Solve Consistency

Each step samples 64 unique problems with 8 rollouts each. Over 28 steps, 1,792 problem-batches:

| Category | Count | % | Description |
|----------|-------|---|-------------|
| solve_none | 1,161 | **64.8%** | All 8 rollouts failed (reward=0.0 for all) |
| solve_partial | 614 | **34.3%** | Some rollouts succeeded, some failed |
| solve_all | 17 | **0.9%** | All 8 rollouts succeeded (reward=1.0 for all) |

**Analysis:**
- **64.7% of problems are never solved** in any of 8 attempts → zero advantage signal for these samples (all rewards identical → RLOO advantage = 0)
- **1.0% always solved** → also zero advantage signal (all rewards identical)
- **34.3% partially solved** → these are the ONLY samples that provide non-zero advantage signal
- Effective training signal comes from only ~34% of the batch in each step
- This means ~66% of compute (rollout time) generates no gradient signal at all, even before token mismatch filtering

**Implication for RLOO:** The RLOO advantage estimator requires variance within rollout groups. With 65.7% of problems producing uniform outcomes (all-fail or all-success), the advantage distribution is extremely sparse. This partially explains the weak learning signal — even with more optimizer steps (v6.2/v6.3), the fundamental constraint is that only ~34% of samples carry any gradient information.

### 4. Token Mismatch Root Cause Analysis

**Scale:** 3,667 retokenization mismatch warnings logged across 26 steps. With 512 trajectories/step × 26 steps = 13,312 total, the mismatch rate is ~27.5% (consistent with the per-step mean of 29.5%).

**Pattern from log analysis:**

Typical mismatch example:
```
Position 6335: Expected [52604, 11525, 13, 13824, 11], Got [422, 37527, 316, 347, 2425]
Position 7508: Expected [272, 20008, 7197, 13, 3197], Got [13665, 2539, 7197, 13, 3197]
Position 8214: Expected [1890, 4835, 49253, 2578, 13216], Got [76168, 49253, 2578, 13216, 15279]
```

**Root cause mechanism:**
1. During rollout, vLLM generates multi-step responses. Each step's response tokens are stored individually
2. During `assemble_steps()`, all step responses are concatenated into one long sequence
3. This concatenated sequence is re-tokenized from text to verify token alignment
4. The re-tokenization produces different token IDs at step boundaries because tokenizers use **context-dependent BPE merges** — the same text produces different tokens depending on surrounding context
5. When mismatch is detected, `response_masks` is set to all-zeros → the trajectory contributes zero gradient signal

**The mismatch occurs at step boundaries (positions 6335, 7508, 8214 — mid-response)** where the concatenation of two separately-tokenized segments produces different BPE merges than tokenizing the full concatenation at once. This is a fundamental property of BPE tokenization with Qwen3's vocabulary.

**Impact:** 29.5% of trajectories × 86% rollout cost = ~25% of total GPU-hours produce zero gradient. Combined with the 65.7% solve_none rate (which produces zero advantage), the effective useful compute fraction is approximately:
- 34.3% × 70.5% = **24.2%** of trajectories provide actual gradient signal

### 5. Response Length Analysis

| Metric | Value |
|--------|-------|
| Mean response length | 15,911 tokens |
| Min step-mean | 14,436 tokens (step 7) |
| Max step-mean | 16,992 tokens (step 25) |
| Max possible | 32,768 tokens |
| Mean clip ratio | 1.08% (hit max length) |
| Clip ratio range | 0.39% — 2.34% |
| Aborted ratio | 0.0% (all steps) |

**Response length per step (no trend):**
```
Steps  1-5:  15608, 15588, 16543, 15585, 16890
Steps  6-10: 16360, 14436, 15640, 15939, 15952
Steps 11-15: 15597, 15325, 15588, 16102, 16670
Steps 16-20: 15763, 15919, 16285, 15435, 16316
Steps 21-26: 15022, 16217, 15821, 16148, 16992, 15950
```

No upward or downward trend in response length — the policy is not becoming more or less verbose over training. This is consistent with the flat entropy trajectory.

**Correlation with score:** High-score steps don't systematically have different response lengths. Step 11 (score=0.194, highest) has mean length 15,597 — almost exactly the overall mean. Step 15 (score=0.077, lowest) has 16,670 — slightly above average. No meaningful length-score correlation.

### 6. CPU Memory Growth

| Step | CPU Memory (GB) | Delta |
|------|----------------|-------|
| 1 | 72.3 | — |
| 5 | 73.9 | +1.6 |
| 10 | 80.2 | +6.3 |
| 15 | 80.9 | +0.7 |
| 20 | 82.2 | +1.3 |
| 25 | 82.8 | +0.6 |
| 26 | 82.9 | +0.1 |
| 27 | 83.1 | +0.2 |
| 28 | **84.2** | **+1.1** ← accelerating |

**Growth pattern:**
- Total growth: 72.3 → 84.2 GB = **+11.9 GB over 28 steps**
- Average rate: **0.44 GB/step**
- Growth is NOT linear — fastest between steps 5-10 (+6.3 GB), slowing after step 15, then re-accelerating at step 28 (+1.1 GB)
- The step 5→10 jump coincides with the first validation (step 10), which creates 500 additional environments
- **The step 27→28 jump (+1.1 GB) preceded the step 29 OOM crash**

**Projection and outcome:**
- Step 28: 84.2 GB → crash during step 29 rollout
- The crash confirms CPU memory pressure is a real operational risk
- The growth may be from:
  1. Ray object store accumulation (trajectory results not fully GC'd)
  2. Python garbage collector not reclaiming large trajectory buffers promptly
  3. Sandbox connection pool metadata growth
  4. Long-tail trajectories holding more concurrent state (step 28 had llm_time_max=5378s)

**GPU memory:** Stable at 158.3 GB allocated / 176.2 GB reserved per GPU across all 28 steps. No GPU memory growth.

### 7. Cross-Step Problem Overlap & Advantage Distribution

**Problem sampling statistics** (500 problems, 64 per step, 28 steps):

| Metric | Value |
|--------|-------|
| Total problem-samples | 1,792 |
| Expected samples per problem | 3.6 |
| Problems never seen | ~11 (2.2%) |
| Problems seen at least once | ~489 (97.8%) |

**Simulated sampling distribution:**
```
0 times:  11 problems (2.2%)
1 time:   52 problems (10.4%)
2 times:  99 problems (19.8%)
3 times:  118 problems (23.6%)  ← mode
4 times:  104 problems (20.8%)
5 times:  71 problems (14.2%)
6 times:  35 problems (7.0%)
7+ times: 10 problems (2.0%)
```

Most problems are sampled 2-4 times over 26 steps. The dataset is well-utilized — 97% of problems are seen at least once.

**Advantage distribution:**

| Metric | Value |
|--------|-------|
| Advantage mean (across steps) | -0.0085 |
| Advantage mean range | -0.0353 to +0.0198 |
| Advantage max | 1.0 |
| Advantage min | -1.0 (most steps), -0.857 (some steps) |

The advantage mean is near zero (as expected for RLOO), but the per-step mean fluctuates around zero with no trend. The advantage range is always [-1, 1] or [-0.857, 1]:
- Max advantage = 1.0: a trajectory that solved when all 7 siblings failed (reward 1.0, mean 1/8 ≈ 0.125, advantage ≈ 0.875, clipped to 1.0)
- Min advantage = -1.0: a trajectory that failed when all siblings also failed but one solved, yielding negative advantage
- Min = -0.857 = -6/7: exactly 1 of 8 rollouts solved (advantage for failing trajectories = 0 - 1/8 × 7/6 ≈ -0.857 under RLOO normalization)

**The advantage signal is extremely sparse.** With 64.7% solve_none (all 8 fail → advantage = 0 for all) and 1.0% solve_all (all 8 succeed → advantage = 0 for all), only the ~34% solve_partial problems generate non-zero advantages. Within those, the advantage is binary: +1 for success, ~-0.14 to -1.0 for failure. This is a very noisy, high-variance signal.

### pg_loss Behavior

| Metric | Value |
|--------|-------|
| pg_loss mean | 1.17 |
| pg_loss range | -2.46 to 4.75 |
| pg_loss trend | No trend (oscillating) |

The pg_loss fluctuates widely between steps (-2.46 to +4.75) with no trend. This is consistent with the sparse, high-variance advantage signal — each step's loss depends heavily on which specific problems were sampled and whether any rollout groups had mixed outcomes.

---

## Summary: Why v6.4 Doesn't Learn

The convergence failure is multi-factorial:

1. **Too few optimizer steps (4):** Full-batch PPO with lr=1e-6 moves the policy so little that pg_clipfrac = 0.0 for all 28 steps. The trust region constraint is never active.

2. **Sparse reward signal:** Only 34.3% of problem-batches have mixed solve outcomes (the only source of non-zero RLOO advantage). 65.7% of batches produce zero gradient.

3. **Token mismatch:** An additional 29.5% of trajectories lose gradient signal due to retokenization mismatches, bringing effective useful trajectories down to ~24%.

4. **High-variance advantage:** Among the ~24% of useful trajectories, the advantage is binary (+1 or ~-0.14 to -1.0), creating noisy gradient estimates that require many optimizer steps to average out — which v6.4's 4 steps cannot provide.

5. **CPU memory leak:** Memory grew from 72.3 to 84.2 GB over 28 steps (0.44 GB/step), causing an OOM crash during step 29 rollout.

6. **Result:** The policy oscillated randomly within a narrow score band (0.077-0.194) with no upward trend across 28 steps, consuming ~2,400 GPU-hours of compute before crashing. Last usable checkpoint: step 25.
