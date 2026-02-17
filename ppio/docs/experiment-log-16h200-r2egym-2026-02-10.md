# 16xH200 R2E-Gym Training Experiment Log

Date: 2026-02-10
Cluster: 2 nodes x 8x NVIDIA H200 (141GB HBM3e each)
Model: Qwen3-32B
Script: `ppio/scripts/train_qwen3_32b_16h200.sh`
Dataset: R2E_Gym_Subset (train: 4578), SWE_Bench_Verified (val: 500)

## Cluster Topology

| Internal IP    | Hostname             | Role   |
|----------------|----------------------|--------|
| 10.83.115.14   | host-10-83-115-14    | HEAD   |
| 10.83.115.17   | host-10-83-115-17    | Worker |

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| nnodes | 2 |
| n_gpus_per_node | 8 |
| train_batch_size | 16 |
| ppo_mini_batch_size | 16 |
| rollout_n | 8 |
| tensor_parallel | 8 |
| sequence_parallel | 8 |
| max_prompt_length | 4096 |
| max_response_length | 32768 |
| gpu_memory_utilization | 0.7 |
| total_epochs | 200 |
| total_training_steps | 57,200 |
| agent.max_steps | 50 |
| trajectory_timeout | 5400s |
| sandbox_pause | False |

## Training Progress

Ran for **5 steps** (~12 hours of wall time) before being manually stopped.

### Step-by-step Metrics

| Step | score/mean | pg_loss | grad_norm | steps_mean | token_mismatch | response_len/mean | step_time (s) |
|------|-----------|---------|-----------|------------|----------------|-------------------|---------------|
| 1 | 0.0 | 0.0 | 0.0 | 4.42 | 0.80 | 4,172 | 7,667 |
| 2 | 0.0 | 0.0 | 0.0 | 0.45 | 0.97 | 410 | 7,218 |
| 3 | 0.0 | 0.0 | 0.0 | 3.46 | 0.80 | 3,669 | 8,974 |
| 4 | 0.0 | 0.0 | 0.0 | 0.36 | 0.96 | 459 | 7,422 |
| 5 | 0.0 | 0.0 | 0.0 | 3.77 | 0.82 | 3,644 | 14,286* |

*Step 5 includes validation (test_score=0.0927, pass@k=0.084) and checkpoint save.

### Validation at Step 5
- `val/test_score/unknown`: 0.0927
- `val/test_score/pass@k/unknown`: 0.084 (8.4%)

## Issues Found

### 1. Hydra Config: `sandbox_pause` Key Not in Struct

**Symptom:**
```
Could not override 'rllm.env.env_args.sandbox_pause'.
Key 'sandbox_pause' is not in struct
```

**Fix:** Changed `rllm.env.env_args.sandbox_pause=False` to `+rllm.env.env_args.sandbox_pause=False` (Hydra `+` prefix for adding new keys).

### 2. Zero Reward / Zero Gradient — No Learning Signal

**Symptom:** Across all 5 training steps:
- `critic/score/mean = 0.0`
- `critic/rewards/mean = 0.0`
- `actor/pg_loss = 0.0`
- `actor/grad_norm = 0.0`
- `critic/advantages/mean = 0.0`
- `batch/solve_none = 16/16` (no batch solved any problem)

The model produces no gradient update at all. PPO is effectively a no-op.

**Root Cause:** The combination of two issues:
1. High token mismatch rate (80-97%) causes `response_masks` to be zeroed out, eliminating reward signal even when trajectories complete.
2. Massive sandbox checkout failures (see below) produce dummy results with reward=0.

### 3. Token Mismatch (retokenization) — 80-97% of Trajectories

**Symptom:**
```
WARNING - When assemble steps, detect the trajectory not accumulative at position XXXX.
Expected: [...], Got: [...]. Setting response_masks to all 0s.
This is likely due to retokenization.
```

`traj/token_mismatch_mean` ranges from 0.80 to 0.97 across steps.

**Impact:** When response_masks are all zeros, the trajectory's reward cannot be backpropagated through the policy loss, even if the trajectory succeeded. This is the primary reason all rewards and gradients are zero.

**Root Cause:** In multi-step agent trajectories, the `agent_execution_engine.py:assemble_steps()` concatenates observation and action tokens from multiple turns. When the concatenated sequence is re-tokenized, boundary tokens may differ from the original per-turn tokenization. The mismatch detection at `agent_execution_engine.py:~line 200+` then zeros out the entire response mask.

**Potential Fixes:**
- Investigate the token assembly logic in `agent_execution_engine.py` to handle boundary token mismatches more gracefully (e.g., allow small mismatches at turn boundaries instead of zeroing all masks)
- Use token IDs from the original generation rather than re-tokenizing the assembled text

### 4. Massive Sandbox Checkout Failures (R2E-Gym Dataset)

**Symptom:**
```
RuntimeError: Failed to setup repository: Checkout failed (exit=128):
fatal: reference is not a tree: <commit_hash>
```

**Statistics:**
- 6,137 checkout failures total
- 662 trajectories permanently failed (returned dummy results)
- 602 trajectories completed (but mostly reward=0 due to token mismatch)

**Root Cause:** The R2E_Gym_Subset dataset contains references to git commits that don't exist in the PPIO sandbox's pre-cloned repository images. Unlike SWE-Bench Verified (which has 100% template coverage), R2E-Gym samples reference arbitrary commits across many repos, and the sandbox templates don't always have the full git history.

**Impact:** ~50%+ of trajectories fail at the env.reset() stage. Combined with token mismatch, effectively <5% of trajectory data contributes valid training signal.

### 5. Alternating Short/Long Steps

Steps 2 and 4 have extremely low `traj/steps_mean` (0.45, 0.36) compared to steps 1, 3, 5 (3.5-4.4). This is because most trajectories in those batches fail at checkout (dummy result with 0 steps), leaving only a handful of real trajectories.

### 6. Slow update_actor

`timing_s/update_actor` consistently takes ~6,300s (~105 min) per step regardless of trajectory content. This suggests FSDP param/optimizer offload overhead dominates training time even when there's little actual gradient computation (since all gradients are zero).

## Conclusions

1. **R2E-Gym on PPIO sandbox is not viable without template coverage fixes.** ~50% of samples fail at checkout. The dataset assumes full git history access, which PPIO sandbox templates don't provide.

2. **Token mismatch is the critical blocking issue for multi-step training.** Even when trajectories complete, 80-97% have their reward signal zeroed out. This must be fixed in the assembly logic before meaningful RL training can occur.

3. **SWE-Bench Verified is a better starting point** — it has 100% template coverage on PPIO sandbox and 500 well-defined samples. Switching to this dataset eliminates the checkout failure problem entirely.

4. **Per-step wall time is ~2-4 hours**, making the current setup expensive for iteration. Consider reducing batch_size for faster debugging cycles.

## Next Steps

- Switch training data to SWE-Bench Verified (100% template coverage)
- Investigate and fix the token mismatch issue in `agent_execution_engine.py`
- Consider smaller batch size for faster iteration
