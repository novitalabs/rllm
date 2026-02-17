# 16xH200 SWE-Bench Verified v1 Training Experiment Log

Date: 2026-02-11
Cluster: 2 nodes x 8x NVIDIA H200 (141GB HBM3e each)
Model: Qwen3-32B
Script: `ppio/scripts/train_swebench_verified_16h200.sh`
Dataset: SWE_Bench_Verified (train: 500, val: 500 — same dataset for both)

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
| train_batch_size | 8 |
| ppo_mini_batch_size | 8 |
| rollout_n | 8 |
| tensor_parallel | 8 |
| sequence_parallel | 8 |
| max_prompt_length | 8192 |
| max_response_length | 32768 |
| gpu_memory_utilization | 0.7 |
| ppo_max_token_len_per_gpu | 32000 |
| optim.lr | 1e-6 |
| clip_ratio_high | 0.28 |
| kl_loss_coef | 0.001 |
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

Ran for **28 steps** before crashing (NCCL communication failure on step 28).
Total trajectories: ~4,268 (444 reward=1.0, 3,824 reward=0.0 — ~10.4% solve rate).

### Step-by-step Metrics

| Step | score/mean | pg_loss | grad_norm | entropy | pg_clipfrac | token_mismatch | resp_len/mean | step_time (s) |
|------|-----------|---------|-----------|---------|-------------|----------------|---------------|---------------|
| 1 | 0.026 | 1.92 | 266 | 6,721 | 0.0 | 0.33 | 15,658 | 4,028 |
| 2 | 0.031 | 4.02 | 264 | 6,531 | 0.0 | 0.36 | 14,918 | 3,974 |
| 3 | 0.172 | -17.44 | 788 | 7,145 | 0.0 | 0.27 | 16,280 | 3,851 |
| 4 | — | — | — | — | — | — | — | — |
| 5 | — | — | — | — | — | — | — | ~12,500* |
| 6 | — | — | — | — | — | — | — | — |
| 7 | 0.109 | 8.24 | 624 | 8,584 | 0.0 | 0.28 | 18,090 | 4,345 |
| 8 | 0.141 | -2.43 | 639 | 5,545 | 0.0 | 0.39 | 15,649 | 4,073 |
| 9 | 0.055 | 5.88 | 1,594 | 7,483 | 0.0 | — | 15,999 | 12,501* |
| 10 | — | — | — | — | — | — | — | — |
| 11 | — | — | — | — | — | — | — | — |
| 12 | 0.138 | 10.19 | 619 | 6,252 | 0.0 | 0.19 | 16,000 | 3,993 |
| 13 | 0.233 | 3.98 | 525 | 6,270 | 0.0 | 0.33 | 14,784 | 4,379 |
| 14 | 0.172 | 3.03 | 697 | 6,125 | 0.0 | 0.23 | 19,031 | 4,503 |
| 15 | 0.219 | -2.76 | 219 | 7,332 | 0.0 | — | 16,729 | 13,605* |
| 16 | 0.0 | 0.0 | 0.0 | 7,641 | 0.0 | 0.47 | 13,820 | 4,863 |
| 17 | 0.109 | 4.35 | 368 | 6,311 | 0.0 | 0.23 | 16,804 | 4,640 |
| 18 | 0.281 | 10.66 | 653 | 6,130 | 0.0 | 0.23 | 15,516 | 4,921 |
| 19 | 0.094 | -6.04 | 607 | 6,530 | 0.0 | 0.25 | 18,268 | 4,663 |
| 20 | 0.047 | 4.32 | 534 | 7,006 | 0.0 | — | 17,254 | 13,091* |
| 21 | 0.078 | 7.73 | 421 | 6,677 | 0.0 | 0.25 | 16,139 | 4,822 |
| 22 | 0.148 | 1.96 | 633 | 7,493 | 0.0 | 0.34 | 15,918 | 4,590 |
| 23 | 0.203 | -4.38 | 519 | 5,517 | 0.0 | 0.31 | 16,141 | 4,749 |
| 24 | 0.203 | 12.45 | 685 | 5,194 | 0.0 | 0.34 | 12,840 | 4,781 |
| 25 | 0.083 | -0.08 | 587 | 7,821 | 0.0 | — | 16,612 | 12,135* |
| 26 | 0.078 | 5.41 | 952 | 5,726 | 0.0 | 0.20 | 14,815 | 4,904 |
| 27 | 0.078 | -3.17 | 386 | 6,734 | 0.0 | 0.20 | 16,102 | 4,674 |
| 28 | — | — (NCCL crash) | — | — | — | — | — | — |

*Steps 5, 9, 15, 20, 25 include validation and checkpoint save (step_time includes testing time of ~8,000-8,500s).
Step 16 had zero gradients due to high token mismatch (0.47).

### Validation Scores

| Step | test_score | pass@k | Notes |
|------|-----------|--------|-------|
| 5 | — | — | Not captured in logs |
| 10 | 0.102 | 0.09 | |
| 15 | **0.121** | **0.11** | **Best validation** |
| 20 | 0.093 | 0.086 | Declining |
| 25 | 0.089 | 0.08 | Continued decline |

### Key Observations

- **Score/mean peaked at 0.281** (step 18) — first (and only) batch with `batch/solve_all=1`
- **Token mismatch stabilized at ~0.20-0.35** (down from ~0.80-0.97 in the R2E-Gym experiment, thanks to SWE-Bench's simpler prompts)
- **ppo_kl=0 and pg_clipfrac=0 across all 28 steps** — the policy barely changes, PPO clipping never triggers
- **Entropy generally declining**: 6,721 (step 1) → 5,194 (step 24), with spikes on validation/zero-gradient steps
- **grad_norm highly variable**: ranges from 0 (step 16) to 1,594 (step 9), typical range 200-700

## Bug Fixes Applied During Experiment

### 1. `compute_data_metrics` Empty Tensor Crash

**File:** `rllm/trainer/verl/agent_ppo_trainer.py`

**Symptom:** Training crashed when a batch had no valid trajectories (all masked by token mismatch), causing `torch.stack` on an empty list.

**Fix:** Added safe wrapper around `compute_data_metrics` to handle empty tensor edge cases — returns zero metrics instead of crashing when no valid samples exist in a batch.

### 2. Sandbox Health Check (Stale Sandbox Detection)

**File:** `rllm/environments/swe_ppio/ppio_reward.py`

**Symptom:** Some sandboxes became unresponsive after long trajectories, causing reward computation to hang or return incorrect results.

**Fix:** Added sandbox health check before reward evaluation. If the sandbox is detected as stale/unresponsive, the trajectory is marked as failed with reward=0 instead of hanging indefinitely.

## Issues Found

### 1. pg_clipfrac=0 and ppo_kl=0 — Policy Not Updating

**Symptom:** Across all 28 training steps:
- `actor/pg_clipfrac = 0.0` (PPO clipping never triggers)
- `actor/ppo_kl = 0.0` (KL divergence between old and new policy is zero)

**Impact:** The policy is barely changing between steps. Despite non-zero `pg_loss` and `grad_norm`, the actual parameter updates are too small to measurably shift the policy distribution.

**Root Cause:** Learning rate of 1e-6 is too conservative for this model size and loss landscape. Combined with FSDP offloading (which distributes optimizer state across all 16 GPUs), the effective update magnitude per parameter is negligible.

### 2. Entropy Drop Indicates Premature Collapse Risk

Entropy declined from ~7,000 to ~5,200 over 28 steps, despite the policy barely updating (pg_clipfrac=0). This suggests the model is losing exploration diversity through whatever small updates do occur, without gaining corresponding task performance.

### 3. Validation Score Decline After Step 15

- Step 10: 0.102
- Step 15: 0.121 (peak)
- Step 20: 0.093 (-23% from peak)
- Step 25: 0.089 (-26% from peak)

The model improves through step 15 then deteriorates, suggesting overfitting or policy degradation. Given that pg_clipfrac=0, the degradation is subtle — small weight changes accumulate over many steps to shift the output distribution away from useful behaviors.

### 4. NCCL Communication Failure (Step 28 Crash)

**Symptom:** Training crashed at step 28 with NCCL errors:
```
NCCL communicator was aborted on rank N
```

**Root Cause:** Multi-node NCCL communication timeout/failure. Likely caused by network instability between the two nodes during the `update_actor` FSDP all-reduce phase.

**Impact:** Training stopped at step 28. Last checkpoint saved at step 25.

### 5. Step 16 — Full Zero-Gradient Batch

Step 16 had `token_mismatch=0.47` (highest of any step), resulting in:
- `score/mean = 0.0`
- `pg_loss = 0.0`
- `grad_norm = 0.0`
- `batch/solve_none = 8/8`

An entire training step wasted with no learning signal.

## Conclusions

1. **SWE-Bench Verified works well as training data**: 100% sandbox template coverage, reasonable solve rates (~10%), and stable trajectory collection. Much better than R2E-Gym on PPIO sandbox.

2. **Learning rate 1e-6 is too conservative**: pg_clipfrac=0 across all steps indicates the policy never changes enough to trigger PPO clipping. The model is barely learning despite non-zero gradients.

3. **Best checkpoint is global_step_15** (val test_score=0.121, pass@k=0.11). Performance degrades after this point.

4. **Token mismatch at ~25% is manageable** but still wastes ~1/4 of trajectory data. Future work should investigate the retokenization issue in `agent_execution_engine.py`.

5. **Batch size 8 gives high variance**: score/mean swings wildly between steps (0.0 to 0.281), making learning noisy.

## v2 Hyperparameter Changes (Rationale)

| Parameter | v1 Value | v2 Value | Rationale |
|-----------|----------|----------|-----------|
| optim.lr | 1e-6 | **5e-6** | 5x increase; pg_clipfrac=0 shows policy barely updates at 1e-6 |
| train_batch_size | 8 | **16** | 2x batch for better gradient estimates, reduce variance |
| ppo_mini_batch_size | 8 | **16** | Match train_batch_size |
| ppo_max_token_len_per_gpu | 32000 | **64000** | Accommodate doubled batch size |
| resume_from | — | v1 step 15 | Best validation checkpoint |

See: `ppio/scripts/train_swebench_verified_16h200_v2.sh`
