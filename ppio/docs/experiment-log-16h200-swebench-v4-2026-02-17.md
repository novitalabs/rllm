# 16xH200 SWE-Bench Verified v4 Training Experiment Log

Date: 2026-02-17 ~ 2026-02-18
Cluster: 2 nodes x 8x NVIDIA H200 (141GB HBM3e each)
Model: Qwen3-32B (resumed from v1 step 15 checkpoint)
Script: `ppio/scripts/train_swebench_verified_16h200_v4.sh`
Dataset: SWE_Bench_Verified (500 samples)

## Objective

Test whether **disabling FSDP CPU offloading** speeds up `update_actor`, which consumed 76-87% of step time in v1-v3. H200 has 143GB HBM — hypothesis was that offloading is unnecessary overhead.

Sub-phase timing instrumentation (`ppio/scripts/patch_update_actor_timing.py`) applied to verl on both nodes to get detailed breakdown of update_actor phases.

## Changes from v3

| Parameter | v3 Value | v4 Value | Rationale |
|-----------|----------|----------|-----------|
| param_offload | True | **False** | Eliminate CPU↔GPU transfer overhead |
| optimizer_offload | True | **False** | Eliminate CPU↔GPU transfer overhead |
| train_batch_size | 32 | **16** | Fit in GPU memory without offloading |
| ppo_mini_batch_size | 32 | **16** | Match train_batch_size |
| ppo_max_token_len_per_gpu | 128000 | **64000** | Match v2 for bs=16 |

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| nnodes | 2 |
| n_gpus_per_node | 8 |
| train_batch_size | **16** |
| ppo_mini_batch_size | **16** |
| rollout_n | 8 |
| n_parallel_agents | 128 |
| tensor_parallel | 8 |
| sequence_parallel | 8 |
| max_prompt_length | 8192 |
| max_response_length | 32768 |
| gpu_memory_utilization | 0.7 |
| ppo_max_token_len_per_gpu | **64000** |
| ppo_micro_batch_size_per_gpu | 1 |
| ppo_epochs | 4 (default) |
| optim.lr | 1e-6 |
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
| pool_size | 512 |
| fsdp param_offload | **False** |
| fsdp optimizer_offload | **False** |
| val_before_train | False |

## Training Progress

Ran for **8+ steps** (still running at time of document creation). Checkpoint saved at step 5.

### Step-by-step Metrics

| Step | score/mean | pg_loss | grad_norm | entropy | pg_clipfrac | token_mismatch | resp_len/mean | solve (n/p/a) | step_time (s) |
|------|-----------|---------|-----------|---------|-------------|----------------|---------------|---------------|---------------|
| 1 | 0.026 | -0.21 | 276 | 7,294 | 0.0 | 0.29 | 16,305 | 13/3/0 | 7,741 |
| 2 | 0.119 | -5.08 | 554 | 6,791 | 0.0 | 0.34 | 17,854 | 9/7/0 | 7,904 |
| 3 | 0.043 | 0.99 | 199 | 6,601 | 0.0 | 0.31 | 13,978 | 14/2/0 | 7,698 |
| 4 | 0.141 | 1.12 | 489 | 6,974 | 0.0 | 0.38 | 16,278 | 10/6/0 | 8,197 |
| 5* | 0.102 | 4.96 | 394 | 6,387 | 0.0 | 0.31 | 13,522 | 12/4/0 | 19,145 |
| 6 | 0.092 | 0.62 | 402 | 6,727 | 0.0 | 0.30 | 15,762 | 10/6/0 | 7,839 |
| 7 | 0.129 | -0.58 | 428 | 6,140 | 0.0 | 0.31 | 16,214 | 9/7/0 | 8,282 |
| 8 | 0.159 | 0.77 | 237 | 7,763 | 0.0 | 0.38 | 15,492 | 12/2/2 | 7,894 |

\*Step 5 includes validation (11,240s) + checkpoint save (89s).

### Validation (Step 5)

| Metric | Value |
|--------|-------|
| test_score | 0.098 |
| pass@k | 0.09 |

### Timing Breakdown (per step average, non-validation steps)

| Phase | Average (s) | % of step |
|-------|------------|-----------|
| collect_trajectory | 1,046 | 13.0% |
| old_log_prob | 114 | 1.4% |
| **update_actor** | **6,772** | **84.3%** |
| transform + adv | ~115 | 1.4% |
| **Total (non-val step)** | **~8,037** | 100% |

### update_actor Sub-phase Timing (from instrumentation)

This is the key finding of v4. Sub-phase timing was added via `patch_update_actor_timing.py`.

| Sub-phase | Average (s) | % of update_actor |
|-----------|------------|-------------------|
| load_model_to_gpu | **~0.0000005** | **0.0%** |
| load_optimizer_to_gpu | **~0.0000005** | **0.0%** |
| **forward** | **1,732** | **25.6%** |
| **backward** | **5,034** | **74.3%** |
| optim_step | 0.12 | 0.0% |
| offload_model_to_cpu | ~0.0000004 | 0.0% |
| offload_optimizer_to_cpu | ~0.0000003 | 0.0% |
| **update_policy (total)** | **6,772** | 100% |

**n_micro_batches = 64** (4 ppo_epochs × 16 samples / 1 micro_batch_per_gpu)
**n_mini_batches = 1** (16 batch / 16 mini_batch = 1 per epoch)

Per micro-batch: ~27s forward, ~79s backward

### Memory Usage

| Metric | Value (v4, no offload) | Value (v3, offload) |
|--------|----------------------|---------------------|
| max_memory_allocated_gb | 132.1 - 135.6 | 132.8 - 135.8 |
| max_memory_reserved_gb | 147.1 - 156.2 | 146.8 - 147.3 |
| cpu_memory_used_gb | 70.0 - 78.0 | 73.0 - 76.5 |
| H200 total memory | 143 GB | 143 GB |

Memory usage is **virtually identical** whether offloading is enabled or not.

### Timing Comparison Across All Versions

| Phase | v1 (bs=8) | v2 (bs=16) | v3 (bs=32) | v4 (bs=16, no offload) |
|-------|-----------|------------|------------|------------------------|
| collect_trajectory | ~700s | ~1,100s | ~1,568s | ~1,046s |
| update_actor | ~3,200s | ~6,600s | ~13,100s | **~6,772s** |
| Total step | ~4,200s | ~8,100s | ~15,100s | ~8,037s |
| update_actor % | 76% | 83% | 87% | 84% |

v4 update_actor ≈ v2 update_actor (both bs=16), confirming **offloading had negligible impact**.

## Key Findings

### 1. FSDP Offloading is NOT the Bottleneck

**This was the primary finding of v4.** Load/offload times are in the nanosecond range because `_is_offload_param` and `_is_offload_optimizer` are False, so the load/offload functions are skipped. But even comparing v4 (no offload, bs=16) to v2 (offload, bs=16), update_actor times are nearly identical (~6,770s vs ~6,600s).

Memory allocated is also nearly identical (132-136GB) whether offloading is enabled or not. This means verl's FSDP implementation was NOT actually offloading model parameters to CPU in v1-v3, or the offloading overhead was negligible compared to compute.

### 2. 64 Micro-Batches is the Real Bottleneck

With `ppo_epochs=4` (default), `ppo_mini_batch_size=16`, and `ppo_micro_batch_size_per_gpu=1`:
- 4 epochs × 1 mini-batch × 16 micro-batches = **64 micro-batch iterations**
- Each micro-batch: ~27s forward + ~79s backward = ~106s
- Total: 64 × 106 = ~6,784s ≈ observed ~6,772s

**Backward pass dominates at 74.3%** of update_actor time (3x slower than forward).

### 3. update_actor Scales Linearly with batch_size

| Batch Size | Mini-batches/epoch | Micro-batches | update_actor (s) | Ratio |
|------------|-------------------|---------------|------------------|-------|
| 8 (v1) | 1 | 32 | ~3,200 | 1.0x |
| 16 (v2/v4) | 1 | 64 | ~6,700 | 2.1x |
| 32 (v3) | 1 | 128 | ~13,100 | 4.1x |

Perfectly linear: 2x batch → 2x micro-batches → 2x time.

### 4. pg_clipfrac=0 Persists

Still zero across all 8 steps. This is independent of:
- Learning rate (tested 1e-6 and 5e-6 in v2)
- Batch size (tested 8, 16, 32)
- FSDP offloading (on/off)
- Malware presence (v4 ran on clean nodes)

### 5. Per-Micro-Batch Compute Times

| Operation | Time per micro-batch (s) | Notes |
|-----------|-------------------------|-------|
| Forward | ~27 | Qwen3-32B, SP=8, gradient checkpointing |
| Backward | ~79 | ~3x forward (expected with grad checkpointing) |
| Optimizer step | ~0.003 | Negligible |

Backward is ~3x forward, which is consistent with gradient checkpointing (recomputes activations during backward).

## Conclusions

1. **Offloading was a red herring.** The update_actor bottleneck is purely compute-bound: 64 sequential forward+backward passes through a 32B model. No amount of memory optimization will help.

2. **To reduce update_actor time, reduce the number of micro-batches.** Options:
   - Reduce `ppo_epochs` (from 4 to 1 → 4x speedup)
   - Increase `ppo_micro_batch_size_per_gpu` (from 1 to 2 → 2x speedup, if memory allows)
   - Accept the cost and increase `train_batch_size` for better gradient signal per step

3. **Memory is not a constraint on H200.** 135.5GB/143GB allocated without offloading, leaving ~7.5GB headroom. This means micro_batch_size_per_gpu could potentially be increased to 2.

4. **The pg_clipfrac=0 issue is not related to FSDP, memory, or malware.** It's a fundamental PPO hyperparameter or algorithmic issue that needs separate investigation.

## v5 Direction

Based on user direction: use v3 hyperparameters (offloading re-enabled as baseline) with `train_batch_size=64` and `ppo_mini_batch_size=8`. This gives:
- 8 mini-batches per ppo_epoch (64/8 = 8), 32 optimizer steps per step (4 epochs × 8)
- More gradient signal per step (64 unique samples × 8 rollouts = 512 trajectories)
- More frequent optimizer updates (32 per step vs 4 in v3)
- Expected: longer steps but more learning per step
