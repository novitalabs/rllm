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

1. **Offloading was a red herring** (partially corrected by v5, see below). The v4 top-level load/offload timing showed nanoseconds because `_is_offload_param=False` skipped those calls. Per-micro-batch time was ~106s in v4 (no offload, bs=16) vs ~103s in v3 (offload, bs=32), suggesting offloading overhead is small when memory is sufficient.

2. **To reduce update_actor time, reduce the number of micro-batches.** Options:
   - Reduce `ppo_epochs` (from 4 to 1 → 4x speedup)
   - Increase `ppo_micro_batch_size_per_gpu` (from 1 to 2 → 2x speedup, if memory allows)
   - Accept the cost and increase `train_batch_size` for better gradient signal per step

3. **Memory is not a constraint on H200 at bs=16.** 135.5GB/143GB allocated without offloading, leaving ~7.5GB headroom.

4. **The pg_clipfrac=0 issue is not related to FSDP, memory, or malware.** It's a fundamental PPO hyperparameter or algorithmic issue that needs separate investigation.

---

## v5 Experiment (bs=64, mini_batch=8, offloading=True)

### v5 Configuration Changes from v3

| Parameter | v3 Value | v5 Value | Rationale |
|-----------|----------|----------|-----------|
| train_batch_size | 32 | **64** | More gradient signal per step |
| ppo_mini_batch_size | 32 | **8** | More optimizer updates per step |
| pool_size | 512 | **1024** | Accommodate 64×8=512 trajectories |

### v5 Step 1 Results

Ran 1 step before being stopped due to unacceptable step time.

| Metric | v5 (bs=64) | v4 (bs=16) | Ratio |
|--------|-----------|-----------|-------|
| **step_time** | **110,036s (30.6h)** | 7,741s (2.1h) | 14.2x |
| update_actor | 105,902s (29.4h) | 6,747s | 15.7x |
| collect_trajectory | 3,854s | 877s | 4.4x |
| n_micro_batches | 256 | 64 | 4x |
| n_mini_batches | 8 | 1 | 8x |
| forward/micro-batch | **105s** | 27s | **3.9x** |
| backward/micro-batch | **309s** | 79s | **3.9x** |
| memory_allocated_gb | **139.1** | 135.5 | +3.6GB |
| score/mean | 0.077 | 0.026 | — |
| **pg_clipfrac** | **0.000115** | 0.0 | **non-zero!** |
| ppo_kl | 2.96e-05 | 0.0 | non-zero |
| solve (n/p/a) | 49/14/1 | 13/3/0 | — |

### v5 Key Findings

1. **pg_clipfrac non-zero for the first time!** `pg_clipfrac=0.000115` and `ppo_kl=2.96e-05` — the smaller mini-batch size (8 vs 16-32) with 32 optimizer steps per step finally pushed the policy to update. This suggests `ppo_mini_batch_size` is a critical parameter.

2. **Per micro-batch 3.9x slower with offloading + large batch.** At 139.1/143GB (97.3% utilization), FSDP offloading overhead becomes severe. The real offloading cost is NOT in top-level load/offload calls, but in FSDP's per-layer parameter fetching during forward/backward. Under extreme memory pressure, this fetching dramatically slows down.

3. **Corrects v4 conclusion about offloading.** v4 showed offloading overhead was negligible by comparing top-level timings, but the real overhead lives inside forward/backward. When GPU memory is sufficient (~95% utilization), overhead is small. When near-saturated (~97%), it becomes catastrophic (4x per micro-batch).

4. **30.6 hours per step is not viable.** Even though each step processes 4x more data with 8x more optimizer updates, the 14x wall-clock increase negates the benefit.

### v5 Revised Direction

Reduce batch_size to 32 (same as v3) while keeping `ppo_mini_batch_size=8` to retain the benefit of more optimizer updates (16 per step instead of v3's 4). This keeps memory in the safe zone (~135GB) while testing whether smaller mini-batches help pg_clipfrac.

---

## v5 Experiment Phase 2 (bs=32, mini_batch=8, offloading=True)

Date: 2026-02-19
Script: `ppio/scripts/train_swebench_verified_16h200_v5.sh` (modified)

### v5 Phase 2 Configuration

| Parameter | v3 Value | v5 Phase 2 Value | Rationale |
|-----------|----------|-------------------|-----------|
| train_batch_size | 32 | **32** | Same as v3, avoids memory pressure |
| ppo_mini_batch_size | 32 | **8** | More optimizer updates per step (16 vs 4) |
| ppo_max_token_len_per_gpu | 128000 | **128000** | Same as v3 |
| pool_size | 512 | **512** | Same as v3 |
| param_offload | True | **True** | Same as v3 |
| optimizer_offload | True | **True** | Same as v3 |

With bs=32, mini_batch=8: 4 mini-batches/epoch × 4 epochs = 16 optimizer steps per step (vs v3's 4).
n_micro_batches = 4 epochs × 4 mini-batches × 32/1 = 128 micro-batches (same as v3).

### v5 Phase 2 Run 1 Results (Feb 19, Steps 1-2)

Completed 2 steps before being interrupted by cryptocurrency miner reinfection.

| Step | score/mean | pg_clipfrac | collect_trajectory (s) | step_time (s) | Notes |
|------|-----------|-------------|----------------------|---------------|-------|
| 1 | — | **9.26e-05** | ~1,099 | — | pg_clipfrac non-zero! |
| 2 | — | **6.97e-05** | ~5,222 | — | collect_trajectory anomalously slow |

**Key result: pg_clipfrac confirmed non-zero** with mini_batch_size=8, consistent with the v5 bs=64 finding. Entropy dropped noticeably between steps (6,847 → 4,682).

### Infrastructure Issues During v5

#### 1. Cryptocurrency Miner Reinfection (Third Occurrence)

During step 2 of v5 Phase 2 Run 1, a ProgPoW Zano cryptocurrency miner (`/var/tmp/.tmp/python3.7.3`) was detected on **both nodes**, connecting to a new mining pool at `172.86.68.38:8443` (previously `45.61.148.247` in earlier infections). The miner consumed GPU resources, causing:

- Step 2 collect_trajectory anomaly: 5,222s vs expected ~1,100s (miner competing for GPU)
- Step 3 NCCL deadlock: training hung during update_actor with 0% GPU utilization on all GPUs. Workers stuck on `ep_poll`. Root cause: miner disrupted NCCL communication during update_actor, and killing the miner left broken NCCL connections.

**Actions taken:**
- Killed miners on both nodes: `kill -9` + removed `/var/tmp/.tmp/` files
- Applied iptables firewall rules on **both host machines** to block external access to Ray ports:
  ```
  ACCEPT tcp from 10.83.115.14 (head) to ports 6379, 8265, 10001, 10002-19999
  ACCEPT tcp from 10.83.115.17 (worker) to ports 6379, 8265, 10001, 10002-19999
  ACCEPT tcp from 127.0.0.1 to ports 6379, 8265
  DROP tcp from 0.0.0.0/0 to ports 6379, 8265, 10001, 10002-19999
  ```
- Firewall effectiveness confirmed: 546GB cluster traffic accepted, 4,219 external connection attempts blocked

**Note:** iptables rules are not persisted (lost on reboot). Infection vector is the unauthenticated Ray dashboard exposed to the internet.

#### 2. NVML/CUDA Failure in Containers

After killing the miner and restarting training, both nodes experienced NVML initialization failures inside Docker containers, despite GPUs being functional on the host (`nvidia-smi` works).

**Symptom:** `torch.cuda.is_available() = False` inside container, `Can't initialize NVML` warning. This caused `torch.distributed.init_process_group()` to fail with:
```
ValueError: Duplicate device type cpu in backend string: nccl
Custom backend string: cpu:gloo,cpu:nccl (should be cpu:gloo,cuda:nccl)
```

Because CUDA was unavailable, NCCL was assigned to CPU device type instead of CUDA, creating the duplicate `cpu` entry.

**Root cause:** Docker containers lose their NVML connection to the host GPU driver after GPU state disruption (miner process corruption, zombie processes, etc.). The NVIDIA runtime device files become stale inside the container even though the host GPUs are fine.

**Fix:** `docker restart deepswe-train` on the affected node(s) restores the NVML connection. This was required on:
- Worker node (10.83.115.17): restarted once during v5 Phase 2 Run 1 launch
- Head node (10.83.115.14): restarted during v5 Phase 2 Run 2 launch (discovered head also lost CUDA via Ray worker GPU test)

**Timeline of v5 Phase 2 launch attempts:**
1. **Run 1** (Feb 19 ~14:00): Started successfully. Completed 2 steps. Miner reinfection → NCCL deadlock during step 3.
2. **Attempts 2-4** (Feb 19 ~19:00-22:00): Failed with `Duplicate device type cpu` error. Worker container restarted once, but head node container also had stale NVML (not discovered until GPU test via Ray actor). Multiple restarts of Ray without fixing the root cause.
3. **Run 2** (Feb 19 ~22:28): After restarting **both** containers + Ray cluster, training started successfully. NCCL initialized correctly, CUDA graphs captured, rollout phase active with PPIO sandboxes.

### v5 Phase 2 Run 2 Results

Started Feb 19 22:28 UTC. Configuration identical to Run 1. Completed 5 steps; step 6 update_actor in progress as of Feb 21 01:00 UTC.

#### Step-by-step Metrics

| Step | score/mean | pg_loss | grad_norm | entropy | pg_clipfrac | token_mismatch | resp_len/mean | solve (n/p/a) | step_time (s) |
|------|-----------|---------|-----------|---------|-------------|----------------|---------------|---------------|---------------|
| 1 | 0.079 | -3.36 | 551 | 6,509 | **1.22e-04** | 0.32 | 16,376 | 21/11/0 | 14,558 |
| 2 | 0.094 | -0.01 | 399 | 7,107 | **8.52e-05** | 0.30 | 15,825 | 26/6/0 | 14,781 |
| 3 | 0.137 | 7.14 | 609 | 6,385 | **1.63e-04** | 0.28 | 15,436 | 21/11/0 | 15,307 |
| 4 | 0.145 | 3.01 | 419 | 6,816 | **1.14e-04** | 0.34 | 15,499 | 21/10/1 | 15,410 |
| 5* | 0.121 | 1.23 | 589 | 7,029 | **9.74e-05** | 0.29 | 17,512 | 22/9/1 | 25,830 |

\*Step 5 includes validation (10,072s) + checkpoint save (83s).

#### Validation (Step 5)

| Metric | Value |
|--------|-------|
| test_score | **0.111** |
| pass@k | **0.106** |

Comparable to v1 step 15 best (0.121), significantly better than v4 step 5 (0.098). v5 started from v1 step 15 checkpoint, so effectively at global step 20.

#### Timing Breakdown (per step average, non-validation steps 1-4)

| Phase | Average (s) | % of step |
|-------|------------|-----------|
| collect_trajectory | 1,686 | 11.2% |
| old_log_prob | 154 | 1.0% |
| **update_actor** | **13,173** | **87.6%** |
| transform + adv | ~156 | 1.0% |
| **Total (non-val step)** | **~15,014** | 100% |

#### update_actor Sub-phase Timing

| Sub-phase | Average (s) | % of update_actor |
|-----------|------------|-------------------|
| load_model_to_gpu | 0.30 | 0.0% |
| load_optimizer_to_gpu | 0.47 | 0.0% |
| **forward** | **3,376** | **25.6%** |
| **backward** | **9,785** | **74.3%** |
| optim_step | 0.53 | 0.0% |
| offload_model_to_cpu | 1.92 | 0.0% |
| offload_optimizer_to_cpu | 1.61 | 0.0% |
| **update_policy (total)** | **~13,173** | 100% |

**n_micro_batches = 128** (4 ppo_epochs × 4 mini-batches × 8 micro-batches per mini-batch)
**n_mini_batches = 4** (32 batch / 8 mini_batch = 4 per epoch)

Per micro-batch: ~26s forward, ~76s backward (~102s total, consistent with v3).

#### Memory Usage

| Metric | Value |
|--------|-------|
| max_memory_allocated_gb | 135.97 - 136.43 |
| max_memory_reserved_gb | 147.33 - 147.94 |
| cpu_memory_used_gb | 71.2 - 75.4 |

Memory consistent with v3 (~136GB allocated), safely below 143GB limit.

#### Key Findings

1. **pg_clipfrac consistently non-zero** across all 5 steps (range: 8.52e-05 to 1.63e-04). This confirms the v5 bs=64 finding that `ppo_mini_batch_size=8` (4 mini-batches per epoch, 16 optimizer updates per step) produces measurable policy updates, unlike v1-v4 where `ppo_mini_batch_size=batch_size` (1 mini-batch per epoch, 4 optimizer updates per step) always gave pg_clipfrac=0.

2. **update_actor ~13,173s matches v3 exactly** (~13,100s). With identical n_micro_batches (128) and similar memory (~136GB), per micro-batch time is ~102s (v3: ~103s). Smaller mini_batch_size has no impact on compute time — it only changes when optimizer steps happen within the same number of forward/backward passes.

3. **Validation score 0.111 (pass@k=0.106)** — comparable to v1's best (0.121 at step 15). However, v5 is only at step 5 (started from v1 step 15 checkpoint), so effectively at step 20. The score is trending well.

4. **score/mean trending up**: 0.079 → 0.094 → 0.137 → 0.145 → 0.121 (step 5 slightly lower, possibly batch variance).

5. **Entropy stable** around 6,385-7,107 (vs v4 which had similar range 6,140-7,763). No dramatic entropy collapse.

6. **Step timing ~4.2h per step** (non-validation), ~7.2h for validation steps. At this rate:
   - 5 steps completed in ~26h (including 1 validation step)
   - save_freq=5 / test_freq=5 means validation every 5 steps
   - Expected ~4.2h/step × 4 + 7.2h × 1 = ~24h per 5-step cycle

#### Comparison: v5 (mini_batch=8) vs v3 (mini_batch=32)

| Metric | v3 (mini_batch=32) | v5 (mini_batch=8) |
|--------|-------------------|-------------------|
| optimizer steps per step | 4 | **16** |
| pg_clipfrac | 0.0 | **~1e-04** |
| update_actor time | ~13,100s | ~13,173s |
| per micro-batch time | ~103s | ~102s |
| n_micro_batches | 128 | 128 |

**Conclusion: `ppo_mini_batch_size=8` is strictly better than 32.** Same compute cost, same number of micro-batches, but 4x more optimizer updates and non-zero policy gradient clipping. The additional optimizer steps come "for free" because they just reorganize when gradients are applied within the same forward/backward passes.
