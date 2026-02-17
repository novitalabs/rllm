# 16xH200 SWE-Bench Verified v3 Training Experiment Log

Date: 2026-02-15 ~ 2026-02-16
Cluster: 2 nodes x 8x NVIDIA H200 (141GB HBM3e each)
Model: Qwen3-32B (resumed from v1 step 15 checkpoint)
Script: `ppio/scripts/train_swebench_verified_16h200_v3.sh`
Dataset: SWE_Bench_Verified (500 samples)

## Changes from v2

| Parameter | v2 Value | v3 Value | Rationale |
|-----------|----------|----------|-----------|
| train_batch_size | 16 | **32** | 4x original; more gradient signal per step |
| ppo_mini_batch_size | 16 | **32** | Match train_batch_size |
| optim.lr | 5e-6 | **1e-6** | Revert: 5x lr increase didn't fix pg_clipfrac=0 |
| ppo_max_token_len_per_gpu | 64000 | **128000** | Accommodate 4x batch |
| pool_size | 32 (default) | **512** | Fix sandbox concurrency bottleneck |

## Infrastructure Fixes

### 1. Sandbox Concurrency Lock Fix

**File:** `rllm/environments/swe_ppio/ppio_reward.py`

**Problem:** `SandboxPool._pool_lock` was held during the entire `_create_with_retry()` call (a network operation taking seconds). This serialized ALL sandbox creation, limiting concurrency to <100 even with batch_size=16.

**Fix:** Moved `_create_with_retry()` outside the lock. Lock is now only held briefly for pool lookup and storage. Sandbox creation happens concurrently.

**Result:** Sandbox indices reaching 247+ (well beyond old pool_size=32 limit), confirming concurrent creation works.

### 2. libnvidia-ml.so Library Fix

**Problem:** `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.126.09` was 0 bytes (empty stub) inside both Docker containers. NCCL requires this library during `dist._broadcast_coalesced` in FSDP initialization.

**Fix:** Copied the real 2.2MB library from the host into both containers.

### 3. Ray Worker Port Conflict Fix

**Problem:** Ray worker node's randomly assigned `dashboard_agent_grpc` port fell within the `worker_ports` range (10002-19999).

**Fix:** Explicitly set `--dashboard-agent-grpc-port=9990 --dashboard-agent-listen-port=9991` on worker node start.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| nnodes | 2 |
| n_gpus_per_node | 8 |
| train_batch_size | **32** |
| ppo_mini_batch_size | **32** |
| rollout_n | 8 |
| n_parallel_agents | 256 |
| tensor_parallel | 8 |
| sequence_parallel | 8 |
| max_prompt_length | 8192 |
| max_response_length | 32768 |
| gpu_memory_utilization | 0.7 |
| ppo_max_token_len_per_gpu | **128000** |
| optim.lr | **1e-6** |
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
| pool_size | **512** |
| fsdp param_offload | True |
| fsdp optimizer_offload | True |
| val_before_train | False |

## Training Progress

Ran for **4 steps** before crashing (NCCL connection closed, step 5).
No checkpoint was saved (save_freq=5, crash at step 5 before save).

### Step-by-step Metrics

| Step | score/mean | pg_loss | grad_norm | entropy | pg_clipfrac | token_mismatch | resp_len/mean | solve (n/p/a) | step_time (s) |
|------|-----------|---------|-----------|---------|-------------|----------------|---------------|---------------|---------------|
| 1 | 0.065 | 0.69 | 195 | 6,822 | 0.0 | 0.32 | 15,112 | 22/10/0 | 14,413 |
| 2 | 0.093 | 0.61 | 175 | 6,947 | 0.0 | 0.34 | 14,895 | 25/7/0 | 14,789 |
| 3 | 0.162 | 0.82 | 414 | 6,293 | 0.0 | 0.27 | 14,863 | 17/15/0 | 14,900 |
| 4 | 0.108 | 0.29 | 246 | 7,133 | 0.0 | 0.32 | 15,670 | 23/9/0 | 16,353 |

### Timing Breakdown (per step average)

| Phase | Average (s) | % of step |
|-------|------------|-----------|
| collect_trajectory | 1,568 | 10.4% |
| old_log_prob | 151 | 1.0% |
| **update_actor** | **13,094** | **86.9%** |
| transform + adv | ~152 | 1.0% |
| **Total (non-val step)** | **~15,100** | 100% |

### Timing Comparison Across Versions

| Phase | v1 (bs=8) | v2 (bs=16) | v3 (bs=32) |
|-------|-----------|------------|------------|
| collect_trajectory | ~700s | ~1,100s | ~1,568s |
| update_actor | ~3,200s | ~6,600s | **~13,100s** |
| Total step | ~4,200s | ~8,100s | **~15,100s** |
| update_actor % | 76% | 83% | **87%** |

**update_actor scales ~linearly with batch_size** (2x batch → 2x time), and dominates increasingly as batch grows.

### Memory Usage

| Metric | Value |
|--------|-------|
| max_memory_allocated_gb | 132.8 - 135.8 |
| max_memory_reserved_gb | 146.8 - 147.3 |
| cpu_memory_used_gb | 73.0 - 76.5 |
| H200 total memory | 143 GB |

GPU memory nearly saturated at ~136/143 GB allocated.

## Crash Analysis

### NCCL Connection Closed (Step 5)

**Timeline:**
- Step 4 completed at ~21:42 on Feb 15
- Step 5 trajectory collection started, completed ~22:15
- update_actor began ~22:15
- Head node rank 7 NCCL connection closed: Feb 15 11:23 (during earlier step, non-fatal)
- **Fatal crash**: Feb 16 08:59 — worker node lost all NCCL connections to head node

**Error:** `NCCL WARN socketProgress: Connection closed by remote peer host-10-83-115-14`

**Root cause: Cryptocurrency miner consuming GPU resources** (see Security Incident below).

## Security Incident: Cryptocurrency Miner Infection

### Discovery

During investigation of the NCCL crash, GPU utilization was found at 100% with no training running. A cryptocurrency miner was discovered running on both nodes.

### Malware Inventory

Found on **head node** container (`/var/tmp/` and `/tmp/`):

| File | Size | Description |
|------|------|-------------|
| `/var/tmp/.tmp/python3.7.3` | ~8MB | **ProgPoW Zano GPU miner** (main executable) |
| `/var/tmp/.gpu-helper` | 13MB | GPU mining helper binary |
| `/var/tmp/.nethelper-3233` | 23MB | Network communication tool |
| `/var/tmp/.systemd-worker` | 8MB | Miner disguised as systemd service |
| `/var/tmp/.lolminer-config.json` | 222B | lolMiner configuration |
| `/var/tmp/.cache-config` | 24KB | Configuration cache |
| `/tmp/1.98/mine_*.sh` | Multiple | Mining scripts for ETH, Zano, Grin, Conflux, Radiant, etc. |

**Worker node** container had the same `/tmp/1.98/` mining scripts.

### Miner Process Details

```
python3.7.3 -a progpowz -o stratum+tcp://45.61.148.247:8443 \
  -u KrQtbtsrPTqSTzQwZZisiyJxgtcDMwrdVrQ -w r
```

- **Algorithm:** ProgPoW (Zano cryptocurrency)
- **Pool:** `stratum+tcp://45.61.148.247:8443`
- **Wallet:** `KrQtbtsrPTqSTzQwZZisiyJxgtcDMwrdVrQ`
- **Watchdog:** Parent process (pid watchdog) auto-restarts the miner if killed

### Impact on Training

The miner was likely active since **Feb 10-11** (file creation dates), affecting:

1. **v1 experiment** (Feb 11): NCCL crash at step 28 — miner competing for GPU memory during FSDP all-reduce
2. **v2 experiment** (Feb 13-14): 11 steps completed, but GPU contention may have inflated update_actor times
3. **v3 experiment** (Feb 15-16): NCCL crash at step 5 — miner restarted after container restart, consuming GPU resources during long update_actor phase

The miner running at GPU priority `SN` (nice level) would intermittently cause NCCL timeouts when it competed for GPU memory during the 3.5-hour update_actor FSDP operations.

### Remediation

1. Killed miner processes on both nodes (kill -9)
2. Removed all malware files from `/var/tmp/` and `/tmp/`
3. Verified host systems are clean (malware was container-only)
4. Full security audit of containers pending

### Infection Vector (Suspected)

Likely entry via SSH (authorized_keys has 5 public keys) or through the Docker container's exposed ports. The `/tmp/1.98/` directory suggests a known mining toolkit (version 1.98 of lolMiner or similar).

## Key Observations

1. **pg_clipfrac=0 persists across all three versions** (v1: lr=1e-6, v2: lr=5e-6, v3: lr=1e-6) — the fundamental PPO issue remains unsolved regardless of lr or batch_size changes

2. **update_actor is the critical bottleneck** (87% of step time):
   - Scales linearly with batch_size
   - FSDP param_offload + optimizer_offload adds significant CPU↔GPU transfer overhead
   - With 143GB H200 memory, offloading may be unnecessary
   - ppo_max_token_len_per_gpu=128000 may cause excessive padding

3. **Sandbox concurrency fix worked**: collect_trajectory only increased ~2.2x despite 4x batch, confirming concurrent sandbox creation

4. **score/mean improved in step 3** (0.162) with 15/32 batches having partial solves — but this is training set performance, not validation

5. **All training progress lost**: 4 steps (~18 hours GPU time) with no checkpoint saved due to save_freq=5

## v4 Optimization Directions

1. **Reduce update_actor time**:
   - Disable param_offload and/or optimizer_offload (143GB H200 should fit 32B model + optimizer states)
   - Reduce ppo_max_token_len_per_gpu (check actual token usage vs 128000 budget)
   - Add sub-phase timing to identify exact bottleneck within update_actor

2. **Address pg_clipfrac=0**:
   - Investigate if FSDP offloading causes precision loss in optimizer states
   - Try bf16 optimizer states instead of fp32
   - Consider alternative optimizers (e.g., 8-bit Adam)

3. **Security hardening**:
   - Audit SSH keys and remove unauthorized access
   - Restrict container network access to only necessary endpoints
   - Monitor GPU utilization for anomalous activity
