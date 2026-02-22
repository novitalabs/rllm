# Experiment Log: SWE-Bench Verified v6 — 64x H200 (8 nodes x 8 GPUs)

**Date:** 2026-02-21
**Status:** Running (step 5 in progress)

## Critical Discovery: RDMA/RoCE Network Not Used

### Problem

Previous multi-node training (v1-v5 on 2 nodes) was running NCCL over **TCP Socket at ~400MB/s** (b_manage0 interface, ~1Gbps effective). This made:
- FSDP parameter sync extremely slow (32B model broadcast across nodes)
- update_actor timing dominated by backward_s (~10000s per step on 2 nodes)
- NCCL timeout crashes during FSDP initialization on 8 nodes (600s default timeout exceeded)

### Root Cause

Each node has **8x Mellanox ConnectX-7 400Gbps RoCE NICs** — one dedicated per GPU:

```
Node 14 (head):
  GPU0: 10.83.116.6/26    (mlx5_7,  400Gbps)
  GPU1: 10.83.116.70/26   (mlx5_11, 400Gbps)
  GPU2: 10.83.116.134/26  (mlx5_6,  400Gbps)
  GPU3: 10.83.116.198/26  (mlx5_8,  400Gbps)
  GPU4: 10.83.117.6/26    (mlx5_5,  400Gbps)
  GPU5: 10.83.117.70/26   (mlx5_2,  400Gbps)
  GPU6: 10.83.117.134/26  (mlx5_0,  400Gbps)
  GPU7: 10.83.117.198/26  (mlx5_1,  400Gbps)

Management: b_manage0 10.83.115.14/24 (25Gbps bonded)
```

All 8 nodes have identical topology. Cross-node GPU subnets (10.83.116.x/26, 10.83.117.x/26) are directly connected with 0.1ms latency.

However, Docker containers were created **without `/dev/infiniband` mounted**, causing NCCL to fall back to Socket transport:

```
NCCL INFO NET/IB : No device found.
NCCL INFO NET/Socket : Using [0]b_manage0:10.83.115.14<0>
NCCL INFO Initialized NET plugin Socket
```

### Fix

1. **Mount `/dev/infiniband` into containers:**
```bash
docker run -d \
    --name deepswe-train \
    --network host \
    --ipc host \
    --gpus all \
    --shm-size=64g \
    -v /root/develop:/root/develop \
    -v /data:/data \
    -v /root/.ssh:/root/.ssh:ro \
    -v /tmp:/tmp \
    -v /dev/infiniband:/dev/infiniband \
    --ulimit memlock=-1:-1 \
    deepswe-train:latest
```

2. **Set NCCL environment:**
```bash
export NCCL_IB_DISABLE=0           # Enable IB/RoCE
export NCCL_SOCKET_IFNAME=b_manage0  # OOB control channel
export NCCL_DEBUG=INFO              # Verify RDMA is used
```

3. **Copy `libnvidia-ml.so.1` into new node containers** (NVIDIA Container Toolkit bug — doesn't mount this library on some containerd v2.2 configurations):
```bash
docker cp /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.126.09 deepswe-train:/usr/lib/x86_64-linux-gnu/
docker exec deepswe-train ln -sf libnvidia-ml.so.580.126.09 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
docker exec deepswe-train ldconfig
```

### Expected Impact

| Metric | Socket (before) | RDMA/RoCE (after) | Speedup |
|--------|----------------|-------------------|---------|
| Inter-node bandwidth | ~400 MB/s | ~50 GB/s (400Gbps) | **~125x** |
| FSDP broadcast 32B model | ~160s+ (timeout) | ~1-2s | **~100x** |
| backward_s (gradient allreduce) | ~10000s | est. ~200-500s | **~20-50x** |
| Step time (2 nodes, v5) | ~4.2h | est. ~1h | **~4x** |

### Other Issues Found During 8-Node Deployment

1. **Ray port collision**: Ray's random agent ports (dashboard_agent_grpc, runtime_env_agent, metrics_export) can collide with the worker port range (default 10002-19999). Fix: pin agent ports explicitly:
   ```bash
   ray start --dashboard-agent-grpc-port=50001 --runtime-env-agent-port=50002 --metrics-export-port=50003
   ```

2. **iptables blocking Ray**: Head node had firewall DROP rules from the 2-node setup. New nodes couldn't reach GCS port 6379. Fix: add ACCEPT rules for all node IPs before the DROP rules.

3. **libnvidia-ml.so.1 missing in containers**: NVIDIA Container Toolkit + containerd v2.2.1 doesn't properly inject all driver libraries. CUDA works but `nvidia-smi` and NCCL fail. Fix: manually copy from host.

---

## v6 Configuration

| Parameter | v5 Value | v6 Value | Rationale |
|-----------|----------|----------|-----------|
| nodes | 2 | **8** | Scale to 64 GPUs |
| train_batch_size | 32 | **8** | Fast iteration |
| ppo_mini_batch_size | 8 | **8** | = batch_size |
| param_offload | True | **False** | DP=8 sharding sufficient |
| optimizer_offload | True | **False** | DP=8 sharding sufficient |
| ppo_max_token_len_per_gpu | 128000 | **32000** | Match v1 for bs=8 |
| NCCL transport | Socket (~400MB/s) | **RDMA/RoCE (400Gbps)** | /dev/infiniband mounted |
| Resume from | v1 step 15 | **v5 step 5** | Continue from latest |

## v5 Results (for reference)

v5 ran 6 steps on 2 nodes before being stopped for v6 migration:

| Step | Score | Solve Partial | pg_clipfrac | Entropy | Step Time |
|------|-------|---------------|-------------|---------|-----------|
| 1 | 0.094 | 6/32 | 8.5e-5 | 7107 | ~4.1h |
| 2 | 0.094 | 6/32 | 8.5e-5 | 7107 | ~4.1h |
| 3 | 0.137 | 11/32 | 1.6e-4 | 6385 | ~4.3h |
| 4 | 0.145 | 10/32 | 1.1e-4 | 6816 | ~4.3h |
| 5 | 0.121 | 9/32 | 9.7e-5 | 7029 | ~7.2h (incl val+save) |
| 6 | 0.152 | 14/32 | 2.0e-4 | 7115 | ~4.3h |

Step 5 validation: test_score=0.111, pass@k=0.106

---

## NCCL 64-Rank RoCE Setup

### Critical Fix: NCCL_NVLS_ENABLE=0

NCCL 2.27.3 with 64 ranks across 8 nodes **hung during NCCL communicator initialization** with NVLS enabled (default). The hang occurred even though:
- All 7 pair-wise node connectivity tests passed (23-57 GB/s RDMA)
- 16-rank (2 nodes) test with two communicators passed
- Both Socket and RoCE transport exhibited the same hang

Root cause: NVLS multicast fails with `CUDA error 802 'system not yet initialized'` on these nodes. With 8 nodes, the NVLS fallback path appears to deadlock during the second communicator's QP setup.

**Fix:** `export NCCL_NVLS_ENABLE=0`

### Full NCCL Configuration

```bash
export NCCL_IB_DISABLE=0          # Enable RoCE (unconditional, not ${:-0})
export NCCL_IB_GID_INDEX=3        # RoCE v2 GID with routable IPv4
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11
export NCCL_SOCKET_IFNAME=b_manage0
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0         # Prevents 64-rank hang
export NCCL_DEBUG=INFO
```

Key: `NCCL_IB_GID_INDEX=3` selects the RoCE v2 GID with IPv4-mapped address. Without this, NCCL uses link-local fe80:: GIDs that aren't routable across nodes.

Important: Use unconditional `export NCCL_IB_DISABLE=0` (not `${:-0}`) because Docker containers may have `NCCL_IB_DISABLE=1` in their environment.

### PPIO_API_KEY Forwarding

Added `"PPIO_"` to `FORWARD_PREFIXES` in `rllm/trainer/verl/ray_runtime_env.py` so the API key is forwarded to Ray workers on all nodes.

### Resume Mode Issue

`trainer.resume_from_path` is ignored when `trainer.resume_mode` defaults to `"auto"`. The auto mode only checks `latest_checkpointed_iteration.txt` in the output directory. For future runs, must set `trainer.resume_mode=resume_path` and include `global_step_N` in the path.

v6 step 1 is effectively training from base Qwen3-32B (not v5 checkpoint).

---

## v6 Results

### Summary Table (Steps 1-4)

| Step | Score | Solve | pg_clipfrac | pg_loss | grad_norm | Entropy | token_mismatch | Step Time |
|------|-------|-------|-------------|---------|-----------|---------|----------------|-----------|
| 1 | 0.026 | 0/0/1 partial | 0.0 | -5.67 | 179.7 | 6408 | 28.1% | 65 min |
| 2 | 0.000 | 0/0/0 | 0.0 | 0.0 | 0.0 | 7296 | 37.5% | 67 min |
| 3 | **0.109** | **0/0/2 partial** | 0.0 | -10.12 | **635.2** | 6659 | 17.2% | 61 min |
| 4 | 0.078 | 0/0/3 partial | 0.0 | 10.92 | 397.3 | 6719 | 31.3% | 54 min |

Solve format: all/none/partial out of 8 prompts. Each prompt has 8 rollouts (rollout_n=8), 64 trajectories per step.

### Reward Distribution (Steps 1-4, 256 trajectories + step 5 partial)

| Reward | Count | Meaning |
|--------|-------|---------|
| 1.0 | 51 | Fully solved (all f2p tests pass) |
| 0.5 | 6 | Half of f2p tests pass |
| 0.429 | 1 | 3/7 f2p tests pass |
| 0.333 | 7 | 1/3 f2p tests pass |
| 0.286 | 2 | 2/7 f2p tests pass |
| 0.0 | 476 | No f2p tests pass or no submit |

Overall non-zero reward rate: ~12.3% (67/543 trajectories).

### Per-Step Timing Breakdown

| Module | Step 1 | Step 2 | Step 3 | Step 4 | Avg |
|--------|--------|--------|--------|--------|-----|
| **collect_trajectory** | 3,781s (63.0 min) | 3,907s (65.1 min) | 3,593s (59.9 min) | 3,167s (52.8 min) | 3,612s (60.2 min) |
| transform_trajectory | 0.3s | 0.6s | 0.4s | 0.4s | 0.4s |
| old_log_prob | 15.5s | 10.3s | 10.9s | 11.0s | 11.9s |
| adv (advantage) | 15.5s | 10.3s | 11.0s | 11.0s | 12.0s |
| **update_actor** | 90.8s (1.5 min) | 93.2s (1.6 min) | 85.0s (1.4 min) | 91.0s (1.5 min) | 90.0s (1.5 min) |
| **step (total)** | 3,887s (64.8 min) | 4,012s (66.9 min) | 3,689s (61.5 min) | 3,269s (54.5 min) | 3,714s (61.9 min) |

**Bottleneck: collect_trajectory dominates** (97.3% of step time on average). The FSDP update_actor phase is only 2.4% of step time — RDMA/RoCE delivers massive speedup over v5's Socket transport.

### Per-Step Trajectory Timing

| Metric | Step 1 | Step 2 | Step 3 | Step 4 |
|--------|--------|--------|--------|--------|
| traj/steps_mean | 16.8 | 17.4 | 17.2 | 19.3 |
| traj/steps_max | 30 | 30 | 30 | 30 |
| traj/llm_time_mean | 1,251s | 1,476s | 1,232s | 1,241s |
| traj/llm_time_max | 3,614s | 3,579s | 3,099s | 2,978s |
| traj/env_time_mean | 47s | 79s | 71s | 77s |
| traj/env_time_max | 245s | 427s | 369s | 1,465s |
| traj/total_time_max | 3,680s | 3,704s | 3,347s | 2,989s |
| response_length/mean | 15,311 | 14,684 | 16,197 | 18,401 |

The longest trajectory (~60 min llm_time) is the bottleneck for rollout completion. All other trajectories wait for the slowest one.

### Per-Step Memory & Compute

| Metric | Step 1 | Step 2 | Step 3 | Step 4 |
|--------|--------|--------|--------|--------|
| perf/mfu/actor | 6.5% | 6.0% | 7.2% | 7.7% |
| max_memory_allocated_gb | 109.1 | 113.0 | 113.0 | 113.0 |
| max_memory_reserved_gb | 120.5 | 124.6 | 124.6 | 124.6 |
| cpu_memory_used_gb | 63.2 | 63.9 | 64.2 | 64.7 |
| update_actor (ms/token) | 0.081 | 0.087 | 0.073 | 0.070 |

Notes:
- Step 1 used less GPU memory (109GB vs 113GB) because it was the first step and CUDA memory pools hadn't fully expanded
- CPU memory slowly increasing (63→65 GB) — monitor for leaks over many steps
- MFU improving slightly (6.5%→7.7%) as batch packing becomes more efficient

### Step-by-Step Notes

**Step 1:**
- NCCL RoCE GDRDMA confirmed working (GDR=1, 64 ranks, 8 nodes)
- update_actor 2x faster than v5 (91s vs ~175s) despite same batch size
- pg_clipfrac=0 persists — lr=1e-6 still too conservative
- Model started from base Qwen3-32B (resume mode bug, see above)
- 1 prompt partially solved (score=0.026 across all 64 trajectories)

**Step 2:**
- **Complete blank**: score=0.0, grad_norm=0.0, pg_loss=0.0
- All 8 prompts had zero reward across all rollouts
- token_mismatch=37.5% — highest of any step, 24/64 trajectories had zeroed response masks
- With all rewards=0, RLOO advantages are all 0, so no gradient signal at all
- Entropy jumped to 7296 (from 6408), suggesting model is exploring more randomly

**Step 3:**
- **Best step so far**: score=0.109, reward_max=1.0 (at least one fully solved instance)
- grad_norm spiked to 635 (from 180 in step 1) — strong gradient signal from the solved instance
- token_mismatch dropped to 17.2% — lowest of any step
- 2/8 prompts partially solved
- collect_trajectory faster (59.9 min) due to shorter max trajectory (3,099s llm_time_max)

**Step 4:**
- 3/8 prompts partially solved — most partial solves so far
- Fastest step (54.5 min) due to shorter max trajectory (2,978s llm_time_max)
- response_length/mean increased to 18,401 — model generating longer outputs
- env_time_max=1,465s — one sandbox had a very long eval (24 min)

---

## Issues Encountered

### 1. Retokenization / Token Mismatch (Critical)

**113 retokenization warnings** across steps 1-5. When assembling multi-step trajectories, token IDs at step boundaries don't match between the original generation and the re-tokenized full sequence. Affected trajectories have `response_masks` set to all zeros, meaning they contribute no gradient signal.

```
WARNING - When assemble steps, detect the trajectory not accumulative at position 6450.
Expected: [1648, 59, 77, 286, 671], Got: [89049, 77, 286, 671, 1416].
Setting response_masks to all 0s. This is likely due to retokenization.
```

Impact per step:
| Step | token_mismatch_mean | Affected trajectories (est.) |
|------|--------------------|-----------------------------|
| 1 | 28.1% | ~18/64 |
| 2 | 37.5% | ~24/64 |
| 3 | 17.2% | ~11/64 |
| 4 | 31.3% | ~20/64 |

This is a **major issue**: on average ~25% of trajectories are wasted (generate rollouts but don't contribute to training). The root cause is likely in multi-turn tokenization where environment observation text is spliced between model turns, and the concatenated sequence tokenizes differently than the individual turns.

### 2. pg_clipfrac=0 Across All Steps (Critical)

Policy gradient clipping fraction is exactly 0.0 on every step. Combined with ppo_kl=0.0, this means the policy is barely changing. At lr=1e-6 with the current advantage scale, the policy update is too small to exceed the clip ratio threshold (0.28).

Step 2 had grad_norm=0.0 because all rewards were 0, giving zero advantages under RLOO. The model effectively didn't train at all on that step.

**Recommendation for v7:** Increase lr to 5e-6 or 1e-5.

### 3. PPIO Sandbox 502 Bad Gateway Errors

**15 occurrences** of HTTP 502 from PPIO sandbox endpoints. These appear to be transient — the sandbox instances occasionally return 502 during file read/write operations. The training pipeline retries and doesn't crash, but it adds latency to affected trajectories.

```
HTTP Request: GET https://49983-iuv9nraibvpxnaiy1l5sg-ab9a1a45.sandbox.ppio.cn/files "HTTP/1.1 502 Bad Gateway"
```

### 4. Long-Tail Trajectory Bottleneck

The rollout phase is bound by the slowest trajectory. With max_steps=30 and trajectory_timeout=3600, a single trajectory can take up to 60 min. The step can't proceed to update_actor until all 64 trajectories complete.

| Step | Fastest traj | Slowest traj | Wait ratio |
|------|-------------|-------------|------------|
| 1 | 4s | 3,680s | 920x |
| 2 | 85s | 3,704s | 44x |
| 3 | 83s | 3,347s | 40x |
| 4 | 86s | 2,989s | 35x |

Step 1's fastest trajectory (4s) was likely a very short failed attempt. Steps 2-4 show more consistent minimum times (83-86s).

### 5. Resume Mode Bug

`trainer.resume_from_path` is ignored when `trainer.resume_mode` defaults to `"auto"`. v6 started from base Qwen3-32B instead of v5 checkpoint. See NCCL section above for details.

### 6. Step 2 Zero-Gradient Anomaly

Step 2 produced **zero gradient signal**: all 64 trajectories had reward=0.0. With RLOO (leave-one-out baseline), if all rewards in a prompt's rollout group are identical (all 0), the advantages are all 0, and grad_norm=0. Combined with 37.5% token mismatch, this step was a complete waste of ~67 minutes of compute.

This highlights a weakness of small batch sizes (bs=8): if the 8 sampled problems are all too hard, the entire step produces no learning signal.

### 7. Validation Bottleneck at test_freq=5 (Critical)

`test_freq=5` triggers full validation at step 5, running `_validate_agent()` over the entire val_dataloader (500 SWE-Bench samples). The validation iterates through **8 batches** (500 / val_batch_size=64 ≈ 8), each running 64 full sandbox trajectories — the same workload as a training rollout.

**Timeline (step 5):**

| Phase | Start (UTC) | Duration |
|-------|-------------|----------|
| Step 5 training rollout | 13:42 | ~50 min |
| Val batch 1 (64 traj) | 14:36 | 90 min |
| Val batch 2 (64 traj) | 16:06 | 91 min |
| Val batch 3 (64 traj) | 17:37 | 98 min |
| Val batch 4 (64 traj) | 19:15 | 110 min |
| Val batch 5 (64 traj) | 21:05 | 96 min |
| Val batch 6 (64 traj) | 22:41 | 107 min |
| Val batch 7 (64 traj) | 00:28 (Feb 22) | in progress |
| Val batch 8 (52 traj) | pending | est. ~90 min |

**Total validation time: ~13h** for 500 samples, vs ~5h for 5 training steps. Validation is **2.6x slower than training** because it evaluates 500 samples (8 batches × 64) vs training's 8 samples per step.

Each validation batch takes ~90-110 min — longer than training rollouts (~55 min) — because `val_kwargs.n=1` still runs 64 concurrent trajectories (val_batch_size=64 × n=1), and the same long-tail trajectory bottleneck applies.

**Impact:** Step 5 wall clock time is ~14h (50 min training + ~13h validation), making the total time for 5 steps **~18h** instead of the expected ~5h.

**Recommendation for v7:** Either reduce `val_batch_size` (e.g., 64→16 for a ~3h validation pass) or increase `test_freq` to 10-20 to validate less frequently.
