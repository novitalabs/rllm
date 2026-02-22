# Multi-Node Training Architecture Analysis: 64x H200 (8 Nodes × 8 GPUs)

**Date:** 2026-02-22
**Training:** SWE-Bench Verified v6.1/v6.2, Qwen3-32B, verl hybrid engine
**Cluster:** 8 nodes × 8 H200 GPUs (143 GiB each), ConnectX-7 400Gbps RoCE

---

## 1. Architecture Overview

### Worker Topology

```
Ray Cluster (64 GPUs)
├── Head Node (10.83.115.14) — 8 H200 GPUs
│   ├── TaskRunner (pid=77843, CPU-only Ray actor)
│   │   └── Orchestrates: sandbox envs, trajectory collection, training loop
│   ├── AsyncActorRolloutRefWorker[0-7] (1 per GPU)
│   │   └── vLLM TP group 0 (TP=8, single inference instance)
│   └── FSDP DP rank 0 (8 GPUs)
│
├── Worker Node 1 (10.83.115.17) — 8 H200 GPUs
│   ├── AsyncActorRolloutRefWorker[8-15]
│   │   └── vLLM TP group 1
│   └── FSDP DP rank 1
│
├── ... (6 more worker nodes)
│
└── Worker Node 7 (10.83.115.28) — 8 H200 GPUs
    ├── AsyncActorRolloutRefWorker[56-63]
    │   └── vLLM TP group 7
    └── FSDP DP rank 7
```

### Key Design: Hybrid Engine (Time-Multiplexed GPU Sharing)

GPUs are **not** shared concurrently between inference and training. Instead, the hybrid engine **time-multiplexes**:

```
GPU Timeline per Step:
┌──────────────────────────────────┬───────┬─────┬─────┬──────────────────┐
│       collect_trajectory         │ xform │ olp │ adv │  update_actor    │
│        (vLLM inference)          │       │     │     │  (FSDP training) │
│         ~97% of step             │ <1s   │~12s │~12s │   ~2.4% of step  │
└──────────────────────────────────┴───────┴─────┴─────┴──────────────────┘
       ← Rollout Phase →                            ← Training Phase →
```

### Parallelism Scheme

| Dimension | Value | Scope |
|-----------|-------|-------|
| Tensor Parallel (TP) | 8 | Within each node (NVLink) |
| Data Parallel (DP) | 8 | Across 8 nodes (RoCE RDMA) |
| Sequence Parallel (SP) | 8 | Ulysses, within each node |
| Pipeline Parallel (PP) | 1 | Not used |

---

## 2. Per-Phase Node Activity Analysis

### Phase 1: collect_trajectory (Rollout / Inference)

**Duration:** ~3200-3900s (53-65 min), **97.3% of step time**

**What happens:**
1. TaskRunner (head node, CPU) creates 64 sandbox environments (8 prompts × 8 rollouts)
2. Each trajectory runs an agentic loop: LLM generation → sandbox execution → observe → repeat (up to 30 steps)
3. LLM generation requests are dispatched to the 8 vLLM instances (one per node, TP=8 each)
4. Sandbox HTTP calls go to PPIO cloud API (CPU, no GPU)

**Per-node roles:**

| Component | Head Node | Worker Nodes |
|-----------|-----------|--------------|
| TaskRunner (orchestrator) | ✅ Runs here | ❌ |
| Sandbox HTTP calls | ✅ Initiated here | ❌ |
| vLLM inference (TP=8) | ✅ One of 8 instances | ✅ Each has one instance |
| Inter-node GPU comm | ❌ None during inference | ❌ None during inference |
| CPU load | ~3% (sandbox HTTP + Ray) | ~0.3% (idle) |

**GPU utilization snapshot during rollout (v6.1 step 1, 20/64 trajectories):**

| Node | GPU Util | Mem Controller | Memory Used | Power Draw |
|------|----------|----------------|-------------|------------|
| 10.83.115.14 (HEAD) | 94-96% | 47-48% | ~107 GiB (74%) | 618-658W |
| 10.83.115.17 | 93-96% | 46-48% | ~107 GiB (74%) | 593-662W |
| 10.83.115.22 | 95-96% | 47% | ~107 GiB (74%) | 596-632W |
| 10.83.115.23 | 99-100% | 13-17% | ~107 GiB (74%) | 266-295W |
| 10.83.115.25 | 99-100% | 11-15% | ~107 GiB (74%) | 266-295W |
| 10.83.115.26 | 98-100% | 10-17% | ~107 GiB (74%) | 266-300W |
| 10.83.115.27 | 99-100% | 11-19% | ~107 GiB (74%) | 258-321W |
| 10.83.115.28 | 99-100% | 10-14% | ~107 GiB (74%) | 271-299W |

**Key observation: Two tiers of node activity during rollout.**

- **Tier 1 (active inference)**: Nodes 14, 17, 22 — high memory controller utilization (46-48%), high power (600-660W). These nodes' vLLM instances are actively processing generation requests.
- **Tier 2 (idle/waiting)**: Nodes 23, 25, 26, 27, 28 — low memory controller utilization (10-19%), low power (260-320W), but 99-100% SM utilization. The high SM% with low memory bandwidth indicates GPUs are spin-waiting (CUDA stream synchronization or vLLM idle loop), not doing useful compute.

**Why only 2-3/8 nodes active? Root Cause Analysis:**

The idle node problem stems from the interaction between trajectory execution patterns and vLLM request dispatch:

1. **Trajectory execution is sandbox-bound, not LLM-bound.** Each trajectory runs an agentic loop of ~30 steps. Each step involves:
   - LLM generation (~50s per call) — needs GPU
   - Sandbox execution (edit file, run tests, etc.) — needs PPIO HTTP API, no GPU
   - At any moment, most trajectories are waiting for sandbox HTTP responses

2. **Request dispatch path** (`VerlEngine → AsyncLLMServerManager → server_handles`):
   - `AsyncLLMServerManager` holds 8 `server_handles` (one per node's vLLM TP group)
   - `generate()` in `verl_engine.py:80` dispatches each request to one vLLM instance
   - vLLM's continuous batching within each instance means it will absorb multiple pending requests into one forward pass
   - With only ~10-20 of 64 trajectories requesting LLM generation simultaneously, 2-3 vLLM instances absorb the entire load

3. **The imbalance is inherent to SWE-Bench trajectories:**
   - LLM call duration: ~50s (single generation)
   - Sandbox round-trip: ~3-5s (but trajectories spend most time waiting for test execution: 10-30s)
   - Duty cycle of LLM usage per trajectory: `50s / (50s + 15s) ≈ 77%`, but trajectories are out of phase
   - At any moment: `64 trajectories × 77% LLM duty / (batch_size_per_instance ≈ 10-20) = ~2.5-5 nodes needed`
   - Observed: 2-3 nodes active, consistent with the model

**Network traffic during rollout:**
- Inter-node GPU (RoCE): **~0 bytes** — vLLM instances are independent per node
- Management network (b_manage0): Active — PPIO sandbox HTTP, Ray coordination
- GPU-direct NIC counters (GPU0-GPU7): ~14 MB total, accumulated from NCCL init, not from rollout traffic

---

### Phase 2: transform_trajectory

**Duration:** ~0.3-0.6s, **negligible**

**What happens:** TaskRunner assembles multi-step trajectories, tokenizes, creates attention masks, pads sequences.

| Component | Head Node | Worker Nodes |
|-----------|-----------|--------------|
| CPU processing | ✅ TaskRunner does this | ❌ Idle |
| GPU | ❌ Not used | ❌ Not used |
| Network | ❌ Local only | ❌ |

---

### Phase 3: old_log_prob (Reference Log Probabilities)

**Duration:** ~10-15s, **0.4% of step time**

**What happens:**
1. TaskRunner distributes batch to all 64 workers (8 DP ranks × 8 TP ranks)
2. Each DP rank receives `batch_size / 8` = 1 sample
3. FSDP forward pass computes log probabilities
4. Requires NCCL AllGather for FSDP parameter reconstruction

**Per-node activity:**

| Component | Head Node | Worker Nodes | Communication |
|-----------|-----------|--------------|---------------|
| FSDP forward pass | ✅ DP rank 0 | ✅ DP rank 1-7 | Intra-node: NVLink (TP) |
| NCCL AllGather | ✅ | ✅ | **Inter-node: RoCE RDMA** |
| Data distribution | ✅ Sends batch shards | ✅ Receives shard | Ray object store |

All 8 nodes active equally. ~2s compute + ~10s NCCL sync overhead.

---

### Phase 4: adv (Advantage Computation)

**Duration:** ~10-15s, **0.4% of step time**

**What happens:** RLOO (Reinforcement Learning with Leave-One-Out baseline) advantage computation.

| Component | Head Node | Worker Nodes |
|-----------|-----------|--------------|
| Advantage calc | ✅ TaskRunner (CPU/GPU) | ❌ Idle |
| Network | ❌ Local | ❌ |

The `adv` timer in the logs likely includes the FSDP synchronization overhead from old_log_prob, as both are reported with the same timing (~10-15s).

---

### Phase 5: update_actor (FSDP Training)

**Duration:** ~85-93s (1.4-1.6 min), **2.4% of step time**

**What happens:**
1. Batch distributed across 8 DP ranks (1 sample per rank)
2. 4 PPO epochs × 1 mini-batch = 4 optimizer steps
3. Each optimizer step: FSDP forward → backward → AllReduce gradients → optimizer step

**Per-node activity:**

| Component | Head Node | Worker Nodes | Communication |
|-----------|-----------|--------------|---------------|
| FSDP forward | ✅ DP rank 0 | ✅ DP rank 1-7 | NVLink (TP within node) |
| FSDP backward | ✅ | ✅ | NVLink (TP) |
| Gradient AllReduce | ✅ | ✅ | **RoCE RDMA (400Gbps)** |
| Optimizer step | ✅ | ✅ | Local per shard |
| Parameter broadcast | ✅ | ✅ | **RoCE RDMA** |

**All 8 nodes are equally active during update_actor.** This is the only phase with heavy inter-node GPU communication.

**NCCL communication volume estimate (per optimizer step):**

```
Model: Qwen3-32B (BF16) = 32B params × 2 bytes = 64 GB
FSDP shards across 8 DP ranks = 8 GB per rank
AllReduce gradient volume ≈ 2 × model_size = 128 GB total ring traffic
At 400 Gbps (50 GB/s) per NIC: ~2.6s per AllReduce
4 PPO epochs × 2.6s = ~10.4s NCCL communication
Remaining ~80s = compute (forward + backward on 8 nodes)
```

**GPU utilization during update_actor (estimated from v6.0 MFU data):**

| Metric | Value |
|--------|-------|
| MFU (Model FLOPS Utilization) | 6.5-7.7% |
| GPU compute utilization | ~90-95% (FSDP compute + NCCL wait) |
| Memory controller utilization | ~60-80% (gradient computation) |
| Power draw | ~700W+ per GPU (peak compute) |
| update_actor ms/token | 0.070-0.087 |

---

## 3. Head Node vs Worker Node Differences

### Summary

| Aspect | Head Node (10.83.115.14) | Worker Nodes (×7) |
|--------|--------------------------|-------------------|
| **Ray Role** | Head + GCS server | Worker |
| **TaskRunner** | ✅ Runs here (CPU actor) | ❌ |
| **Sandbox orchestration** | ✅ All PPIO HTTP calls | ❌ |
| **Data loading/batching** | ✅ train_dataloader | ❌ |
| **vLLM inference** | ✅ TP group 0 (8 GPUs) | ✅ TP group 1-7 |
| **FSDP training** | ✅ DP rank 0 | ✅ DP rank 1-7 |
| **Checkpoint saving** | ✅ Coordinates | ✅ Each saves own shard |
| **Logging/metrics** | ✅ Aggregates & logs | ❌ |
| **CPU usage** | ~3% (sandbox + Ray overhead) | ~0.3% (Ray worker only) |
| **CPU memory** | ~75 GiB (data + Ray objects) | ~60 GiB |
| **GPU usage** | Identical to workers | Identical to head |
| **Process count** | ~1272 (incl. Ray head) | ~817 |

### Key Differences

1. **During rollout**: Head node is **busier on CPU** (TaskRunner manages all 64 sandbox environments, makes HTTP calls, processes observations). Worker nodes' CPUs are idle — they only have GPU workers.

2. **During training**: All nodes are **identical in GPU workload**. Each node holds 1 DP shard of the model and processes 1/8 of the batch. NCCL communication is symmetric.

3. **The head node is a potential bottleneck** during rollout: all sandbox interactions, trajectory assembly, and reward computation flow through the TaskRunner on the head. This is a CPU-bound single-process bottleneck.

---

## 4. Per-Step Timing Breakdown (v6.0, Steps 1-4)

### Absolute Timing

| Phase | Step 1 | Step 2 | Step 3 | Step 4 | Average | % of Step |
|-------|--------|--------|--------|--------|---------|-----------|
| **collect_trajectory** | 3,781s | 3,907s | 3,593s | 3,167s | 3,612s | **97.3%** |
| transform_trajectory | 0.3s | 0.6s | 0.4s | 0.4s | 0.4s | 0.01% |
| old_log_prob | 15.5s | 10.3s | 10.9s | 11.0s | 11.9s | 0.3% |
| adv | 15.5s | 10.3s | 11.0s | 11.0s | 12.0s | 0.3% |
| **update_actor** | 90.8s | 93.2s | 85.0s | 91.0s | 90.0s | **2.4%** |
| **Total step** | 3,887s | 4,012s | 3,689s | 3,269s | 3,714s | 100% |

### collect_trajectory Internal Timing Breakdown (v6.1 Step 1)

The `collect_trajectory` phase (97.3% of step time) breaks down into LLM inference time vs sandbox/environment execution time:

| Metric | Value | % of Trajectory |
|--------|-------|-----------------|
| `traj/llm_time_mean` | 1,522s (25.4 min) | **94.8%** |
| `traj/env_time_mean` | 84s (1.4 min) | **5.2%** |
| `traj/total_time_mean` | 1,607s (26.8 min) | 100% |
| `traj/total_time_max` (slowest) | 3,685s (61.4 min) | — |
| `collect_trajectory` total | 3,751s (62.5 min) | — |

**Key observations:**

1. **LLM inference dominates** at 94.8% of per-trajectory time. Each trajectory makes ~30 LLM calls × ~50s each = ~1500s, matching the measured mean.

2. **Sandbox execution is fast** at only 84s mean (~2.8s per step). The PPIO sandbox API responds quickly; the bottleneck is GPU inference, not environment interaction.

3. **Long-tail problem**: Mean trajectory time is 1,607s but the maximum is 3,685s (2.3× the mean). The `collect_trajectory` phase cannot proceed until ALL 64 trajectories complete, so one slow trajectory extends the phase by ~35 min beyond the mean.

4. **Step time ≈ slowest trajectory**: `collect_trajectory` total (3,751s) is dominated by the slowest trajectory (3,685s), with ~66s overhead for setup and finalization.

```
collect_trajectory composition (v6.1 step 1):
├── Per trajectory (mean of 64):
│   ├── LLM inference: 1,522s (94.8%) ← GPU-bound, distributed across 8 vLLM instances
│   └── Env/sandbox:      84s  (5.2%) ← HTTP to PPIO API, CPU-only
├── Wall clock: 3,751s (bound by slowest trajectory at 3,685s)
└── Parallelism: 64 trajectories concurrent, but long-tail extends wall time 2.3×
```

### What Each Phase Means for the Cluster

```
collect_trajectory (97.3%):
  ┌─────────────────────────────────────────────────────────┐
  │ Head: TaskRunner orchestrates 64 trajectories           │
  │   └── HTTP calls to PPIO sandboxes (CPU-bound)          │
  │ All 8 nodes: vLLM generates tokens (GPU, TP=8/node)    │
  │   └── Only 2-3 nodes actively inferencing at any time   │
  │ Network: Management only (HTTP), zero inter-node GPU    │
  │ Bottleneck: Slowest trajectory (up to 60 min)           │
  └─────────────────────────────────────────────────────────┘

transform_trajectory (0.01%):
  ┌─────────────────────────────────────────────────────────┐
  │ Head: TaskRunner tokenizes & assembles (CPU, <1s)       │
  │ Workers: Idle                                           │
  └─────────────────────────────────────────────────────────┘

old_log_prob + adv (0.6%):
  ┌─────────────────────────────────────────────────────────┐
  │ All 8 nodes: FSDP forward pass (GPU)                    │
  │ Network: NCCL AllGather over RoCE RDMA                  │
  │ ~12s = mostly NCCL sync overhead                        │
  └─────────────────────────────────────────────────────────┘

update_actor (2.4%):
  ┌─────────────────────────────────────────────────────────┐
  │ All 8 nodes: FSDP forward + backward + optimizer (GPU)  │
  │ Network: NCCL AllReduce gradients over RoCE RDMA        │
  │ 4 PPO epochs × (forward + backward + AllReduce)         │
  │ ~90s total, all nodes equally loaded                    │
  └─────────────────────────────────────────────────────────┘
```

---

## 5. GPU Memory Phase Transition (Measured, v6.1 Step 1)

The phase monitor captured the exact GPU memory transitions on the head node during step 1:

```
Time (PST)   Phase                  GPU0 Mem (MiB)   GPU Util   State
─────────────────────────────────────────────────────────────────────────
02:38:12     Rollout (62/64)        107,332          95%        vLLM active: model + KV-cache
02:38:42     vLLM offload            9,704           95%        vLLM offloaded to CPU
02:39:12     FSDP loading (64/64)   36,282          100%        FSDP model shards loading onto GPU
02:39:43     old_log_prob           37,226          100%        FSDP forward pass (ref log-probs)
02:40:13     adv computation        37,230          100%        Advantage calculation
02:40:43     update_actor           41,440          100%        FSDP training (forward + backward)
02:41:13     vLLM reload           113,910           96%        vLLM reloaded for step 2 rollout
02:41:44     Step 2 rollout        113,910           94%        vLLM serving inference requests
```

**Key observations:**

1. **Massive memory swing**: 107 GiB (rollout) → 10 GiB (offload) → 37 GiB (FSDP) → 114 GiB (rollout reload)
2. **vLLM offload/reload takes ~30s each**: The transition windows (02:38:12→02:39:12 and 02:40:43→02:41:13) account for ~60s of the ~87s update_actor timing
3. **FSDP training uses only 37-41 GiB**: Much less than rollout (107 GiB) because FSDP shards the model across 64 GPUs (DP=8 × TP=8), while vLLM only shards across 8 GPUs (TP=8 within node)
4. **GPU util hits 100% during FSDP**: All phases after offload show 100% utilization (pure compute, no I/O waiting)
5. **Step 2 starts at 114 GiB**: Higher than step 1's 107 GiB because CUDA memory pools expanded

---

## 6. Resource Consumption Summary

### Per-GPU Memory by Phase (Measured)

| Phase | GPU Memory | What's in GPU |
|-------|-----------|---------------|
| **Rollout (vLLM)** | **107-114 GiB (74-79%)** | Model weights/8 (TP=8) + KV-cache + CUDA buffers |
| **vLLM offload** | **~10 GiB** | CUDA context only, model + KV-cache moved to CPU |
| **old_log_prob (FSDP)** | **36-37 GiB (25%)** | FSDP model shard/64 + activations |
| **update_actor (FSDP)** | **37-41 GiB (27%)** | FSDP shard + gradients + optimizer states + activations |
| **vLLM reload** | **113-114 GiB (79%)** | Model weights/8 + fresh KV-cache |

### Power Consumption Per Node

| Phase | Active Inference Node | Idle/Waiting Node | Notes |
|-------|----------------------|-------------------|-------|
| Rollout | 600-660W × 8 GPUs = 4.8-5.3 kW | 260-320W × 8 GPUs = 2.1-2.6 kW | 5/8 nodes idle |
| Training | ~700W × 8 GPUs = 5.6 kW (est.) | N/A (all active) | All 8 nodes active |

### Network Bandwidth

| Phase | Inter-Node GPU (RoCE) | Management (TCP) |
|-------|----------------------|------------------|
| Rollout | **0 bytes/s** | ~100 Mbps (sandbox HTTP) |
| old_log_prob | ~5-10 GB/s (AllGather) | Minimal |
| update_actor | **~30-50 GB/s (AllReduce)** | Minimal |

---

## 6. Efficiency Analysis

### GPU Utilization Efficiency

During a full step of ~62 min average:
- **Active inference time**: ~62 min (all GPUs allocated), but only 2-3 nodes doing useful work at any moment
- **Active training time**: ~90s (all 8 nodes, all 64 GPUs fully utilized)
- **Effective GPU utilization**: Very low — most of the step time is rollout, and 5/8 nodes are idle-spinning during rollout

```
Effective utilization estimate:
  Rollout: 3/8 nodes × 95% GPU util × 97.3% time = ~36% effective
  Training: 8/8 nodes × 95% GPU util × 2.4% time  = ~2.3% effective
  Overall: ~38% effective utilization across the cluster
```

### Why So Low?

1. **Rollout is the bottleneck (97% of time)**, and it's bound by sandbox latency (HTTP calls, test execution), not GPU compute
2. **vLLM inference is sparsely scheduled** — trajectories alternate between LLM calls and sandbox execution, so only a fraction need generation at any moment
3. **Long-tail trajectory** — all 64 trajectories must complete before training can start; the slowest one determines the step time
4. **8 DP shards for batch_size=8** — each node processes only 1 sample during training, so the per-GPU compute is very small

### Recommendations

1. **Increase batch_size** to utilize more nodes during training (e.g., batch_size=32 with 4 mini-batches)
2. **Reduce max_steps or trajectory_timeout** to limit the long-tail
3. **Consider fewer nodes with higher batch_size** — 4 nodes × DP=4 might be more efficient since rollout doesn't scale with nodes
4. **Overlap rollout and training** — true async overlap (not just async vLLM) could keep training nodes busy while rollout collects next batch

---

## 7. Appendix: Monitoring Snapshots

### Rollout Phase GPU Snapshot (v6.1, Step 1, 20/64 trajectories, 2026-02-22 01:50 UTC)

```
Node              GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    Avg Power
10.83.115.14 (H)  96/47%  96/48%  95/47%  95/47%  95/47%  95/48%  96/48%  95/47%  638W
10.83.115.17      94/46%  96/47%  94/46%  93/46%  95/47%  95/47%  96/48%  93/46%  624W
10.83.115.22      96/47%  94/46%  96/47%  96/47%  95/47%  95/47%  95/47%  95/47%  605W
10.83.115.23     100/13%  99/15%  99/14%  99/17% 99/13%  99/13%  100/13% 100/13% 282W
10.83.115.25      98/17% 100/11% 100/11% 100/11% 100/11% 100/11% 100/11% 100/11% 289W
10.83.115.26     100/11%  99/14%  100/11%  99/14% 100/14%  99/17%  99/17%  99/19% 303W
10.83.115.27     100/10% 100/10% 100/10% 100/10%  99/15%  100/13%  99/14%  99/12% 274W
10.83.115.28     100/10% 100/11%  99/14%  100/10% 100/11% 100/12% 100/10% 100/11% 280W

Format: SM_util/mem_controller_util
Active (>40% mem_ctrl): nodes 14, 17, 22 — vLLM inference running
Idle (<20% mem_ctrl): nodes 23, 25, 26, 27, 28 — GPUs spin-waiting
```

### Node Activity Rotation During Rollout

At 20/64 trajectories, nodes 14/17/22 were active (inference). At 33/64, the active set changed:

```
20/64 trajectories:                       33/64 trajectories:
  Node 14: ACTIVE (47% mem_ctrl)            Node 14: ACTIVE (47%)
  Node 17: ACTIVE (46-48%)                  Node 17: ? (not sampled)
  Node 22: ACTIVE (47%)                     Node 22: ? (not sampled)
  Node 23: idle (13-17%)                    Node 23: idle (12-18%)
  Node 27: idle (10-19%)         →          Node 27: ACTIVE (47%)  ← switched!
  Node 28: idle (10-14%)                    Node 28: ? (not sampled)
```

This confirms vLLM load **dynamically rotates** across nodes as trajectories alternate between LLM generation (needs GPU) and sandbox execution (needs HTTP/CPU only).

### Rollout Phase GPU Snapshot (v6.1 step 1, 33/64 trajectories, 2026-02-22 02:11 PST)

```
Node              GPU0     GPU1     GPU2     GPU3     GPU4     GPU5     GPU6     GPU7     Avg Power
10.83.115.14 (H)  95/47%   96/47%   95/47%   96/48%   94/47%   94/47%   96/48%   95/48%   627W
10.83.115.23      99/15%   98/18%   99/14%   99/13%   99/16%  100/12%   99/15%   99/16%   290W
10.83.115.27      96/47%   96/47%   96/47%   96/47%   95/47%   95/47%   96/47%   95/47%   607W
```

### Phase Transition Memory Trace (v6.1 Step 1, Head Node, 30s intervals)

```
Time (PST)   Traj Count    GPU0 Mem (MiB)    GPU0 Util   Phase
────────────────────────────────────────────────────────────────
02:38:12     62/64         107,332           95%          Rollout (vLLM active)
02:38:42     62/64           9,704           95%          vLLM → CPU offload
02:39:12     64/64          36,282          100%          FSDP model loading
02:39:43     64/64          37,226          100%          old_log_prob (FSDP fwd)
02:40:13     64/64          37,230          100%          adv computation
02:40:43     step:1         41,440          100%          update_actor (FSDP train)
02:41:13     step:1        113,910           96%          vLLM reload complete
02:41:44     step:1        113,910           94%          Step 2 rollout starts
02:43:44     6/64          114,026           99%          Step 2 rollout running
```

### CPU and Memory Snapshot

```
Node              CPU Usage  Load Avg   CPU Mem Used  CPU Mem Total  Processes  Zombies
10.83.115.14 (H)  0.7%       2.94       75 GiB        2 TiB          1272       780
10.83.115.17      0.4%       3.64       60 GiB        2 TiB          817        534

Head node: higher process count (Ray head, TaskRunner, dashboard) and memory
(training data, Ray object store for trajectory buffers).
```

### v6.1 Steps 1-3 Metrics

**Timing (seconds):**

| Phase | Step 1 | Step 2 | Step 3 | Average |
|-------|--------|--------|--------|---------|
| collect_trajectory | 3,751 | 3,731 | 3,375 | 3,619 |
| transform_trajectory | 0.4 | 0.3 | 0.3 | 0.3 |
| old_log_prob | 14.9 | 10.0 | 9.0 | 11.3 |
| adv | 14.9 | 10.0 | 9.0 | 11.3 |
| update_actor | 87.0 | 84.8 | 84.9 | 85.6 |
| **step total** | **3,853** | **3,826** | **3,469** | **3,716** |

**Trajectory Timing (per-trajectory means):**

| Metric | Step 1 | Step 2 | Step 3 | Average |
|--------|--------|--------|--------|---------|
| llm_time_mean | 1,522s | 1,452s | 1,240s | 1,405s |
| env_time_mean | 84s | 63s | 69s | 72s |
| total_time_mean | 1,607s | 1,515s | 1,309s | 1,477s |
| total_time_max | 3,685s | 3,535s | 3,195s | 3,472s |
| LLM % of trajectory | 94.8% | 95.8% | 94.7% | 95.1% |

**Training Metrics:**

| Metric | Step 1 | Step 2 | Step 3 |
|--------|--------|--------|--------|
| score | 0.021 | 0.016 | 0.109 |
| solve (none/all/partial) | 7/0/1 | 7/0/1 | 5/0/3 |
| entropy | 7,804 | 7,847 | 6,596 |
| pg_clipfrac | 0.0 | 0.0 | 0.0 |
| grad_norm | 166.1 | 343.8 | 520.9 |
| token_mismatch | 28.1% | 35.9% | 29.7% |
| MFU | 7.2% | 7.0% | 6.3% |
| max_mem_alloc (GiB) | 108.7 | 112.8 | 112.8 |

---

## 8. v6.2 Analysis (train_batch_size=64, 512 Trajectories)

### 8.1 Configuration Change

| Parameter | v6.1 | v6.2 | Impact |
|-----------|------|------|--------|
| train_batch_size | 8 | **64** | 8× more samples per step |
| ppo_mini_batch_size | 8 | **16** | 4 mini-batches/epoch |
| ppo_max_token_len_per_gpu | 32000 | **64000** | 2 samples/GPU per mini-batch |
| Trajectories per step | 64 | **512** | 64 samples × 8 rollouts |
| Optimizer steps per step | 4 | **16** | 4 epochs × 4 mini-batches |
| Samples per GPU per mini-batch | 1 | **2** | 16/8 DP ranks = 2 |

---

### 8.2 Per-Phase Node Activity (v6.2 Step 1, Measured)

#### Phase 1: collect_trajectory — 4,314s (85.8% of step)

**What happens:**
1. TaskRunner (head node) creates 512 sandbox environments (64 prompts × 8 rollouts)
2. vLLM instances on all 8 nodes `wake_up()` (NCCL barrier)
3. 512 concurrent async trajectories execute agentic loop: LLM call → sandbox step → observe → repeat (up to 30 steps)
4. Trajectories are collected as they complete; entire phase waits for the slowest one
5. vLLM instances `sleep()` (NCCL barrier, offload to CPU)

**Per-node GPU utilization during rollout (4 snapshots over 20 min, 288-427/512 trajectories):**

```
Snapshot   Traj     HEAD .14    WRKR .17    WRKR .22    WRKR .23    WRKR .25    WRKR .26    WRKR .27    WRKR .28
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1 (21:43)  288/512  47%  634W   47%  630W   47%  346W   15%  334W   13%  317W   14%  330W   46%  607W   23%  348W
           Status:  ACTIVE      ACTIVE      ★ACTIVE     IDLE        IDLE        IDLE        IDLE        IDLE
2 (21:44)  293/512  47%  634W   47%  630W   22%  346W   15%  334W   13%  317W   14%  330W   46%  607W   23%  348W
           Status:  ACTIVE      ACTIVE      IDLE        IDLE        IDLE        IDLE        ★ACTIVE     IDLE
3 (21:47)  310/512  47%  634W   47%  631W   11%  321W   17%  352W   15%  324W   20%  342W   46%  609W   14%  322W
           Status:  ACTIVE      ACTIVE      IDLE        IDLE        IDLE        IDLE        ACTIVE      IDLE
4 (22:04)  427/512  47%  636W   47%  632W   18%  320W   17%  317W   14%  329W   20%  312W   47%  611W   17%  303W
           Status:  ACTIVE      ACTIVE      IDLE        IDLE        IDLE        IDLE        ACTIVE      IDLE

Format: mem_controller_avg%  power_avg
ACTIVE = mem_ctrl > 30% (actively serving inference), IDLE = mem_ctrl < 25% (spin-waiting)
★ = changed status from previous snapshot
```

**Key observation: Even with 8× more trajectories (512 vs 64), still only 3/8 nodes active.**

The third active node rotates (22→27 between snapshots 1→2), but nodes 14 and 17 are always active. This pattern is identical to v6.1 — the bottleneck is not the number of trajectories but the concurrency of LLM requests.

**Why?** (detailed in Section 2):
- At any instant, ~10-20 of the remaining trajectories are requesting LLM generation
- Others are waiting for sandbox HTTP responses (test execution, file edits)
- vLLM's continuous batching absorbs 10-20 concurrent requests into 2-3 nodes
- 5 nodes' vLLM instances have zero pending requests and spin-wait at 100% SM utilization

**Network traffic during rollout (measured 30s delta on head node):**
- RoCE RDMA (GPU NICs): **~330 bytes** — essentially zero (no inter-node GPU comm during inference)
- Management network: Active (PPIO sandbox HTTP, ~100 Mbps)
- Note: NCCL uses /dev/infiniband verbs directly, bypassing kernel /proc/net/dev counters

**CPU and memory during rollout (all 8 nodes):**

```
Node              CPU%   Load Avg   RAM Used  Processes
HEAD 10.83.115.14 0.5%   12.4       75G/2T    1506
WRKR 10.83.115.17 0.1%   1.8        63G/2T     847
WRKR 10.83.115.22 0.2%   5.9        58G/2T    1112
WRKR 10.83.115.23 0.2%   5.0        62G/2T     847
WRKR 10.83.115.25 0.1%   5.1        60G/2T    1103
WRKR 10.83.115.26 1.4%   6.5        60G/2T    1103
WRKR 10.83.115.27 0.2%   1.4        60G/2T     856
WRKR 10.83.115.28 1.2%   7.7        65G/2T    1103

Head node: 75G RAM, 1506 procs (TaskRunner + Ray head + 512 async trajectories)
Workers: 58-65G RAM, 847-1103 procs (Ray worker + vLLM only)
```

---

#### Rollout Phase Internal Substeps

Within `collect_trajectory`, the execution follows this flow:

```
collect_trajectory (4,314s total)
│
├─ 1. init_envs_and_agents() [HEAD only, ~0.1s]
│     Create 512 env + agent instances, ThreadPoolExecutor(512 workers)
│
├─ 2. vLLM wake_up() [ALL 8 nodes, ~1-2s]
│     NCCL barrier across 64 GPUs; all vLLM TP groups resume from sleep
│
├─ 3. Launch 512 async trajectory tasks [HEAD only, ~0.1s]
│     asyncio.create_task() × 512, with Semaphore(512) for concurrency control
│
├─ 4. Per-trajectory agentic loop (512 concurrent, ~4,300s wall clock)
│     │
│     ├─ 4a. env.reset() — acquire sandbox from pool [HEAD ThreadPool → PPIO HTTP]
│     │     sandbox_idx = trajectory_idx % pool_size (deterministic reuse)
│     │     Create/resume PPIO sandbox, set up git safe.directory
│     │     ~2-5s per trajectory, parallelized across 512
│     │
│     ├─ 4b. Agent reset + prompt tokenization [HEAD CPU, ~0.5s]
│     │     Initialize chat_completions, tokenize problem statement
│     │     Reject if prompt > 8192 tokens (set token_mismatch=1.0)
│     │
│     ├─ 4c. For step_idx in range(30):  [per-trajectory loop, 15-60 min]
│     │     │
│     │     ├─ LLM generation [HEAD async → vLLM on any of 8 nodes, TP=8]
│     │     │   Route via AsyncLLMServerManager.generate()
│     │     │   vLLM continuous batching absorbs concurrent requests
│     │     │   ~50s per call (includes queuing, prefill, decode)
│     │     │
│     │     ├─ Action extraction + sandbox execution [HEAD ThreadPool → PPIO HTTP]
│     │     │   Parse LLM response into bash command or file edit
│     │     │   sandbox.commands.run(action, timeout=30)
│     │     │   ~3-15s per step (test execution can take 30s+)
│     │     │
│     │     └─ Token accumulation [HEAD CPU, ~0.01s]
│     │         Tokenize assistant response + env feedback
│     │         Track response_token_len for truncation detection
│     │
│     ├─ 4d. Final reward computation [HEAD ThreadPool → PPIO HTTP]
│     │     Run pytest in sandbox: sandbox.commands.run("python -m pytest ...")
│     │     ~1-10s depending on test suite size
│     │
│     └─ 4e. Trajectory assembly [HEAD CPU, ~0.01s]
│           Convert tokens to tensors, compute masks
│           Return {prompt_tokens, response_tokens, response_masks, reward}
│
├─ 5. Collect all 512 trajectories [HEAD, async as_completed]
│     Yield results as trajectories finish (sorted by idx afterward)
│     Bottleneck: slowest trajectory (3,683s in step 1, 6.1h without parallelism)
│
├─ 6. vLLM sleep() [ALL 8 nodes, ~1-2s]
│     NCCL barrier; offload vLLM model+KV-cache to CPU RAM
│     GPU memory drops: 107 GiB → 10 GiB
│
└─ 7. transform_trajectory() [HEAD CPU, ~1.0s]
      Left-pad prompts to 8192, right-pad responses to 32768
      Create DataProto: (512, 40960) tensor batch
```

**Per-trajectory timing breakdown (v6.2 step 1 means):**

| Metric | Value | % of Trajectory |
|--------|-------|-----------------|
| llm_time_mean | 1,048s (17.5 min) | **93.4%** |
| env_time_mean | 74s (1.2 min) | **6.6%** |
| total_time_mean | 1,122s (18.7 min) | 100% |
| total_time_max (slowest) | 3,683s (61.4 min) | — |
| steps_mean | 14.6 steps | — |
| steps_max | 30 steps | — |
| collect_trajectory wall clock | 4,314s (71.9 min) | — |

**Long-tail problem**: Mean trajectory time is 1,122s but max is 3,683s (3.3× the mean). The slowest trajectory bounds the entire phase.

---

#### Phase 2: transform_trajectory — 1.0s (0.02% of step)

HEAD node CPU only. Tokenizes and pads 512 trajectories into DataProto batch.

| Component | Head Node | Worker Nodes |
|-----------|-----------|--------------|
| CPU processing | ✅ (1.0s) | ❌ Idle |
| GPU | ❌ Not used | ❌ Not used |

---

#### Phase 3: old_log_prob — 53.2s (1.1% of step)

**All 8 nodes active.** FSDP forward pass computes reference log-probabilities.

With batch_size=64 and DP=8, each DP rank processes 64/8 = 8 samples.
GPU memory: ~31-32 GiB per GPU (FSDP sharded model + activations for 8 samples).

| Component | Head Node | Worker Nodes | Communication |
|-----------|-----------|--------------|---------------|
| FSDP forward | ✅ DP rank 0 (8 samples) | ✅ DP rank 1-7 (8 each) | NVLink (TP within node) |
| NCCL AllGather | ✅ | ✅ | **RoCE RDMA** |

Compared to v6.1 (14.9s with 1 sample/rank): 53.2s = ~3.6× slower due to 8× more samples per rank.

---

#### Phase 4: adv — 53.3s (1.1% of step)

Advantage computation (RLOO). Reported timing overlaps with old_log_prob sync.

---

#### Phase 5: update_actor — 657.2s (13.1% of step)

**All 8 nodes active.** FSDP training with 16 optimizer steps.

```
update_actor breakdown:
  4 PPO epochs × 4 mini-batches (64/16) = 16 optimizer steps
  Each step: FSDP forward → backward → AllReduce gradients → optimizer
  Per-step estimate: 657.2 / 16 ≈ 41s per optimizer step

  vs v6.1: 4 optimizer steps × ~22s each = 87s total
  v6.2: 16 optimizer steps × ~41s each = 657s total (2× per step due to 2 samples/GPU)
```

**GPU utilization during update_actor (all 8 nodes, measured 22:35 CST):**

```
Node              GPU0             GPU1             GPU2-6 (avg)     GPU7            Avg Power
HEAD .14          100%/21% 43.2G   100%/21% 43.7G   100%/21% 43.6G   100%/21% 42.9G   404W
WRKR .17          100%/23% 43.1G   100%/21% 43.3G   100%/22% 43.3G   100%/21% 42.5G   411W
WRKR .22          100%/22% 43.1G   100%/22% 43.3G   100%/21% 43.3G   100%/23% 42.6G   410W
WRKR .23          100%/21% 42.4G   100%/21% 42.6G   100%/21% 42.6G   100%/21% 41.9G   456W
WRKR .25          100%/22% 41.8G   100%/21% 42.0G   100%/21% 42.0G   100%/20% 41.2G   406W
WRKR .26          100%/14% 43.7G   99%/18%  43.9G   100%/14% 43.9G   100%/15% 43.2G   386W
WRKR .27          100%/14% 44.5G   100%/14% 44.6G   100%/14% 44.6G   100%/14% 43.9G   341W
WRKR .28          100%/18% 41.6G   100%/17% 41.8G   100%/16% 41.8G   100%/16% 41.1G   373W

All nodes 100% GPU util, 14-23% mem controller, 341-456W
Memory: 41-45 GiB per GPU (FSDP sharded model + gradients + optimizer states + activations for 2 samples)
```

**All 8 nodes are symmetrically loaded during training.** No distinction between head and worker — every node has identical FSDP compute. The mem_ctrl variation (14-23%) across nodes reflects micro-timing differences in NCCL AllReduce synchronization.

---

#### Phase 6: vLLM Reload — included in update_actor timing

After the last optimizer step, vLLM model is reloaded from CPU to GPU:
- GPU memory: 43 GiB → 114 GiB
- Takes ~30s (measured from phase monitor)

---

### 8.3 Phase Transition Memory Trace (v6.2 Step 1, Head Node, 15s intervals)

```
Time (CST)   Traj Count    GPU0 Mem (MiB)    GPU0 Util   Mem Ctrl   Phase
────────────────────────────────────────────────────────────────────────────────
22:29:27     510/512       107,608           96%         46%        Rollout (vLLM active, last 2 trajs)
22:29:43     512/512        25,898          100%         10%        vLLM → CPU offload (in progress)
22:29:59     512/512        31,418           97%         45%        FSDP model loading
22:30:14     512/512        31,418          100%         31%        old_log_prob (FSDP forward, 8 samples/rank)
22:30:30     512/512        31,418           96%         46%        old_log_prob continues
22:30:46     512/512        42,270           98%         29%        update_actor starting (FSDP train)
22:31:02     512/512        43,216          100%         19%        update_actor (opt step 2/16)
   ... 16 optimizer steps, ~41s each ...
22:41:25     512/512        43,244          100%         17%        update_actor (opt step 16/16)
22:41:41     step:1         41,440          100%         11%        vLLM reloading from CPU
22:41:57     step:1        113,910           96%         46%        Step 2 rollout begins
```

**Key observations (v6.2 vs v6.1):**

| Metric | v6.1 (bs=8) | v6.2 (bs=64) | Ratio |
|--------|-------------|--------------|-------|
| Rollout GPU memory | 107 GiB | 108 GiB | ~same |
| FSDP training memory | 37-41 GiB | **42-45 GiB** | ~1.1× (2 samples/GPU vs 1) |
| old_log_prob time | 14.9s | **53.2s** | 3.6× (8 samples/rank vs 1) |
| update_actor time | 87s | **657s** | 7.6× (16 opt steps × 2 samples/GPU) |
| update_actor % of step | 2.4% | **13.1%** | 5.5× better utilization of training |

---

### 8.4 Phase Transition Across All 8 Nodes (v6.2 Step 1)

Captures from the training phase capture script show per-node memory during the transition (22:29-22:31 CST):

```
Time       Phase           HEAD .14  WRKR .17  WRKR .22  WRKR .23  WRKR .25  WRKR .26  WRKR .27  WRKR .28
22:29:45   vLLM offload    26.3G     26.1G     26.3G     25.8G     24.9G     26.7G     27.1G     25.3G
22:29:59   FSDP loading    31.4G     31.0G     31.2G     30.4G     28.9G     31.9G     32.6G     29.6G
22:30:13   old_log_prob    31.4G     -         31.2G     30.6G     29.2G     31.9G     32.6G     29.7G
22:30:26   old_log_prob    31.4G     -         31.3G     30.6G     29.2G     31.9G     32.6G     -
22:30:45   update_actor    42.2G     42.1G     42.2G     41.5G     40.0G     42.8G     43.5G     -
22:31:11   update_actor    43.2G     43.1G     43.1G     42.4G     41.0G     43.7G     44.5G     41.5G
```

**Observations:**
1. **All 8 nodes transition synchronously** — NCCL barriers enforce lockstep phase changes
2. **Memory varies slightly per node** (41-45 GiB) — due to different data shard sizes and CUDA memory pool behavior
3. **The offload phase reduces memory uniformly** — all nodes drop from ~107 GiB to ~25-27 GiB in one 15s window
4. **Node 27 consistently has the highest memory** (44.5 GiB) — likely due to slightly longer CUDA memory pool expansion

---

### 8.5 Head Node vs Worker Node Differences (v6.2)

| Aspect | Head Node (.14) | Worker Nodes (×7) |
|--------|-----------------|-------------------|
| **TaskRunner** | ✅ CPU actor, orchestrates all 512 trajectories | ❌ |
| **Sandbox HTTP calls** | ✅ 512 concurrent PPIO API connections | ❌ |
| **Data loading** | ✅ train_dataloader, batch distribution | ❌ |
| **vLLM inference (rollout)** | ✅ TP group 0 — **always active** | 1-2 other nodes active, 5 idle |
| **FSDP training (update_actor)** | ✅ DP rank 0, identical workload | ✅ DP ranks 1-7, identical |
| **Checkpoint saving** | ✅ Coordinates, saves rank 0 shard | ✅ Each saves own FSDP shard |
| **CPU during rollout** | ~0.5%, load 12.4, 75G RAM | ~0.1-1.4%, load 1-8, 58-65G RAM |
| **GPU during rollout** | 96%/47%, 634W (**always active**) | Active: 96%/47%, 630W; Idle: 100%/15%, 320W |
| **GPU during training** | 100%/21%, 404W | 100%/14-23%, 341-456W |
| **Process count** | 1506 | 847-1103 |

**Key difference summary:**
- During **rollout (86% of step)**: Head node does all the CPU work (sandbox orchestration, 512 trajectory management). Head node's vLLM is always active. 5/8 nodes' GPUs are idle-spinning.
- During **training (13% of step)**: All 8 nodes are identical — same FSDP compute, same NCCL communication, same GPU memory. No head/worker distinction.

---

### 8.6 v6.2 Step 1 Full Metrics

**Timing:**

| Phase | Duration | % of Step |
|-------|----------|-----------|
| collect_trajectory | **4,314s** (71.9 min) | **85.8%** |
| transform_trajectory | 1.0s | 0.02% |
| old_log_prob | 53.2s | 1.1% |
| adv | 53.3s | 1.1% |
| update_actor | **657.2s** (11.0 min) | **13.1%** |
| **step total** | **5,026s** (83.8 min) | 100% |

**Trajectory stats:**

| Metric | Value |
|--------|-------|
| steps_mean / max | 14.6 / 30 |
| llm_time_mean | 1,048s (93.4%) |
| env_time_mean | 74s (6.6%) |
| total_time_mean / max | 1,122s / 3,683s |
| token_mismatch | 44.1% |

**Training metrics:**

| Metric | Value | vs v6.1 step 1 |
|--------|-------|----------------|
| score | 0.060 | 0.021 (3× better) |
| solve (none/all/partial) | 48/0/16 | 7/0/1 |
| entropy | 7,044 | 7,804 |
| **pg_clipfrac** | **6.8e-5** | **0.0** (now non-zero!) |
| ppo_kl | 4.2e-6 | 0.0 (now non-zero!) |
| grad_norm | 323.5 | 166.1 |
| MFU | 6.0% | 7.2% |
| max_mem_alloc | 113.1 GiB | 108.7 GiB |

---

### 8.7 v6.1 vs v6.2 Comparison

| Metric | v6.1 (bs=8) | v6.2 (bs=64) | Change |
|--------|-------------|--------------|--------|
| Trajectories per step | 64 | 512 | 8× |
| collect_trajectory | 3,751s | 4,314s | +15% (more trajectories, similar wall clock) |
| update_actor | 87s | 657s | 7.6× (16 opt steps vs 4, 2× per step) |
| update_actor % of step | 2.4% | **13.1%** | 5.5× |
| Step total | 3,853s | 5,026s | +30% |
| Nodes active during rollout | 3/8 | **3/8** | Same (!) |
| Nodes active during training | 8/8 | 8/8 | Same |
| pg_clipfrac | 0.0 | **6.8e-5** | Now non-zero |
| Effective GPU utilization | ~38% | ~42% | Improved by longer training phase |

**Key insights:**
1. **Rollout doesn't scale with batch size.** 512 trajectories use the same 3 nodes as 64 trajectories. The bottleneck is LLM request concurrency, not trajectory count.
2. **Training scales linearly.** 8× batch → 7.6× longer update_actor (16 opt steps at 2 samples/GPU vs 4 at 1/GPU).
3. **Training/rollout ratio improved.** From 2.4% to 13.1% of step time, meaning GPUs spend more time doing useful FSDP work.
4. **Policy is finally updating.** `pg_clipfrac > 0` with 16 optimizer steps, suggesting the larger batch + more opt steps overcome the policy update threshold.
