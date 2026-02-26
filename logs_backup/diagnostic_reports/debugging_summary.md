# Training Hang Root Cause Analysis

**Date**: 2026-02-22 14:23

## Problem Summary

Training hangs during initialization, before model loading, regardless of whether checkpoint is loaded or not.

## Root Cause Identified

**Ray Cluster Configuration Issue**: Training code creates a single-node local Ray instance instead of connecting to a multi-node cluster.

### Evidence

1. **Two Ray sessions found** (13:10 and 13:22)
   - Training creates its own local Ray instance via `ray.init()` in `main_ppo.py:74`
   - Without `address` parameter, `ray.init()` creates local instance with only 8 GPUs

2. **TP=16 requires 16 GPUs across 2 nodes**
   - Config: `actor_rollout_ref.rollout.tensor_model_parallel_size=16`
   - Local Ray instance only sees 8 GPUs on single node
   - vLLM TP=16 initialization fails/hangs when only 8 GPUs available

3. **Hang pattern consistent regardless of checkpoint**
   - With checkpoint: hangs after dataset filtering
   - Without checkpoint: hangs after dataset filtering
   - Both cases: GPU memory = 0 MiB (model never loads)

### Technical Details

From code inspection:
- `verl/trainer/main_ppo.py:74`: `ray.init(**OmegaConf.to_container(ray_init_kwargs))`
- Config file doesn't set `ray_kwargs.ray_init.address`
- Without address, Ray creates local instance instead of connecting to cluster
- Training script config: `trainer.nnodes=2, trainer.n_gpus_per_node=8`
- But Ray initialization doesn't honor multi-node config

### Ray Logs Analysis

Worker logs show:
```
Error reporting lease backlog information: RpcError: RPC error: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.83.115.10:32719: Failed to connect to remote host: Connection refused
```

This indicates raylet process issues, but root cause is the single-node Ray instance attempting TP=16.

## Solutions

### Option A: Fix Multi-Node Ray Initialization (Complex)

Need to properly configure Ray for multi-node:
1. Start Ray cluster with `ray start --head` on node 1
2. Start Ray worker with `ray start --address=...` on node 2
3. Configure training to connect: `ray_kwargs.ray_init.address='auto'` or specific address

**Problem**: verl's default workflow doesn't handle multi-node Ray setup

### Option B: Use TP=8 Single-Node (Recommended)

Reduce tensor parallelism to fit single node:
- `actor_rollout_ref.rollout.tensor_model_parallel_size=8`
- Model fits on 8x H200 GPUs
- Still use Ulysses=8 for FSDP across 2 nodes (different from TP)
- Simpler, less error-prone

**Benefits**:
- Avoids complex multi-node Ray setup
- vLLM TP=8 fully within single node (no cross-node communication)
- Can still use 2 nodes for FSDP training phase (Ulysses=8)

### Option C: Use torchrun for Multi-Node (Alternative)

verl may support torchrun-based multi-node:
```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=10.83.115.10:29500 \
    -m rllm.trainer.verl.train_agent_ppo ...
```

**Need to verify**: Whether verl's Ray initialization works with torchrun

## Recommendation

**Proceed with Option B (TP=8)** as immediate solution:
1. Simple configuration change
2. Avoids multi-node Ray complexity
3. H200 GPUs have enough memory for Qwen3-32B with TP=8
4. Can still use 2 nodes for FSDP training (Ulysses sequence parallel)

This will allow us to:
1. Verify training can initialize and run
2. Check if vLLM throughput is normal (~20 tok/s)
3. If normal, prove throughput degradation was related to multi-node issues
4. If still degraded, identifies different root cause

## Next Step

Create `train_deepswe_tp8.sh` with:
- `actor_rollout_ref.rollout.tensor_model_parallel_size=8`
- All other settings same
- Test if training initializes successfully
