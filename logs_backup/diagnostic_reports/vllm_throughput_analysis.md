# vLLM Throughput Degradation Analysis

## Problem Summary

vLLM inference throughput severely degraded starting at Step 161, causing training to produce zero gradients.

## Throughput Timeline

### Normal Performance (Step 157 and earlier)
- **llm_time**: ~1101s per step
- **Estimated throughput**: ~21 tok/s
- **Training**: Producing valid gradients

### Degraded Performance

**Step 161 (Worst)**:
- llm_time_mean: 4611s
- Estimated throughput: ~5 tok/s
- **Degradation**: 4.2x slower than normal
- Training: batch/solve_partial:1, some gradient

**Step 165**:
- llm_time_mean: 3008s
- Estimated throughput: ~7 tok/s
- **Degradation**: 2.7x slower than normal
- Training: batch/solve_none:4, zero gradient (pg_loss:0.0)

**Steps 166-170**:
- llm_time_mean: 2410-2750s
- Estimated throughput: ~8-9 tok/s
- **Degradation**: 2.2-2.5x slower than normal
- Training: All steps show batch/solve_none:4, pg_loss:0.0, ppo_kl:0.0

## Impact on Training

### Why Zero Gradients?

With trajectory_timeout=3600s and severely degraded vLLM throughput:

1. **Normal Case** (~21 tok/s, 1101s llm_time):
   - Trajectories complete 20+ steps
   - Agents solve tasks successfully
   - Training produces valid gradients

2. **Degraded Case** (~5-9 tok/s, 2400-4600s llm_time):
   - Most trajectory time consumed by slow inference
   - Only 3-6 agent steps completed before ENV_TIMEOUT
   - All 32 trajectories fail (batch/solve_none:4 means all 4 rollout workers had no solves)
   - Zero rewards → zero advantages → zero gradient

## Infrastructure Checks

### ✅ Network Health
- Inter-node ping latency: 0.274ms (normal)
- Network interface b_manage0: 0 RX errors, 0 TX errors
- No packet loss or connectivity issues

### ✅ GPU Topology
- All GPUs connected via NV18 (NVLink)
- Worker node NVLink status: 26.562 GB/s on all links (normal H200 bandwidth)
- No hardware degradation detected

### ✅ Ray Cluster
- 2 nodes active (correct configuration)
- Head: 10.83.115.10 (8x H200)
- Worker: 10.83.115.12 (8x H200)
- No additional nodes or topology changes

### ✅ NCCL Configuration
From train_deepswe_full.sh:
```bash
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_IFNAME=b_manage0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
```

Configuration appears standard for this setup.

## Unresolved Questions

### 1. What caused the initial degradation at Step 161?

The throughput was normal through Step 160, then suddenly degraded 4x at Step 161. Possible causes:

- **NCCL communication issue**: Cross-node TP=16 communication slowdown
- **vLLM worker scheduling**: Ray placement or resource allocation changed
- **GPU Direct RDMA**: Network path for GPU-GPU communication degraded
- **System event**: Background process, kernel task, or cluster management action

### 2. Why did it partially recover?

- Step 161: 4611s (worst)
- Step 165: 3008s (better, but still 2.7x slow)
- Steps 166-170: 2400-2750s (stabilized at 2.2-2.5x slow)

This partial recovery suggests:
- Not a permanent hardware failure
- Possibly a resource contention or scheduling issue that partially resolved
- May be related to checkpoint loading or Ray worker reinitialization

### 3. Why does training hang after checkpoint loading?

Current symptoms:
- All 16 ranks successfully load checkpoint
- GPU memory reaches 96GB on all GPUs
- Training hangs indefinitely, never proceeds to "step 161 started"
- GPU utilization: 0-3% (idle)

This appears to be a deadlock in:
- FSDP collective operations after checkpoint restore
- Ray placement group or actor initialization
- vLLM TP=16 worker communication handshake

## Diagnostic Tests Needed

### 1. NCCL Bandwidth Test
```bash
# Test cross-node all-reduce bandwidth
# Install nccl-tests if not available
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make MPI=1 MPI_HOME=/path/to/mpi
mpirun -np 16 -N 8 --host 10.83.115.10,10.83.115.12 \
    ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

Expected: >150 GB/s algbw for H200 NVLink
If significantly lower: NCCL communication is degraded

### 2. vLLM TP Communication Test
Run a simple vLLM inference with TP=16 across nodes and profile:
- Time to first token
- Token generation throughput
- Ray object store latency

### 3. Ray Cluster Debug
```bash
ray status --verbose
ray memory --stats-only
# Check for stale placement groups or leaked resources
```

### 4. Compare Against Fresh Initialization

Restart from Step 160 checkpoint but with:
1. Fresh Ray cluster (ray stop --force on both nodes, then ray start)
2. Clear all Ray shared memory (/tmp/ray/, /dev/shm/ray*)
3. Monitor vLLM throughput from first step after restart

If throughput is normal → issue is with stale Ray/vLLM state
If throughput is still degraded → issue is with checkpoint or model state

## Recommendations

### Immediate Actions

1. **Test from earlier checkpoint**: Use global_step_160 to restart
2. **Fresh Ray cluster**: Complete cleanup and restart
3. **Monitor first rollout**: Watch llm_time_mean on first step after restart
   - If <1500s: Throughput is recovering, continue training
   - If >2500s: Throughput still degraded, investigate further

### If Throughput Still Degraded

1. **Try Step 157 or earlier checkpoint** (if available)
2. **NCCL bandwidth test** to verify cross-node communication
3. **Check system logs** around Feb 21-22 for events at Step 161 time
4. **Consider training without TP=16**: Try TP=8 to isolate cross-node communication

### If Training Hangs Persist

1. **Add timeout and auto-restart**: Detect hang and restart automatically
2. **Simplify initialization**: Skip some checkpoint restore components
3. **Debug with Ray profiling**: Enable Ray timeline and analyze where hang occurs
4. **Check FSDP**: May need to adjust FSDP synchronization parameters

## Next Steps

Priority order:
1. Kill current hung training
2. Fresh Ray cluster restart (both nodes)
3. Restart from Step 160 checkpoint
4. Monitor first rollout llm_time_mean
5. If <1500s: Continue and monitor for stability
6. If >2500s: Run NCCL bandwidth test and deeper diagnostics
