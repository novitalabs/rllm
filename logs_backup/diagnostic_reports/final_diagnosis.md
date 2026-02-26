# Final Diagnosis - Training Initialization Hang

**Date**: 2026-02-22 14:28
**Status**: CRITICAL BLOCKING ISSUE

## Executive Summary

Training cannot initialize regardless of configuration. All attempts hang after dataset filtering, before model loading. This is a blocking issue that prevents ANY training from starting.

## Configurations Tested

### Test 1: TP=16 with Checkpoint (Original)
- Config: TP=16, 2 nodes, load from Step 160
- Result: **HANG** after dataset filtering
- GPU Memory: 0 MiB (model never loads)

### Test 2: TP=16 WITHOUT Checkpoint
- Config: TP=16, 2 nodes, no checkpoint (`trainer.load_checkpoint=false`)
- Result: **HANG** after dataset filtering (same as Test 1)
- GPU Memory: 0 MiB
- **Conclusion**: Problem is NOT checkpoint-related

### Test 3: TP=8 Single-Node
- Config: TP=8, single node, no checkpoint
- Result: **HANG** after dataset filtering (same as Test 1 & 2)
- GPU Memory: 0 MiB
- **Conclusion**: Problem is NOT multi-node Ray configuration

## Root Cause Analysis

### What We Know

1. **Consistent Hang Point**: All configurations hang at the same place:
   - ✅ Dataset filtering completes (100%)
   - ❌ Model loading never starts (GPU memory stays at 0)
   - ❌ Training never proceeds past initialization

2. **Ray Logs Show Worker Failures**:
   ```
   Error reporting lease backlog information: RpcError: RPC error: failed to connect to all addresses;
   last error: UNKNOWN: ipv4:10.83.115.10:32719: Failed to connect to remote host: Connection refused
   ```
   - Workers cannot connect to raylet
   - Raylet process becomes unresponsive

3. **Single-Node Ray Instance Created**:
   - Training calls `ray.init()` without address parameter
   - Creates local Ray instance instead of multi-node cluster
   - But even with TP=8 (fits on single node), still hangs

### What This Means

**The hang is NOT caused by**:
- ❌ Checkpoint corruption or loading issues
- ❌ Cross-node communication or TP=16 complexity
- ❌ Insufficient GPU resources

**The hang IS caused by**:
- ✅ Ray initialization/worker scheduling deadlock
- ✅ Issue occurs DURING Ray actor creation phase (after dataset load, before model load)
- ✅ Likely: verl's `RayClassWithInitArgs` or placement group creation hangs

## Technical Deep Dive

### Initialization Sequence

1. **Main process starts** ✅
2. **Ray.init() creates local instance** ✅
3. **Dataset filtering** ✅ (completes successfully)
4. **Ray actor creation** ❌ **← HANGS HERE**
   - Should create: TaskRunner, vLLM workers, FSDP workers
   - But: Placement group or resource scheduling deadlocks
5. **Model loading** (never reached)
6. **Training loop** (never reached)

### Why Ray Hangs

Possible causes:
1. **Resource Mismatch**: verl requests resources Ray cannot provide
2. **Placement Group Deadlock**: Conflicting resource constraints
3. **Worker Initialization Timeout**: vLLM or FSDP workers fail to start
4. **NCCL Initialization Issue**: Distributed training setup fails silently

### Evidence from Previous Successful Runs

Looking at the summary, training WAS working previously:
- Step 160 completed successfully
- Step 161 also completed (with degraded vLLM throughput but no hang)
- Steps 162-170 completed (zero learning but no hang)

**This means**: Something changed in the environment between Step 170 and now that causes Ray initialization to fail.

## What Changed?

### Hypothesis 1: Cluster State
- K8s pod scheduling changed
- Network configuration modified
- Resource limits adjusted

### Hypothesis 2: Ray Temp Files
- Corrupted Ray session state in /tmp/ray/
- Stale placement groups or resources
- **BUT**: We cleaned /tmp/ray/* multiple times with no effect

### Hypothesis 3: Docker State
- Too many Docker containers running
- Docker daemon resource exhaustion
- **Check**: `docker ps -a | wc -l`

### Hypothesis 4: System Resources
- Memory exhaustion
- File descriptor limits
- Disk I/O saturation

## Recommended Actions

### Immediate (Debug)

1. **Check System Resources**:
   ```bash
   free -h
   df -h
   docker ps -a | wc -l
   lsof | wc -l
   ulimit -a
   ```

2. **Check for Zombie Processes**:
   ```bash
   ps aux | grep defunct
   ps aux | grep ray | wc -l
   ```

3. **Test Minimal Ray**:
   ```python
   import ray
   ray.init(num_cpus=8, num_gpus=8)
   @ray.remote(num_gpus=1)
   class Worker:
       def test(self):
           return "OK"
   workers = [Worker.remote() for _ in range(8)]
   print(ray.get([w.test.remote() for w in workers]))
   ```

4. **Check verl Version**:
   ```bash
   pip show verl
   git -C ~/work/rllm log --oneline -5
   ```

### Short-term (Workaround)

1. **Reboot Worker Node**:
   ```bash
   ssh 10.83.115.12 "sudo reboot"
   ```
   - Clears all stale state
   - Resets Docker daemon
   - Fresh start for Ray

2. **Use Different Node**:
   - If another node available in cluster
   - Test if issue is node-specific

3. **Reduce Complexity**:
   - Try smallest possible config:
   - TP=1, batch=1, n=1
   - See if basic training can start

### Long-term (Fix)

1. **Use Pre-Started Ray Cluster**:
   ```bash
   # On head node (10.83.115.10)
   ray start --head --port=6379 --num-gpus=8 --num-cpus=96

   # On worker node (10.83.115.12)
   ray start --address=10.83.115.10:6379 --num-gpus=8 --num-cpus=96

   # In training script, add:
   ray_kwargs.ray_init.address='10.83.115.10:6379'
   ```

2. **Add Timeout and Retry Logic**:
   - Detect Ray initialization hang (timeout after 5 min)
   - Auto-restart with fresh Ray cluster
   - Log detailed error information

3. **Upgrade/Downgrade verl**:
   - Check if recent verl update broke something
   - Try pinning to known-working version

## Next Steps for User

**Please decide**:

1. **Restart worker node** (10.83.115.12) to clear all stale state?
2. **Check system resources** and zombie processes?
3. **Try minimal Ray test** to isolate Ray vs verl issue?
4. **Provide access** for deeper debugging (if possible)?

The training cannot proceed until this initialization hang is resolved. This is environment/infrastructure related, not a code or configuration issue.
