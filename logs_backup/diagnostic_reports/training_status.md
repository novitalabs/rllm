# DeepSWE Training Status - CRITICAL ISSUE (2026-02-22 12:07)

## ⚠️ RECURRING HANG PROBLEM ⚠️

### Symptom Pattern (Observed 4+ Times)
Training **consistently hangs** after checkpoint loading completes:
1. ✓ Ray cluster starts successfully
2. ✓ Training process connects to Ray
3. ✓ vLLM workers load checkpoint shards (17/17 complete)
4. ✓ FSDP actors load model/optimizer/rng/lr_scheduler
5. ✓ GPU memory fills to ~96GB
6. ❌ **HANGS HERE - no progress for 5+ minutes**
7. Never reaches: "step 161 started" or rollout phase

### Timeline Today (2026-02-22)

| Time | Event | Outcome |
|------|-------|---------|
| 10:24-10:41 | First restart (after placement group deadlock) | Hung at checkpoint loading |
| 11:38-11:40 | Second restart (force Ray cleanup) | Connection error (old placement groups) |
| 11:44-11:59 | Third restart (complete /tmp/ray/ cleanup) | Hung at checkpoint loading |
| 11:46-12:07 | **Current state** | **HUNG at 12:02:17 for 5+ minutes** |

### Current State (12:07)

**System:**
- Ray cluster: ALIVE (2 nodes, 16 GPUs)
- Training PID: 224261
- GPU memory: 96087MB (all ranks loaded)
- Log stopped: 12:02:17 (5 minutes ago)
- No Docker containers running

**Last Log Messages (12:02):**
```
TaskRunner initialized: Using QwenChatTemplateParser
vLLM engine initialized: V1 LLM engine with TP=16
[Rank 8] Loaded model from .../global_step_160/actor/model_world_size_16_rank_8.pt
FSDP warnings about deprecated state_dict_type()
```

**What's Missing:**
- "Found checkpoint: global_step_160"
- "Resuming from ..."
- "step 161 started"
- "AgentLoopManager: ..."
- No rollout activity

---

## Root Cause Hypothesis

Based on repeated failures at the same point, possible causes:

### 1. **TaskRunner Initialization Deadlock** (Most Likely)
- TaskRunner actor created but waiting for something
- Possible waiting on:
  - vLLM HTTP server readiness check (never completes)
  - Distributed coordinator initialization
  - Ray object store ready signal

### 2. **FSDP Checkpoint Loading Issue**
- All ranks load individual checkpoint files
- Possible barrier/sync after loading never completes
- One rank stuck → all ranks wait

### 3. **Ray Communication Problem**
- Placement groups exist but actors can't communicate
- GCS (Global Control Service) latency
- TCP store issues between nodes

---

## Failed Recovery Attempts

1. ❌ Simple Ray restart → Placement group conflict
2. ❌ Force kill + Ray restart → Old session symlink issue
3. ❌ Complete /tmp/ray/ cleanup → Still hangs at same point

---

## Diagnostic Checklist

### Pre-Checkpoint Loading (All Pass ✓)
- ✓ Ray cluster healthy (0 pending demands)
- ✓ Training connects to Ray successfully
- ✓ vLLM workers spawn and load shards
- ✓ FSDP actors create placement groups

### Checkpoint Loading (All Pass ✓)
- ✓ All 16 ranks load checkpoint shards (17/17)
- ✓ Model weights loaded (GPU memory rises to ~22GB → 96GB)
- ✓ Optimizer/RNG/LR scheduler loaded per rank
- ✓ No CUDA OOM errors

### Post-Checkpoint Initialization (FAILS ❌)
- ❌ Never logs "Found checkpoint" or "Resuming from"
- ❌ Never starts "step 161"
- ❌ TaskRunner/AgentLoopManager silent
- ❌ No Docker containers spawn (rollout never starts)

---

## Required Investigation

1. **Check Ray actor logs:**
   ```bash
   ray logs actor_<taskrunner_id>
   ray logs actor_<vllm_id>
   ```

2. **Check GCS logs:**
   ```bash
   tail -100 /tmp/ray/session_*/logs/gcs_server.out
   ```

3. **Check if actors are stuck:**
   ```bash
   ray list actors --detail
   ray timeline  # if available
   ```

4. **Potential code issues to examine:**
   - `rllm/trainer/verl/train_agent_ppo.py:37` - ray.init() and training loop
   - `verl/trainer/ppo/ray_trainer.py` - coordinator initialization
   - `verl/workers/rollout/vllm_rollout_spmd.py` - vLLM worker startup
   - TaskRunner actor creation and initialization

---

## What We Know From Previous Context

- **Steps 161-170 had zero-learning** due to vLLM throughput degradation
- **Step 161 previously completed** (2/32 reward=1) before ZMQ crash
- **ZMQ fix was applied** (retry on EAGAIN) in vllm_rollout_spmd.py
- **ray_trainer.py patched** to skip _check_resource_available()
- **Checkpoints exist:** step 160, 162 (step 164 was incomplete/deleted)

---

## Immediate Action Needed

This is a **blocking system-level issue**, not a transient hang. Options:

### Option A: Deep Debug (Time-consuming)
1. Attach to hanging process with py-spy/gdb
2. Examine Ray actor states and logs
3. Identify exact blocking call
4. Patch or work around

### Option B: Alternative Approach (Faster)
1. Try loading from step 162 instead of 160 (might skip problem state)
2. Try single-node training first (eliminate cross-node sync issues)
3. Try synchronous rollout mode instead of async
4. Check if issue exists with smaller batch size

### Option C: Rollback Patches
1. Revert ray_trainer.py patch (restore _check_resource_available)
2. Test if that was causing initialization to skip critical steps

---

## Next Steps (Awaiting User Decision)

Training is currently **STUCK** at 12:02 for 5+ minutes. Recommend:
1. Kill current training (PID 224261)
2. Decide on investigation approach (A/B/C above)
3. Implement chosen approach
4. Restart training

**Docker cleanup status:** 3.2TB free, no immediate disk issues.
