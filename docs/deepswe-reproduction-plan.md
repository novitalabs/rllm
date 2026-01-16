# Plan: DeepSWE Reproduction on 4x H100 GPUs with PPIO Sandbox

## Repository Information

| Item | Value |
|------|-------|
| Repository | https://github.com/novitalabs/rllm.git |
| Upstream | https://github.com/rllm-org/rllm.git |
| Branch | `feature/swe-bench` |
| R2E-Gym Reference | https://github.com/agentica-project/R2E-Gym |

## Hardware Configuration

| Original (DeepSWE) | Target |
|--------------------|--------|
| 64 H100 GPUs (8 nodes × 8 GPUs) | 4 H100 GPUs (1 node × 4 GPUs) |
| 5,120 GB total VRAM | 320 GB total VRAM |
| 6 days training | ~12-24 days (scaled estimate) |

## Training Configuration

### Model Selection

**Option A: Qwen3-32B** (Original DeepSWE model)
- Requires: tensor_parallel=4, sequence_parallel=4
- Risk: Memory tight, may need aggressive offloading

**Option B: Qwen3-14B or Qwen3-8B** (Recommended for initial run)
- Lower memory, faster iteration
- Can validate pipeline before scaling up

### Key Parameter Adjustments

```bash
# Cluster configuration
trainer.n_gpus_per_node=4        # (was 8)
trainer.nnodes=1                  # (was 8)

# Parallelism (for 32B model on 4 GPUs)
actor_rollout_ref.actor.ulysses_sequence_parallel_size=4  # (was 8)
actor_rollout_ref.rollout.tensor_model_parallel_size=4    # (was 8)

# Batch sizes (scaled down 16x from 64 GPUs)
data.train_batch_size=2           # (was 8, scale: 8/4=2 min)
actor_rollout_ref.actor.ppo_mini_batch_size=2

# Rollouts per sample
actor_rollout_ref.rollout.n=4     # (was 8, reduce for memory)

# Sequence lengths (keep aggressive for learning)
data.max_prompt_length=4096
data.max_response_length=16384    # Need long context for SWE

# Memory optimization
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
```

### GRPO++ Features to Enable

| Feature | Config | Status |
|---------|--------|--------|
| RLOO advantage estimator | `algorithm.adv_estimator=rloo` | ✓ Enabled |
| Compact filtering (overlong) | `rllm.agent.overlong_filter=True` | ✓ Enabled |
| No entropy loss | `actor_rollout_ref.actor.entropy_coeff=0.0` | ✓ Enabled |
| Length normalization | `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum` | ✓ Enabled |
| High clip ratio | `actor_rollout_ref.actor.clip_ratio_high=0.28` | ✓ Enabled |

### Dataset

- **Training**: R2E-Gym Subset (4,500 problems)
  - Path: `${RLLM_DIR}/data/swe/R2E_Gym_Subset.parquet`
  - Need to prepare if not exists
- **Validation**: SWE-Bench Verified (500 problems)
  - Path: `${RLLM_DIR}/data/swe/SWE_Bench_Verified.parquet`

### Environment

- **Environment**: `swe_ppio` (PPIO Sandbox)
- **Agent**: `sweagent`
- **Max steps**: 50
- **Trajectory timeout**: 5400s (90 min)

### PPIO Sandbox Pool Configuration (Critical for avoiding 429)

The `swe_ppio` environment now uses a **sandbox pool** to avoid API rate limits:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_pool` | `True` | Enable sandbox pooling (recommended) |
| `pool_size` | `16` | Number of sandboxes to reuse |

**Key optimizations implemented:**

1. **Sandbox Reuse**: `trajectory_idx % pool_size` maps trajectories to pool slots
2. **Pause/Resume**: Sandboxes are paused (not destroyed) between uses
3. **Exponential Backoff**: Automatic retry with backoff on 429 errors
4. **Global Pool**: Singleton pool shared across all environment instances

**Configuration in training script:**
```bash
# PPIO sandbox pool settings (in environment config or code)
# Default: use_pool=True, pool_size=16
# For higher parallelism, increase pool_size (max recommended: 64)
```

**Before training, clean up any stale sandboxes:**
```bash
# If you have verl repo with cleanup script:
python /path/to/verl/ppio/scripts/cleanup_sandboxes.py --concurrency 100

# Or manually via PPIO dashboard
```

## Implementation Steps

### Step 1: Prepare R2E-Gym Dataset
```bash
cd /home/claude/work/rllm
python3 scripts/prepare_r2e_gym_data.py
```
If script doesn't exist, create data prep script following existing patterns.

### Step 2: Create 4xH100 Training Script
Create `/home/claude/work/rllm/experiments/swebench_ppio/train_qwen3_4b_swe_ppio_h100.sh`:
- Base on `train_deepswe_32b.sh`
- Adjust parallelism and batch sizes for 4 GPUs
- Use `swe_ppio` environment instead of `swe`

### Step 3: Verify PPIO Environment
```bash
# Test PPIO sandbox connection
python3 -c "from rllm.environments.swe_ppio import SWEBenchPPIOEnv; print('OK')"
```

### Step 4: Run Training
```bash
cd /home/claude/work/rllm/experiments/swebench_ppio
./train_qwen3_4b_swe_ppio_h100.sh
```

### Step 5: Monitor Progress
- Watch validation score improvement (target: 23% → 42%)
- Monitor VRAM usage with `nvidia-smi`
- Check for truncated trajectories (should decrease over time)

## Verification

1. **Environment test**: Run `swe_ppio.py` main block to verify PPIO connectivity
2. **Data validation**: Check parquet files exist and have correct schema
3. **Dry run**: Start training with `trainer.total_epochs=1` to verify pipeline
4. **Full training**: Run for 100+ epochs, monitoring validation metrics

## Expected Outcomes

- **Pass@1 target**: 15-25% (scaled from DeepSWE's 42% due to fewer GPUs, less parallelism)
- **Training time**: ~2-4 weeks for meaningful results
- **Key metric**: Validation score improvement over baseline

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OOM with 32B model | Start with Qwen3-8B or 14B |
| PPIO API rate limits (429) | ✓ **Fixed**: Sandbox pool with reuse + exponential backoff |
| Long training time | Use checkpoint saving, monitor early |
| Truncated trajectories | Increase max_response_length if memory allows |

## PPIO 429 Rate Limit Fix (2026-01-16)

### Problem

Previous implementation created a new sandbox for every `reset()` call, quickly exhausting PPIO's rate limit (max 4000 concurrent sandboxes).

### Solution

Implemented `SandboxPool` in `rllm/environments/swe_ppio/ppio_reward.py`:

```python
# Sandbox reuse via pool
sandbox_idx = trajectory_idx % pool_size  # Maps to pool slot [0, pool_size-1]
sandbox = pool.get_sandbox(sandbox_idx)   # Reuse or create

# On release, pause instead of destroy
pool.release_sandbox(trajectory_idx, pause=True)
```

### Key Features

1. **Singleton Pool**: Global pool shared across all environments
2. **Modulo Mapping**: `trajectory_idx % pool_size` ensures bounded sandbox count
3. **Pause/Resume**: `sandbox.beta_pause()` / `sandbox.connect()` for fast reuse
4. **Exponential Backoff**: Automatic retry on 429 with increasing delays
5. **Thread-Safe**: Locking for concurrent trajectory execution

### Files Modified

- `rllm/environments/swe_ppio/ppio_reward.py`: Added `SandboxPool` class
- `rllm/environments/swe_ppio/swe_ppio.py`: Use pool in `reset()` and `close()`
