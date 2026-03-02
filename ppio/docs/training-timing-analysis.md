# DeepSWE 32B RL Training - Timing & Resource Analysis

## Overview

This document analyzes the per-module timing breakdown and resource consumption of the DeepSWE 32B RL training pipeline, based on **6 completed training steps (steps 4-9)** from the K8s deployment on 4×H200 nodes (32 GPUs).

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-32B (32B params, BF16) |
| Algorithm | RLOO (no critic) |
| Batch size | 8 prompts × 8 rollouts = 64 trajectories/step |
| Max prompt length | 4096 tokens |
| Max response length | 32768 tokens |
| Tensor parallel | 8 (within single node) |
| Ulysses SP | 8 |
| FSDP | param_offload + optimizer_offload |
| vLLM | 0.10.2, TP=8, gpu_mem_util=0.7, CUDA graph |
| Environment | SWE Docker (R2E-Gym), trajectory_timeout=3600s, max_steps=30 |
| Infrastructure | 4 nodes × 8 H200 (140GB), K8s StatefulSet, DinD |

---

## Training Step Pipeline

Each training step follows this sequential pipeline, timed by `marked_timer()`:

```
step (total)
├── init_envs_and_agents     (not individually timed)
├── collect_trajectory        (SWE agent rollout via vLLM + Docker environments)
│   ├── [per-trajectory] llm_time    (model inference via vLLM API)
│   ├── [per-trajectory] env_time    (Docker container execution)
│   └── [per-trajectory] reward_time (final reward computation)
├── transform_trajectory      (tokenize, pad, assemble DataProto)
├── adv                       (advantage computation - RLOO baseline)
│   ├── rejection_sampling    (filter all-pass/all-fail UIDs)
│   └── compute_advantage     (RLOO: leave-one-out baseline)
├── old_log_prob              (recompute log-probs with current actor)
├── ref                       (reference policy log-probs, if use_reference_policy)
├── update_actor              (PPO policy gradient update via FSDP)
├── testing                   (validation, every test_freq steps)
└── save_checkpoint           (FSDP checkpoint save, every save_freq steps)
```

### Phases Not Used in Current Config
- **values** / **update_critic**: Disabled (RLOO algorithm, `use_critic=False`)
- **gen** / **gen_max**: Used in standard PPO rollout, replaced by `collect_trajectory` in agent mode
- **reward**: Rewards come from SWE environment, not a separate reward model

---

## Timing Instrumentation Details

### Framework-Level Timing (`marked_timer`)

Source: `verl/utils/profiler/performance.py`

```python
@contextmanager
def marked_timer(name, timing_raw, color=None, domain=None, category=None):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last  # accumulates if called multiple times
```

Each timer records wall-clock seconds into `timing_raw[name]`. After the step, metrics are computed:

| Metric Key | Unit | Formula |
|------------|------|---------|
| `timing_s/{name}` | seconds | Raw elapsed time |
| `timing_per_token_ms/{name}` | ms/token | `timing_s * 1000 / num_tokens` |
| `perf/total_num_tokens` | count | Sum of all token lengths in batch |
| `perf/time_per_step` | seconds | `timing_raw["step"]` |
| `perf/throughput` | tokens/s/GPU | `total_tokens / (step_time * n_gpus)` |

**Token normalization rules** (from `compute_timing_metrics`):
- `gen`: Uses response tokens only (`num_response_tokens`)
- `ref`, `values`, `adv`, `update_critic`, `update_actor`: Uses all tokens (`num_prompt_tokens + num_response_tokens`)

### Trajectory-Level Timing (Per-Agent)

Source: `rllm/engine/agent_execution_engine.py`, method `run_agent_trajectory_async()`

Each of the 64 parallel agent trajectories records:

| Metric | Description |
|--------|-------------|
| `llm_time` | Total time waiting for vLLM model responses (all turns) |
| `env_time` | Total time executing Docker environment steps (all turns) |
| `reward_time` | Time for `compute_final_reward()` (patch evaluation) |
| `total_time` | Wall-clock time from start to end of trajectory |
| `steps` | Number of agent interaction steps |
| `token_mismatch` | 1.0 if retokenization failed, else 0.0 |

These are aggregated into `traj/{metric}_mean`, `traj/{metric}_min`, `traj/{metric}_max` in the training metrics.

---

## Expected Timing Breakdown (SWE Agent RL)

### Phase 1: `collect_trajectory` (DOMINANT)

This is the **bottleneck** phase for SWE agent RL training. Unlike standard LLM RL (where generation takes ~30-60% of step time), SWE agent rollout involves:

1. **Multi-turn LLM inference**: Each trajectory runs up to 50 agent steps, each requiring a full vLLM chat_completion call with growing context (up to 32K tokens)
2. **Docker environment execution**: Each agent step triggers code execution in an isolated Docker container (running tests, applying patches, etc.)
3. **64 parallel trajectories**: All 64 trajectories run concurrently with 64 Docker containers
4. **Trajectory timeout**: 5400 seconds (90 minutes) maximum per trajectory

**Key factors affecting this phase:**
- SWE problem difficulty determines number of agent steps (1-50)
- Container startup and test execution time varies by repository
- vLLM serving throughput is shared across 64 concurrent requests
- Early completion (ENV_DONE) vs timeout/truncation dramatically affects wall-clock time

### Phase 2: `transform_trajectory`

Converts raw trajectories into padded tensor batches:
- Left-pad prompts to `max_prompt_length` (4096)
- Right-pad responses to `max_response_length` (32768)
- Build attention masks, position IDs, response masks
- Apply overlong filter (mask out TRUNCATION/TIMEOUT trajectories)

Expected time: seconds to low tens of seconds (CPU-bound, tokenization + padding).

### Phase 3: `adv` (Advantage Computation)

For RLOO algorithm:
- Computes leave-one-out baseline per UID group (8 rollouts per prompt)
- Rejection sampling: filters groups where all 8 pass or all 8 fail
- CPU computation on driver process

Expected time: sub-second (pure CPU, small computation).

### Phase 4: `old_log_prob` (Log Probability Recomputation)

- Runs current actor model forward pass on all batch sequences
- Uses FSDP distributed computation across 64 GPUs
- Processes sequences of length up to 4096 + 32768 = 36864 tokens
- `log_prob_use_dynamic_bsz=True`, `log_prob_micro_batch_size_per_gpu=1`

Expected time: tens of seconds to minutes (GPU compute, proportional to total tokens).

### Phase 5: `ref` (Reference Policy Log-Probs)

- Same computation as `old_log_prob` but with frozen reference policy
- FSDP with `param_offload=True`
- Requires loading reference weights from CPU to GPU

Expected time: similar to `old_log_prob`, possibly slower due to param offload I/O.

### Phase 6: `update_actor` (Policy Gradient Update)

- PPO clipped policy gradient loss computation
- FSDP backward pass across 64 GPUs
- `ppo_mini_batch_size=8`, `ppo_micro_batch_size_per_gpu=1`
- `gradient_checkpointing=True` to reduce memory
- `param_offload=True`, `optimizer_offload=True`
- Ulysses sequence parallel size = 8

Expected time: minutes (GPU compute + CPU<->GPU data movement for FSDP offload).

### Phases 7-8: `testing` and `save_checkpoint`

- Testing: Runs full validation rollout (500 SWE-bench Verified problems, n=1), only every `test_freq=10` steps
- Save checkpoint: FSDP distributed checkpoint, only every `save_freq=10` steps

Expected time: Testing dominates when triggered (~hours for full validation). Checkpoint save is I/O-bound.

---

## Timing Data Collection

### Where Timing Metrics Are Logged

The `AgentPPOTrainer.fit_agent()` method logs all metrics via:
```python
logger.log(data=metrics, step=self.global_steps)
```

With `trainer.logger=[console]`, metrics are printed to stdout. Look for lines containing:
- `timing_s/step`, `timing_s/collect_trajectory`, `timing_s/transform_trajectory`
- `timing_s/adv`, `timing_s/old_log_prob`, `timing_s/ref`, `timing_s/update_actor`
- `timing_per_token_ms/...` (per-token normalized)
- `perf/throughput`, `perf/time_per_step`, `perf/total_num_tokens`
- `traj/llm_time_mean`, `traj/env_time_mean`, `traj/total_time_mean`

### Log File Locations

| Log | Path |
|-----|------|
| Main training log | `/tmp/train_run34.log` |
| TaskRunner stdout | `/tmp/ray/session_latest/logs/worker-...-3049795.out` |
| TaskRunner stderr | `/tmp/ray/session_latest/logs/worker-...-3049795.err` |
| vLLM server | `/tmp/ray/session_latest/logs/worker-...-3066090.err` |
| FSDP workers | `/tmp/ray/session_latest/logs/worker-...-305177*.err` |

---

## Actual Timing Data (K8s Run, Steps 1-9, Post-Fix)

**Configuration**: 4 nodes × 8 H200, K8s StatefulSet, DinD, torch 2.8.0 + vLLM 0.10.2
**Fixes applied**: token accumulation (0% mismatch), max_steps=30, trajectory_timeout=3600s, gpu_mem_util=0.7

**Metrics collection**: `./ppio/scripts/fetch_k8s_metrics.sh | python3 ppio/scripts/parse_training_metrics.py -`

### Per-Step Timing Breakdown

| Step | Total (min) | collect_traj | % | update_actor | % | old_log_prob | MFU |
|------|-------------|-------------|-------|--------------|-------|-------------|-----|
| 1 | 60.3 | 3412s | 94.4% | 175s | 4.9% | 27.2s | 9.3% |
| 2 | 74.0 | 4278s | 96.3% | 146s | 3.3% | 17.7s | 10.2% |
| 3 | 65.3 | 3744s | 95.6% | 154s | 3.9% | 17.0s | 8.6% |
| 4 | 58.9 | 3357s | 95.1% | 153s | 4.3% | 20.9s | 9.5% |
| 5 | 65.1 | 3738s | 95.8% | 148s | 3.8% | 16.7s | 8.7% |
| 6 | 46.7 | 2622s | 93.7% | 153s | 5.5% | 23.8s | 9.2% |
| 7 | 65.5 | 3762s | 95.7% | 150s | 3.8% | 18.1s | 9.1% |
| 8 | 60.1 | 3433s | 95.2% | 153s | 4.2% | 20.0s | 9.1% |
| 9 | 67.1 | 3859s | 95.9% | 148s | 3.7% | 17.6s | 8.6% |
| **AVG** | **62.5** | **3578s** | **95.4%** | **153s** | **4.1%** | **19.9s** | **9.1%** |

### Per-Step Trajectory Metrics

| Step | LLM mean | LLM max | Env mean | Env max | Total max | Tail ratio | Agent Steps | Mismatch |
|------|----------|---------|----------|---------|-----------|------------|-------------|----------|
| 1 | 937s | 3290s | 32s | 363s | 3292s | 3.4x | 22.5 | **0.0%** |
| 2 | 925s | 4237s | 53s | 513s | 4238s | 4.3x | 21.7 | **0.0%** |
| 3 | 864s | 3616s | 43s | 386s | 3618s | 4.0x | 21.5 | **0.0%** |
| 4 | 1046s | 3234s | 41s | 563s | 3236s | 3.0x | 20.5 | **0.0%** |
| 5 | 1091s | 3608s | 62s | 477s | 3614s | 3.1x | 19.0 | **0.0%** |
| 6 | 949s | 2506s | 47s | 455s | 2512s | 2.5x | 19.9 | **0.0%** |
| 7 | 1063s | 3635s | 67s | 406s | 3639s | 3.2x | 21.8 | **0.0%** |
| 8 | 1105s | 3303s | 49s | 632s | 3395s | 2.9x | 21.7 | **0.0%** |
| 9 | 1127s | 3665s | 58s | 458s | 3670s | 3.1x | 21.8 | **0.0%** |
| **AVG** | **1012s** | **3455s** | **50s** | **473s** | **3469s** | **3.3x** | **21.2** | **0.0%** |

### Per-Step Training Metrics

| Step | Score | solve_none | solve_partial | PG loss | Grad Norm | Clipfrac | Entropy | KL | Resp Len | Advantages |
|------|-------|------------|---------------|---------|-----------|----------|---------|-----|----------|------------|
| 1 | 0.031 | 6 | 2 | -5.63 | 489 | 0.0% | 7829 | 0.0 | 18,354 | 0.006 |
| 2 | 0.172 | 5 | 3 | 33.54 | 848 | 0.0% | 6839 | 0.0 | 19,140 | -0.040 |
| 3 | 0.031 | 6 | 2 | 1.74 | 505 | 0.0% | 6672 | 0.0 | 17,387 | -0.002 |
| 4 | 0.016 | 7 | 1 | -7.60 | 366 | 0.0% | 6945 | 0.0 | 18,649 | 0.010 |
| 5 | 0.188 | 5 | 3 | 21.05 | 730 | 0.0% | 6409 | 0.0 | 17,131 | -0.027 |
| 6 | 0.094 | 6 | 2 | 18.02 | 605 | 0.0% | 5926 | 0.0 | 17,962 | -0.026 |
| 7 | 0.156 | 6 | 2 | 18.00 | 546 | 0.0% | 6672 | 0.0 | 17,770 | -0.022 |
| 8 | 0.203 | 4 | 4 | 33.41 | 864 | 0.0% | 7049 | 0.0 | 18,301 | -0.039 |
| 9 | 0.047 | 6 | 2 | 8.74 | 521 | 0.0% | 6798 | 0.0 | 16,882 | -0.011 |
| **AVG** | **0.104** | | | **13.47** | **608** | **0.0%** | **6793** | **0.0** | **17,953** | |

### Trajectory Outcome Distribution (576 Trajectories over 9 Steps)

| Category | Count | % |
|----------|-------|-----|
| ENV_DONE | 508 | 88.2% |
| MAX_STEPS | 53 | 9.2% |
| TRUNCATION | 10 | 1.7% |
| **Reward = 1.0** | **60** | **10.4%** |
| Reward = 0.0 | 516 | 89.6% |
| Masked (overlong) | 63 | 10.9% |
| Retokenization mismatch | **0** | **0.0%** |

### Sequence Length Stats

| Step | Prompt mean | Prompt max | Resp mean | Resp max | Resp min | Clip ratio |
|------|-------------|------------|-----------|----------|----------|------------|
| 1 | 1800 | 1876 | 18,354 | 32,768 | 1,530 | 1.6% |
| 2 | 1804 | 1899 | 19,140 | 32,768 | 5,053 | 1.6% |
| 3 | 1806 | 1863 | 17,387 | 32,294 | 5,497 | 0.0% |
| 4 | 1869 | 2157 | 18,649 | 32,750 | 4,780 | 0.0% |
| 5 | 1782 | 1876 | 17,131 | 30,768 | 6,357 | 0.0% |
| 6 | 1820 | 1976 | 17,962 | 32,768 | 5,420 | 6.2% |
| 7 | 1802 | 1950 | 17,770 | 32,768 | 5,230 | 1.6% |
| 8 | 1785 | 1903 | 18,301 | 31,808 | 4,455 | 0.0% |
| 9 | 1782 | 1860 | 16,882 | 29,517 | 4,513 | 0.0% |

### GPU Memory Usage

| Step | Max Allocated (GB) | Max Reserved (GB) | CPU Memory (GB) |
|------|-------------------|--------------------|-----------------|
| 1 | 112.9 | 127.6 | 85.6 |
| 2-9 | 120.2 | 131.6 | 87.3 → 88.5 |

---

## Resource Consumption

### GPU Utilization (Snapshot During collect_trajectory)

All 32 GPUs across 4 nodes show **0% compute utilization** during trajectory collection:

| Node | GPU Util | Memory Used (GB) | Memory Total (GB) | Mem % |
|------|----------|-------------------|-------------------|-------|
| 10.83.115.18 (node-2) | 0% | 88.4-89.1 | 143.8 | 62% |
| 10.83.115.21 (node-1) | 0% | 91.0-91.8 | 143.8 | 64% |
| 10.83.115.22 (node-3) | 0-1% | 88.2-88.9 | 143.8 | 62% |
| 10.83.115.23 (node-0) | 0% | 90.3-91.1 | 143.8 | 63% |

**GPU memory breakdown** (per GPU, estimated):
- vLLM KV cache + model weights: ~70 GB (gpu_memory_utilization=0.7)
- FSDP sharded params (offloaded to CPU): ~28 GB reserved but mostly on host
- Peak during update_actor: 120.2 GB allocated, 131.6 GB reserved

### CPU & System Memory

| Metric | Value |
|--------|-------|
| CPU memory per pod | 85-89 GB (growing slowly, ~0.4 GB/step) |
| Total system memory | ~2 TB per node |
| Memory utilization | ~5% |
| CPU utilization (during trajectory) | Low (Docker container scheduling) |

### Docker Image Cache (DinD)

| Metric | Value |
|--------|-------|
| Cached images | 150 (of 4578 total) |
| Unique repos | 10 (namanjain12/*_final) |
| Total cache size | 164.7 GB |
| Reclaimable | 47 GB (28%) |
| Docker storage disk | 7.0 TB NVMe, 6.0 TB available |

#### Full Image Set Storage Estimate

Training uses 4578 unique Docker images (one per git commit across 10 repos). Due to large per-commit unique layers, full pre-pull is **not feasible**:

| Repo | Images | Avg Unique Size | Est. Total |
|------|--------|-----------------|------------|
| pandas_final | 1444 | 2.4 GB | 3.4 TB |
| numpy_final | 781 | 1.3 GB | 1.0 TB |
| pillow_final | 620 | 1.2 GB | 0.7 TB |
| orange3_final | 482 | 3.1 GB | 1.5 TB |
| aiohttp_final | 299 | 1.2 GB | 0.4 TB |
| tornado_final | 261 | 0.9 GB | 0.2 TB |
| scrapy_final | 215 | 0.9 GB | 0.2 TB |
| pyramid_final | 189 | 0.9 GB | 0.2 TB |
| datalad_final | 179 | 1.7 GB | 0.3 TB |
| coveragepy_final | 108 | 0.8 GB | 0.1 TB |
| **Total** | **4578** | | **~8.0 TB** |

**Conclusion**: All 4578 images require ~8TB, exceeding 6TB available disk. On-demand pull is the only viable strategy. Impact is minimal: first-encounter pull adds ~30-60s per image, but env_time is only 3.3% of trajectory time and amortized across training as cache grows.

---

## Key Findings & Time Budget Analysis

### Average Step Time Budget (Post-Fix, 9 Steps)

```
Total step time:                3752s (62.5 min)
├── collect_trajectory:         3578s (95.4%)  ← BOTTLENECK
│   ├── LLM inference (mean):  1012s (95.3% of trajectory time)
│   ├── Env execution (mean):    50s ( 4.7% of trajectory time)
│   └── Wall-clock = max traj: 3469s (blocked by slowest of 64)
├── update_actor (PPO):          153s ( 4.1%)
├── old_log_prob + adv:           20s ( 0.5%)
├── transform_trajectory:        0.6s ( 0.0%)
└── overhead:                    0.4s ( 0.0%)
```

### Critical Bottleneck: collect_trajectory (95.4% of time)

1. **Long-tail problem**: Max trajectory time is **3.3× the mean**
   - Mean trajectory: 1063s (~18 min)
   - Max trajectory: 3469s (~58 min)
   - The batch is blocked by the single slowest trajectory

2. **LLM inference dominates**: 95.3% of trajectory time is waiting for vLLM
   - Average ~21 agent steps per trajectory
   - Each step requires a full chat_completion with growing context (up to 32K tokens)
   - 64 concurrent trajectories share vLLM TP=8 serving across 4 nodes

3. **GPU utilization = 0%** during trajectory collection
   - vLLM runs on the same GPUs as FSDP training
   - During collect_trajectory, vLLM serves inference (but GPU compute shows 0% in `nvidia-smi` because inference is memory-bound, not compute-bound)
   - During update_actor, MFU = 8.6-10.2%

4. **Token mismatch: FIXED (0%)** — all 576 trajectories over 9 steps contribute gradient
   - Previously 37% mismatch rate, now 0% via token accumulation fix
   - Effective batch size: 64/64 (was ~40/64)

5. **Clipfrac = 0%**: PPO clip ratio never triggers
   - Indicates policy updates are very small (lr=1e-6)
   - KL divergence also 0.0 — policy has barely moved from initial weights after 9 steps

6. **Grad norm highly variable**: 366 to 864 across steps (mean 608)
   - Correlated with score — high-reward steps have larger gradients

### GPU Waste Quantification

```
Per step:   32 GPUs × 3578s idle = 31.8 GPU-hours wasted
Per epoch:  572 steps × 31.8 = 18,190 GPU-hours wasted
Daily:      ~23 steps × 31.8 = 731 GPU-hours wasted/day
```

The GPUs are allocated but effectively idle during 95.4% of training time. Only during `update_actor` (4.1% of time) do the GPUs perform compute work.

---

## Optimization Recommendations

### P0: Reduce collect_trajectory Wall-Clock Time

#### 1. ~~Tighten trajectory timeout~~ [DONE]
- ~~Current: `trajectory_timeout=5400s` (90 min)~~
- Applied: `trajectory_timeout=3600s` (60 min)
- Result: tail trajectory time reduced from 3951s → 3292s (-17%)

#### 2. ~~Reduce max_steps~~ [DONE]
- ~~Current: `max_steps=50`~~
- Applied: `max_steps=30`
- Result: combined with timeout reduction, step time 4366s → 3615s (-17%)

#### 3. ~~Fix retokenization mismatch~~ [DONE]
- ~~37% of trajectories contribute zero gradient due to token mismatch~~
- Fix: accumulate token IDs across agent steps, send directly to vLLM (bypass re-tokenization)
- Files: `rllm/engine/rollout/verl_engine.py` (accept `prompt_token_ids`), `rllm/engine/agent_execution_engine.py` (accumulate & forward)
- Result: **token_mismatch_mean: 0.37 → 0.0** — all 64 trajectories now contribute gradient

#### 4. ~~Increase vLLM throughput~~ [DONE]
- ~~`gpu_memory_utilization=0.6` leaves 40% of GPU memory unused~~
- Applied: `gpu_memory_utilization=0.7`
- Result: larger KV cache, supports more concurrent decoding

#### Summary: P0 Fixes Combined Impact (K8s Run, Step 1 Post-Fix)

| Metric | Before (steps 4-9 avg) | After (step 1) | Improvement |
|--------|----------------------|----------------|-------------|
| token_mismatch_mean | 37% | **0%** | Eliminated |
| step time | 4366s (73 min) | 3615s (60 min) | -17% |
| collect_trajectory | 4204s | 3412s | -19% |
| tail trajectory | 3951s | 3292s | -17% |
| effective batch (contributing gradient) | ~40/64 | **64/64** | +60% |

### P0.5: Docker Image Pre-Pull [WON'T FIX]

#### 5. Pre-pull Docker images for training tasks
- **Status**: Evaluated, not needed
- Training dataset uses 4578 unique Docker images across 10 repos (one image per git commit)
- Full pre-pull requires ~8TB storage, exceeds available 6TB NVMe disk — **not feasible**
- Current strategy: on-demand pull during `env.reset()`, ~30-60s per uncached image
- Impact is minimal: `env_time_mean=32s` (3.3% of trajectory time), dominated by LLM inference (96.7%)
- Cache grows naturally over training: 150/4578 (3.3%) cached after initial steps, hit rate improves progressively
- Existing `prewarm-images.py` (disabled by default) could pre-pull a subset if needed
- **Per-batch pre-pull** (pre-pulling 8 images per step before trajectory collection) is possible but low ROI: 8 parallel pulls ≈ 30-60s vs 3400s trajectory time (<2% overhead)
- **Recommendation**: Keep on-demand pull. Monitor `env_time_max` for outlier pulls; if any single pull >300s, investigate Docker registry performance

### P1: Improve GPU Utilization During Trajectory Collection

#### 5. Async training pipeline (Hard, Very High Impact)
- Overlap `update_actor` from step N with `collect_trajectory` from step N+1
- verl's async rollout mode partially supports this, but the FSDP ↔ vLLM weight sync complicates true overlap
- Potential: reduce effective step time from 4366s to ~4204s (save the 145s update time, which would overlap)

#### 6. Separate inference and training GPUs (Architecture change, Very High Impact)
- Current: same 32 GPUs do both vLLM inference and FSDP training
- Alternative: dedicate e.g. 8 GPUs (1 node) for persistent vLLM serving, 24 GPUs (3 nodes) for FSDP training
- Pro: vLLM can serve continuously without weight load/unload overhead
- Con: requires verl architecture changes, 25% fewer training GPUs

### P2: Improve Training Phase Efficiency

#### 7. Reduce FSDP offload overhead (Medium, Low Impact)
- `param_offload=True` + `optimizer_offload=True` adds CPU↔GPU transfer time
- Currently update_actor takes 145s (3.3%) — already fast relative to total
- Consider: disable `param_offload` if GPU memory permits (keep optimizer offload)
- Memory check: 106 GB used, 143.8 GB total → ~38 GB headroom, may suffice

#### 8. Increase effective batch size (Easy, Medium Impact)
- With 37% mismatch rate, effective training batch size is ~40/64 = 62.5%
- Fix mismatch (P0 #3) to get full 64-trajectory training signal
- Alternatively: increase `data.train_batch_size` to compensate, but this increases trajectory collection time proportionally

---

## LLM Inference Metrics (K8s Run, Steps 4-9)

Derived from per-trajectory `llm_time` and `response_length` metrics. All values are end-to-end including vLLM scheduling, prefill, and decode.

### Average Trajectory

| Metric | Value |
|--------|-------|
| Model | Qwen3-32B (BF16) |
| vLLM config | TP=8, gpu_mem_util=0.6, CUDA graph, 2 replicas |
| Concurrent trajectories | 64 |
| Agent steps/trajectory | 25.1 |
| Prompt tokens | 1,807 |
| Response tokens | 15,529 |
| LLM time | 983s (16.4 min) |
| LLM time/agent step | 39.2s |
| Effective tok/s (per-traj) | 15.8 |
| End-to-end ms/tok | 63.3 |

### Per-Step Inference Breakdown

| Step | LLM/step | Tok/traj | Tok/s/traj | Batch tok/s | ms/tok | Agent Steps |
|------|----------|----------|-----------|-------------|--------|-------------|
| 4 | 39.3s | 17,837 | 17.5 | 270 | 57.2 | 26.0 |
| 5 | 41.3s | 14,927 | 16.1 | 233 | 62.0 | 22.4 |
| 6 | 37.6s | 16,459 | 16.3 | 281 | 61.4 | 26.9 |
| 7 | 39.4s | 14,260 | 15.7 | 196 | 63.7 | 23.1 |
| 8 | 40.4s | 14,945 | 14.6 | 235 | 68.5 | 25.3 |
| 9 | 37.7s | 14,747 | 14.6 | 213 | 68.5 | 26.8 |
| **AVG** | **39.2s** | **15,529** | **15.8** | **238** | **63.6** | **25.1** |

### Tail Trajectory Inference (Slowest Per Batch)

| Step | LLM time | Steps | LLM/step | Resp tokens | Tok/s | ms/tok |
|------|----------|-------|----------|-------------|-------|--------|
| 4 | 3,934s | 50 | 78.7s | 32,593 | 8.3 | 120.7 |
| 5 | 3,828s | 50 | 76.6s | 32,572 | 8.5 | 117.5 |
| 6 | 3,494s | 50 | 69.9s | 32,768 | 9.4 | 106.6 |
| 7 | 4,171s | 50 | 83.4s | 32,768 | 7.9 | 127.3 |
| 8 | 3,725s | 50 | 74.5s | 32,447 | 8.7 | 114.8 |
| 9 | 3,531s | 50 | 70.6s | 32,768 | 9.3 | 107.8 |
| **AVG** | **3,781s** | **50** | **75.6s** | **32,653** | **8.6** | **115.8** |

### Key Inference Observations

1. **Per-trajectory decode throughput**: 15.8 tok/s average, dropping to 8.6 tok/s for the tail (1.84× slowdown)
2. **End-to-end latency**: 63 ms/tok average, 116 ms/tok for tail — dominated by multi-turn round-trip overhead, not raw decode speed
3. **Batch throughput**: ~238 tok/s effective across all 64 trajectories. This is the aggregate rate at which training tokens are generated.
4. **Per-agent-step LLM time**: 39s average → includes prefill (growing context), decode, and vLLM scheduling. For the tail, 76s/step due to ~37K token context.

### Training Phase Per-Token Metrics

| Step | update_actor (ms/tok) | adv (ms/tok) |
|------|-----------------------|--------------|
| 4 | 0.1194 | 0.0163 |
| 5 | 0.1350 | 0.0141 |
| 6 | 0.1296 | 0.0179 |
| 7 | 0.1382 | 0.0146 |
| 8 | 0.1297 | 0.0136 |
| 9 | 0.1339 | 0.0159 |
| **AVG** | **0.131** | **0.015** |

---

## Pure Inference Benchmark (Qwen3-32B Standalone vLLM)

Benchmark date: 2026-02-28. Single node (8× H200), no training workload, matching rollout config.

### Server Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-32B (BF16) |
| vLLM | 0.10.2 V1, TP=8, CUDA graph |
| gpu_memory_utilization | 0.7 |
| max_model_len | 36,864 |
| Prefix caching | Enabled (98% hit rate with repeated prompts) |
| Chunked prefill | Enabled |

### Results: Concurrency Scaling (4K prompt, 512 max output)

| Concurrency | Batch tok/s | Per-req tok/s | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) | Req time (s) |
|-------------|------------|---------------|---------------|---------------|--------------|-------------|
| 1 | 10.6 | 10.6 | 192.5 | 614.9 | 94.05 | 43.7 |
| 8 | 72.9 | 10.1 | 465.5 | 609.3 | 98.19 | 45.3 |
| 16 | 157.0 | 10.0 | 461.1 | 798.3 | 99.71 | 50.6 |
| 32 | 300.7 | 9.5 | 660.5 | 887.6 | 103.75 | 52.9 |
| 64 | 535.2 | 8.7 | 1095.6 | 1856.8 | 112.36 | 56.0 |

### Results: Context Length Scaling (concurrency=8, 512 max output)

| Prompt tokens | Batch tok/s | Per-req tok/s | TTFT avg (ms) | ITL avg (ms) |
|---------------|------------|---------------|---------------|--------------|
| 1K | 77.0 | 10.2 | 289.4 | 97.66 |
| 4K | 72.9 | 10.1 | 465.5 | 98.19 |
| 16K | 80.5 | 10.1 | 1030.0 | 97.44 |
| 32K | FAILED | - | - | - |

Note: 32K prompt fails with "Chunk too big" — chunked prefill limit exceeded. Needs `--max-num-batched-tokens` tuning or `--enable-chunked-prefill=false`.

### Results: Long Output (2048 max output)

| Config | Batch tok/s | Per-req tok/s | TTFT avg (ms) | ITL avg (ms) | Req time (s) |
|--------|------------|---------------|---------------|--------------|-------------|
| c=8, 4K prompt | 60.9 | 10.2 | 349.6 | 97.90 | 147.5 |
| c=16, 4K prompt | 141.0 | 10.1 | 571.1 | 99.13 | 178.8 |

### Results: Training-Matching Scenarios (concurrency=8)

| Scenario | Prompt | Max out | Batch tok/s | Per-req tok/s | TTFT avg (ms) | ITL avg (ms) |
|----------|--------|---------|------------|---------------|---------------|--------------|
| Early step | 4K | 512 | 70.5 | 10.1 | 394.5 | 98.32 |
| Mid trajectory | 16K | 512 | 77.4 | 10.2 | 506.8 | 97.05 |
| Late trajectory | 32K | 256 | FAILED | - | - | - |

### Key Findings: Pure Inference vs Training Rollout

| Metric | Pure Inference (c=8) | Training Rollout (avg) | Training Rollout (tail) | Ratio (pure/train avg) |
|--------|---------------------|----------------------|----------------------|----------------------|
| Per-req tok/s | 10.1 | 15.8 | 8.6 | 0.64× |
| ITL ms/tok | 98 | 63 | 116 | 1.55× (slower) |
| Batch tok/s | 72.9 | 238 | - | 0.31× |
| TTFT (4K ctx) | 466ms | - | - | - |

**Analysis:**

1. **Pure inference is SLOWER per-request (10.1 vs 15.8 tok/s)**. This is counterintuitive but explainable:
   - Training rollout metrics compute tok/s as `response_tokens / llm_time` which includes multi-turn round-trips but also reflects vLLM's continuous batching of 64 concurrent trajectories each issuing sequential requests
   - The benchmark uses 512-token max output vs training's ~620 tokens/step (avg 15,529 / 25.1 steps), with different generation patterns
   - Benchmark prefix caching hit rate is 98% (identical prompts), inflating effective throughput

2. **Batch throughput: 72.9 vs 238 tok/s (single replica)**. Training had 4 vLLM replicas across 4 nodes (K8s 4-node setup). Per-replica: 238/4 ≈ 60 tok/s, close to our 72.9 tok/s.

3. **ITL is remarkably stable**: 94-98ms at c=1 to c=8, only reaching 112ms at c=64. The decode phase is memory-bandwidth-bound, not compute-bound.

4. **TTFT scales linearly with context**: 192ms (4K) → 466ms (4K cached) → 1030ms (16K). Prefill is compute-bound and scales with sequence length.

5. **Concurrency scaling is near-linear**: Batch throughput scales from 10.6 (c=1) to 535.2 (c=64) = 50.5× at 64× concurrency, only 21% overhead.

6. **32K context limit**: Chunked prefill rejects prompts >32K as a single chunk. Training works around this because vLLM processes growing contexts incrementally (prefix caching + append). A standalone benchmark with 32K prompt hits the single-request chunk limit.

7. **Theoretical max throughput**: At c=64 (matching training's 64 trajectories but on 1 replica), batch throughput is 535 tok/s — 2.25× the training's 238 tok/s. This suggests room for improvement in how training dispatches inference requests.

---

## Appendix A: Training Step Pipeline

Each step follows this sequential pipeline (timed by `marked_timer()`):

```
step (total)
├── init_envs_and_agents     (not individually timed)
├── collect_trajectory        (SWE agent rollout via vLLM + Docker environments)
│   ├── [per-trajectory] llm_time    (model inference via vLLM API)
│   ├── [per-trajectory] env_time    (Docker container execution)
│   └── [per-trajectory] reward_time (final reward computation)
├── transform_trajectory      (tokenize, pad, assemble DataProto)
├── adv                       (advantage computation - RLOO baseline)
│   ├── rejection_sampling    (filter all-pass/all-fail UIDs)
│   └── compute_advantage     (RLOO: leave-one-out baseline)
├── old_log_prob              (recompute log-probs with current actor)
├── update_actor              (PPO policy gradient update via FSDP)
├── testing                   (validation, every test_freq=10 steps)
└── save_checkpoint           (FSDP checkpoint save, every save_freq=10 steps)
```

Not used in current config: `values`, `update_critic` (RLOO), `ref` (use_kl_loss=False), `gen`/`gen_max` (replaced by collect_trajectory in agent mode).

## Appendix B: Source Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| Training loop | `rllm/trainer/verl/agent_ppo_trainer.py` | `fit_agent()` L126-454 |
| Agent trajectory | `rllm/engine/agent_execution_engine.py` | `run_agent_trajectory_async()` L180-430 |
| Timer impl | `verl/utils/profiler/performance.py` | `marked_timer()` L171-195 |
| Timing metrics | `verl/trainer/ppo/metric_utils.py` | `compute_timing_metrics()` L227-266 |
| Throughput metrics | `verl/trainer/ppo/metric_utils.py` | `compute_throughout_metrics()` L269-302 |
| Parent trainer | `verl/trainer/ppo/ray_trainer.py` | `fit()` L1009-1285 |

## Appendix C: Metrics Collection Scripts

### Fetch & Parse (one command)

```bash
./ppio/scripts/fetch_k8s_metrics.sh | python3 ppio/scripts/parse_training_metrics.py -
```

### Scripts

| Script | Purpose |
|--------|---------|
| `ppio/scripts/fetch_k8s_metrics.sh` | Fetch raw logs from K8s pods (incl. rotated log files) |
| `ppio/scripts/parse_training_metrics.py` | Parse step metrics, trajectory events, training stats (6 tables + summary) |
| `ppio/scripts/extract_llm_metrics.py` | Extract per-trajectory LLM inference metrics |
| `ppio/scripts/analyze_long_tail.py` | Analyze long-tail trajectory distribution |

### Data Sources

- K8s container logs: `kubectl -n deepswe logs deepswe-training-0`
- Host rotated logs: `/var/log/pods/deepswe_deepswe-training-0_<uid>/training/`
- Head node: 10.83.115.23 (node-3)
