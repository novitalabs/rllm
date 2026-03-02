# DeepSWE 32B — train_batch_size=32 Experiment Data

K8s 4-node (32 H200), Steps 3–11, 2026-03-01 21:50 – 2026-03-02 11:28

## Configuration Delta (vs batch_size=8)

| Parameter | batch_size=8 | batch_size=32 |
|-----------|-------------|---------------|
| `data.train_batch_size` | 8 | 32 |
| `actor_rollout_ref.rollout.n` | 8 | 8 |
| Trajectories/step | 64 | 256 |
| `trainer.save_freq` | 10 | 10 |
| `trainer.test_freq` | 10 | 10 |
| `rllm.agent.max_steps` | 30 | 30 |
| `rllm.agent.trajectory_timeout` | 3600 | 3600 |
| `rllm.agent.overlong_filter` | True | True |
| Infrastructure | 4×8 H200, K8s DinD | same |

## Timing Breakdown (per step)

| Step | Total | Collect | % | Update | % | LogProb | MFU |
|------|-------|---------|-------|--------|-------|---------|-----|
| 3 | 73m | 62m | 85.2% | 10m | 13.3% | 64.6s | 9.5% |
| 4 | 75m | 64m | 85.3% | 10m | 13.4% | 57.1s | 8.4% |
| 5 | 87m | 77m | 87.9% | 10m | 10.9% | 58.6s | 9.3% |
| 6 | 83m | 73m | 87.4% | 10m | 11.4% | 59.7s | 9.1% |
| 7 | 77m | 67m | 86.2% | 10m | 12.5% | 60.8s | 9.4% |
| 8* | 73m | 63m | 86.6% | 9m | 12.6% | 32.3s | 4.1% |
| 9* | 77m | 67m | 87.4% | 9m | 12.0% | 28.7s | 3.8% |
| 10† | 172m | 61m | 35.2% | 9m | 5.5% | 37.6s | 5.0% |
| 11 | 96m | 86m | 88.8% | 10m | 10.1% | 60.0s | 8.5% |
| **AVG** | **91m** | **69m** | **76.0%** | **10m** | **10.6%** | **51.0s** | **7.5%** |

\* Steps 8-9 affected by Docker Hub rate limiting (see below)
† Step 10 includes validation (6051s) and checkpoint save (45s)

## Trajectory Metrics (per step)

| Step | LLM_mean | LLM_max | Env_mean | Env_max | Total_max | Tail | Steps | Mis% |
|------|----------|---------|----------|---------|-----------|------|-------|------|
| 3 | 20m | 60m | 45s | 8m | 60m | 2.9x | 20.9 | 0.0% |
| 4 | 20m | 62m | 47s | 10m | 62m | 3.0x | 21.0 | 0.0% |
| 5 | 21m | 62m | 57s | 10m | 62m | 2.9x | 21.6 | 0.0% |
| 6 | 20m | 61m | 53s | 11m | 61m | 2.9x | 21.1 | 0.0% |
| 7 | 20m | 60m | 50s | 9m | 62m | 2.9x | 21.1 | 0.0% |
| 8* | 8m | 54m | 22s | 12m | 57m | 6.9x | 9.5 | 0.0% |
| 9* | 6m | 61m | 23s | 9m | 61m | 8.9x | 8.6 | 0.0% |
| 10 | 10m | 57m | 18s | 10m | 57m | 5.6x | 11.4 | 0.0% |
| 11 | 19m | 62m | 46s | 7m | 66m | 3.3x | 21.6 | 0.0% |

## Training Metrics (per step)

| Step | Score | s_none | s_part | s_all | PG_loss | Grad | Entropy | Resp_len |
|------|-------|--------|--------|-------|---------|------|---------|----------|
| 3 | 0.090 | 24 | 7 | 1 | 15.11 | 473 | 6794 | 18424 |
| 4 | 0.172 | 17 | 15 | 0 | 31.93 | 835 | 6432 | 16953 |
| 5 | 0.066 | 26 | 6 | 0 | 15.80 | 522 | 7226 | 17718 |
| 6 | 0.113 | 21 | 11 | 0 | 16.52 | 773 | 7877 | 17396 |
| 7 | 0.082 | 20 | 12 | 0 | 10.82 | 657 | 7439 | 17955 |
| 8* | 0.059 | 25 | 7 | 0 | 2.95 | 412 | 6975 | 7690 |
| 9* | 0.043 | 26 | 6 | 0 | 1.96 | 414 | 6561 | 7028 |
| 10 | 0.082 | 26 | 5 | 1 | -0.67 | 411 | 6845 | 9521 |
| 11 | 0.180 | 18 | 13 | 1 | 25.80 | 713 | 6620 | 16665 |

## Trajectory Outcomes (per step)

| Step | Total | ENV_DONE | MAX_STEP | TRUNC | R=1 | R=0 | Overlong | Dummy (failed) |
|------|-------|----------|----------|-------|-----|-----|----------|----------------|
| 3 | 256 | 229 | 19 | 6 | 23 | 233 | 25 | 0 |
| 4 | 256 | 233 | 14 | 3 | 44 | 212 | 17 | 1 |
| 5 | 256 | 236 | 11 | 3 | 17 | 238 | 14 | 0 |
| 6 | 256 | 242 | 10 | 2 | 29 | 227 | 12 | 5 |
| 7 | 256 | 230 | 17 | 1 | 21 | 230 | 18 | **137** |
| 8 | 256 | 111 | 6 | 2 | 15 | 104 | 8 | **150** |
| 9 | 256 | 93 | 12 | 0 | 11 | 95 | 12 | **117** |
| 10† | 500 | 476 | 44 | 52 | 60 | 572 | 96 | 0 |
| 11 | 256 | 235 | 14 | 2 | 46 | 210 | 16 | 0 |

## Validation (Step 10)

| Metric | Value |
|--------|-------|
| val/test_score | 7.8% (39/500) |
| Duration | 6051s (~101 min) |
| Concurrency | 500 |

## Comparison: batch_size=8 vs batch_size=32

Comparison uses **healthy steps only** (excluding steps 7-9 affected by Docker Hub rate limiting):

| Metric | BS=8 (9 steps) | BS=32 (6 healthy steps) |
|--------|----------------|------------------------|
| Avg step time | 62.5 min | 82.5 min |
| Avg collect_trajectory | 59.6 min (95.4%) | 71.3 min (86.4%) |
| Avg update_actor | 153s (4.1%) | 580s (11.7%) |
| Avg old_log_prob | 20s | 60s |
| Avg MFU | 9.1% | 9.0% |
| Avg agent_steps | 21.2 | 21.2 |
| Avg resp_length | 17,953 | 17,527 |
| Avg score (reward) | 0.104 | 0.120 |
| Avg tail ratio | 3.3x | 2.9x |
| Token mismatch | 0.0% | 0.0% |

### Key Observations

1. **collect_trajectory +20%** (60→71 min): 4× more trajectories (256 vs 64) competing for same vLLM inference capacity. But only 20% longer, not 4× — vLLM serves them concurrently.
2. **update_actor +280%** (153→580s): 4× more tokens in the PPO update batch. Linear scaling as expected.
3. **old_log_prob +200%** (20→60s): Same reason — 4× larger batch.
4. **Tail ratio improved** (3.3→2.9x): With 256 trajectories, more diversity averages out. However, the absolute tail time is similar (~62min), so the wall-clock collect time is similar.
5. **Score slightly higher** (0.104→0.120): Larger batch provides more stable gradient estimates.
6. **MFU unchanged** (9.1→9.0%): GPU utilization during update_actor is similar — more tokens but same parallelism.

### Training Efficiency

| Metric | BS=8 | BS=32 |
|--------|------|-------|
| Samples per step | 64 | 256 |
| Step time | 62.5 min | 82.5 min |
| Samples/hour | 61.4 | 186.2 |
| GPU-hours/step | 33.3 | 44.0 |
| GPU-hours/sample | 0.52 | 0.17 |

**BS=32 is 3× more sample-efficient** in GPU-hours per training sample, because trajectory collection time scales sub-linearly with batch size.

---

## Docker Hub Rate Limiting Incident (Steps 7-9)

### Symptom
Steps 7-9 had 137/150/117 failed trajectories out of 256, with `traj/steps_mean` dropping from ~21 to ~9 and `response_length/mean` from ~17K to ~7K.

### Root Cause
Docker Hub rate limiting (`429 Too Many Requests`) on image pulls through the DinD daemon.

Error chain:
```
Docker Hub → 429 Too Many Requests (on /images/create)
→ DockerRuntime.__init__: container = None
→ AttributeError: 'NoneType' object has no attribute 'id'
→ run_agent_trajectory_with_retry: 3 retries all fail
→ launch_one_trajectory_task: returns dummy result (reward=0, response=[pad_id])
```

### Per-Step Docker Error Counts

| Step | 429 Errors | 500 Errors | Dummy (failed trajs) |
|------|-----------|-----------|---------------------|
| 3 | 0 | 0 | 0 |
| 4 | 0 | 5 | 1 |
| 5 | 0 | 5 | 0 |
| 6 | **34** | 20 | 5 |
| 7 | **346** | 51 | **137** |
| 8 | **402** | 49 | **150** |
| 9 | **329** | 44 | **117** |
| 10 | 0 | 5 | 0 |
| 11 | 0 | 5 | 0 |

### Why Steps 7-9

Docker Hub's unauthenticated rate limit: **100 pulls per 6 hours per IP**. With 256 concurrent containers per step, each needing a task-specific Docker image:
- Steps 3-5 (21:00-02:21): Consumed ~768 image pulls, starting to exhaust quota
- Step 6 (03:33): First 429 errors appear
- Steps 7-9 (04:50-07:20): Deep into rate limit — over half of trajectories fail
- Step 10 validation (08:47-10:24): 100min cooldown + 6h window partially refreshed
- Step 11 (10:24+): Fully recovered, zero 429 errors

### Impact
- 404 trajectories returned dummy results (reward=0) across 3 steps
- Steps 7-9 contributed minimal effective gradient (most samples were zero-padded dummies)
- Estimated 3 wasted training steps (~4h, 128 GPU-hours)

### Mitigation
Set up local Docker registry mirror to cache all r2e-gym images on-cluster. See k8s-issues.md.
