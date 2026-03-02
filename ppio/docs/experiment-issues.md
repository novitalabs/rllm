# DeepSWE 32B RL Training - Experiment Issues Index

Issues encountered during DeepSWE training experiments, organized by deployment mode.

## Bare-Metal Training (8 nodes × 8 H200)

[bare-metal-issues.md](bare-metal-issues.md) — 23 issues

Software stack compatibility (torch/vLLM/flash-attn ABI), verl patches, CUDA OOM, vLLM deadlocks, trajectory timeout design flaws, retokenization mismatch, etc.

## K8s Training (4 nodes × 8 H200, StatefulSet + DinD)

[k8s-issues.md](k8s-issues.md) — 11 issues

hostNetwork pitfalls (hostname, Ray IP, port conflicts, Gloo interface), DinD proxy, rolling update vs GCS mismatch, NCCL NVLS warnings, API server unreachable, etc.

## Code Issues (rllm Framework Bugs & Fixes)

[code-issues.md](code-issues.md) — 5 issues

Retokenization mismatch (37%→0%), overlong prompt crash, checkpoint/validation ordering, validation concurrency bottleneck.

## Experiment Data

[batch32-experiment-data.md](batch32-experiment-data.md)

train_batch_size=32 (256 trajectories/step) experiment metrics: Steps 3-11, timing, training, trajectory outcomes, Docker Hub rate limiting incident, comparison with batch_size=8.

## Performance Analysis

[training-timing-analysis.md](training-timing-analysis.md)

Per-step timing breakdown (steps 4-9), trajectory-level metrics, GPU utilization, resource consumption, and optimization recommendations.

[long-tail-trajectory-analysis.md](long-tail-trajectory-analysis.md)

Deep analysis of why the slowest trajectory is 3.85× the mean, per-step LLM/env breakdown, completion order, GPU waste quantification, and mitigation strategies.

## Working Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| torch | 2.8.0+cu128 | NOT 2.10.0 (has "Cannot access data pointer" bug) |
| vLLM | 0.10.2 | NOT 0.11.2 (ABI incompatible with torch 2.8.0) |
| flash-attn | 2.8.3 | Rebuilt from source against torch 2.8.0 |
| verl | 0.6.1 | With init_app_state, enable_sleep_mode, _loop_forever patches |
| NVIDIA driver | 580.126.09 | |

## Key Patches

| Patch | File | Issue |
|-------|------|-------|
| `init_app_state` 4-arg call | `verl/.../vllm_async_server.py` | bare-metal #18 |
| `enable_sleep_mode` = `free_cache_engine` | `verl/.../vllm_async_server.py` | bare-metal #13 |
| `_loop_forever` no-break | `verl/.../vllm_rollout_spmd.py` | bare-metal #12 |
| `extra_info` JSON deserialization | `verl/.../rl_dataset.py` | k8s #1 |
