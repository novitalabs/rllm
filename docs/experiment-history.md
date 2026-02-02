# DeepSWE Experiment History

This document tracks all rllm/DeepSWE training and evaluation experiments conducted on the novitalabs infrastructure.

## Experiment Timeline

| Date | Experiment | Model | Infrastructure | Result |
|------|------------|-------|----------------|--------|
| 2026-01-14 | PPIO Validation | - | 196.2 host | 9/9 API tests passed |
| 2026-01-15 | verl+PPIO Setup | Qwen3-4B | 8x RTX 4090 | Blocked (GLIBC) |
| 2026-01-15 | GRPO Training v1 | Qwen3-4B | 4x RTX 4090 | 0.87%-0.58% validation |
| 2026-01-16 | Training v4-v9 | Qwen3-4B/8B | 8x RTX 4090 | Memory issues |
| 2026-01-28 | Qwen3-32B Eval | Qwen3-32B | K8s 4x H100 | 30% → 66.7% |
| 2026-01-31 | Tinker+PPIO | Qwen3-32B | K8s 4x H100 | 66.7% (2/3) confirmed |
| 2026-02-02 | Test Verification Fix | - | 196.3 | REPO_TEST_CMDS added |

---

## Experiment Details

### 1. PPIO Sandbox Validation (2026-01-14)

**Objective:** Validate PPIO sandbox API for SWE-bench environment execution.

**Results:**
- 9/9 API endpoint tests passed
- Sandbox creation: ~5-10 seconds
- Command execution: stable
- File upload/download: working with timeout handling

**Configuration:**
```python
PPIO_API_KEY = "sk-***"
SANDBOX_TIMEOUT = 1200  # 20 minutes
MAX_CONCURRENT = 32
```

---

### 2. verl+PPIO GRPO Training v1 (2026-01-15)

**Objective:** Initial GRPO training with PPIO sandbox backend.

**Model:** Qwen3-4B (4B parameters)

**Infrastructure:**
- GPUs: 4x RTX 4090 (24GB each)
- Framework: verl 0.6.1 + vLLM 0.10.2

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Learning rate | 1e-6 |
| Entropy coefficient | 0.0 |
| Temperature | 0.6 |
| Top-p | 0.95 |
| Max prompt tokens | 2048 |
| Max response tokens | 4096 |

**Results:**
- Validation accuracy: 0.87% → 0.58% (slight decrease)
- Training stable but low reward signal
- Most samples truncated due to response length

**Issues:**
- Response length (4096) too short for SWE-bench patches
- RTX 4090 memory insufficient for longer sequences

---

### 3. Training v4-v9 on RTX 4090 (2026-01-16)

**Objective:** Scale up training with larger models and longer sequences.

**Experiments:**

| Version | Model | Sequence Length | Status |
|---------|-------|-----------------|--------|
| v4 | Qwen3-4B | 6144 | OOM after 50 steps |
| v5 | Qwen3-4B | 8192 | Failed to start |
| v6 | Qwen3-8B | 4096 | OOM on entropy calc |
| v7 | Qwen3-4B | 6144 (FSDP offload) | 69/7500 steps, zero reward |
| v8 | Qwen3-8B | 4096 | Training started |
| v9 | Qwen3-8B | 4096 (pool=32) | Sandbox pool implemented |

**v7 Detailed Results:**
```
Steps completed: 69/7500 (0.9%)
All samples truncated (batch/solve_none: 4)
Actor/pg_loss: 0.0 (zero learning)
Response length clip ratio: 93.75%
```

**Key Finding:**
RTX 4090 (24GB) cannot handle:
- Qwen3-8B+ with sequences > 4096
- Entropy calculation materializes full vocab (151,936 tokens, ~5GB overhead)

---

### 4. Qwen3-32B BF16 Evaluation (2026-01-28 to 2026-01-30)

**Objective:** Evaluate Qwen3-32B on SWE-bench Verified with PPIO backend.

**Model:** Qwen/Qwen3-32B (32B dense, ~64GB BF16)

**Infrastructure:**
- GPUs: 4x H100 on Kubernetes cluster
- Deployment: SGLang with tensor_parallel=4
- Backend: PPIO sandbox (replaced Docker)

**Dataset:** SWE-bench Verified (500 instances, 12 repos)

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Temperature | 0.6 |
| Top-p | 0.95 |
| Max tokens | 16384 |
| Tensor parallel | 4 |

**Evaluation Progression:**

| Stage | Accuracy | Notes |
|-------|----------|-------|
| Initial (vLLM) | 30% (3/10) | Docker backend |
| After tool fixes | 66.7% (2/3) | PPIO backend, django+sympy resolved |
| Tinker+PPIO | 66.7% (2/3) | Identical results |

**Critical Bug Fixes Applied:**
1. **Tool name dispatch:** `file_editor` parameter parsing fixed
2. **Search output format:** Match counts per file corrected
3. **str_replace feedback:** Must show edited snippet
4. **Test verification:** Django uses `runtests.py`, not pytest

---

### 5. Tinker+PPIO Training v2 (2026-01-31 to 2026-02-02)

**Objective:** RL training with Tinker API and PPIO sandbox.

**Configuration:**
```python
# Training parameters
NUM_BATCHES = 50
GROUP_SIZE = 4
MAX_STEPS = 30
POOL_SIZE = 32

# Sampling parameters
temperature = 0.6
top_p = 0.95
max_new_tokens = 16384
```

**Template Configuration:**
| Template ID | Name | Contents |
|-------------|------|----------|
| `8k14f17ixvkgis2guf51` | r2e-gym-scientific | Dependencies only |
| `xzegiq0xmuinwrclqr8a` | r2e-gym-orange3 | Pre-cloned orange3 |
| `f62brfz8qc6cjsz96kpz` | r2e-gym-pillow | Pillow dependencies |

**Status:** Training pipeline validated, full-scale training pending GPU allocation.

---

## Infrastructure Notes

### PPIO Sandbox Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Pool size | 16-32 | Parallel sandbox instances |
| Template | volcengine/sandbox-fusion | Base image |
| Max concurrent | 32 | Concurrent task executions |
| Timeout | 1200s | For GitHub clones |
| API rate limit | 5 req/s | Per sandbox |

### Memory Requirements

| Model | BF16 VRAM | FP8 VRAM | Min GPUs |
|-------|-----------|----------|----------|
| Qwen3-4B | ~8GB | ~4GB | 1x RTX 4090 |
| Qwen3-8B | ~16GB | ~8GB | 1x RTX 4090 |
| Qwen3-32B | ~64GB | ~32GB | 4x H100 |

### Known Issues

1. **RTX 4090 Entropy Bottleneck:**
   - Entropy calculation materializes full vocab (151,936 tokens)
   - ~5GB overhead, blocks sequences >6k tokens

2. **verl + vLLM Compatibility:**
   - verl 0.6.1 works with vLLM 0.10.2
   - vLLM 0.11.0+ has symmetric memory conflict with hybrid engine

3. **Mint API Limitations:**
   - Session timeout: ~30-40 minutes
   - `save_weights` unstable under concurrent load
   - Requires tinker==0.6.3 for API key format compatibility

---

## R2E-Gym Alignment Features

Features implemented for R2E-Gym compatibility:

| Feature | Status | Description |
|---------|--------|-------------|
| Full history | ✅ | No 15-entry limit |
| Token limit checking | ✅ | MAX_CONTEXT_TOKENS=65536 |
| Concise view (AST-based) | ✅ | Grep skeleton, body elision |
| Linting (ast.parse) | ✅ | Syntax error detection |
| Observation formatting | ✅ | "Execution output of [function]:" prefix |
| REPO_TEST_CMDS | ✅ | Django, sympy, pytest-specific commands |

---

## DeepSWE Baseline Comparison

| Metric | DeepSWE (Together AI) | Our Results |
|--------|----------------------|-------------|
| Model | Qwen3-32B | Qwen3-32B |
| Pre-RL baseline | 23% Pass@1 | ~30% (initial) |
| Post-RL (200 steps) | 42.2% Pass@1 | TBD |
| Training data | 4,500 R2E-Gym instances | R2E-Gym subset |
| Infrastructure | 64 H100, 6 days | 4 H100, scaled |

---

## File Locations

```
/home/claude/work/rllm/
├── experiments/swebench_ppio/
│   ├── swebench_ppio_eval.py           # vLLM+PPIO eval
│   ├── swebench_tinker_ppio_eval.py    # Tinker+PPIO eval
│   ├── tinker_r2e_training_v2.py       # RL training script
│   └── template/                        # PPIO template dockerfiles
├── rllm/environments/swe_ppio/
│   ├── swe_ppio.py                     # Environment implementation
│   └── ppio_reward.py                  # Reward computation + SandboxPool
└── data/swe/
    ├── R2E_Gym_Subset.parquet          # 4,578 instances
    └── SWE_Bench_Verified.parquet      # 500 instances
```

---

## Next Steps

1. **Full-scale RL training** with 4x H100 on R2E-Gym dataset
2. **Hyperparameter tuning** based on DeepSWE paper recommendations
3. **Evaluation** on full SWE-bench Verified (500 instances)
4. **Comparison** with DeepSWE 42.2% baseline

---

*Last updated: 2026-02-02*
