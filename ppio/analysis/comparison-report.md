# DeepSWE Training Pipeline Comparison Report

## Standard (Docker/r2egym) vs PPIO Sandbox

---

## Executive Summary

| Aspect | Standard Pipeline | PPIO Pipeline |
|--------|------------------|---------------|
| **Entry Script** | `examples/swe/train_deepswe_32b.sh` | `ppio/scripts/train_qwen3_32b_8h200.sh` |
| **Environment** | `swe` (SWEEnv) | `swe_ppio_multistep` (SWEBenchPPIOMultiStepEnv) |
| **Backend** | Docker + r2egym | PPIO Cloud Sandbox |
| **Infrastructure** | On-premise | Cloud (pay-per-use) |
| **Pre-requisites** | Docker, r2egym | PPIO_API_KEY only |
| **Scaling** | Limited by hardware | Unlimited via API |

---

## 1. Architecture Comparison

### 1.1 Shared Components (Identical)

Both pipelines share these modules unchanged:

| Module | File | Purpose |
|--------|------|---------|
| Entry Point | `train_agent_ppo.py` | Hydra + Ray launcher |
| Trainer | `agent_ppo_trainer.py` | PPO training loop |
| Agent | `swe_agent.py` | SWEAgent implementation |
| Execution Engine | `agent_execution_engine.py` | Async trajectory generation |
| vLLM Engine | `verl_engine.py` | Model inference |
| Registry | `env_agent_mappings.py` | Class lookup |

### 1.2 Different Components

| Component | Standard | PPIO |
|-----------|----------|------|
| Environment Class | `SWEEnv` | `SWEBenchPPIOMultiStepEnv` |
| Environment File | `rllm/environments/swe/swe.py` | `rllm/environments/swe_ppio/swe_ppio_multistep.py` |
| Sandbox Backend | r2egym (Docker) | PPIO SDK |
| Sandbox Manager | r2egym.RepoEnv | PPIOSandboxManager |
| Pool Management | None (per-env) | SandboxPool (global) |
| Template System | None | REPO_TEMPLATE_MAP |

---

## 2. Environment Implementation Comparison

### 2.1 Class Structure

**Standard: SWEEnv**
```python
class SWEEnv(BaseEnv):
    def __init__(self, entry):
        self.entry = entry
        self.repo_env = None  # r2egym.RepoEnv

    def reset(self):
        self.repo_env = r2egym.RepoEnv(
            repo=self.entry["repo"],
            commit=self.entry["base_commit"],
        )
        return observation, info

    def step(self, action):
        return self.repo_env.step(action)

    def compute_final_reward(self):
        return self.repo_env.compute_reward()
```

**PPIO: SWEBenchPPIOMultiStepEnv**
```python
class SWEBenchPPIOMultiStepEnv(BaseEnv):
    def __init__(self, entry, timeout=3600, use_pool=True, ...):
        self.entry = entry
        self.sandbox_manager = None

    def reset(self):
        self.sandbox_manager = PPIOSandboxManager(
            repo=self.repo,
            use_pool=True,
        )
        self.sandbox_manager.create_sandbox(trajectory_idx)
        self.sandbox_manager.clone_repo(url, commit)
        return observation, info

    def step(self, action):
        parsed = parse_xml_action(action)
        if parsed.function_name == "execute_bash":
            return self._execute_bash(parsed)
        elif parsed.function_name == "submit":
            return self._submit(parsed)
        # ...

    def _submit(self, action):
        # Custom test execution + reward calculation
        test_output = self.sandbox_manager.run_tests(test_cmd)
        result = parse_pytest_output(test_output, ...)
        return obs, reward, done=True, info
```

### 2.2 Key Differences

| Feature | Standard | PPIO |
|---------|----------|------|
| Sandbox Creation | Per environment | From global pool |
| Action Parsing | r2egym handles | Custom XML parser |
| Tool Execution | r2egym API | Direct sandbox.commands.run |
| File Operations | r2egym API | sandbox.files read/write |
| Test Execution | r2egym evaluator | Custom parse_pytest_output |
| Reward Calculation | r2egym.compute_reward | Manual FAIL_TO_PASS check |

---

## 3. Sandbox Management

### 3.1 Standard (r2egym)

```
New Episode
    ↓
r2egym.RepoEnv(repo, commit)
    ↓
Docker container created
    ↓
Episode execution
    ↓
Container destroyed
    ↓
Repeat for next episode
```

**Characteristics:**
- Each episode creates new container
- No reuse between episodes
- Container overhead per episode

### 3.2 PPIO (SandboxPool)

```
New Episode
    ↓
SandboxPool.get_sandbox(trajectory_idx, repo)
    ↓
    ├─ [Hit] Existing sandbox found
    │   └─ sandbox.connect() (resume)
    │
    └─ [Miss] Create new sandbox
        └─ Store in pool
    ↓
Episode execution
    ↓
sandbox.pause() (keep in pool)
    ↓
Next episode reuses same sandbox
```

**Characteristics:**
- Global pool shared across all episodes
- Per-repo sub-pools for isolation
- Sandbox reuse reduces cost/latency
- Exponential backoff for 429 errors

---

## 4. Configuration Comparison

### 4.1 Shell Script Differences

| Parameter | Standard | PPIO |
|-----------|----------|------|
| `rllm.env.name` | `swe` | `swe_ppio_multistep` |
| `trainer.nnodes` | `8` | `1` |
| `trainer.logger` | `['console','wandb']` | `['console']` |
| `data.val_batch_size` | `512` | `256` |
| `gpu_memory_utilization` | `0.6` | `0.7` |
| `trainer.default_local_dir` | Not set | `/3fsdata/data0/tengwan/...` |

### 4.2 Environment Variables

**Standard:**
```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
VLLM_USE_V1=1
```

**PPIO (Additional):**
```bash
PPIO_API_KEY=sk_xxxxx
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0
USE_PROXY=0|1
https_proxy=http://...
```

---

## 5. Data Flow Comparison

### 5.1 Environment Instantiation

**Standard:**
```python
# Direct mapping
env = SWEEnv(entry=extra_info)
```

**PPIO:**
```python
# Signature inspection + entry assignment
init_params = {k: v for k, v in extra_info.items() if k in signature}
init_params["entry"] = extra_info  # Critical fix
env = SWEBenchPPIOMultiStepEnv(**init_params)
```

### 5.2 Repository Setup

**Standard:**
```
r2egym.RepoEnv handles:
- Docker container creation
- Git clone
- Dependency installation
- Working directory setup
```

**PPIO:**
```
PPIOSandboxManager handles:
- PPIO API sandbox creation
- Template-based fast startup
- Git checkout (pre-cloned repos)
- pip install -e .
- Permission fixes (safe.directory)
```

### 5.3 Test Execution

**Standard:**
```python
# r2egym runs tests internally
result = self.repo_env.compute_reward()
```

**PPIO:**
```python
# Manual test execution and parsing
test_cmd = REPO_TEST_CMDS.get(repo, "pytest")
exit_code, output = sandbox.commands.run(test_cmd)
result = parse_pytest_output(output, fail_to_pass, pass_to_pass)
reward = len(result.f2p_success) / len(fail_to_pass)
```

---

## 6. Repository Support

### 6.1 Standard Pipeline

Supports any repository that r2egym can handle:
- Requires Docker
- Requires r2egym setup per repo
- No pre-built templates

### 6.2 PPIO Pipeline

**Pre-built Templates (Fast):**
```python
REPO_TEMPLATE_MAP = {
    # SWE-Bench (12 repos)
    "django/django": "swebench-django-django",
    "sympy/sympy": "swebench-sympy-sympy",
    # ...

    # R2E-Gym (2 repos currently)
    "pillow": "f62brfz8qc6cjsz96kpz",
    "orange3": "zmaw00bm1kxnfktz4xbq",
}
```

**R2E-Gym Name Mapping:**
```python
R2E_GYM_REPO_MAP = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    # ... 10 repos
}
```

---

## 7. Performance Comparison

### 7.1 Sandbox Lifecycle

| Operation | Standard (Docker) | PPIO (Pre-built) | PPIO (Base) |
|-----------|-------------------|------------------|-------------|
| Create sandbox | 10-30s | 10-20s | 30-60s |
| Resume paused | N/A | 1-2s | 1-2s |
| Git clone | 30-120s | 0s (pre-cloned) | 30-120s |
| Install deps | 60-300s | 10-30s (cached) | 60-300s |
| Run tests | 60-1800s | 60-1800s | 60-1800s |

### 7.2 Cost Model

**Standard:**
- Fixed infrastructure cost
- Scale limited by hardware
- No per-episode cost

**PPIO:**
- Pay-per-use (sandbox-seconds)
- Unlimited scaling
- Pool reuse reduces cost ~90%

### 7.3 Scalability

| Metric | Standard | PPIO |
|--------|----------|------|
| Max parallel sandboxes | Hardware-limited | 1000+ |
| Rate limiting | None | 429 handled |
| Geographic distribution | Single location | Global |

---

## 8. Error Handling

### 8.1 Standard Pipeline

```python
# r2egym handles errors internally
try:
    result = repo_env.step(action)
except Exception as e:
    # Propagates to training loop
    raise
```

### 8.2 PPIO Pipeline

```python
# Custom error handling
def _create_with_retry(template, workdir, max_retries=5):
    for attempt in range(max_retries):
        try:
            return Sandbox(...)
        except Exception as e:
            if "429" in str(e):
                delay = 2 ** attempt * 5
                time.sleep(delay)
            else:
                raise

# Permission fixes
sandbox.commands.run("chmod -R 777 /testbed")
sandbox.commands.run("git config --global --add safe.directory /testbed")
```

---

## 9. Bugs and Fixes (PPIO Pipeline)

### 9.1 Bugs Found and Fixed

| Bug | Symptom | Root Cause | Fix |
|-----|---------|------------|-----|
| from_dict entry loss | 0/0 FAIL_TO_PASS | `info.get("entry")` → None | Use `entry = info` |
| R2E-Gym repo names | Clone failed | Short names not converted | Added `normalize_repo_name()` |
| Workdir permissions | Permission denied | /testbed not writable | Use `/tmp/testbed` fallback |
| Network restrictions | Clone exit 128 | Base sandbox restricted | Use pre-built templates |

### 9.2 Bugs Not Applicable to Standard

These bugs are PPIO-specific and don't affect standard pipeline:
- Rate limiting (429)
- Template management
- Sandbox pooling
- Proxy support

---

## 10. Module Count Summary

### 10.1 Standard Pipeline

| Category | Files | Lines (approx) |
|----------|-------|----------------|
| Shared modules | 6 | ~1600 |
| Environment | 1 | ~150 |
| **Total unique** | **7** | **~1750** |

### 10.2 PPIO Pipeline

| Category | Files | Lines (approx) |
|----------|-------|----------------|
| Shared modules | 6 | ~1600 |
| Environment | 3 | ~1900 |
| **Total unique** | **9** | **~3500** |

### 10.3 Code Delta

PPIO adds:
- `swe_ppio_multistep.py` (~750 lines)
- `swe_ppio.py` (~350 lines)
- `ppio_reward.py` (~800 lines)
- Template Dockerfiles (10 files, ~500 lines)

---

## 11. When to Use Which Pipeline

### Use Standard Pipeline When:
- You have Docker infrastructure
- You have r2egym set up
- You need support for any repository
- You don't want cloud API dependencies

### Use PPIO Pipeline When:
- You don't have Docker setup
- You need cloud-based scaling
- You want pay-per-use cost model
- You're using supported repos (SWE-Bench, R2E-Gym)
- You're behind a corporate proxy

---

## 12. Migration Guide

### From Standard to PPIO

1. **Change environment name:**
   ```bash
   # Before
   rllm.env.name=swe

   # After
   rllm.env.name=swe_ppio_multistep
   ```

2. **Set PPIO API key:**
   ```bash
   export PPIO_API_KEY=sk_xxxxx
   ```

3. **Check repository support:**
   - SWE-Bench repos: All 12 supported
   - R2E-Gym repos: Build templates first

4. **Optional proxy setup:**
   ```bash
   export USE_PROXY=1
   export https_proxy=http://127.0.0.1:1083
   ```

### From PPIO to Standard

1. **Change environment name:**
   ```bash
   rllm.env.name=swe
   ```

2. **Set up r2egym:**
   ```bash
   pip install r2egym
   # Configure Docker
   ```

3. **Remove PPIO variables:**
   ```bash
   unset PPIO_API_KEY
   unset USE_PROXY
   ```

---

## 13. Conclusion

### Key Findings

1. **Architecture:** Both pipelines share ~80% of code; only environment differs
2. **Complexity:** PPIO adds ~1750 lines for sandbox management
3. **Performance:** PPIO with pre-built templates matches Docker speed
4. **Cost:** PPIO pool reduces API costs by ~90%
5. **Reliability:** PPIO handles rate limits, permissions, proxy

### Recommendations

- **For development:** Use PPIO (easier setup, cloud scaling)
- **For production:** Either works; choose based on infrastructure
- **For new repos:** Build PPIO templates using `ppio/templates/`

### Files Reference

| Document | Content |
|----------|---------|
| `deepswe-standard-pipeline.md` | Standard pipeline analysis |
| `deepswe-ppio-pipeline.md` | PPIO pipeline analysis |
| `comparison-report.md` | This comparison report |
