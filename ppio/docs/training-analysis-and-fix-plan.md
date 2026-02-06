# PPIO Training Analysis and Fix Plan

## 1. Problem Summary

The Qwen3-32B training using `ppio/scripts/train_qwen3_32b_8h200.sh` shows **0% solve rate** after 19 training steps. All trajectories terminate with reward 0.0.

### Training Status (Last Run: 2026-02-03)
| Metric | Value |
|--------|-------|
| Steps Completed | 19/200 |
| Solve Rate | 0% |
| Reward Mean | 0.0 |
| Trajectories per Step | 64 (8 samples x 8 rollouts) |
| Termination Reasons | MAX_STEPS, TRUNCATION |

## 2. Root Cause Analysis

### 2.1 Critical Mismatch: Environment vs Agent Architecture

The fundamental issue is an **architectural mismatch** between the environment (`swe_ppio`) and the agent (`sweagent`).

#### Current Configuration (ppio/scripts/train_qwen3_32b_8h200.sh):
```bash
rllm.env.name=swe_ppio        # Single-step patch-based environment
rllm.agent.name=sweagent      # Multi-step tool-calling agent
```

#### Reference Configuration (examples/swe/train_deepswe_32b.sh):
```bash
rllm.env.name=swe             # Multi-step r2egym-based environment
rllm.agent.name=sweagent      # Multi-step tool-calling agent
```

### 2.2 Environment Comparison

| Feature | `swe` (Reference) | `swe_ppio` (Current) |
|---------|-------------------|---------------------|
| Backend | r2egym + Docker/K8s | PPIO Sandbox |
| Interaction | Multi-step (50 steps) | **Single-step (1 patch)** |
| Tools | file_editor, execute_bash, search, finish | None |
| Agent Protocol | XML function calls | **Direct patch extraction** |
| Reward Signal | After each step via `compute_reward()` | After single patch application |

### 2.3 Why Training Fails

1. **SWEAgent** generates multi-step tool calls like:
   ```xml
   <function=execute_bash>
   <parameter=command>ls -la /testbed</parameter>
   </function>
   ```

2. **SWEBenchPPIOEnv.step()** expects a complete git patch and calls:
   ```python
   patch = extract_patch_from_response(action)  # Returns "" for tool calls
   if not patch:
       return "No valid patch found", 0.0, False, {"error": "No patch found"}
   ```

3. Result: **All 50 steps return 0 reward** because the agent never generates a patch in the expected format.

### 2.4 Code Evidence

**swe_ppio.py:233-252** (step method):
```python
def step(self, action: str) -> tuple[str, float, bool, dict]:
    self.total_steps += 1

    # Expects action to be a complete patch - FAILS with tool calls
    patch = extract_patch_from_response(action)
    if not patch:
        return "No valid patch found in response.", 0.0, False, {
            "error": "No patch found"
        }
    # ... apply patch and run tests
    done = True  # Terminates after ONE step
```

**swe.py:128-151** (step method):
```python
def step(self, action: str | Action) -> tuple[str, float, bool, bool, dict]:
    if isinstance(action, str):
        action_obj: Action = Action.from_string(action)  # Parses tool calls
    else:
        action_obj = action

    # r2egym RepoEnv handles tool execution
    obs, reward, done, info = self.env.step(action_obj)
    return str(obs), reward, done, info
```

## 3. Solution Options

### Option A: Implement Multi-Step PPIO Environment (Recommended)

Create a new `SWEBenchPPIOMultiStepEnv` that:
1. Supports r2egym-style tool calls (file_editor, execute_bash, search, submit)
2. Executes commands in PPIO sandbox instead of Docker
3. Computes reward on `submit` action

**Pros:**
- Full compatibility with existing SWEAgent
- Same training dynamics as reference implementation
- Proper exploration before patching

**Cons:**
- Requires significant implementation effort
- Need to implement tool execution layer for PPIO

### Option B: Create Patch-Only Agent

Create a new agent that generates patches directly without multi-step exploration:
1. System prompt instructs to generate `diff --git` format directly
2. Single-step interaction

**Pros:**
- Works with current `swe_ppio` environment
- Simpler implementation

**Cons:**
- Harder task for model (no exploration)
- Different from DeepSWE training paradigm
- Lower expected performance

### Option C: Use Original Environment with PPIO Reward (Hybrid)

Modify training to:
1. Use `swe` environment for exploration (r2egym + local Docker)
2. Use PPIO sandbox only for final reward verification

**Pros:**
- Minimal changes
- Proven exploration dynamics

**Cons:**
- Still requires Docker/K8s for exploration
- More complex infrastructure

## 4. Recommended Fix Plan

### Phase 1: Implement Multi-Step PPIO Environment

**File: `rllm/environments/swe_ppio/swe_ppio_multistep.py`**

```python
class SWEBenchPPIOMultiStepEnv(BaseEnv):
    """Multi-step SWE environment using PPIO sandbox"""

    def __init__(self, ...):
        self.tools = {
            'execute_bash': self._execute_bash,
            'file_editor': self._file_editor,
            'search': self._search,
            'submit': self._submit,
        }

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        # Parse XML tool call
        action_obj = self._parse_action(action)

        if action_obj.function_name == 'submit':
            # Apply patch and compute reward
            return self._handle_submit(action_obj)
        else:
            # Execute tool and return observation
            return self._execute_tool(action_obj)
```

### Phase 2: Implement Tool Executors

| Tool | Implementation |
|------|---------------|
| `execute_bash` | `sandbox.commands.run(cmd)` |
| `file_editor` | `sandbox.files.read/write()` + custom logic |
| `search` | `sandbox.commands.run(f"grep -rn {term} {path}")` |
| `submit` | Extract patch from git diff, apply, run tests |

### Phase 3: Update Training Script

```bash
# train_qwen3_32b_8h200.sh
rllm.env.name=swe_ppio_multistep   # New multi-step environment
rllm.agent.name=sweagent           # Keep existing agent
```

### Phase 4: Verification Steps

1. Test single trajectory with PPIO sandbox
2. Verify tool execution works correctly
3. Run short training (10 steps) to validate reward signal
4. Full training run (200 steps)

## 5. Implementation Details

### 5.1 New File Structure

```
rllm/environments/swe_ppio/
├── __init__.py
├── swe_ppio.py              # Existing single-step (keep for reference)
├── swe_ppio_multistep.py    # NEW: Multi-step environment
├── ppio_reward.py           # Existing reward utilities
└── tool_executors.py        # NEW: Tool implementation
```

### 5.2 Key Implementation Points

1. **Action Parsing**: Use existing `parse_xml_response()` from SWEAgent
2. **State Management**: Track file states for `undo_edit`
3. **Timeout Handling**: Per-tool timeouts (bash: 90s, tests: 1800s)
4. **Error Recovery**: Graceful handling of sandbox errors

### 5.3 Environment Mapping Update

```python
# rllm/trainer/env_agent_mappings.py
ENV_CLASSES = {
    ...
    "swe_ppio": safe_import("rllm.environments.swe_ppio.swe_ppio", "SWEBenchPPIOEnv"),
    "swe_ppio_multistep": safe_import("rllm.environments.swe_ppio.swe_ppio_multistep", "SWEBenchPPIOMultiStepEnv"),
}
```

## 6. Alternative Quick Fix (If Time Constrained)

If full multi-step implementation is not feasible immediately, apply this minimal fix:

### 6.1 Modify System Prompt for Patch Generation

Create a simplified prompt that instructs the model to:
1. Analyze the issue
2. Generate a complete patch in first response
3. Use `<function=submit>` format

### 6.2 Modify Environment to Accept Submit Action

```python
# swe_ppio.py
def step(self, action: str) -> tuple[str, float, bool, dict]:
    # Try to extract from submit action first
    if '<function=submit>' in action:
        patch = self._extract_patch_from_submit(action)
    else:
        patch = extract_patch_from_response(action)
```

**Warning**: This is a workaround, not a proper fix. Expected performance will be lower than multi-step approach.

## 7. Estimated Timeline

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Design & scaffold multi-step env | 1 day |
| 2 | Implement tool executors | 2 days |
| 3 | Integration testing | 1 day |
| 4 | Training validation | 1 day |
| **Total** | | **5 days** |

## 8. Success Criteria

1. Training shows non-zero solve rate after 10 steps
2. Solve rate increases over training (learning signal present)
3. Final solve rate comparable to reference (~40% after 200 steps)

## 9. Appendix: Training Log Evidence

### Sample Log Showing Zero Rewards
```
step:19 - batch/solve_none:8 - batch/solve_all:0 - batch/solve_partial:0
critic/rewards/mean:0.0 - critic/rewards/max:0.0 - critic/rewards/min:0.0
```

### Trajectory Termination Patterns
```
Trajectory 228 completed due to: MAX_STEPS. Reward is 0.0.
Trajectory 61 completed due to: TRUNCATION. Reward is 0.0.
Trajectory 100 completed due to: MAX_STEPS. Reward is 0.0.
```

All trajectories either:
- Hit MAX_STEPS (50) without generating valid patch
- Hit response length limit (TRUNCATION) without generating valid patch

---

## 10. Critical Bug Analysis (2026-02-05)

### 10.1 Training Results After 49 Steps

After running the training with `swe_ppio_multistep` environment for 49 steps:

| Metric | Value |
|--------|-------|
| Steps Completed | 49/200 |
| Checkpoints | global_step_10, 20, 30, 40 |
| **Resolved Rate** | **0%** (all steps) |
| Samples per Step | 64 (training), 244 (validation) |

**All trajectories show `FAIL_TO_PASS Success: 0/0` - tests are not being executed!**

### 10.2 Root Cause: `from_dict` Data Passing Bug

#### Issue Location
- `rllm/environments/swe_ppio/swe_ppio_multistep.py:718-730`
- `rllm/environments/swe_ppio/swe_ppio.py:338-346`

#### Data Flow Analysis

1. **Data Preparation** (`rllm/data/dataset.py:412-432`):
   ```python
   def apply_verl_postprocessing(cls, data):
       processed_entry = {
           "prompt": [...],
           "reward_model": {...},
           "extra_info": entry,  # Original data goes here
       }
   ```

2. **Trainer** (`rllm/trainer/verl/agent_ppo_trainer.py:95-103`):
   ```python
   env_args = batch.non_tensor_batch["extra_info"].tolist()
   # env_args[i] contains: {instance_id, repo, FAIL_TO_PASS, ...}

   return i, self.env_class.from_dict({**env_args[i], **base_env_args})
   ```

3. **PPIO Environment (INCORRECT)** (`swe_ppio_multistep.py:718-730`):
   ```python
   @staticmethod
   def from_dict(info: dict) -> "SWEBenchPPIOMultiStepEnv":
       return SWEBenchPPIOMultiStepEnv(
           entry=info.get("entry"),  # BUG: Returns None!
           ...
       )
   ```

4. **Standard SWEEnv (CORRECT)** (`swe.py:162-188`):
   ```python
   @staticmethod
   def from_dict(extra_info: dict | str) -> "SWEEnv":
       # ... extract init params ...
       init_params["entry"] = extra_info  # CORRECT: Use entire info as entry
       return SWEEnv(**init_params)
   ```

#### Why Bug Causes Zero Rewards

1. `from_dict` receives: `{instance_id: "...", repo: "...", FAIL_TO_PASS: [...], ...}`
2. `info.get("entry")` returns `None` (no "entry" key exists)
3. Environment initialized with `entry=None`
4. In `_submit` method: `fail_to_pass = self.entry.get("FAIL_TO_PASS", [])` returns `[]`
5. Test command runs without specific tests → FAIL_TO_PASS Success: 0/0
6. **Reward = 0.0 for all samples**

### 10.3 Secondary Issue: R2E-Gym Data Format

| Dataset | FAIL_TO_PASS | expected_output_json |
|---------|-------------|---------------------|
| R2E_Gym_Subset (train) | Empty | Present |
| SWE_Bench_Verified (val) | Present | Empty |

R2E-Gym training data uses `expected_output_json` format, but the current implementation only supports `FAIL_TO_PASS`. This is a secondary issue that only matters after fixing the primary bug.

### 10.4 Fix Implementation

#### Fix for `swe_ppio_multistep.py`

Replace lines 718-730:

```python
@staticmethod
def from_dict(info: dict | str) -> "SWEBenchPPIOMultiStepEnv":
    """Create environment from dictionary.

    Args:
        info: Dictionary containing task data. The entire dict will be used
              as 'entry', and any keys matching __init__ parameters will be
              extracted and passed.
    """
    import inspect
    import json

    if isinstance(info, str):
        info = json.loads(info)

    # Extract init params that match __init__ signature
    sig = inspect.signature(SWEBenchPPIOMultiStepEnv.__init__)
    init_params = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name in info:
            init_params[param_name] = info[param_name]

    # Use entire info as entry (matches standard SWEEnv behavior)
    init_params["entry"] = info

    return SWEBenchPPIOMultiStepEnv(**init_params)
```

#### Fix for `swe_ppio.py`

Apply same fix to `from_dict` method in `swe_ppio.py:338-346`.

### 10.5 Verification Plan

After applying fix:

1. **Unit Test**:
   ```python
   extra_info = {"instance_id": "test", "FAIL_TO_PASS": '["test1"]', ...}
   env = SWEBenchPPIOMultiStepEnv.from_dict(extra_info)
   assert env.entry is not None
   assert env.entry["instance_id"] == "test"
   ```

2. **Short Training Run**:
   - Run 1-2 steps
   - Check chat_completions for "FAIL_TO_PASS Success: X/Y" (not 0/0)
   - Verify reward computation works

3. **Full Training**:
   - Restart training from scratch
   - Monitor for non-zero rewards

---

## 11. Implementation Status (2026-02-04)

### Completed Changes

1. **Created `swe_ppio_multistep.py`** - Multi-step PPIO environment supporting:
   - `execute_bash`: Execute bash commands in sandbox
   - `str_replace_editor`: View/edit/create files with undo support
   - `submit`: Submit solution and compute reward via test execution
   - `search`: Search for terms in files

2. **Updated `__init__.py`** - Export new environment class

3. **Updated `env_agent_mappings.py`** - Registered `swe_ppio_multistep` environment

4. **Updated `train_qwen3_32b_8h200.sh`**:
   - Changed `rllm.env.name=swe_ppio_multistep`
   - Added proxy configuration support (`USE_PROXY=1`)

### Usage

```bash
# Without proxy
./ppio/scripts/train_qwen3_32b_8h200.sh

# With proxy (for network issues)
USE_PROXY=1 https_proxy=http://127.0.0.1:1083 ./ppio/scripts/train_qwen3_32b_8h200.sh
```

### Expected Behavior

With the new multi-step environment:
- SWEAgent tool calls will now be executed in PPIO sandbox
- Agent can explore, edit files, and run commands before submitting
- Reward computed on `submit` action via test execution
- Training should show non-zero rewards when agent successfully fixes issues

---

## 11. R2E-Gym Repository Name Mapping Fix (2026-02-05)

### 11.1 Issue Description

After fixing the `from_dict` bug and restarting training, a new error occurred:

```
RuntimeError: Failed to setup repository: Clone failed (exit=1): Command failed: Command exited with code 1 and error:
```

### 11.2 Root Cause

R2E-Gym training data uses **short repository names** instead of full GitHub paths:

| Dataset | Example Repo Value |
|---------|-------------------|
| R2E_Gym (training) | `pandas` |
| SWE-Bench (validation) | `pandas-dev/pandas` |

The code tried to clone:
- `https://github.com/pandas.git` ❌ (doesn't exist)

Instead of:
- `https://github.com/pandas-dev/pandas.git` ✅

### 11.3 Fix Implementation

Added `R2E_GYM_REPO_MAP` and `normalize_repo_name()` function to `ppio_reward.py`:

```python
# R2E-Gym Short Name to Full GitHub Path Mapping
R2E_GYM_REPO_MAP = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    "pillow": "python-pillow/Pillow",
    "orange3": "biolab/orange3",
    "aiohttp": "aio-libs/aiohttp",
    "tornado": "tornadoweb/tornado",
    "scrapy": "scrapy/scrapy",
    "pyramid": "Pylons/pyramid",
    "datalad": "datalad/datalad",
    "coveragepy": "nedbat/coveragepy",
}

def normalize_repo_name(repo: str) -> str:
    """Convert R2E-Gym short repo name to full GitHub path if needed."""
    if "/" in repo:
        return repo  # Already a full path
    return R2E_GYM_REPO_MAP.get(repo, repo)
```

### 11.4 Files Updated

1. **`rllm/environments/swe_ppio/ppio_reward.py`**:
   - Added `R2E_GYM_REPO_MAP` dictionary
   - Added `normalize_repo_name()` function
   - Updated `SandboxPool.get_sandbox()` to use normalized names
   - Updated `PPIOSandboxManager.__init__()` to use normalized names
   - Updated `PPIOSandboxManager.create_sandbox()` to use normalized names

2. **`rllm/environments/swe_ppio/swe_ppio_multistep.py`**:
   - Import `normalize_repo_name` from `ppio_reward`
   - Updated repo URL construction: `full_repo = normalize_repo_name(repo)`
   - Updated template check to cover both short and full names

3. **`rllm/environments/swe_ppio/swe_ppio.py`**:
   - Same updates as swe_ppio_multistep.py

### 11.5 Summary of Changes

Before:
```python
repo_url = f"https://github.com/{repo}.git"  # Fails for "pandas"
```

After:
```python
full_repo = normalize_repo_name(repo)  # "pandas" → "pandas-dev/pandas"
repo_url = f"https://github.com/{full_repo}.git"  # Works!
```

---

## 12. PPIO Sandbox Network Issue (2026-02-05)

### 12.1 Issue Description

After fixing the repo name mapping, git clone operations in PPIO sandbox fail with exit code 128.

### 12.2 Root Cause Analysis

- **git is available** in the sandbox (version 2.39.5)
- **directory creation works** (`mkdir -p /tmp/testbed` succeeds)
- **git clone fails** with exit code 128 (generic fatal error)

Exit code 128 typically indicates:
1. Network connectivity issue (can't reach GitHub)
2. SSL/TLS certificate verification failed
3. Firewall/proxy blocking GitHub access

The PPIO base sandbox may have network restrictions that prevent direct GitHub access.

### 12.3 Solution: Build Pre-built Templates

The proper solution is to build the R2E-Gym templates created in `ppio/templates/`:

```bash
cd ppio/templates
./build_r2e_templates.sh all
```

This will:
1. Build Docker images with repos pre-cloned
2. Upload to PPIO as templates
3. Templates have full network access during build time

### 12.4 Files Fixed During Debugging

1. **`ppio_reward.py`** - Multiple fixes:
   - Added `normalize_repo_name()` function for R2E-Gym short names
   - Added `FALLBACK_WORKDIR = "/tmp/testbed"` for writable directory
   - Updated `create_sandbox()` to use fallback workdir for non-prebuilt templates
   - Improved error handling in `_run_command()`
   - Added git version check before clone

2. **`swe_ppio_multistep.py`** - Import and use `normalize_repo_name`

3. **`swe_ppio.py`** - Import and use `normalize_repo_name`

### 12.5 Bugs Fixed in This Session

| Bug | Description | Status |
|-----|-------------|--------|
| `from_dict` bug | Entry data not passed to environment | ✅ Fixed |
| R2E-Gym repo names | Short names not mapped to full paths | ✅ Fixed |
| `/testbed` permission | Can't create dir in root | ✅ Fixed (use /tmp/testbed) |
| Git clone in sandbox | Network access blocked | ⚠️ Needs pre-built templates |

### 12.6 Next Steps

1. **Build R2E-Gym templates**:
   ```bash
   cd ppio/templates
   ./build_r2e_templates.sh all
   ```

2. **Update REPO_TEMPLATE_MAP**:
   ```bash
   python update_repo_map.py
   ```

3. **Restart training** - with pre-built templates, sandbox will use pre-cloned repos
