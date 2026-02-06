# DeepSWE PPIO Sandbox Training Pipeline Analysis

## Entry Point
**Script:** `ppio/scripts/train_qwen3_32b_8h200.sh`

## Overview

This document analyzes the PPIO sandbox-based DeepSWE training pipeline, which replaces Docker/r2egym with PPIO cloud sandboxes.

---

## 1. Configuration Summary

```bash
# Environment (KEY DIFFERENCE)
rllm.env.name=swe_ppio_multistep
rllm.agent.name=sweagent
rllm.agent.max_steps=50
rllm.agent.trajectory_timeout=5400

# Model
actor_rollout_ref.model.path=/models/models/Qwen3-32B

# Data
data.train_files=R2E_Gym_Subset.parquet
data.val_files=SWE_Bench_Verified.parquet
data.train_batch_size=8
data.max_prompt_length=4096
data.max_response_length=32768

# Distributed (single node)
trainer.n_gpus_per_node=8
trainer.nnodes=1  # 8 GPUs total
actor_rollout_ref.rollout.n=8
```

---

## 2. Module Dependency Tree

```
ppio/scripts/train_qwen3_32b_8h200.sh
└── python3 -m rllm.trainer.verl.train_agent_ppo
    ├── rllm/trainer/verl/train_agent_ppo.py (Entry Point)
    │   └── [Same as standard]
    │
    ├── rllm/trainer/verl/agent_ppo_trainer.py (Trainer)
    │   └── [Same as standard]
    │
    ├── rllm/trainer/env_agent_mappings.py (Registry)
    │   ├── ENV_CLASS_MAPPING["swe_ppio_multistep"] → SWEBenchPPIOMultiStepEnv
    │   └── AGENT_CLASS_MAPPING["sweagent"] → SWEAgent
    │
    ├── rllm/environments/swe_ppio/ (PPIO Environment) ★ DIFFERENT
    │   ├── __init__.py
    │   ├── swe_ppio_multistep.py (Main)
    │   │   ├── SWEBenchPPIOMultiStepEnv
    │   │   │   ├── reset()
    │   │   │   ├── step()
    │   │   │   ├── _execute_bash()
    │   │   │   ├── _str_replace_editor()
    │   │   │   ├── _search()
    │   │   │   ├── _submit()
    │   │   │   └── from_dict()
    │   │   └── parse_xml_action()
    │   ├── swe_ppio.py (Legacy single-step)
    │   │   └── SWEBenchPPIOEnv
    │   └── ppio_reward.py (Sandbox Management)
    │       ├── SandboxPool (Singleton)
    │       ├── PPIOSandboxManager
    │       ├── REPO_TEMPLATE_MAP
    │       ├── R2E_GYM_REPO_MAP
    │       ├── normalize_repo_name()
    │       ├── parse_pytest_output()
    │       └── setup_proxy_tunnel()
    │
    ├── rllm/agents/swe_agent.py (Agent)
    │   └── [Same as standard]
    │
    ├── rllm/engine/agent_execution_engine.py (Execution)
    │   └── [Same as standard]
    │
    └── External: ppio_sandbox (PPIO SDK)
        └── ppio_sandbox.core.Sandbox
```

---

## 3. PPIO-Specific Modules Detail

### 3.1 Environment: `swe_ppio_multistep.py`

**File:** `rllm/environments/swe_ppio/swe_ppio_multistep.py`

**Class:** `SWEBenchPPIOMultiStepEnv(BaseEnv)`

**Initialization:**
```python
def __init__(
    entry: Optional[dict] = None,      # Task data
    timeout: int = 3600,               # Sandbox timeout
    workdir: str = "/testbed",         # Working directory
    use_pool: bool = True,             # Use sandbox pool
    pool_size: int = 32,               # Pool size per repo
    step_timeout: int = 90,            # Per-step timeout
    max_steps: int = 50,               # Max steps
    reward_timeout: int = 1800,        # Test timeout
)
```

**Key Methods:**

#### `reset() -> tuple[str, dict]`
```python
def reset(self):
    # 1. Release previous sandbox to pool
    if self.sandbox_manager:
        self.sandbox_manager.cleanup(pause=True)

    # 2. Get sandbox from pool (or create new)
    self.sandbox_manager = PPIOSandboxManager(
        repo=self.repo,
        use_pool=True,
    )
    self.sandbox_manager.create_sandbox(trajectory_idx=self._get_trajectory_idx())

    # 3. Setup repository
    self.sandbox_manager.clone_repo(repo_url, commit)
    self.sandbox_manager.install_deps(install_cmd)

    return {"task_instruction": problem_statement}, info
```

#### `step(action: str) -> tuple[str, float, bool, dict]`
```python
def step(self, action: str):
    # Parse XML action
    thought, parsed_action = parse_xml_action(action)

    # Route to handler
    if parsed_action.function_name == "execute_bash":
        return self._execute_bash(parsed_action)
    elif parsed_action.function_name in ["str_replace_editor", "file_editor"]:
        return self._str_replace_editor(parsed_action)
    elif parsed_action.function_name == "submit":
        return self._submit(parsed_action)
    elif parsed_action.function_name == "search":
        return self._search(parsed_action)
```

#### `_submit() -> tuple[str, float, bool, dict]`
```python
def _submit(self, action):
    # 1. Get git diff
    _, patch = self.sandbox_manager._run_command("git diff HEAD")

    # 2. Get test lists from entry
    fail_to_pass = self.entry.get("FAIL_TO_PASS", [])
    pass_to_pass = self.entry.get("PASS_TO_PASS", [])

    # 3. Run tests
    test_cmd = get_test_cmd_for_repo(self.repo)
    exit_code, output = self.sandbox_manager.run_tests(test_cmd)

    # 4. Parse results
    result = parse_pytest_output(output, fail_to_pass, pass_to_pass)

    # 5. Calculate reward
    if result.resolved:
        reward = 1.0
    elif len(fail_to_pass) > 0:
        reward = len(result.f2p_success) / len(fail_to_pass)
    else:
        reward = 0.0

    return observation, reward, done=True, info
```

#### `from_dict(info: dict) -> SWEBenchPPIOMultiStepEnv`
```python
@staticmethod
def from_dict(info: dict | str):
    if isinstance(info, str):
        info = json.loads(info)

    # Extract init params
    init_params = {}
    for param_name in signature.parameters:
        if param_name in info:
            init_params[param_name] = info[param_name]

    # CRITICAL: Use entire info as entry
    init_params["entry"] = info  # Contains FAIL_TO_PASS, repo, etc.

    return SWEBenchPPIOMultiStepEnv(**init_params)
```

### 3.2 Sandbox Management: `ppio_reward.py`

**File:** `rllm/environments/swe_ppio/ppio_reward.py`

#### `SandboxPool` (Singleton)
```python
class SandboxPool:
    """Global sandbox pool to prevent rate-limit errors (429)."""

    _instance = None  # Singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def configure(self, api_key, pool_size=32, timeout=3600, template="base"):
        """Configure pool parameters."""
        self._api_key = api_key
        self._pool_size = pool_size
        self._timeout = timeout
        self._default_template = template
        self._pools = {}  # Per-repo pools

    def get_sandbox(self, trajectory_idx, repo, workdir):
        """Get sandbox from per-repo pool."""
        # Normalize repo name (R2E-Gym compatibility)
        full_repo = normalize_repo_name(repo)

        # Get template
        template = REPO_TEMPLATE_MAP.get(repo) or \
                   REPO_TEMPLATE_MAP.get(full_repo) or \
                   self._default_template

        # Use repo-specific pool
        pool_key = repo
        sandbox_idx = trajectory_idx % self._pool_size

        if pool_key not in self._pools:
            self._pools[pool_key] = {}

        # Reuse or create
        if sandbox_idx in self._pools[pool_key]:
            sandbox = self._pools[pool_key][sandbox_idx]
            sandbox.connect()  # Resume paused sandbox
        else:
            sandbox = self._create_with_retry(template, workdir)
            self._pools[pool_key][sandbox_idx] = sandbox

        return sandbox, using_prebuilt

    def _create_with_retry(self, template, workdir, max_retries=5):
        """Create sandbox with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return Sandbox(
                    template=template,
                    timeout=self._timeout,
                    workdir=workdir,
                )
            except Exception as e:
                if "429" in str(e):  # Rate limit
                    delay = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                    time.sleep(delay)
                else:
                    raise
```

#### `PPIOSandboxManager`
```python
class PPIOSandboxManager:
    """Manages PPIO sandbox lifecycle for a single task."""

    def __init__(self, repo, timeout=3600, workdir=DEFAULT_WORKDIR,
                 template=None, use_pool=True, pool_size=32):
        self.repo = repo
        self.timeout = timeout
        self.workdir = workdir
        self.use_pool = use_pool

        # Normalize repo for template lookup
        full_repo = normalize_repo_name(repo)
        self.template = template or \
                        REPO_TEMPLATE_MAP.get(repo) or \
                        REPO_TEMPLATE_MAP.get(full_repo) or \
                        "base"

    def create_sandbox(self, trajectory_idx=0):
        """Get sandbox from pool or create new."""
        if self.use_pool:
            pool = get_sandbox_pool()
            self.sandbox, self.using_prebuilt = pool.get_sandbox(
                trajectory_idx, self.repo, self.workdir
            )
        else:
            self.sandbox = Sandbox(template=self.template, ...)
            self.using_prebuilt = self.repo in REPO_TEMPLATE_MAP

        # Fix permissions
        self.sandbox.commands.run(f"chmod -R 777 {self.workdir}")
        self.sandbox.commands.run(f"git config --global --add safe.directory {self.workdir}")

    def _run_command(self, cmd, timeout=90):
        """Execute command in sandbox."""
        result = self.sandbox.commands.run(cmd, timeout=timeout)
        return result.exit_code, result.stdout + result.stderr

    def clone_repo(self, repo_url, commit):
        """Clone and checkout repository."""
        if self.using_prebuilt:
            # Pre-built: just checkout commit
            self._run_command(f"cd {self.workdir} && git checkout -f {commit}")
        else:
            # Base template: clone fresh
            self._run_command(f"git clone {repo_url} {self.workdir}")
            self._run_command(f"cd {self.workdir} && git checkout {commit}")

    def run_tests(self, test_cmd, timeout=1800):
        """Run test suite."""
        return self._run_command(f"cd {self.workdir} && {test_cmd}", timeout)

    def cleanup(self, pause=True):
        """Release sandbox."""
        if self.use_pool and pause:
            self.sandbox.pause()  # Keep for reuse
        else:
            self.sandbox.kill()
```

#### Repository Mappings

```python
# Pre-built templates for fast startup
REPO_TEMPLATE_MAP = {
    # SWE-Bench validation repos
    "django/django": "swebench-django-django",
    "pallets/flask": "swebench-pallets-flask",
    "sympy/sympy": "swebench-sympy-sympy",
    "astropy/astropy": "swebench-astropy-astropy",
    # ... 8 more

    # R2E-Gym training repos (by template ID)
    "pillow": "f62brfz8qc6cjsz96kpz",
    "orange3": "zmaw00bm1kxnfktz4xbq",
}

# Short name to full path conversion
R2E_GYM_REPO_MAP = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    "pillow": "python-pillow/Pillow",
    "orange3": "biolab/orange3",
    # ... 6 more
}

def normalize_repo_name(repo: str) -> str:
    """Convert R2E-Gym short name to full GitHub path."""
    if "/" in repo:
        return repo  # Already full path
    return R2E_GYM_REPO_MAP.get(repo, repo)
```

#### Test Result Parsing

```python
def parse_pytest_output(output, fail_to_pass, pass_to_pass):
    """Parse pytest output and categorize results."""
    # Extract test results
    passed_tests = set()
    failed_tests = set()

    # Parse output for PASSED/FAILED markers
    for line in output.split('\n'):
        if '::' in line:
            test_name = extract_test_name(line)
            if 'PASSED' in line:
                passed_tests.add(test_name)
            elif 'FAILED' in line:
                failed_tests.add(test_name)

    # Categorize
    f2p_success = [t for t in fail_to_pass if t in passed_tests]
    f2p_failure = [t for t in fail_to_pass if t not in passed_tests]
    p2p_success = [t for t in pass_to_pass if t in passed_tests]
    p2p_failure = [t for t in pass_to_pass if t not in passed_tests]

    # Resolved = all FAIL_TO_PASS pass AND no PASS_TO_PASS fail
    resolved = (len(f2p_success) == len(fail_to_pass)) and (len(p2p_failure) == 0)

    return TestResult(
        passed=len(passed_tests),
        failed=len(failed_tests),
        f2p_success=f2p_success,
        f2p_failure=f2p_failure,
        p2p_success=p2p_success,
        p2p_failure=p2p_failure,
        resolved=resolved,
        test_output=output,
    )
```

### 3.3 Legacy: `swe_ppio.py`

**File:** `rllm/environments/swe_ppio/swe_ppio.py`

**Class:** `SWEBenchPPIOEnv`

**Purpose:** Single-step patch-based environment (deprecated)

**Why Deprecated:**
- Expects complete git patch in single response
- Agent generates multi-step tool calls
- Mismatch causes 0% solve rate

---

## 4. Data Flow

### 4.1 Complete Pipeline

```
Parquet Data
    ↓
extra_info = {
    "instance_id": "pandas__123",
    "repo": "pandas",           # R2E-Gym short name
    "base_commit": "abc123",
    "FAIL_TO_PASS": [...],
    "PASS_TO_PASS": [...],
    "problem_statement": "...",
}
    ↓
SWEBenchPPIOMultiStepEnv.from_dict(extra_info)
    ↓
entry = extra_info  # CRITICAL: entire dict
repo = normalize_repo_name("pandas") → "pandas-dev/pandas"
    ↓
reset()
    ↓
SandboxPool.get_sandbox(trajectory_idx, repo)
    ├─ REPO_TEMPLATE_MAP lookup → pre-built template
    └─ Pool reuse or create new
    ↓
PPIOSandboxManager.clone_repo()
    ├─ Pre-built: git checkout {commit}
    └─ Base: git clone + checkout
    ↓
Agent Loop (max 50 steps):
    ├─ Model generates: <function=execute_bash>...</function>
    ├─ parse_xml_action() → ParsedAction
    ├─ _execute_bash() → sandbox.commands.run()
    └─ Return observation
    ↓
_submit()
    ├─ git diff HEAD → patch
    ├─ Get FAIL_TO_PASS from entry
    ├─ run_tests() → sandbox.commands.run(test_cmd)
    ├─ parse_pytest_output()
    └─ Calculate reward (0.0 - 1.0)
    ↓
cleanup(pause=True) → sandbox.pause() → back to pool
    ↓
Trajectory → PPO Update
```

### 4.2 Sandbox Lifecycle

```
Request sandbox
    ↓
SandboxPool.get_sandbox()
    ↓
    ├─ [Pool Hit] sandbox exists for repo+idx
    │       ↓
    │   sandbox.connect() (resume paused)
    │       ↓
    │   Return existing sandbox (fast: ~1-2s)
    │
    └─ [Pool Miss] no sandbox exists
            ↓
        _create_with_retry()
            ↓
        ├─ [429 Rate Limit]
        │       ↓
        │   Exponential backoff (5s, 10s, 20s, ...)
        │       ↓
        │   Retry
        │
        └─ [Success]
                ↓
            Sandbox created (~30-60s)
                ↓
            Store in pool
                ↓
            Return new sandbox
    ↓
Use sandbox (step loop)
    ↓
cleanup(pause=True)
    ↓
sandbox.pause() → Keep in pool for reuse
```

---

## 5. Key Features

### 5.1 Sandbox Pool (Rate Limit Prevention)

**Problem:** PPIO API returns 429 when too many sandboxes created
**Solution:** Global singleton pool with per-repo sub-pools

```python
# Per-repo pools
self._pools = {
    "django/django": {0: sandbox0, 1: sandbox1, ...},
    "pandas-dev/pandas": {0: sandbox0, ...},
}

# Reuse by trajectory index
sandbox_idx = trajectory_idx % pool_size  # e.g., 32
```

### 5.2 Pre-built Templates

**Problem:** Git clone in base sandbox fails (network restrictions)
**Solution:** Templates with pre-cloned repos at /testbed

| Template | Repo | Startup Time |
|----------|------|--------------|
| `swebench-django-django` | django/django | ~10s |
| `swebench-astropy-astropy` | astropy/astropy | ~10s |
| `f62brfz8qc6cjsz96kpz` | python-pillow/Pillow | ~10s |
| Base template | Any | ~60s + clone |

### 5.3 R2E-Gym Compatibility

**Problem:** R2E-Gym uses short names, SWE-Bench uses full paths
**Solution:** Automatic normalization

```python
normalize_repo_name("pandas")     → "pandas-dev/pandas"
normalize_repo_name("django/django") → "django/django"
```

### 5.4 Proxy Support

**Problem:** Corporate environments behind HTTP proxy
**Solution:** Socket-level CONNECT proxy patch

```python
def setup_proxy_tunnel():
    proxy_url = os.environ.get('https_proxy')
    if proxy_url:
        # Patch socket.connect to use HTTP CONNECT
        socket.socket.connect = proxy_connect
```

---

## 6. Bugs Fixed

### Bug #1: `from_dict` Entry Loss
**Symptom:** FAIL_TO_PASS Success: 0/0, all rewards = 0.0
**Root Cause:** `info.get("entry")` returned None
**Fix:** `init_params["entry"] = info`

### Bug #2: R2E-Gym Repo Names
**Symptom:** `Clone failed: https://github.com/pandas.git`
**Root Cause:** Short names not converted to full paths
**Fix:** Added `R2E_GYM_REPO_MAP` and `normalize_repo_name()`

### Bug #3: Workdir Permissions
**Symptom:** `Permission denied: /testbed`
**Root Cause:** Base template lacks /testbed with correct permissions
**Fix:** `FALLBACK_WORKDIR = "/tmp/testbed"` for non-prebuilt

### Bug #4: Network Restrictions
**Symptom:** Git clone exit code 128, empty stderr
**Root Cause:** Base sandbox has network restrictions
**Fix:** Use pre-built templates from REPO_TEMPLATE_MAP

---

## 7. Repo-Specific Test Commands

```python
REPO_TEST_CMDS = {
    "django/django": "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    "sympy/sympy": "bin/test -C --verbose",
    "pytest-dev/pytest": "pytest -rA",
    "matplotlib/matplotlib": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "scikit-learn/scikit-learn": "pytest sklearn/ --no-header -rA --tb=no -p no:cacheprovider",
    "astropy/astropy": "pytest astropy/ --no-header -rA --tb=no -p no:cacheprovider",
    "sphinx-doc/sphinx": "pytest tests/ --no-header -rA --tb=no -p no:cacheprovider",
    "pylint-dev/pylint": "pytest tests/ --no-header -rA --tb=no -p no:cacheprovider",
    "pallets/flask": "pytest tests/ --no-header -rA --tb=no -p no:cacheprovider",
    "psf/requests": "pytest tests/ --no-header -rA --tb=no -p no:cacheprovider",
}
```

---

## 8. External Dependencies

### 8.1 ppio_sandbox (PPIO SDK)
```python
from ppio_sandbox.core import Sandbox

sandbox = Sandbox(
    template="swebench-django-django",
    timeout=3600,
    workdir="/testbed",
)
sandbox.commands.run("ls -la")
sandbox.files.read("/testbed/file.py")
sandbox.files.write("/testbed/file.py", content)
sandbox.pause()  # Keep for reuse
sandbox.kill()   # Destroy
```

### 8.2 Environment Variables
```bash
PPIO_API_KEY=sk_xxxxx          # Required
PPIO_SANDBOX_TEMPLATE=base     # Optional (default template)
https_proxy=http://...         # Optional (proxy support)
USE_PROXY=1                    # Enable proxy
```

---

## 9. Performance Characteristics

| Operation | Time |
|-----------|------|
| Create sandbox (pre-built) | 10-20s |
| Create sandbox (base) | 30-60s |
| Resume paused sandbox | 1-2s |
| Git checkout | 5-10s |
| Install deps (first time) | 30-300s |
| Install deps (cached) | 5-10s |
| Run test suite | 60-1800s |

### Cost Efficiency

Without pool:
- 512 sandboxes per epoch (64 samples × 8 rollouts)
- Many 429 errors
- High API costs

With pool:
- 32 sandboxes per repo
- Reuse across rollouts
- ~90% cost reduction

---

## 10. File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `swe_ppio_multistep.py` | ~750 | Multi-step environment |
| `swe_ppio.py` | ~350 | Legacy single-step env |
| `ppio_reward.py` | ~800 | Sandbox management |
| `__init__.py` | ~20 | Module exports |
