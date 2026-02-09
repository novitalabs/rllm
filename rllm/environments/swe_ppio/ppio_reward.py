#!/usr/bin/env python3
"""
SWE-bench Reward Function using PPIO Sandbox

This module provides a reward function for SWE-bench tasks using PPIO sandbox
instead of Docker. It follows the rllm RewardFunction protocol.
"""

import os
import re
import socket
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Proxy Configuration
# =============================================================================
def setup_proxy_tunnel():
    """Install socket-level proxy patch for environments behind HTTP proxy."""
    proxy_url = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
    if not proxy_url:
        return False

    from urllib.parse import urlparse
    parsed = urlparse(proxy_url)
    proxy_host = parsed.hostname
    proxy_port = parsed.port or 1080

    _original_socket_connect = socket.socket.connect

    def proxy_connect(self, address):
        # Handle Unix socket addresses (strings or paths)
        if isinstance(address, (str, bytes)) or not isinstance(address, tuple):
            return _original_socket_connect(self, address)

        # Handle tuple addresses (host, port)
        if len(address) != 2:
            return _original_socket_connect(self, address)

        host, port = address
        if isinstance(host, str) and (
            host in ('localhost', '127.0.0.1') or
            host.startswith('172.') or
            host.startswith('10.') or
            host.startswith('192.168.')
        ):
            return _original_socket_connect(self, address)
        _original_socket_connect(self, (proxy_host, proxy_port))
        connect_req = f'CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n'
        self.sendall(connect_req.encode())
        response = b''
        while b'\r\n\r\n' not in response:
            chunk = self.recv(1024)
            if not chunk:
                break
            response += chunk
        if b'200' not in response.split(b'\r\n')[0]:
            raise ConnectionError(f'Proxy connection failed: {response}')
        return None

    socket.socket.connect = proxy_connect
    return True


# Setup proxy before importing PPIO
setup_proxy_tunnel()


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class RewardOutput:
    """Output from reward function"""
    reward: float
    metadata: dict


@dataclass
class TestResult:
    """Result from running tests"""
    passed: int
    failed: int
    errors: int
    total: int
    f2p_success: list  # FAIL_TO_PASS tests that now pass
    f2p_failure: list  # FAIL_TO_PASS tests that still fail
    p2p_success: list  # PASS_TO_PASS tests that still pass
    p2p_failure: list  # PASS_TO_PASS tests that now fail
    resolved: bool
    test_output: str


# =============================================================================
# Pre-built PPIO Templates for SWE-bench repos
# =============================================================================
# These templates have repos pre-cloned at /testbed with dependencies installed
REPO_TEMPLATE_MAP = {
    "pallets/flask": "swebench-pallets-flask",
    "psf/requests": "swebench-psf-requests",
    "pytest-dev/pytest": "swebench-pytest-dev-pytest",
    "pylint-dev/pylint": "swebench-pylint-dev-pylint",
    "django/django": "swebench-django-django",
    "sympy/sympy": "swebench-sympy-sympy",
    "sphinx-doc/sphinx": "swebench-sphinx-doc-sphinx",
    "matplotlib/matplotlib": "swebench-matplotlib-matplotlib",
    "scikit-learn/scikit-learn": "swebench-scikit-learn-scikit-learn",
    "astropy/astropy": "swebench-astropy-astropy",
    "pydata/xarray": "swebench-pydata-xarray",
    "mwaskom/seaborn": "swebench-mwaskom-seaborn",
}

# Default workdir for pre-built templates
DEFAULT_WORKDIR = "/testbed"  # For pre-built templates
FALLBACK_WORKDIR = "/code"   # For base template (writable directory)


# =============================================================================
# Sandbox Pool Manager (Singleton)
# =============================================================================
import threading
from typing import Dict, List


class SandboxPool:
    """
    Global sandbox pool to avoid 429 rate limit errors.

    Key optimizations:
    1. Use pre-built templates per repo (REPO_TEMPLATE_MAP)
    2. Per-repo sandbox pools for efficient reuse
    3. Exponential backoff retry for rate limits
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # Per-repo sandbox pools: repo -> {sandbox_idx -> sandbox}
        self._repo_pools: Dict[str, Dict[int, any]] = {}
        self._pool_lock = threading.Lock()
        self._api_key: str = ""
        self._timeout: int = 3600
        self._pool_size: int = 32  # Pool size per repo (skill recommends 32+)
        self._max_retries: int = 5
        self._base_delay: float = 2.0  # Base delay for exponential backoff
        self._default_template: str = os.environ.get("PPIO_SANDBOX_TEMPLATE", "base")

    def configure(self, api_key: str, pool_size: int = 32, timeout: int = 3600, template: str = None):
        """Configure the pool parameters."""
        self._api_key = api_key
        self._pool_size = pool_size
        self._timeout = timeout
        if template:
            self._default_template = template
        print(f"[SandboxPool] Configured: pool_size={pool_size}, timeout={timeout}")

    def get_sandbox(self, trajectory_idx: int, repo: str = "", workdir: str = DEFAULT_WORKDIR):
        """
        Get a sandbox for the given trajectory index and repo.
        Uses per-repo pools with pre-built templates when available.
        """
        # Normalize repo name for template lookup
        normalized_repo = normalize_repo_name(repo)
        # Determine template to use
        template = REPO_TEMPLATE_MAP.get(normalized_repo, self._default_template)
        using_prebuilt = normalized_repo in REPO_TEMPLATE_MAP

        # Use repo-specific pool key
        pool_key = repo if repo else "_default"
        sandbox_idx = trajectory_idx % self._pool_size

        with self._pool_lock:
            # Initialize repo pool if needed
            if pool_key not in self._repo_pools:
                self._repo_pools[pool_key] = {}

            pool = self._repo_pools[pool_key]

            if sandbox_idx in pool and pool[sandbox_idx] is not None:
                sandbox = pool[sandbox_idx]
                # Try to resume if paused
                try:
                    sandbox.connect()
                    print(f"[SandboxPool] Reusing sandbox [{pool_key}][{sandbox_idx}] for trajectory {trajectory_idx}")
                    return sandbox, using_prebuilt
                except Exception as e:
                    print(f"[SandboxPool] Failed to resume sandbox [{pool_key}][{sandbox_idx}]: {e}")
                    pool[sandbox_idx] = None

            # Create new sandbox with retry
            sandbox = self._create_with_retry(sandbox_idx, template, pool_key, workdir)
            pool[sandbox_idx] = sandbox
            return sandbox, using_prebuilt

    def _create_with_retry(self, sandbox_idx: int, template: str, pool_key: str, workdir: str):
        """Create sandbox with exponential backoff retry for rate limits."""
        from ppio_sandbox.core import Sandbox

        last_error = None
        for attempt in range(self._max_retries):
            try:
                sandbox = Sandbox.create(
                    api_key=self._api_key,
                    timeout=self._timeout,
                    template=template
                )
                # Fix permissions and git safe.directory for pre-built templates
                sandbox.commands.run(f"chmod -R 777 {workdir} 2>/dev/null || true", timeout=30)
                sandbox.commands.run(f"git config --global --add safe.directory {workdir}", timeout=10)
                print(f"[SandboxPool] Created sandbox [{pool_key}][{sandbox_idx}] with template={template} (attempt {attempt + 1})")
                return sandbox
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    delay = self._base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[SandboxPool] Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{self._max_retries})")
                    time.sleep(delay)
                else:
                    print(f"[SandboxPool] Sandbox creation failed: {e}")
                    raise

        raise RuntimeError(f"Failed to create sandbox after {self._max_retries} retries: {last_error}")

    def release_sandbox(self, trajectory_idx: int, repo: str = "", pause: bool = True):
        """
        Release sandbox back to pool.
        If pause=True, pause the sandbox to save resources.
        """
        pool_key = repo if repo else "_default"
        sandbox_idx = trajectory_idx % self._pool_size

        with self._pool_lock:
            if pool_key not in self._repo_pools:
                return
            pool = self._repo_pools[pool_key]
            if sandbox_idx in pool and pool[sandbox_idx] is not None:
                sandbox = pool[sandbox_idx]
                if pause:
                    try:
                        sandbox.beta_pause()
                        print(f"[SandboxPool] Paused sandbox [{pool_key}][{sandbox_idx}]")
                    except Exception as e:
                        print(f"[SandboxPool] Failed to pause sandbox [{pool_key}][{sandbox_idx}]: {e}")

    def cleanup_all(self):
        """Kill all sandboxes in all pools."""
        with self._pool_lock:
            for pool_key, pool in self._repo_pools.items():
                for idx, sandbox in pool.items():
                    if sandbox is not None:
                        try:
                            sandbox.kill()
                            print(f"[SandboxPool] Killed sandbox [{pool_key}][{idx}]")
                        except:
                            pass
                pool.clear()
            self._repo_pools.clear()
            print(f"[SandboxPool] Cleaned up all sandboxes")

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def active_count(self) -> int:
        with self._pool_lock:
            return sum(1 for s in self._pool.values() if s is not None)


# Global pool instance
_sandbox_pool = SandboxPool()


def get_sandbox_pool() -> SandboxPool:
    """Get the global sandbox pool instance."""
    return _sandbox_pool


# =============================================================================
# PPIO Sandbox Manager
# =============================================================================
class PPIOSandboxManager:
    """Manages PPIO sandbox lifecycle for SWE-bench evaluation"""

    def __init__(self, api_key: str, timeout: int = 3600, workdir: str = DEFAULT_WORKDIR,
                 use_pool: bool = True, trajectory_idx: int = 0, pool_size: int = 32,
                 template: str = None, repo: str = ""):
        self.api_key = api_key
        self.timeout = timeout
        # Auto-select workdir based on template
        # Pre-built templates use /testbed, base template uses /code
        normalized_repo = normalize_repo_name(repo)
        if normalized_repo in REPO_TEMPLATE_MAP:
            self.workdir = workdir  # /testbed for pre-built
        else:
            self.workdir = FALLBACK_WORKDIR  # /code for base template
        self.sandbox = None
        self.use_pool = use_pool
        self.trajectory_idx = trajectory_idx
        self.pool_size = pool_size
        self.repo = repo
        self.using_prebuilt = False  # Will be set when sandbox is created
        # Template: can be "base", repo-specific template, etc.
        self.template = template or REPO_TEMPLATE_MAP.get(repo) or os.environ.get("PPIO_SANDBOX_TEMPLATE", "base")

        # Configure pool if using it
        if use_pool:
            pool = get_sandbox_pool()
            pool.configure(api_key, pool_size=pool_size, timeout=timeout, template=self.template)

    def create_sandbox(self):
        """Create or get a sandbox from pool"""
        if self.use_pool:
            pool = get_sandbox_pool()
            self.sandbox, self.using_prebuilt = pool.get_sandbox(
                self.trajectory_idx, repo=self.repo, workdir=self.workdir
            )
        else:
            # Legacy: create new sandbox directly
            from ppio_sandbox.core import Sandbox
            self.sandbox = Sandbox.create(
                api_key=self.api_key,
                timeout=self.timeout,
                template=self.template
            )
            # Fix permissions and git safe.directory for pre-built templates
            self.sandbox.commands.run(f"chmod -R 777 {self.workdir} 2>/dev/null || true", timeout=30)
            self.sandbox.commands.run(f"git config --global --add safe.directory {self.workdir}", timeout=10)
            self.using_prebuilt = normalize_repo_name(self.repo) in REPO_TEMPLATE_MAP
        return self.sandbox

    def run_command(self, cmd: str, timeout: int = 60) -> tuple[int, str]:
        """Public wrapper for running commands in sandbox"""
        return self._run_command(cmd, timeout)

    def _run_command(self, cmd: str, timeout: int = 60) -> tuple[int, str]:
        """Run command and return (exit_code, output), catching exceptions"""
        try:
            result = self.sandbox.commands.run(cmd, timeout=timeout)
            return result.exit_code, result.stdout
        except Exception as e:
            # CommandExitException includes exit code and error in message
            error_str = str(e)
            # Extract exit code if present
            if "exit code" in error_str.lower():
                return 1, error_str
            return 1, f"Command failed: {error_str}"

    def clone_repo(self, repo_url: str, commit: Optional[str] = None) -> tuple[bool, str]:
        """Clone repository and checkout specific commit.

        If using a pre-built template, the repo is already cloned at workdir.
        Just need to fetch and checkout the specific commit.
        """
        if self.using_prebuilt:
            # Pre-built template: repo already cloned at /testbed
            # Just need to reset and checkout the specific commit
            print(f"[PPIOSandboxManager] Using pre-built template, checking out {commit}")

            # Reset any local changes (hard reset is more reliable for pre-built templates)
            self._run_command(f"cd {self.workdir} && git reset --hard HEAD 2>/dev/null || true", timeout=30)
            self._run_command(f"cd {self.workdir} && git clean -fd 2>/dev/null || true", timeout=30)

            if commit and commit != "HEAD":
                # Fetch to get latest refs
                self._run_command(f"cd {self.workdir} && git fetch origin", timeout=300)
                # Try direct checkout
                exit_code, output = self._run_command(f"cd {self.workdir} && git checkout -f {commit} 2>&1", timeout=60)
                if exit_code != 0:
                    # Fetch the specific commit
                    self._run_command(f"cd {self.workdir} && git fetch origin {commit}", timeout=120)
                    exit_code, output = self._run_command(f"cd {self.workdir} && git checkout -f {commit} 2>&1", timeout=60)
                    if exit_code != 0:
                        return False, f"Checkout failed (exit={exit_code}): {output}"
            return True, "Success (pre-built template)"

        # Not using pre-built template: clone from scratch
        # Clean up workdir first
        # Create workdir first (ensure it exists before cd)
        self._run_command(f"mkdir -p {self.workdir}", timeout=30)
        # Clean up any existing contents
        self._run_command(f"rm -rf {self.workdir}/* {self.workdir}/.[!.]* 2>/dev/null || true", timeout=30)

        if commit and commit != "HEAD":
            # For specific commits, need full clone or fetch
            cmd = f"cd {self.workdir} && git clone {repo_url} . 2>&1"
            exit_code, output = self._run_command(cmd, timeout=300)
            if exit_code != 0:
                return False, f"Clone failed (exit={exit_code}): {output}"
            # Checkout specific commit
            exit_code, output = self._run_command(f"cd {self.workdir} && git checkout {commit} 2>&1", timeout=60)
            if exit_code != 0:
                return False, f"Checkout failed (exit={exit_code}): {output}"
            return True, "Success"
        else:
            cmd = f"cd {self.workdir} && git clone --depth 1 {repo_url} . 2>&1"
            exit_code, output = self._run_command(cmd, timeout=300)
            if exit_code != 0:
                return False, f"Clone failed (exit={exit_code}): {output}"
            return True, "Success"

    def apply_patch(self, patch: str) -> tuple[bool, str]:
        """Apply patch to the repository"""
        import base64

        # Verify we're in a git repo
        exit_code, output = self._run_command(f"cd {self.workdir} && git status 2>&1", timeout=30)
        if exit_code != 0:
            return False, f"Not in a git repository: {output}"

        # Write patch file to workdir (more reliable than /tmp)
        patch_file = f"{self.workdir}/patch.diff"

        # Use base64 encoding to avoid issues with special characters
        patch_b64 = base64.b64encode(patch.encode()).decode()
        exit_code, _ = self._run_command(
            f"echo '{patch_b64}' | base64 -d > {patch_file}",
            timeout=30
        )
        if exit_code != 0:
            # Fallback: try using PPIO files API
            try:
                self.sandbox.files.write(patch_file, patch)
            except Exception as e:
                return False, f"Failed to write patch file: {e}"

        # Try different apply methods
        apply_cmds = [
            f"git apply --verbose {patch_file}",
            f"git apply --verbose --reject {patch_file}",
            f"patch --batch --fuzz=5 -p1 -i {patch_file}",
        ]

        outputs = []
        for cmd in apply_cmds:
            exit_code, result_output = self._run_command(f"cd {self.workdir} && {cmd} 2>&1", timeout=60)
            outputs.append(f"{cmd}: exit={exit_code}, output={result_output[:500]}")
            if exit_code == 0:
                # Clean up patch file
                self._run_command(f"rm -f {patch_file}", timeout=10)
                return True, result_output

        # Clean up patch file even on failure
        self._run_command(f"rm -f {patch_file}", timeout=10)
        return False, "\n".join(outputs)

    def install_deps(self, install_cmd: str = "pip install -e . 2>&1") -> bool:
        """Install dependencies.

        For pre-built templates, dependencies are already installed.
        Just run install again to handle any version-specific requirements.
        """
        if self.using_prebuilt:
            # Pre-built template: try quick reinstall (may fail, that's ok)
            print(f"[PPIOSandboxManager] Pre-built template, running install: {install_cmd}")
        exit_code, output = self._run_command(f"cd {self.workdir} && {install_cmd}", timeout=600)
        return exit_code == 0

    def run_tests(self, test_cmd: str, timeout: int = 1800) -> tuple[int, str]:
        """Run tests and return exit code and output"""
        return self._run_command(f"cd {self.workdir} && {test_cmd} 2>&1", timeout=timeout)

    def cleanup(self, pause: bool = True):
        """Release sandbox back to pool (pause) or kill if not using pool"""
        if self.sandbox:
            if self.use_pool:
                pool = get_sandbox_pool()
                pool.release_sandbox(self.trajectory_idx, repo=self.repo, pause=pause)
            else:
                try:
                    self.sandbox.kill()
                except:
                    pass
            self.sandbox = None


# =============================================================================
# Test Output Parser
# =============================================================================
def parse_pytest_output(output: str, fail_to_pass: list, pass_to_pass: list) -> TestResult:
    """Parse pytest output and categorize test results"""
    # Initialize result tracking
    f2p_success = []
    f2p_failure = []
    p2p_success = []
    p2p_failure = []

    # Parse individual test results
    test_results = {}
    for line in output.split('\n'):
        line = line.strip()
        # Match patterns like "test_file.py::test_name PASSED" or "FAILED"
        if '::' in line and (' PASSED' in line or ' FAILED' in line or ' ERROR' in line):
            parts = line.split()
            if len(parts) >= 2:
                test_name = parts[0]
                status = parts[1] if len(parts) > 1 else "UNKNOWN"
                # Normalize test name (remove module prefix variations)
                test_results[test_name] = status == "PASSED"

    # Categorize tests
    for test in fail_to_pass:
        # Check if test passed (try different name formats)
        passed = False
        for result_name, result_passed in test_results.items():
            if test in result_name or result_name in test:
                passed = result_passed
                break
        if passed:
            f2p_success.append(test)
        else:
            f2p_failure.append(test)

    for test in pass_to_pass:
        # Check if test still passes
        passed = True  # Default to pass if not found
        for result_name, result_passed in test_results.items():
            if test in result_name or result_name in test:
                passed = result_passed
                break
        if passed:
            p2p_success.append(test)
        else:
            p2p_failure.append(test)

    # Calculate totals from parsed output
    passed = sum(1 for v in test_results.values() if v)
    failed = sum(1 for v in test_results.values() if not v)

    # Check if resolved: all FAIL_TO_PASS must pass, no PASS_TO_PASS can fail
    resolved = (len(f2p_failure) == 0 and len(p2p_failure) == 0 and len(f2p_success) > 0)

    return TestResult(
        passed=passed,
        failed=failed,
        errors=0,
        total=passed + failed,
        f2p_success=f2p_success,
        f2p_failure=f2p_failure,
        p2p_success=p2p_success,
        p2p_failure=p2p_failure,
        resolved=resolved,
        test_output=output
    )


# =============================================================================
# Patch Extraction
# =============================================================================
def extract_patch_from_response(response: str) -> str:
    """Extract patch from model response"""
    import re

    # Look for diff format
    patch_start = response.find("diff --git")
    if patch_start == -1:
        # Try alternative formats
        patch_start = response.find("--- a/")
        if patch_start == -1:
            return ""

    # Find end of patch (usually marked by ```)
    patch_end = response.find("```", patch_start)
    if patch_end == -1:
        patch_end = len(response)

    patch = response[patch_start:patch_end].strip()

    # Clean up line number prefixes (e.g., "3: " at start of lines)
    lines = patch.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove line number prefix like "3: " or "123: "
        cleaned = re.sub(r'^\d+:\s?', '', line)
        cleaned_lines.append(cleaned)

    return '\n'.join(cleaned_lines)


# =============================================================================
# Main Reward Function
# =============================================================================
def swebench_ppio_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    Calculate reward for SWE-bench task using PPIO sandbox.

    Args:
        task_info: Dictionary containing:
            - instance_id: Unique identifier for the instance
            - repo: Repository in format "owner/repo"
            - base_commit: Git commit to checkout
            - FAIL_TO_PASS: List of tests that should change from fail to pass
            - PASS_TO_PASS: List of tests that should remain passing
            - test_cmd: Optional pytest command to run (default: auto-detect)
            - install_cmd: Optional install command (default: pip install -e .)
        action: Model's response containing the patch

    Returns:
        RewardOutput with reward (0.0 or 1.0) and metadata
    """
    # Load API key
    api_key = os.environ.get("PPIO_API_KEY")
    if not api_key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("PPIO_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    if not api_key:
        return RewardOutput(reward=0.0, metadata={"error": "PPIO_API_KEY not found"})

    # Extract task info
    instance_id = task_info.get("instance_id", "unknown")
    repo = task_info.get("repo", "")
    base_commit = task_info.get("base_commit", "HEAD")
    fail_to_pass = task_info.get("FAIL_TO_PASS", [])
    pass_to_pass = task_info.get("PASS_TO_PASS", [])
    test_cmd = task_info.get("test_cmd", "pytest -xvs")
    install_cmd = task_info.get("install_cmd", "pip install -e . 2>&1")

    # Extract patch from response
    patch = extract_patch_from_response(action)
    if not patch:
        return RewardOutput(reward=0.0, metadata={
            "instance_id": instance_id,
            "error": "No patch found in response"
        })

    # Create sandbox and run evaluation
    manager = PPIOSandboxManager(api_key=api_key)
    try:
        print(f"[{instance_id}] Creating sandbox...")
        manager.create_sandbox()

        # Clone repo
        repo_url = f"https://github.com/{repo}.git"
        print(f"[{instance_id}] Cloning {repo_url}...")
        success, output = manager.clone_repo(repo_url, base_commit)
        if not success:
            return RewardOutput(reward=0.0, metadata={
                "instance_id": instance_id,
                "error": f"Failed to clone repository: {output}"
            })

        # Apply patch
        print(f"[{instance_id}] Applying patch...")
        success, output = manager.apply_patch(patch)
        if not success:
            return RewardOutput(reward=0.0, metadata={
                "instance_id": instance_id,
                "error": f"Failed to apply patch: {output[:500]}"
            })

        # Install dependencies
        print(f"[{instance_id}] Installing dependencies...")
        if not manager.install_deps(install_cmd):
            return RewardOutput(reward=0.0, metadata={
                "instance_id": instance_id,
                "error": "Failed to install dependencies"
            })

        # Build test command from FAIL_TO_PASS tests
        if fail_to_pass:
            tests_to_run = " ".join(fail_to_pass)
            full_test_cmd = f"{test_cmd} {tests_to_run}"
        else:
            full_test_cmd = test_cmd

        # Run tests
        print(f"[{instance_id}] Running tests: {full_test_cmd}")
        exit_code, test_output = manager.run_tests(full_test_cmd, timeout=1800)

        # Parse results
        result = parse_pytest_output(test_output, fail_to_pass, pass_to_pass)

        # Calculate reward
        # Full resolution = 1.0, partial = proportion of tests passed
        if result.resolved:
            reward = 1.0
        elif len(fail_to_pass) > 0:
            reward = len(result.f2p_success) / len(fail_to_pass)
        else:
            reward = 0.0

        return RewardOutput(reward=reward, metadata={
            "instance_id": instance_id,
            "resolved": result.resolved,
            "f2p_success": result.f2p_success,
            "f2p_failure": result.f2p_failure,
            "p2p_success": result.p2p_success,
            "p2p_failure": result.p2p_failure,
            "test_passed": result.passed,
            "test_failed": result.failed,
            "test_output_preview": test_output[:1000] if test_output else ""
        })

    except Exception as e:
        import traceback
        return RewardOutput(reward=0.0, metadata={
            "instance_id": instance_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        manager.cleanup()


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # Simple test with click repo
    task_info = {
        "instance_id": "pallets__click-test",
        "repo": "pallets/click",
        "base_commit": "HEAD",
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "test_cmd": "pytest tests/test_basic.py -v",
        "install_cmd": "pip install -e . pytest 2>&1",
    }

    # Action contains a simple patch
    action = '''
Here is the fix:

```diff
diff --git a/src/click/core.py b/src/click/core.py
--- a/src/click/core.py
+++ b/src/click/core.py
@@ -1,3 +1,4 @@
+# Test patch applied by PPIO reward function
 from __future__ import annotations

 import collections.abc as cabc
```
'''

    print("Testing PPIO SWE-bench reward function...")
    result = swebench_ppio_reward_fn(task_info, action)
    print(f"\nResult: {result}")


# Mapping of short repo names to full GitHub org/repo format
REPO_NAME_MAP = {
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
    """Convert short repo names to full GitHub org/repo format."""
    if "/" in repo:
        return repo  # Already full format
    return REPO_NAME_MAP.get(repo, repo)
