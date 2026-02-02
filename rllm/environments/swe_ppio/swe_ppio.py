#!/usr/bin/env python3
"""
SWE-bench Environment using PPIO Sandbox

A minimal environment implementation that follows the rllm BaseEnv protocol,
using PPIO sandbox for code execution instead of Docker.
"""

import os
import json
import re
import socket
from pathlib import Path
from typing import Any, Optional

from rllm.environments.base.base_env import BaseEnv
from .ppio_reward import (
    PPIOSandboxManager,
    RewardOutput,
    extract_patch_from_response,
    parse_pytest_output,
    setup_proxy_tunnel,
    get_sandbox_pool,
)


# Setup proxy
setup_proxy_tunnel()


# Hardcoded test commands for common repos (from SWE-bench harness)
REPO_TEST_CMDS = {
    "django/django": "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    "sympy/sympy": "bin/test -C --verbose",
    "pytest-dev/pytest": "pytest -rA",
    "matplotlib/matplotlib": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "scikit-learn/scikit-learn": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "astropy/astropy": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "sphinx-doc/sphinx": "tox -e py39 --",
    "pylint-dev/pylint": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "pallets/flask": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "psf/requests": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "pydata/xarray": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "mwaskom/seaborn": "pytest --no-header -rA --tb=no -p no:cacheprovider",
}


def get_test_cmd_for_repo(repo: str) -> str:
    """Get repo-specific test command."""
    return REPO_TEST_CMDS.get(repo, "pytest -xvs")


def convert_test_names_for_django(fail_to_pass):
    """Convert unittest test names to Django module format.

    Django tests use format: test_method(module.TestClass)
    Need to extract module path for runtests.py
    """
    modules = set()
    for test in fail_to_pass:
        match = re.match(r"[^(]+\(([^)]+)\)", test)
        if match:
            full_path = match.group(1)
            parts = full_path.rsplit(".", 1)
            if len(parts) >= 1:
                modules.add(parts[0])
        else:
            modules.add(test)
    return " ".join(sorted(modules))


def parse_fail_to_pass(fail_to_pass):
    """Handle JSON string parsing (some datasets store as string)."""
    if isinstance(fail_to_pass, str):
        try:
            return json.loads(fail_to_pass)
        except json.JSONDecodeError:
            return [fail_to_pass]
    return fail_to_pass if fail_to_pass else []


class SWEBenchPPIOEnv(BaseEnv):
    """SWE-bench Environment using PPIO Sandbox"""

    # Class-level trajectory counter for pool indexing
    _trajectory_counter = 0
    _counter_lock = None

    def __init__(
        self,
        entry: Optional[dict] = None,
        dataset: Optional[Any] = None,
        idx: Optional[int] = None,
        timeout: int = 3600,
        workdir: str = "/home/user/testbed",
        use_pool: bool = True,
        pool_size: int = 16,
    ):
        """Initialize the environment.

        Args:
            entry: Task dictionary with instance info
            dataset: Dataset to select from (optional)
            idx: Index in dataset (optional)
            timeout: Sandbox timeout in seconds
            workdir: Working directory in sandbox
            use_pool: Whether to use sandbox pool (recommended to avoid 429)
            pool_size: Size of sandbox pool for reuse
        """
        import threading
        if SWEBenchPPIOEnv._counter_lock is None:
            SWEBenchPPIOEnv._counter_lock = threading.Lock()

        self.entry = entry
        self.dataset = dataset
        self._idx = idx
        self.timeout = timeout
        self.workdir = workdir
        self.use_pool = use_pool
        self.pool_size = pool_size

        # Assign trajectory index from counter
        with SWEBenchPPIOEnv._counter_lock:
            self.trajectory_idx = SWEBenchPPIOEnv._trajectory_counter
            SWEBenchPPIOEnv._trajectory_counter += 1

        self.sandbox_manager: Optional[PPIOSandboxManager] = None
        self.api_key = self._load_api_key()
        self.task_instruction = ""
        self.total_steps = 0

    @property
    def idx(self) -> Any:
        return self._idx

    @idx.setter
    def idx(self, value: Any):
        self._idx = value

    def _load_api_key(self) -> str:
        """Load PPIO API key"""
        api_key = os.environ.get("PPIO_API_KEY")
        if api_key:
            return api_key

        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("PPIO_API_KEY="):
                        return line.split("=", 1)[1].strip()
        raise ValueError("PPIO_API_KEY not found")

    def reset(self) -> tuple[dict, dict]:
        """Reset the environment to initial state.

        Returns:
            Tuple of (observation_dict, info_dict)
        """
        # Release sandbox back to pool (pause, don't destroy)
        if self.sandbox_manager:
            self.sandbox_manager.cleanup(pause=True)
            self.sandbox_manager = None

        # Get sandbox from pool (reuses existing or creates new)
        self.sandbox_manager = PPIOSandboxManager(
            api_key=self.api_key,
            timeout=self.timeout,
            workdir=self.workdir,
            use_pool=self.use_pool,
            trajectory_idx=self.trajectory_idx,
            pool_size=self.pool_size,
        )
        self.sandbox_manager.create_sandbox()

        # Clone repository if entry is provided
        if self.entry:
            repo = self.entry.get("repo", "")
            base_commit = self.entry.get("base_commit", "HEAD")

            if repo:
                repo_url = f"https://github.com/{repo}.git"
                print(f"[reset] Cloning {repo_url} at {base_commit}...")
                try:
                    success, output = self.sandbox_manager.clone_repo(repo_url, base_commit)
                    if not success:
                        raise RuntimeError(f"Failed to clone repository: {output}")
                    print(f"[reset] Clone successful")
                except Exception as e:
                    print(f"[reset] Clone failed: {e}")
                    raise

            # Install dependencies
            install_cmd = self.entry.get("install_cmd", "pip install -e . 2>&1")
            self.sandbox_manager.install_deps(install_cmd)

            # Build task instruction
            problem_statement = self.entry.get("problem_statement", "")
            self.task_instruction = f"""You are a software engineer working on a bug fix.

Repository: {repo}
Commit: {base_commit}

Issue Description:
{problem_statement}

Please analyze the issue and provide a fix in the form of a git patch.
Your response should include a patch in unified diff format starting with "diff --git".
"""

        self.total_steps = 0

        return {"task_instruction": self.task_instruction}, {
            "instance_id": self.entry.get("instance_id", "unknown") if self.entry else "unknown",
            "repo": self.entry.get("repo", "") if self.entry else "",
        }

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute an action in the environment.

        For SWE-bench, the action is typically a patch that gets applied and tested.

        Args:
            action: The model's response containing a patch

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.total_steps += 1

        # Extract patch from action
        patch = extract_patch_from_response(action)
        if not patch:
            return "No valid patch found in response.", 0.0, False, {
                "error": "No patch found"
            }

        # Apply patch
        success, output = self.sandbox_manager.apply_patch(patch)
        if not success:
            return f"Failed to apply patch: {output}", 0.0, False, {
                "error": "Patch application failed",
                "output": output[:500]
            }

        # Run tests
        fail_to_pass = self.entry.get("FAIL_TO_PASS", []) if self.entry else []
        pass_to_pass = self.entry.get("PASS_TO_PASS", []) if self.entry else []

        # Parse JSON strings if needed (some datasets store as string)
        fail_to_pass = parse_fail_to_pass(fail_to_pass)
        pass_to_pass = parse_fail_to_pass(pass_to_pass)

        # Get repo-specific test command
        repo = self.entry.get("repo", "") if self.entry else ""
        test_cmd = get_test_cmd_for_repo(repo)
        is_django = "runtests.py" in test_cmd

        # Build test command
        if fail_to_pass:
            if is_django:
                tests_to_run = convert_test_names_for_django(fail_to_pass)
            else:
                tests_to_run = " ".join(fail_to_pass)
            full_test_cmd = f"{test_cmd} {tests_to_run}"
        else:
            full_test_cmd = test_cmd

        exit_code, test_output = self.sandbox_manager.run_tests(full_test_cmd, timeout=1800)

        # Parse results
        result = parse_pytest_output(test_output, fail_to_pass, pass_to_pass)

        # Calculate reward
        if result.resolved:
            reward = 1.0
        elif len(fail_to_pass) > 0:
            reward = len(result.f2p_success) / len(fail_to_pass)
        else:
            reward = 0.0

        # Environment is done after one patch attempt
        done = True

        observation = f"""Test Results:
- Tests Passed: {result.passed}
- Tests Failed: {result.failed}
- FAIL_TO_PASS Success: {len(result.f2p_success)}/{len(fail_to_pass)}
- PASS_TO_PASS Success: {len(result.p2p_success)}/{len(pass_to_pass)}
- Resolved: {result.resolved}
"""

        info = {
            "resolved": result.resolved,
            "f2p_success": result.f2p_success,
            "f2p_failure": result.f2p_failure,
            "p2p_success": result.p2p_success,
            "p2p_failure": result.p2p_failure,
            "test_passed": result.passed,
            "test_failed": result.failed,
        }

        return observation, reward, done, info

    def compute_final_reward(self) -> float:
        """Compute the final reward for the episode."""
        # For SWE-bench, the step already computes the final reward
        return 0.0

    def close(self):
        """Clean up resources (pause sandbox, keep in pool)."""
        if self.sandbox_manager:
            self.sandbox_manager.cleanup(pause=True)
            self.sandbox_manager = None

    @classmethod
    def cleanup_pool(cls):
        """Clean up all sandboxes in the pool. Call at end of training."""
        pool = get_sandbox_pool()
        pool.cleanup_all()

    @staticmethod
    def from_dict(info: dict) -> "SWEBenchPPIOEnv":
        """Create environment from dictionary."""
        return SWEBenchPPIOEnv(
            entry=info.get("entry"),
            timeout=info.get("timeout", 3600),
            workdir=info.get("workdir", "/home/user/testbed"),
            use_pool=info.get("use_pool", True),
            pool_size=info.get("pool_size", 16),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        """PPIO sandboxes are isolated, so multithread safe."""
        return True
