#!/usr/bin/env python3
"""
SWE-bench Environment using PPIO Sandbox

A minimal environment implementation that follows the rllm BaseEnv protocol,
using PPIO sandbox for code execution instead of Docker.
"""

import os
import socket
from pathlib import Path
from typing import Any, Optional

from ppio_reward import (
    PPIOSandboxManager,
    RewardOutput,
    extract_patch_from_response,
    parse_pytest_output,
    setup_proxy_tunnel,
)


# Setup proxy
setup_proxy_tunnel()


class SWEBenchPPIOEnv:
    """SWE-bench Environment using PPIO Sandbox"""

    def __init__(
        self,
        entry: Optional[dict] = None,
        dataset: Optional[Any] = None,
        idx: Optional[int] = None,
        timeout: int = 3600,
        workdir: str = "/home/user/testbed",
    ):
        """Initialize the environment.

        Args:
            entry: Task dictionary with instance info
            dataset: Dataset to select from (optional)
            idx: Index in dataset (optional)
            timeout: Sandbox timeout in seconds
            workdir: Working directory in sandbox
        """
        self.entry = entry
        self.dataset = dataset
        self._idx = idx
        self.timeout = timeout
        self.workdir = workdir

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

    def reset(self) -> tuple[str, dict]:
        """Reset the environment to initial state.

        Returns:
            Tuple of (task_instruction, info_dict)
        """
        # Close any existing sandbox
        self.close()

        # Create new sandbox
        self.sandbox_manager = PPIOSandboxManager(
            api_key=self.api_key,
            timeout=self.timeout,
            workdir=self.workdir,
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

        return self.task_instruction, {
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
        test_cmd = self.entry.get("test_cmd", "pytest -xvs") if self.entry else "pytest -xvs"

        # Build test command
        if fail_to_pass:
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
        """Clean up resources."""
        if self.sandbox_manager:
            self.sandbox_manager.cleanup()
            self.sandbox_manager = None

    @staticmethod
    def from_dict(info: dict) -> "SWEBenchPPIOEnv":
        """Create environment from dictionary."""
        return SWEBenchPPIOEnv(
            entry=info.get("entry"),
            timeout=info.get("timeout", 3600),
            workdir=info.get("workdir", "/home/user/testbed"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        """PPIO sandboxes are isolated, so multithread safe."""
        return True


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # Test with a simple entry
    entry = {
        "instance_id": "pallets__click-test",
        "repo": "pallets/click",
        "base_commit": "HEAD",
        "problem_statement": "This is a test problem for validation.",
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "test_cmd": "pytest tests/test_basic.py -v --tb=short",
        "install_cmd": "pip install -e . pytest 2>&1",
    }

    print("Testing SWEBenchPPIOEnv...")
    env = SWEBenchPPIOEnv(entry=entry)

    try:
        # Reset environment
        print("\n[1] Resetting environment...")
        task, info = env.reset()
        print(f"Task instruction:\n{task[:500]}...")
        print(f"Info: {info}")

        # Take a step with a simple patch
        print("\n[2] Taking a step with a patch...")
        action = '''
Here is a simple fix:

```diff
diff --git a/src/click/core.py b/src/click/core.py
--- a/src/click/core.py
+++ b/src/click/core.py
@@ -1,3 +1,4 @@
+# Test comment added by PPIO env
 from __future__ import annotations

 import collections.abc as cabc
```
'''
        obs, reward, done, step_info = env.step(action)
        print(f"Observation:\n{obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {step_info}")

    finally:
        # Cleanup
        print("\n[3] Closing environment...")
        env.close()
        print("Done!")
