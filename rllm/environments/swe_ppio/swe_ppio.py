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
    DEFAULT_WORKDIR,
    REPO_TEMPLATE_MAP,
    normalize_repo_name,
)
from .swe_ppio_multistep import build_eval_script


# Setup proxy
setup_proxy_tunnel()


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
        workdir: str = DEFAULT_WORKDIR,
        use_pool: bool = True,
        pool_size: int = 32,
    ):
        """Initialize the environment.

        Args:
            entry: Task dictionary with instance info
            dataset: Dataset to select from (optional)
            idx: Index in dataset (optional)
            timeout: Sandbox timeout in seconds
            workdir: Working directory in sandbox (default: /testbed for pre-built templates)
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
        self._last_reward: Optional[float] = None

        # Get repo from entry for template selection
        self.repo = entry.get("repo", "") if entry else ""

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

        # Update repo from entry if available
        if self.entry:
            self.repo = self.entry.get("repo", "")

        # Get sandbox from pool (reuses existing or creates new)
        # Pass repo for pre-built template selection
        self.sandbox_manager = PPIOSandboxManager(
            api_key=self.api_key,
            timeout=self.timeout,
            workdir=self.workdir,
            use_pool=self.use_pool,
            trajectory_idx=self.trajectory_idx,
            pool_size=self.pool_size,
            repo=self.repo,
        )
        self.sandbox_manager.create_sandbox()

        # Clone repository if entry is provided
        if self.entry:
            repo = self.entry.get("repo", "")
            base_commit = self.entry.get("base_commit", "HEAD")

            if repo:
                # Normalize R2E-Gym short names to full GitHub paths
                full_repo = normalize_repo_name(repo)
                repo_url = f"https://github.com/{full_repo}.git"
                # Check both short name and full path for template mapping
                using_prebuilt = repo in REPO_TEMPLATE_MAP or full_repo in REPO_TEMPLATE_MAP
                if using_prebuilt:
                    print(f"[reset] Using pre-built template for {repo} ({full_repo}), checking out {base_commit}...")
                else:
                    print(f"[reset] Cloning {repo_url} at {base_commit}...")
                try:
                    success, output = self.sandbox_manager.clone_repo(repo_url, base_commit)
                    if not success:
                        raise RuntimeError(f"Failed to setup repository: {output}")
                    print(f"[reset] Repository setup successful")
                except Exception as e:
                    print(f"[reset] Repository setup failed: {e}")
                    raise

            # Install dependencies (may be skipped for pre-built templates)
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

        # Parse FAIL_TO_PASS / PASS_TO_PASS for grading (NOT for test command)
        fail_to_pass = self.entry.get("FAIL_TO_PASS", []) if self.entry else []
        pass_to_pass = self.entry.get("PASS_TO_PASS", []) if self.entry else []
        fail_to_pass = parse_fail_to_pass(fail_to_pass)
        pass_to_pass = parse_fail_to_pass(pass_to_pass)
        repo = self.entry.get("repo", "") if self.entry else ""

        # Run eval script (uses test_patch directives, matching Standard flow)
        eval_script, test_cmd = build_eval_script(self.entry)
        eval_script = eval_script.replace("__WORKDIR__", self.sandbox_manager.workdir)

        # Write and run eval script
        import base64
        eval_path = f"{self.sandbox_manager.workdir}/_eval.sh"
        script_b64 = base64.b64encode(eval_script.encode()).decode()
        self.sandbox_manager._run_command(
            f"echo '{script_b64}' | base64 -d > {eval_path} && chmod +x {eval_path}",
            timeout=30
        )
        exit_code, output = self.sandbox_manager._run_command(
            f"cd {self.sandbox_manager.workdir} && bash {eval_path} 2>&1",
            timeout=1800
        )
        self.sandbox_manager._run_command(f"rm -f {eval_path}", timeout=10)

        # Extract test output between markers
        start_marker = ">>>>> Start Test Output"
        end_marker = ">>>>> End Test Output"
        if start_marker in output:
            test_output = output.split(start_marker, 1)[1]
            if end_marker in test_output:
                test_output = test_output.split(end_marker, 1)[0]
        else:
            test_output = output

        # Parse results using repo-specific swebench parser
        result = parse_pytest_output(test_output, fail_to_pass, pass_to_pass, repo=repo)

        # Calculate reward
        if result.resolved:
            reward = 1.0
        elif len(fail_to_pass) > 0:
            reward = len(result.f2p_success) / len(fail_to_pass)
        else:
            reward = 0.0

        # Environment is done after one patch attempt
        done = True

        # Store reward for compute_final_reward()
        self._last_reward = reward

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
        """Compute the final reward for the episode.

        Returns the reward computed during step(), which the execution engine
        uses to assign the trajectory reward.
        """
        if self._last_reward is not None:
            return self._last_reward
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
    def from_dict(info: dict | str) -> "SWEBenchPPIOEnv":
        """Create environment from dictionary.

        Args:
            info: Dictionary containing task data. The entire dict will be used
                  as 'entry', and any keys matching __init__ parameters will be
                  extracted and passed separately.

        Returns:
            Initialized SWEBenchPPIOEnv instance
        """
        import inspect

        if isinstance(info, str):
            info = json.loads(info)

        # Extract init params that match __init__ signature
        sig = inspect.signature(SWEBenchPPIOEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in info:
                init_params[param_name] = info[param_name]

        # Use entire info as entry (matches standard SWEEnv behavior)
        init_params["entry"] = info

        return SWEBenchPPIOEnv(**init_params)

    @staticmethod
    def is_multithread_safe() -> bool:
        """PPIO sandboxes are isolated, so multithread safe."""
        return True
