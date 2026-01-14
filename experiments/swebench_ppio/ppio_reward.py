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
# PPIO Sandbox Manager
# =============================================================================
class PPIOSandboxManager:
    """Manages PPIO sandbox lifecycle for SWE-bench evaluation"""

    def __init__(self, api_key: str, timeout: int = 3600, workdir: str = "/home/user/testbed"):
        self.api_key = api_key
        self.timeout = timeout
        self.workdir = workdir
        self.sandbox = None

    def create_sandbox(self):
        """Create a new PPIO sandbox"""
        from ppio_sandbox.core import Sandbox
        self.sandbox = Sandbox.create(api_key=self.api_key, timeout=self.timeout)
        # Create workdir
        self.sandbox.commands.run(f"mkdir -p {self.workdir}", timeout=10)
        return self.sandbox

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
        """Clone repository and checkout specific commit"""
        # Clone with depth 1 if commit is HEAD, otherwise full clone
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
        # Write patch file
        self.sandbox.files.write("/tmp/patch.diff", patch)

        # Verify we're in a git repo
        exit_code, output = self._run_command(f"cd {self.workdir} && git status 2>&1", timeout=30)
        if exit_code != 0:
            return False, f"Not in a git repository: {output}"

        # Try different apply methods
        apply_cmds = [
            "git apply --verbose /tmp/patch.diff",
            "git apply --verbose --reject /tmp/patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]

        outputs = []
        for cmd in apply_cmds:
            exit_code, result_output = self._run_command(f"cd {self.workdir} && {cmd} 2>&1", timeout=60)
            outputs.append(f"{cmd}: exit={exit_code}, output={result_output[:500]}")
            if exit_code == 0:
                return True, result_output
        return False, "\n".join(outputs)

    def install_deps(self, install_cmd: str = "pip install -e . 2>&1") -> bool:
        """Install dependencies"""
        exit_code, _ = self._run_command(f"cd {self.workdir} && {install_cmd}", timeout=600)
        return exit_code == 0

    def run_tests(self, test_cmd: str, timeout: int = 1800) -> tuple[int, str]:
        """Run tests and return exit code and output"""
        return self._run_command(f"cd {self.workdir} && {test_cmd} 2>&1", timeout=timeout)

    def cleanup(self):
        """Kill the sandbox"""
        if self.sandbox:
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

    return response[patch_start:patch_end].strip()


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
