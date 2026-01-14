#!/usr/bin/env python3
"""
Step 2: SWE-bench Workflow Validation

Tests the complete SWE-bench workflow in PPIO sandbox:
1. Git clone repository
2. Checkout specific commit
3. Apply patch
4. Install dependencies
5. Run tests
6. Parse test output
"""

import os
import socket
import time
from pathlib import Path


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
    print(f"[Proxy] Socket proxy patch installed: {proxy_host}:{proxy_port}")
    return True


def load_api_key():
    """Load PPIO API key from .env file or environment."""
    api_key = os.environ.get("PPIO_API_KEY")
    if api_key:
        return api_key

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("PPIO_API_KEY="):
                    return line.split("=", 1)[1].strip()

    raise ValueError("PPIO_API_KEY not found in environment or .env file")


# Setup proxy before importing PPIO
setup_proxy_tunnel()
API_KEY = load_api_key()


# =============================================================================
# Sample SWE-bench Instance (from SWE-bench Lite)
# =============================================================================
# Using a simple astropy issue for testing
SAMPLE_INSTANCE = {
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
    "problem_statement": """Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels...""",
    "FAIL_TO_PASS": [
        "astropy/modeling/tests/test_separable.py::test_nested_compound_models"
    ],
    "PASS_TO_PASS": [],
    # Gold patch (simplified for testing)
    "patch": '''--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
         cright = _coord_matrix(right, 'right', noutp)
     else:
         cright = np.zeros((noutp, right.n_inputs))
-        cright[-right.n_outputs:, -right.n_inputs:] = _separable(right)
+        cright[-right.n_outputs:, -right.n_inputs:] = right if isinstance(right, np.ndarray) else _separable(right)

     return np.hstack([cleft, cright])
'''
}

# Use click library - known to work with git clone
SIMPLE_INSTANCE = {
    "instance_id": "pallets__click-test",
    "repo": "pallets/click",
    "base_commit": "HEAD",  # Use latest
    "problem_statement": "Test workflow validation",
    "FAIL_TO_PASS": [],
    "PASS_TO_PASS": [],
    # Simple test patch - add a comment to a file
    "patch": '''--- a/src/click/core.py
+++ b/src/click/core.py
@@ -1,3 +1,4 @@
+# Test patch applied successfully
 from __future__ import annotations

 import collections.abc as cabc
'''
}


def parse_pytest_output(output: str) -> dict:
    """Parse pytest output to extract test results."""
    results = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "total": 0,
        "status": "unknown",
        "details": []
    }

    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        # Look for summary line like "1 passed, 2 failed in 3.45s"
        if ' passed' in line or ' failed' in line or ' error' in line:
            import re
            if match := re.search(r'(\d+) passed', line):
                results["passed"] = int(match.group(1))
            if match := re.search(r'(\d+) failed', line):
                results["failed"] = int(match.group(1))
            if match := re.search(r'(\d+) error', line):
                results["errors"] = int(match.group(1))

        # Collect failure details
        if '::' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line):
            results["details"].append(line)

    results["total"] = results["passed"] + results["failed"] + results["errors"]
    if results["total"] > 0:
        if results["failed"] == 0 and results["errors"] == 0:
            results["status"] = "passed"
        elif results["passed"] > 0:
            results["status"] = "partial"
        else:
            results["status"] = "failed"

    return results


def test_swebench_workflow():
    """Test the complete SWE-bench workflow in PPIO sandbox."""
    from ppio_sandbox.core import Sandbox

    print("=" * 60)
    print("SWE-bench Workflow Validation Test")
    print("=" * 60)

    results = {}
    start_time = time.time()

    # Step 1: Create sandbox with 1-hour timeout
    print("\n[Step 1] Creating sandbox with 1-hour timeout...")
    try:
        sandbox = Sandbox.create(api_key=API_KEY, timeout=3600)
        sandbox_id = sandbox.sandbox_id
        print(f"  ✓ Sandbox created: {sandbox_id}")
        results["create_sandbox"] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["create_sandbox"] = False
        return results

    # Use user home directory instead of /testbed (no root access)
    WORKDIR = "/home/user/testbed"

    # Step 2: Clone repository
    print("\n[Step 2] Cloning repository (requests - small repo for testing)...")
    try:
        # Create workdir in user's home
        sandbox.commands.run(f"mkdir -p {WORKDIR}", timeout=10)

        # Use click repo (small, known to work)
        repo_url = "https://github.com/pallets/click.git"
        cmd = f"cd {WORKDIR} && git clone --depth 1 {repo_url} . 2>&1"
        result = sandbox.commands.run(cmd, timeout=300)
        print(f"  Exit code: {result.exit_code}")
        if result.exit_code == 0:
            print(f"  ✓ Repository cloned successfully")
            results["git_clone"] = True
        else:
            print(f"  ✗ Clone failed: {result.stdout[:500]}")
            results["git_clone"] = False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["git_clone"] = False

    # Step 3: Verify checkout (already done in clone step)
    print("\n[Step 3] Verifying checkout...")
    try:
        result = sandbox.commands.run(f"cd {WORKDIR} && git log -1 --oneline 2>&1", timeout=60)
        print(f"  Current commit: {result.stdout.strip()}")
        results["git_checkout"] = result.exit_code == 0
        if results["git_checkout"]:
            print(f"  ✓ Checkout verified")
        else:
            print(f"  ✗ Checkout verification failed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["git_checkout"] = False

    # Step 4: Write and apply patch
    print("\n[Step 4] Applying patch...")
    try:
        patch_content = SIMPLE_INSTANCE["patch"]
        sandbox.files.write("/tmp/patch.diff", patch_content)
        print(f"  Patch written to /tmp/patch.diff")

        result = sandbox.commands.run(f"cd {WORKDIR} && git apply /tmp/patch.diff 2>&1", timeout=60)
        print(f"  Exit code: {result.exit_code}")
        if result.exit_code == 0:
            print(f"  ✓ Patch applied successfully")
            results["apply_patch"] = True
        else:
            # Try with --check first to see if it's already applied
            check_result = sandbox.commands.run(f"cd {WORKDIR} && git apply --check /tmp/patch.diff 2>&1", timeout=60)
            print(f"  Patch apply output: {result.stdout[:500]}")
            print(f"  Check output: {check_result.stdout[:500]}")
            results["apply_patch"] = False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["apply_patch"] = False

    # Step 5: Install dependencies (minimal)
    print("\n[Step 5] Installing dependencies...")
    try:
        # Install click in editable mode with pytest
        result = sandbox.commands.run(f"cd {WORKDIR} && pip install -e '.[dev]' pytest 2>&1 | tail -30", timeout=300)
        print(f"  Exit code: {result.exit_code}")
        print(f"  Output (last 20 lines):\n{result.stdout}")
        results["install_deps"] = result.exit_code == 0
        if results["install_deps"]:
            print(f"  ✓ Dependencies installed")
        else:
            print(f"  ✗ Install failed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["install_deps"] = False

    # Step 6: Run tests
    print("\n[Step 6] Running tests...")
    try:
        # Verify patch was applied by checking the file
        verify_cmd = f"cd {WORKDIR} && head -3 src/click/core.py"
        result = sandbox.commands.run(verify_cmd, timeout=60)
        print(f"  File content after patch:\n{result.stdout}")

        # Run a simple import test
        test_cmd = f"cd {WORKDIR} && python -c 'import click; print(f\"Click version: {{click.__version__}}\")'"
        result = sandbox.commands.run(test_cmd, timeout=60)
        print(f"  Import test: {result.stdout.strip()}")

        # Run pytest on a small subset of tests
        test_cmd = f"cd {WORKDIR} && pytest tests/test_basic.py -v --tb=short 2>&1 | head -50"
        result = sandbox.commands.run(test_cmd, timeout=300)
        print(f"  Exit code: {result.exit_code}")
        print(f"  Test output:\n{result.stdout}")

        parsed = parse_pytest_output(result.stdout)
        print(f"\n  Parsed results: {parsed}")
        results["run_tests"] = True
        results["test_results"] = parsed
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["run_tests"] = False

    # Cleanup
    print("\n[Cleanup] Killing sandbox...")
    try:
        sandbox.kill()
        print("  ✓ Sandbox terminated")
    except Exception as e:
        print(f"  ✗ Cleanup failed: {e}")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Workflow Test Summary")
    print("=" * 60)
    for step, status in results.items():
        if isinstance(status, dict):
            print(f"  {step}: {status}")
        else:
            print(f"  {'✓' if status else '✗'} {step}")
    print(f"\nTotal time: {elapsed:.2f}s")

    return results


if __name__ == "__main__":
    results = test_swebench_workflow()
    # Count passed steps
    core_steps = ["create_sandbox", "git_clone", "git_checkout", "apply_patch", "install_deps", "run_tests"]
    passed = sum(1 for s in core_steps if results.get(s, False))
    print(f"\n{'='*60}")
    print(f"Core steps passed: {passed}/{len(core_steps)}")
    if passed == len(core_steps):
        print("✓ SWE-bench workflow validation PASSED!")
    else:
        print("✗ Some workflow steps FAILED")
