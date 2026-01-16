#!/usr/bin/env python3
"""
Test PPIO sandbox with custom template support.

Tests:
1. Load R2E-Gym / SWE-Bench data
2. Create sandbox with template parameter
3. Clone repo and run basic commands
4. Verify template environment (check pre-installed packages)

Usage:
    # Test with default 'base' template
    python test_template.py

    # Test with custom template
    PPIO_SANDBOX_TEMPLATE=r2e-gym-base python test_template.py
"""

import os
import sys
import json
import time

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rllm.environments.swe_ppio.ppio_reward import (
    PPIOSandboxManager,
    setup_proxy_tunnel,
    get_sandbox_pool,
)


def load_test_entry():
    """Load a test entry from SWE-Bench data."""
    import pandas as pd

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data/swe/SWE_Bench_Lite.parquet"
    )

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        # Return a simple test entry
        return {
            "instance_id": "test-entry",
            "repo": "pallets/click",
            "base_commit": "HEAD",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
            "test_cmd": "pytest --version",
            "install_cmd": "pip install -e . 2>&1",
        }

    df = pd.read_parquet(data_path)
    extra = df.iloc[0]["extra_info"]

    # Parse extra_info
    if isinstance(extra, str):
        extra = json.loads(extra)

    return {
        "instance_id": extra.get("instance_id", "unknown"),
        "repo": extra.get("repo", ""),
        "base_commit": extra.get("base_commit", "HEAD"),
        "FAIL_TO_PASS": json.loads(extra.get("FAIL_TO_PASS", "[]")) if isinstance(extra.get("FAIL_TO_PASS"), str) else extra.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": json.loads(extra.get("PASS_TO_PASS", "[]")) if isinstance(extra.get("PASS_TO_PASS"), str) else extra.get("PASS_TO_PASS", []),
    }


def test_template_support():
    """Test sandbox creation with template support."""
    # Setup proxy
    setup_proxy_tunnel()

    # Get template from env or default
    template = os.environ.get("PPIO_SANDBOX_TEMPLATE", "base")
    print(f"\n{'='*60}")
    print(f"Testing PPIO Sandbox with template: {template}")
    print(f"{'='*60}\n")

    # Load API key
    api_key = os.environ.get("PPIO_API_KEY")
    if not api_key:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith("PPIO_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    if not api_key:
        print("ERROR: PPIO_API_KEY not found")
        return False

    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    # Create manager with template
    manager = PPIOSandboxManager(
        api_key=api_key,
        timeout=600,
        workdir="/home/user/testbed",
        use_pool=False,  # Don't use pool for this test
        template=template,
    )

    try:
        # Test 1: Create sandbox
        print("\n[1] Creating sandbox...")
        start = time.time()
        manager.create_sandbox()
        create_time = time.time() - start
        print(f"    Sandbox created in {create_time:.1f}s")

        # Test 2: Check environment
        print("\n[2] Checking environment...")

        # Check Python version
        exit_code, output = manager._run_command("python3 --version", timeout=30)
        print(f"    Python: {output.strip()}")

        # Check if template packages are installed
        packages_to_check = ["pytest", "numpy", "chardet"]
        for pkg in packages_to_check:
            exit_code, output = manager._run_command(f"python3 -c 'import {pkg}; print({pkg}.__version__)'", timeout=30)
            if exit_code == 0:
                print(f"    {pkg}: {output.strip()} ✓")
            else:
                print(f"    {pkg}: NOT INSTALLED")

        # Check uv
        exit_code, output = manager._run_command("which uv || echo 'not found'", timeout=30)
        print(f"    uv: {output.strip()}")

        # Test 3: Clone a small repo
        print("\n[3] Testing git clone...")
        start = time.time()
        success, output = manager.clone_repo("https://github.com/pallets/click.git", "HEAD")
        clone_time = time.time() - start
        if success:
            print(f"    Clone successful in {clone_time:.1f}s ✓")
        else:
            print(f"    Clone failed: {output[:200]}")

        # Test 4: Check repo contents
        print("\n[4] Checking repo contents...")
        exit_code, output = manager._run_command("ls -la /home/user/testbed/", timeout=30)
        files = [f for f in output.split('\n') if f.strip()][:10]
        for f in files:
            print(f"    {f}")

        # Test 5: Install and run pytest
        print("\n[5] Testing pytest...")
        exit_code, output = manager._run_command("cd /home/user/testbed && pip install -e . pytest -q 2>&1 | tail -5", timeout=120)
        print(f"    Install output: {output[:200]}")

        exit_code, output = manager._run_command("cd /home/user/testbed && pytest --version", timeout=30)
        print(f"    Pytest version: {output.strip()}")

        print("\n" + "="*60)
        print("TEST PASSED ✓")
        print("="*60)
        return True

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return False

    finally:
        print("\n[Cleanup] Destroying sandbox...")
        manager.cleanup(pause=False)


def main():
    success = test_template_support()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
