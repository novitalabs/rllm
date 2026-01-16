#!/usr/bin/env python3
"""
Test setup time for each R2E-Gym repo in PPIO sandbox.

Measures:
1. Git clone time
2. pip install time (dependencies)
3. Total setup time

Usage:
    python test_repo_setup_time.py [--repos repo1,repo2] [--timeout 600]
"""

import os
import sys
import time
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rllm.environments.swe_ppio.ppio_reward import (
    PPIOSandboxManager,
    setup_proxy_tunnel,
)


# R2E-Gym repos with GitHub paths
REPO_MAP = {
    'coveragepy': 'nedbat/coveragepy',
    'scrapy': 'scrapy/scrapy',
    'tornado': 'tornadoweb/tornado',
    'pandas': 'pandas-dev/pandas',
    'numpy': 'numpy/numpy',
    'pillow': 'python-pillow/Pillow',
    'aiohttp': 'aio-libs/aiohttp',
    'orange3': 'biolab/orange3',
    'datalad': 'datalad/datalad',
    'pyramid': 'Pylons/pyramid',
}


def test_repo_setup(repo_name: str, api_key: str, timeout: int = 600) -> dict:
    """Test setup time for a single repo."""
    github_repo = REPO_MAP.get(repo_name, repo_name)

    result = {
        "repo": repo_name,
        "github": github_repo,
        "sandbox_create_time": 0,
        "clone_time": 0,
        "install_time": 0,
        "total_time": 0,
        "clone_ok": False,
        "install_ok": False,
        "error": None,
    }

    manager = PPIOSandboxManager(
        api_key=api_key,
        timeout=timeout,
        workdir="/home/user/testbed",
        use_pool=False,
    )

    total_start = time.time()

    try:
        # Step 1: Create sandbox
        print(f"  Creating sandbox...", end=" ", flush=True)
        t0 = time.time()
        manager.create_sandbox()
        result["sandbox_create_time"] = time.time() - t0
        print(f"OK ({result['sandbox_create_time']:.1f}s)")

        # Step 2: Clone repo (with PPIO internal proxy)
        print(f"  Cloning {github_repo}...", end=" ", flush=True)
        t0 = time.time()
        proxy_cmd = "export https_proxy=http://10.1.80.46:1181 && export http_proxy=http://10.1.80.46:1181"
        exit_code, output = manager._run_command(
            f"{proxy_cmd} && git clone --depth 1 https://github.com/{github_repo}.git /home/user/testbed 2>&1",
            timeout=300
        )
        result["clone_time"] = time.time() - t0

        if exit_code != 0:
            # Try without proxy
            print(f"proxy failed, trying direct...", end=" ", flush=True)
            manager._run_command("rm -rf /home/user/testbed", timeout=30)
            t0 = time.time()
            exit_code, output = manager._run_command(
                f"git clone --depth 1 https://github.com/{github_repo}.git /home/user/testbed 2>&1",
                timeout=300
            )
            result["clone_time"] = time.time() - t0

        if exit_code != 0:
            result["error"] = f"Clone failed: {output[:200]}"
            print(f"FAILED")
            return result

        result["clone_ok"] = True
        print(f"OK ({result['clone_time']:.1f}s)")

        # Step 3: Install dependencies
        print(f"  Installing dependencies...", end=" ", flush=True)
        t0 = time.time()

        # First install common dependencies
        manager._run_command(
            "pip install pytest pytest-cov hypothesis coverage -q 2>&1",
            timeout=120
        )

        # Then install the package itself
        exit_code, output = manager._run_command(
            "cd /home/user/testbed && pip install -e . -q 2>&1",
            timeout=300
        )
        result["install_time"] = time.time() - t0

        if exit_code != 0:
            # Try with additional flags
            exit_code, output = manager._run_command(
                "cd /home/user/testbed && pip install -e . --no-build-isolation -q 2>&1",
                timeout=300
            )

        result["install_ok"] = (exit_code == 0)
        status = "OK" if result["install_ok"] else "PARTIAL"
        print(f"{status} ({result['install_time']:.1f}s)")

        if not result["install_ok"]:
            result["error"] = f"Install issues: {output[-500:]}"

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}")
    finally:
        result["total_time"] = time.time() - total_start
        manager.cleanup(pause=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test R2E-Gym repo setup times")
    parser.add_argument("--repos", default="", help="Comma-separated repo names (default: all)")
    parser.add_argument("--timeout", type=int, default=600, help="Sandbox timeout in seconds")
    parser.add_argument("--output", default="", help="Output JSON file")
    args = parser.parse_args()

    setup_proxy_tunnel()

    # Load PPIO API key
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    ppio_key = os.environ.get("PPIO_API_KEY")
    if not ppio_key and os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("PPIO_API_KEY="):
                    ppio_key = line.split("=", 1)[1].strip()

    if not ppio_key:
        print("ERROR: PPIO_API_KEY not found")
        return 1

    # Select repos to test
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",")]
    else:
        repos = list(REPO_MAP.keys())

    print("=" * 60)
    print("R2E-Gym Repo Setup Time Test")
    print("=" * 60)
    print(f"Testing {len(repos)} repos: {', '.join(repos)}")
    print()

    results = []

    for i, repo in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] Testing {repo}")
        print("-" * 40)
        result = test_repo_setup(repo, ppio_key, args.timeout)
        results.append(result)

        print(f"  Total: {result['total_time']:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Repo':<15} {'Clone':<10} {'Install':<10} {'Total':<10} {'Status'}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x['total_time'], reverse=True):
        status = "✓" if r['clone_ok'] and r['install_ok'] else "✗"
        print(f"{r['repo']:<15} {r['clone_time']:<10.1f} {r['install_time']:<10.1f} {r['total_time']:<10.1f} {status}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    slow_repos = [r for r in results if r['total_time'] > 60]
    if slow_repos:
        print("Repos needing custom templates (>60s setup):")
        for r in sorted(slow_repos, key=lambda x: x['total_time'], reverse=True):
            print(f"  - {r['repo']}: {r['total_time']:.1f}s")
    else:
        print("All repos setup in <60s, generic template should work.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
