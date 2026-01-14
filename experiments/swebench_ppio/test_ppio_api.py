#!/usr/bin/env python3
"""
Step 1: PPIO Sandbox API Connectivity Test

Tests basic PPIO sandbox operations required for SWE-bench:
1. Sandbox creation with 1-hour timeout
2. File write/read operations
3. Shell command execution (git, python)
4. Network access (pip install)
"""

import asyncio
import os
import socket
import time
from pathlib import Path

# =============================================================================
# Proxy Configuration (for environments behind HTTP proxy)
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
        """Tunnel through HTTP proxy using CONNECT method."""
        host, port = address

        # Skip proxy for local connections
        if isinstance(host, str) and (
            host in ('localhost', '127.0.0.1') or
            host.startswith('172.') or
            host.startswith('10.') or
            host.startswith('192.168.')
        ):
            return _original_socket_connect(self, address)

        # Connect to proxy first
        _original_socket_connect(self, (proxy_host, proxy_port))

        # Send CONNECT request
        connect_req = f'CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n'
        self.sendall(connect_req.encode())

        # Read response
        response = b''
        while b'\r\n\r\n' not in response:
            chunk = self.recv(1024)
            if not chunk:
                break
            response += chunk

        # Check if tunnel established
        if b'200' not in response.split(b'\r\n')[0]:
            raise ConnectionError(f'Proxy connection failed: {response}')

        return None

    socket.socket.connect = proxy_connect
    print(f"[Proxy] Socket proxy patch installed: {proxy_host}:{proxy_port}")
    return True


# Load API key from .env file
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


# Setup proxy tunnel before importing PPIO
setup_proxy_tunnel()
API_KEY = load_api_key()


def test_sync_api():
    """Test synchronous PPIO API."""
    from ppio_sandbox.core import Sandbox

    print("=" * 60)
    print("PPIO Sandbox API Connectivity Test (Sync)")
    print("=" * 60)

    results = {}

    # Test 1: Create sandbox with 1-hour timeout
    print("\n[Test 1] Creating sandbox with 1-hour timeout...")
    start = time.time()
    try:
        sandbox = Sandbox.create(api_key=API_KEY, timeout=3600)
        sandbox_id = sandbox.sandbox_id
        print(f"  ✓ Sandbox created: {sandbox_id}")
        print(f"  ✓ Time: {time.time() - start:.2f}s")
        results["create_sandbox"] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["create_sandbox"] = False
        return results

    # Test 2: File write operation
    print("\n[Test 2] Writing file to sandbox...")
    try:
        test_content = "Hello from PPIO sandbox test!"
        sandbox.files.write("/tmp/test.txt", test_content)
        print(f"  ✓ File written: /tmp/test.txt")
        results["file_write"] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["file_write"] = False

    # Test 3: File read operation
    print("\n[Test 3] Reading file from sandbox...")
    try:
        content = sandbox.files.read("/tmp/test.txt")
        if content.strip() == test_content:
            print(f"  ✓ File content verified: '{content.strip()}'")
            results["file_read"] = True
        else:
            print(f"  ✗ Content mismatch: got '{content}'")
            results["file_read"] = False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["file_read"] = False

    # Test 4: Shell command - check git
    print("\n[Test 4] Running shell command (git --version)...")
    try:
        result = sandbox.commands.run("git --version")
        print(f"  ✓ Exit code: {result.exit_code}")
        print(f"  ✓ Output: {result.stdout.strip()}")
        results["shell_git"] = result.exit_code == 0
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["shell_git"] = False

    # Test 5: Shell command - check python
    print("\n[Test 5] Running shell command (python3 --version)...")
    try:
        result = sandbox.commands.run("python3 --version")
        print(f"  ✓ Exit code: {result.exit_code}")
        print(f"  ✓ Output: {result.stdout.strip()}")
        results["shell_python"] = result.exit_code == 0
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["shell_python"] = False

    # Test 6: Network access - pip list
    print("\n[Test 6] Testing network access (pip list)...")
    try:
        result = sandbox.commands.run("pip list 2>/dev/null | head -10")
        print(f"  ✓ Exit code: {result.exit_code}")
        print(f"  ✓ Sample packages:\n{result.stdout[:300]}")
        results["network_pip"] = result.exit_code == 0
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["network_pip"] = False

    # Test 7: Git clone (small repo)
    print("\n[Test 7] Testing git clone (small repo)...")
    try:
        result = sandbox.commands.run(
            "cd /tmp && git clone --depth 1 https://github.com/pallets/click.git click_test 2>&1 | tail -5",
            timeout=120
        )
        print(f"  ✓ Exit code: {result.exit_code}")
        print(f"  ✓ Output: {result.stdout.strip()}")

        # Verify clone
        result2 = sandbox.commands.run("ls -la /tmp/click_test | head -10")
        print(f"  ✓ Directory listing:\n{result2.stdout}")
        results["git_clone"] = result.exit_code == 0
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["git_clone"] = False

    # Test 8: Run Python code
    print("\n[Test 8] Running Python code...")
    try:
        code = '''
import sys
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
result = sum(range(100))
print(f"Sum 0-99: {result}")
'''
        sandbox.files.write("/tmp/test_code.py", code)
        result = sandbox.commands.run("python3 /tmp/test_code.py")
        print(f"  ✓ Exit code: {result.exit_code}")
        print(f"  ✓ Output:\n{result.stdout}")
        results["python_code"] = result.exit_code == 0
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["python_code"] = False

    # Test 9: Test pytest availability
    print("\n[Test 9] Checking pytest availability...")
    try:
        result = sandbox.commands.run("pip install pytest -q && pytest --version")
        print(f"  ✓ Exit code: {result.exit_code}")
        print(f"  ✓ Output: {result.stdout.strip()}")
        results["pytest"] = result.exit_code == 0
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["pytest"] = False

    # Clean up
    print("\n[Cleanup] Killing sandbox...")
    try:
        sandbox.kill()
        print("  ✓ Sandbox terminated")
    except Exception as e:
        print(f"  ✗ Cleanup failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, status in results.items():
        print(f"  {'✓' if status else '✗'} {name}")
    print(f"\nTotal: {passed}/{total} tests passed")

    return results


async def test_async_api():
    """Test async PPIO API."""
    from ppio_sandbox.core import AsyncSandbox

    print("\n" + "=" * 60)
    print("PPIO Sandbox API Connectivity Test (Async)")
    print("=" * 60)

    results = {}

    # Test 1: Create async sandbox
    print("\n[Test 1] Creating async sandbox...")
    start = time.time()
    try:
        sandbox = await AsyncSandbox.create(api_key=API_KEY, timeout=3600)
        sandbox_id = sandbox.sandbox_id
        print(f"  ✓ Async sandbox created: {sandbox_id}")
        print(f"  ✓ Time: {time.time() - start:.2f}s")
        results["async_create"] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["async_create"] = False
        return results

    # Test 2: Async command execution
    print("\n[Test 2] Running async command...")
    try:
        result = await sandbox.commands.run("echo 'Async test successful!'")
        print(f"  ✓ Exit code: {result.exit_code}")
        print(f"  ✓ Output: {result.stdout.strip()}")
        results["async_command"] = result.exit_code == 0
    except Exception as e:
        import traceback
        print(f"  ✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["async_command"] = False

    # Test 3: Async file operations
    print("\n[Test 3] Async file operations...")
    try:
        await sandbox.files.write("/tmp/async_test.txt", "Async file content")
        content = await sandbox.files.read("/tmp/async_test.txt")
        print(f"  ✓ File written and read: '{content.strip()}'")
        results["async_files"] = content.strip() == "Async file content"
    except Exception as e:
        import traceback
        print(f"  ✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["async_files"] = False

    # Clean up
    print("\n[Cleanup] Killing async sandbox...")
    try:
        await sandbox.kill()
        print("  ✓ Async sandbox terminated")
    except Exception as e:
        print(f"  ✗ Cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("Async Test Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, status in results.items():
        print(f"  {'✓' if status else '✗'} {name}")
    print(f"\nTotal: {passed}/{total} tests passed")

    return results


def main():
    print("\n" + "#" * 60)
    print("# PPIO Sandbox API Connectivity Test Suite")
    print("#" * 60)

    # Run sync tests
    sync_results = test_sync_api()

    # Run async tests
    print("\n")
    async_results = asyncio.run(test_async_api())

    # Final summary
    all_results = {**sync_results, **async_results}
    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)

    print("\n" + "#" * 60)
    print("# FINAL SUMMARY")
    print("#" * 60)
    print(f"\nAll tests: {passed}/{total} passed")

    if passed == total:
        print("\n✓ All PPIO API tests PASSED - Ready for SWE-bench integration!")
    else:
        print("\n✗ Some tests FAILED - Check errors above")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
