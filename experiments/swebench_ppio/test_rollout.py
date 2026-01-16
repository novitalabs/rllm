#!/usr/bin/env python3
"""
Complete rollout test with R2E-Gym data:
1. Load real data from parquet
2. Generate patch using LLM
3. Apply patch in PPIO sandbox
4. Run tests to verify

Usage:
    python test_rollout.py --model Qwen/Qwen3-8B
"""

import os
import sys
import json
import time
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rllm.environments.swe_ppio.ppio_reward import (
    PPIOSandboxManager,
    setup_proxy_tunnel,
    extract_patch_from_response,
)


def load_r2e_gym_entry(repo_name: str = "coveragepy"):
    """Load a specific entry from R2E-Gym dataset."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data/swe/R2E_Gym_Subset.parquet"
    )
    df = pd.read_parquet(data_path)

    for _, row in df.iterrows():
        if row['extra_info']['repo'] == repo_name:
            return row['extra_info']
    return None


def build_prompt(entry: dict, code_context: str = "") -> str:
    """Build prompt for the LLM."""
    problem = entry['problem_statement']
    modified_files = entry.get('modified_files', [])

    prompt = f"""You are an expert software engineer. Fix the following issue by providing a git patch.

## Issue Description
{problem}

## Files to modify
{modified_files}

## Relevant Code
```python
{code_context}
```

## Instructions
1. Analyze the issue carefully
2. Provide a fix in the form of a unified diff patch
3. Start your patch with "diff --git"
4. Make minimal changes to fix the issue
5. Include correct line numbers based on the code shown above

## Your Response
Provide the patch below:
"""
    return prompt


def fetch_code_context(manager, modified_files: list, workdir: str = "/home/user/testbed") -> str:
    """Fetch relevant code from modified files."""
    context_parts = []
    for filepath in modified_files[:3]:  # Limit to first 3 files
        full_path = f"{workdir}/{filepath}"
        exit_code, content = manager._run_command(f"cat {full_path} 2>&1 | head -200", timeout=30)
        if exit_code == 0 and content:
            context_parts.append(f"# File: {filepath}\n{content}")
    return "\n\n".join(context_parts)


def generate_with_vllm(prompt: str, model_name: str = "Qwen/Qwen3-8B") -> str:
    """Generate response using vLLM with chat template."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch

    print(f"Loading model {model_name}...")

    # Use tensor parallel for large models (>14B)
    num_gpus = torch.cuda.device_count()
    tp_size = 1
    max_len = 16384
    gpu_mem_util = 0.9

    if "32B" in model_name or "70B" in model_name:
        tp_size = min(4, num_gpus)  # Use up to 4 GPUs for large models
    elif "30B" in model_name:
        # MoE 30B models need 4 GPUs and lower memory settings
        tp_size = min(4, num_gpus)
        max_len = 8192  # Reduce context length to save memory
        gpu_mem_util = 0.85
    elif "14B" in model_name:
        tp_size = min(2, num_gpus)

    print(f"    Using tensor_parallel_size={tp_size} with {num_gpus} GPUs available")
    print(f"    max_model_len={max_len}, gpu_memory_utilization={gpu_mem_util}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=tp_size,
    )

    # Apply chat template for instruct models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )

    print("Generating response...")
    outputs = llm.generate([formatted_prompt], sampling_params)
    return outputs[0].outputs[0].text


def generate_with_openai_api(prompt: str, model_name: str, base_url: str, api_key: str) -> str:
    """Generate response using OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4096,
    )

    return response.choices[0].message.content


def test_in_sandbox(entry: dict, patch: str, api_key: str) -> dict:
    """Test the patch in PPIO sandbox."""
    setup_proxy_tunnel()

    # Map repo names to GitHub paths
    repo_map = {
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

    github_repo = repo_map.get(entry['repo'], entry['repo'])

    manager = PPIOSandboxManager(
        api_key=api_key,
        timeout=1200,  # 20 min for slow model loading
        workdir="/home/user/testbed",
        use_pool=False,
    )

    result = {
        "success": False,
        "clone_ok": False,
        "patch_ok": False,
        "test_ok": False,
        "output": "",
    }

    try:
        print("\n[Sandbox] Creating...", end=" ", flush=True)
        manager.create_sandbox()
        print("OK")

        print(f"[Sandbox] Cloning {github_repo}...", end=" ", flush=True)
        # Use PPIO sandbox internal proxy for faster clone
        proxy_cmd = "export https_proxy=http://10.1.80.46:1181 && export http_proxy=http://10.1.80.46:1181"
        exit_code, output = manager._run_command(
            f"{proxy_cmd} && git clone --depth 1 https://github.com/{github_repo}.git /home/user/testbed 2>&1",
            timeout=300
        )
        if exit_code != 0:
            print(f"proxy clone failed, trying direct...")
            manager._run_command("rm -rf /home/user/testbed", timeout=30)
            exit_code, output = manager._run_command(
                f"git clone --depth 1 https://github.com/{github_repo}.git /home/user/testbed 2>&1",
                timeout=600
            )
        if exit_code != 0:
            print(f"FAILED: {output[:300]}")
            return result
        print("OK")
        result["clone_ok"] = True

        print("[Sandbox] Installing deps...", end=" ", flush=True)
        manager._run_command("cd /home/user/testbed && pip install -e . pytest hypothesis -q 2>&1", timeout=180)
        print("OK")

        # Apply patch
        print("[Sandbox] Applying patch...", end=" ", flush=True)
        success, output = manager.apply_patch(patch)
        if not success:
            print(f"FAILED: {output[:200]}")
            result["output"] = output
            return result
        print("OK")
        result["patch_ok"] = True

        # Run tests
        print("[Sandbox] Running tests...", flush=True)

        # Get test files from expected_output_json
        expected = entry.get('expected_output_json', '')
        if expected:
            try:
                expected_tests = json.loads(expected) if isinstance(expected, str) else expected
                # Extract test class/method names
                test_names = list(expected_tests.keys())[:5]  # First 5 tests
                print(f"    Tests: {test_names}")
            except:
                test_names = []

        # Run pytest
        exit_code, output = manager._run_command(
            "cd /home/user/testbed && python -m pytest tests/ -x -v --tb=short 2>&1 | tail -30",
            timeout=120
        )

        result["output"] = output
        result["test_ok"] = exit_code == 0
        result["success"] = result["patch_ok"] and result["test_ok"]

        print(f"    Exit code: {exit_code}")
        for line in output.strip().split('\n')[-10:]:
            print(f"    {line}")

    except Exception as e:
        result["output"] = str(e)
        print(f"ERROR: {e}")
    finally:
        manager.cleanup(pause=False)

    return result


def run_complete_rollout(entry: dict, args, api_key: str) -> dict:
    """Run complete rollout: setup sandbox, fetch code, generate patch, test."""
    setup_proxy_tunnel()

    # Map repo names to GitHub paths
    repo_map = {
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

    github_repo = repo_map.get(entry['repo'], entry['repo'])

    manager = PPIOSandboxManager(
        api_key=api_key,
        timeout=1200,  # 20 min for slow model loading
        workdir="/home/user/testbed",
        use_pool=False,
    )

    result = {
        "success": False,
        "clone_ok": False,
        "patch_ok": False,
        "test_ok": False,
        "output": "",
    }

    try:
        # Step 1: Create sandbox
        print("    Creating sandbox...", end=" ", flush=True)
        manager.create_sandbox()
        print("OK")

        # Step 2: Clone repo
        print(f"    Cloning {github_repo}...", end=" ", flush=True)
        proxy_cmd = "export https_proxy=http://10.1.80.46:1181 && export http_proxy=http://10.1.80.46:1181"
        exit_code, output = manager._run_command(
            f"{proxy_cmd} && git clone --depth 1 https://github.com/{github_repo}.git /home/user/testbed 2>&1",
            timeout=300
        )
        if exit_code != 0:
            print(f"FAILED: {output[:200]}")
            return result
        print("OK")
        result["clone_ok"] = True

        # Step 3: Fetch code context
        print("    Fetching code context...", end=" ", flush=True)
        modified_files = entry.get('modified_files', [])
        # Convert numpy array to list if needed
        if hasattr(modified_files, 'tolist'):
            modified_files = modified_files.tolist()
        if not modified_files or len(modified_files) == 0:
            # Try to find relevant files from problem statement
            modified_files = ['coverage/debug.py']  # Default for coveragepy
        code_context = fetch_code_context(manager, modified_files)
        print(f"OK ({len(code_context)} chars)")

        # Step 4: Build prompt with code context
        print("\n[3] Building prompt with code context")
        prompt = build_prompt(entry, code_context)
        print(f"    Prompt length: {len(prompt)} chars")

        # Step 5: Generate patch
        print(f"\n[4] Generating patch with {args.model}")
        if args.skip_inference:
            print("    (Using mock patch for testing)")
            response = """
Here's the fix:

```diff
diff --git a/coverage/debug.py b/coverage/debug.py
--- a/coverage/debug.py
+++ b/coverage/debug.py
@@ -141,6 +141,7 @@ def info_formatter(info: Iterable[Tuple[str, Any]]) -> Iterable[str]:
     if not info:
         return

+    info = list(info)
     for label, data in info:
```
"""
        elif args.use_api:
            response = generate_with_openai_api(
                prompt, args.model, args.api_base, args.api_key
            )
        else:
            response = generate_with_vllm(prompt, args.model)

        print(f"    Response length: {len(response)} chars")

        # Step 6: Extract patch
        print("\n[5] Extracting patch")
        patch = extract_patch_from_response(response)
        if not patch:
            print("    ERROR: No patch found in response")
            print(f"    Response preview: {response[:500]}")
            return result

        print(f"    Patch:\n{patch[:500]}")

        # Step 7: Install deps
        print("\n[6] Installing dependencies...", end=" ", flush=True)
        manager._run_command("cd /home/user/testbed && pip install -e . pytest hypothesis -q 2>&1", timeout=180)
        print("OK")

        # Step 8: Apply patch
        print("[7] Applying patch...", end=" ", flush=True)
        success, output = manager.apply_patch(patch)
        if not success:
            print(f"FAILED: {output[:200]}")
            result["output"] = output
            return result
        print("OK")
        result["patch_ok"] = True

        # Step 9: Run tests
        print("[8] Running tests...")
        exit_code, output = manager._run_command(
            "cd /home/user/testbed && python -m pytest tests/ -x -v --tb=short 2>&1 | tail -30",
            timeout=120
        )

        result["output"] = output
        result["test_ok"] = exit_code == 0
        result["success"] = result["patch_ok"] and result["test_ok"]

        print(f"    Exit code: {exit_code}")
        for line in output.strip().split('\n')[-10:]:
            print(f"    {line}")

    except Exception as e:
        import traceback
        result["output"] = str(e)
        print(f"ERROR: {e}")
        traceback.print_exc()
    finally:
        manager.cleanup(pause=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="R2E-Gym rollout test")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--repo", default="coveragepy", help="Repository to test")
    parser.add_argument("--use-api", action="store_true", help="Use API instead of local vLLM")
    parser.add_argument("--api-base", default="", help="API base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference, use mock patch")
    args = parser.parse_args()

    print("=" * 60)
    print("R2E-Gym Complete Rollout Test")
    print("=" * 60)

    # Load entry
    print(f"\n[1] Loading R2E-Gym entry: {args.repo}")
    entry = load_r2e_gym_entry(args.repo)
    if not entry:
        print(f"ERROR: No entry found for {args.repo}")
        return 1

    print(f"    instance_id: {entry['instance_id']}")
    print(f"    commit: {entry['base_commit'][:12]}")
    print(f"    problem: {entry['problem_statement'][:80]}...")

    # Load PPIO API key
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    ppio_key = os.environ.get("PPIO_API_KEY")
    if not ppio_key and os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("PPIO_API_KEY="):
                    ppio_key = line.split("=", 1)[1].strip()

    if not ppio_key:
        print("    ERROR: PPIO_API_KEY not found")
        return 1

    # New flow: Create sandbox first to fetch code context
    print("\n[2] Setting up sandbox and fetching code context")
    result = run_complete_rollout(entry, args, ppio_key)

    # Summary
    print("\n" + "=" * 60)
    print("ROLLOUT RESULT")
    print("=" * 60)
    print(f"Clone:   {'✓' if result['clone_ok'] else '✗'}")
    print(f"Patch:   {'✓' if result['patch_ok'] else '✗'}")
    print(f"Tests:   {'✓' if result['test_ok'] else '✗'}")
    print(f"Success: {'✓' if result['success'] else '✗'}")

    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
