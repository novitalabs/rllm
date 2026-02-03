#!/usr/bin/env python3
"""
Qwen3-32B Baseline Evaluation on SWE-bench Verified using PPIO Sandbox

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --tensor-parallel-size 8 --port 30000

    # Then run evaluation:
    export PPIO_API_KEY=sk_xxxxx
    python eval_qwen32b_baseline.py --num-eval 10
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Clear proxy for PPIO
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)

# Add rllm to path
RLLM_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(RLLM_DIR))

import litellm
from rllm.environments.swe_ppio.ppio_reward import PPIOSandboxManager, REPO_TEMPLATE_MAP, DEFAULT_WORKDIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_STEPS = 50
MAX_TOKENS = 4096
TEMPERATURE = 1.0
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:30000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/Qwen3-32B")


SYSTEM_PROMPT = '''You are a programming agent who is provided a github issue and repository bash environment and is tasked to resolve the issue.

You have access to the following functions:

1. execute_bash: Execute a bash command in the repository
   <function=execute_bash>
   <parameter=cmd>your command here</parameter>
   </function>

2. str_replace_editor: Edit files
   <function=str_replace_editor>
   <parameter=command>view|str_replace|create</parameter>
   <parameter=path>/path/to/file</parameter>
   <parameter=old_str>text to replace</parameter>
   <parameter=new_str>replacement text</parameter>
   </function>

3. finish: Submit your solution
   <function=finish>
   </function>

Always use a function call in your response.
'''


def parse_action(response: str) -> Tuple[str, Dict[str, str]]:
    """Parse R2E-Gym XML function format."""
    import re
    fn_match = re.search(r"<function\s*=\s*([^>]+)>", response)
    if not fn_match:
        return "", {}

    function_name = fn_match.group(1).strip()
    parameters = {}

    param_pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
    for match in re.finditer(param_pattern, response, re.DOTALL):
        key = match.group(1).strip()
        value = match.group(2).strip()
        parameters[key] = value

    return function_name, parameters


class SimpleAgent:
    """Simple agent for baseline evaluation."""

    def __init__(self, instance: Dict, manager: PPIOSandboxManager):
        self.instance = instance
        self.manager = manager
        self.history = []
        self.current_step = 0
        self.finished = False

    async def complete(self, messages: List[Dict]) -> str:
        """Call LLM."""
        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=LLM_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                api_base=LLM_BASE_URL,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def execute_action(self, fn_name: str, params: Dict) -> str:
        """Execute action in sandbox."""
        if fn_name == "execute_bash":
            cmd = params.get("cmd", "")
            exit_code, output = self.manager._run_command(cmd, timeout=120)
            return f"Exit code: {exit_code}\n{output[:5000]}"

        elif fn_name == "str_replace_editor":
            command = params.get("command", "view")
            path = params.get("path", "")
            if not path.startswith("/"):
                path = f"{DEFAULT_WORKDIR}/{path}"

            if command == "view":
                exit_code, output = self.manager._run_command(f"cat '{path}'", timeout=30)
                return output[:10000]
            elif command == "str_replace":
                old_str = params.get("old_str", "")
                new_str = params.get("new_str", "")
                # Read file
                _, content = self.manager._run_command(f"cat '{path}'", timeout=30)
                if old_str not in content:
                    return "ERROR: old_str not found in file"
                new_content = content.replace(old_str, new_str, 1)
                # Write file
                import base64
                encoded = base64.b64encode(new_content.encode()).decode()
                self.manager._run_command(f"echo '{encoded}' | base64 -d > '{path}'", timeout=30)
                return "Replacement successful"
            elif command == "create":
                file_text = params.get("file_text", "")
                import base64
                encoded = base64.b64encode(file_text.encode()).decode()
                self.manager._run_command(f"mkdir -p $(dirname '{path}')")
                self.manager._run_command(f"echo '{encoded}' | base64 -d > '{path}'")
                return f"File created: {path}"

        elif fn_name == "finish":
            self.finished = True
            return "Finished"

        return f"Unknown function: {fn_name}"

    async def run(self) -> Dict:
        """Run agent loop."""
        problem = self.instance.get("problem_statement", "")

        while self.current_step < MAX_STEPS and not self.finished:
            self.current_step += 1
            logger.info(f"Step {self.current_step}/{MAX_STEPS}")

            # Build messages
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.append({"role": "user", "content": f"Problem:\n{problem}\n\nStep {self.current_step}/{MAX_STEPS}"})
            for h in self.history[-10:]:  # Keep last 10 exchanges
                messages.append({"role": "assistant", "content": h["action"]})
                messages.append({"role": "user", "content": h["observation"]})

            # Get response
            response = await self.complete(messages)
            if not response:
                break

            # Parse and execute
            fn_name, params = parse_action(response)
            if not fn_name:
                observation = "No function call found. Please use a function."
            else:
                observation = self.execute_action(fn_name, params)

            self.history.append({"action": response, "observation": observation})

        return {"steps": self.current_step, "finished": self.finished}


async def evaluate_instance(instance: Dict) -> Dict:
    """Evaluate a single instance."""
    instance_id = instance.get("instance_id", "unknown")
    repo = instance.get("repo", "")
    commit = instance.get("base_commit", "HEAD")

    logger.info(f"Evaluating: {instance_id}")

    api_key = os.environ.get("PPIO_API_KEY")
    if not api_key:
        return {"instance_id": instance_id, "error": "PPIO_API_KEY not set"}

    manager = PPIOSandboxManager(
        api_key=api_key,
        repo=repo,
        use_pool=False,  # Create fresh sandbox for each instance
        timeout=3600,
    )

    try:
        manager.create_sandbox()

        # Clone repo
        repo_url = f"https://github.com/{repo}.git"
        success, output = manager.clone_repo(repo_url, commit)
        if not success:
            return {"instance_id": instance_id, "error": f"Clone failed: {output}"}

        # Install deps
        install_cmd = instance.get("install_cmd", "pip install -e . 2>&1")
        manager.install_deps(install_cmd)

        # Run agent
        agent = SimpleAgent(instance, manager)
        agent_result = await agent.run()

        # Get patch
        exit_code, patch = manager._run_command(f"cd {DEFAULT_WORKDIR} && git diff", timeout=30)

        # Calculate reward (simplified - just check if patch exists)
        has_patch = len(patch.strip()) > 0

        return {
            "instance_id": instance_id,
            "repo": repo,
            "steps": agent_result["steps"],
            "finished": agent_result["finished"],
            "has_patch": has_patch,
            "patch_length": len(patch),
        }

    except Exception as e:
        import traceback
        return {"instance_id": instance_id, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        manager.cleanup(pause=False)


async def main():
    parser = argparse.ArgumentParser(description="Qwen3-32B Baseline Evaluation")
    parser.add_argument("--data", default="/root/develop/tengwan/rllm/data/swe/SWE_Bench_Verified.parquet")
    parser.add_argument("--num-eval", type=int, default=10)
    parser.add_argument("--output", default="outputs/qwen32b_baseline_results.jsonl")
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.data)
    instances = df.to_dict(orient="records")[:args.num_eval]

    logger.info(f"Evaluating {len(instances)} instances")

    results = []
    for i, instance in enumerate(instances):
        logger.info(f"\nProgress: {i+1}/{len(instances)}")
        result = await evaluate_instance(instance)
        results.append(result)
        logger.info(f"Result: {result}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    finished_count = sum(1 for r in results if r.get("finished"))
    has_patch_count = sum(1 for r in results if r.get("has_patch"))
    error_count = sum(1 for r in results if "error" in r)

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {len(results)} total")
    logger.info(f"  Finished: {finished_count}")
    logger.info(f"  Has patch: {has_patch_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
