#!/usr/bin/env python3
"""
Qwen3-32B Baseline Evaluation on SWE-bench using PPIO Sandbox

Usage:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --port 30000

    # Run evaluation:
    python run_qwen32b_eval.py --num-eval 3
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import re
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Clear proxy for PPIO
for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        del os.environ[key]

# Add rllm to path
RLLM_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(RLLM_DIR))

import pandas as pd
from openai import OpenAI
from ppio_sandbox.core import Sandbox

from rllm.environments.swe_ppio.ppio_reward import REPO_TEMPLATE_MAP, DEFAULT_WORKDIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_STEPS = 50
MAX_TOKENS = 4096
TEMPERATURE = 1.0

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
   <parameter=old_str>text to replace (for str_replace)</parameter>
   <parameter=new_str>replacement text (for str_replace)</parameter>
   <parameter=file_text>file content (for create)</parameter>
   </function>

3. finish: Submit your solution when done
   <function=finish>
   </function>

Always use a function call in your response. Working directory is /testbed.
'''


def parse_action(response: str) -> Tuple[str, Dict[str, str]]:
    """Parse R2E-Gym XML function format."""
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


class PPIOEvalAgent:
    """Agent for SWE-bench evaluation using PPIO sandbox."""

    def __init__(self, instance: Dict, sandbox: Sandbox, client: OpenAI, model: str):
        self.instance = instance
        self.sandbox = sandbox
        self.client = client
        self.model = model
        self.history = []
        self.current_step = 0
        self.finished = False

    def run_cmd(self, cmd: str, timeout: int = 120) -> Tuple[int, str]:
        """Run command in sandbox."""
        try:
            result = self.sandbox.commands.run(cmd, timeout=timeout, cwd=DEFAULT_WORKDIR)
            return result.exit_code, result.stdout[:10000] if result.stdout else ""
        except Exception as e:
            return 1, f"Error: {e}"

    def complete(self, messages: List[Dict]) -> str:
        """Call LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def execute_action(self, fn_name: str, params: Dict) -> str:
        """Execute action in sandbox."""
        if fn_name == "execute_bash":
            cmd = params.get("cmd", "")
            exit_code, output = self.run_cmd(cmd)
            return f"Exit code: {exit_code}\n{output}"

        elif fn_name in ("str_replace_editor", "file_editor"):
            command = params.get("command", "view")
            path = params.get("path", "")
            if not path.startswith("/"):
                path = f"{DEFAULT_WORKDIR}/{path}"

            if command == "view":
                view_range = params.get("view_range")
                if view_range:
                    try:
                        import ast
                        start, end = ast.literal_eval(view_range)
                        _, output = self.run_cmd(f"sed -n '{start},{end}p' '{path}'")
                    except:
                        _, output = self.run_cmd(f"cat -n '{path}'")
                else:
                    _, output = self.run_cmd(f"cat -n '{path}'")
                return output[:10000]

            elif command == "str_replace":
                old_str = params.get("old_str", "")
                new_str = params.get("new_str", "")
                # Read file
                _, content = self.run_cmd(f"cat '{path}'")
                if old_str not in content:
                    return "ERROR: old_str not found in file. Make sure to include exact whitespace."
                new_content = content.replace(old_str, new_str, 1)
                # Write file
                encoded = base64.b64encode(new_content.encode()).decode()
                self.run_cmd(f"echo '{encoded}' | base64 -d > '{path}'")
                return f"File {path} edited successfully."

            elif command == "create":
                file_text = params.get("file_text", "")
                encoded = base64.b64encode(file_text.encode()).decode()
                self.run_cmd(f"mkdir -p $(dirname '{path}')")
                self.run_cmd(f"echo '{encoded}' | base64 -d > '{path}'")
                return f"File created: {path}"

            elif command == "insert":
                insert_line = int(params.get("insert_line", 0))
                new_str = params.get("new_str", "")
                _, content = self.run_cmd(f"cat '{path}'")
                lines = content.split('\n')
                lines.insert(insert_line, new_str)
                new_content = '\n'.join(lines)
                encoded = base64.b64encode(new_content.encode()).decode()
                self.run_cmd(f"echo '{encoded}' | base64 -d > '{path}'")
                return f"Inserted at line {insert_line}"

            return f"Unknown editor command: {command}"

        elif fn_name == "finish":
            self.finished = True
            return "Solution submitted."

        return f"Unknown function: {fn_name}"

    def run(self) -> Dict:
        """Run agent loop."""
        problem = self.instance.get("problem_statement", "")
        repo = self.instance.get("repo", "")

        while self.current_step < MAX_STEPS and not self.finished:
            self.current_step += 1
            logger.info(f"Step {self.current_step}/{MAX_STEPS}")

            # Build messages
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            user_msg = f"Repository: {repo}\n\nProblem:\n{problem}\n\nStep {self.current_step}/{MAX_STEPS}"
            messages.append({"role": "user", "content": user_msg})

            for h in self.history[-15:]:  # Keep last 15 exchanges
                messages.append({"role": "assistant", "content": h["action"]})
                messages.append({"role": "user", "content": h["observation"]})

            # Get response
            response = self.complete(messages)
            if not response:
                logger.warning("Empty LLM response")
                break

            # Parse and execute
            fn_name, params = parse_action(response)
            if not fn_name:
                observation = "No function call found. You MUST use a function call (execute_bash, str_replace_editor, or finish)."
            else:
                logger.info(f"Executing: {fn_name}")
                observation = self.execute_action(fn_name, params)

            self.history.append({"action": response, "observation": observation})

        return {"steps": self.current_step, "finished": self.finished, "history": self.history}


def evaluate_instance(instance: Dict, client: OpenAI, model: str, api_key: str) -> Dict:
    """Evaluate a single instance."""
    instance_id = instance.get("instance_id", "unknown")
    repo = instance.get("repo", "")
    commit = instance.get("base_commit", "HEAD")

    logger.info(f"\n{'='*60}\nEvaluating: {instance_id}\n{'='*60}")
    logger.info(f"Repo: {repo}, Commit: {commit[:12]}")

    # Get template
    template = REPO_TEMPLATE_MAP.get(repo, "base")
    logger.info(f"Using template: {template}")

    sandbox = None
    try:
        # Create sandbox
        sandbox = Sandbox.create(api_key=api_key, template=template, timeout=3600)
        logger.info(f"Created sandbox: {sandbox.sandbox_id}")

        # Setup repo
        sandbox.commands.run(f"git config --global --add safe.directory {DEFAULT_WORKDIR}", timeout=10)

        if template == "base":
            # Need to clone
            repo_url = f"https://github.com/{repo}.git"
            logger.info(f"Cloning {repo_url}...")
            result = sandbox.commands.run(f"git clone --depth 100 {repo_url} {DEFAULT_WORKDIR}", timeout=300)
            if result.exit_code != 0:
                return {"instance_id": instance_id, "error": f"Clone failed: {result.stdout}"}

        # Checkout commit (force to avoid conflicts with pre-built template)
        logger.info(f"Checking out {commit[:12]}...")
        # Reset any local changes first, then checkout
        sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && git reset --hard HEAD", timeout=30)
        sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && git clean -fd", timeout=30)
        result = sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && git fetch origin {commit} && git checkout -f {commit}", timeout=120)

        # Install deps (skip for pre-built templates)
        if template != "base":
            logger.info("Using pre-built template, skipping install")
        else:
            install_cmd = instance.get("install_cmd", "pip install -e .")
            logger.info(f"Installing: {install_cmd[:50]}...")
            sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && {install_cmd}", timeout=600)

        # Run agent
        agent = PPIOEvalAgent(instance, sandbox, client, model)
        agent_result = agent.run()

        # Get patch
        result = sandbox.commands.run(f"cd {DEFAULT_WORKDIR} && git diff", timeout=30)
        patch = result.stdout if result.exit_code == 0 else ""

        return {
            "instance_id": instance_id,
            "repo": repo,
            "template": template,
            "steps": agent_result["steps"],
            "finished": agent_result["finished"],
            "has_patch": len(patch.strip()) > 0,
            "patch_length": len(patch),
            "patch": patch[:5000],
        }

    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        return {"instance_id": instance_id, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        if sandbox:
            try:
                sandbox.kill()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Qwen3-32B Baseline Evaluation")
    parser.add_argument("--data", default="/root/develop/tengwan/rft-tinker-ppio/data/resolved_3.jsonl")
    parser.add_argument("--num-eval", type=int, default=3)
    parser.add_argument("--output", default="outputs/qwen32b_eval_results.jsonl")
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("PPIO_API_KEY")
    if not api_key:
        logger.error("PPIO_API_KEY not set")
        return

    # Create OpenAI client
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    # Load data
    if args.data.endswith('.jsonl'):
        with open(args.data) as f:
            instances = [json.loads(line) for line in f]
    else:
        df = pd.read_parquet(args.data)
        instances = df['extra_info'].tolist()
        if isinstance(instances[0], str):
            instances = [json.loads(x) for x in instances]

    instances = instances[:args.num_eval]
    logger.info(f"Evaluating {len(instances)} instances")

    results = []
    for i, instance in enumerate(instances):
        logger.info(f"\nProgress: {i+1}/{len(instances)}")
        result = evaluate_instance(instance, client, args.model, api_key)
        results.append(result)

        # Print result summary
        if "error" in result:
            logger.error(f"FAILED: {result['error'][:100]}")
        else:
            logger.info(f"Result: steps={result['steps']}, finished={result['finished']}, has_patch={result['has_patch']}")

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
    main()
