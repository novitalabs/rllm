#!/usr/bin/env python3
"""
SWE-bench Docker Training Test (Simplified)
Uses Docker directly without swebench dependencies.
"""

import json
import logging
import os
import re
import sys
import time
import base64
import requests
import subprocess
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
INFERENCE_URL = "http://localhost:8000/v1/chat/completions"
MAX_TOKENS = 4096
TEMPERATURE = 1.0
DEFAULT_REPO_PATH = "/testbed"

SYSTEM_PROMPT = """You are an expert software engineering agent fixing bugs in a code repository.

YOUR MISSION: Fix the bug and submit your solution within {max_steps} steps.

=== AVAILABLE ACTIONS ===
- bash <command>: Execute shell command in /testbed
- read <filepath>: Read file (first 1000 lines by default)
- read <filepath> <start> <end>: Read specific line range
- search <pattern> [path]: Search for pattern (grep -rn)
- find_file <pattern>: Find files by name (supports *.py wildcards)
- list_dir <path>: List directory contents
- edit <filepath> <start_line> <end_line>
<new_content>: Replace lines start_line to end_line with new_content
- submit: Submit your solution for evaluation

=== OUTPUT FORMAT ===
Thought: <your reasoning about what to do next>
Action: <action_type> <arguments>

=== TIPS ===
- The repository is at /testbed
- Use search/find_file to locate the issue
- Use edit to fix the bug
- Always end with submit action"""


class SimpleDockerEnv:
    """Simple Docker environment using subprocess."""

    def __init__(self, instance: Dict):
        self.instance = instance
        self.instance_id = instance.get("instance_id", "unknown")
        self.container_name = f"swe-test-{self.instance_id.replace('/', '-').replace('__', '-')}"
        self.docker_image = instance.get("docker_image") or f"slimshetty/swebench-verified:sweb.eval.x86_64.{self.instance_id}"
        self.container_id = None

    def start(self):
        """Start Docker container."""
        # Remove existing container if any
        subprocess.run(["docker", "rm", "-f", self.container_name],
                      capture_output=True)

        # Start new container
        result = subprocess.run(
            ["docker", "run", "-d", "--name", self.container_name,
             self.docker_image, "/bin/bash", "-c", "sleep infinity"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
        self.container_id = result.stdout.strip()
        logger.info(f"Container started: {self.container_id[:12]}")

        # Setup
        self.run("chmod +x /run_tests.sh 2>/dev/null || true")

    def run(self, cmd: str, timeout: int = 90) -> Tuple[str, int]:
        """Execute command in container."""
        result = subprocess.run(
            ["docker", "exec", self.container_name, "/bin/bash", "-c",
             f"cd /testbed && timeout {timeout} {cmd}"],
            capture_output=True, text=True, timeout=timeout + 10
        )
        output = result.stdout + result.stderr
        # Clean ANSI codes
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, result.returncode

    def get_patch(self) -> str:
        """Get git diff."""
        output, _ = self.run("git add -A && git diff --cached")
        return output

    def run_tests(self, timeout: int = 300) -> Tuple[str, int]:
        """Run test script."""
        output, code = self.run("/run_tests.sh", timeout=timeout)
        return output, code

    def stop(self):
        """Stop and remove container."""
        subprocess.run(["docker", "rm", "-f", self.container_name],
                      capture_output=True)


def generate_response(messages: List[Dict]) -> str:
    """Call inference server."""
    try:
        response = requests.post(
            INFERENCE_URL,
            json={
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def parse_action(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse model response to extract thought and action."""
    lines = response.strip().split("\n")

    thought = None
    for line in lines:
        if line.strip().lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
            break

    # Find action
    action_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("action:"):
            action_start_idx = i
            break

    if action_start_idx is None:
        return thought, None, None

    action_line = lines[action_start_idx].split(":", 1)[1].strip()
    parts = action_line.split(None, 1)
    if not parts:
        return thought, None, None

    action_type = parts[0].lower()
    first_line_content = parts[1] if len(parts) > 1 else ""

    # For edit actions, capture multiline content
    if action_type == "edit":
        remaining_lines = []
        for line in lines[action_start_idx + 1:]:
            if line.strip().lower().startswith(("thought:", "action:", "observation:")):
                break
            remaining_lines.append(line)
        full_content = first_line_content + "\n" + "\n".join(remaining_lines) if remaining_lines else first_line_content
        return thought, action_type, full_content.strip()

    return thought, action_type, first_line_content


def execute_action(env: SimpleDockerEnv, action_type: str, args: str) -> str:
    """Execute action in Docker environment."""
    repo_path = DEFAULT_REPO_PATH

    if action_type == "bash":
        output, _ = env.run(args)
        return output[:5000]

    elif action_type == "read":
        parts = args.split()
        filepath = parts[0]
        if not filepath.startswith("/"):
            filepath = f"{repo_path}/{filepath}"

        if len(parts) == 3:
            start, end = int(parts[1]), int(parts[2])
            output, _ = env.run(f"sed -n '{start},{end}p' {filepath}")
        else:
            output, _ = env.run(f"head -1000 {filepath}")
        return output[:10000]

    elif action_type == "search":
        parts = args.split(None, 1)
        pattern = parts[0]
        path = parts[1] if len(parts) > 1 else repo_path
        output, _ = env.run(f"grep -rn '{pattern}' {path} 2>/dev/null | head -100")
        return output[:5000]

    elif action_type == "find_file":
        output, _ = env.run(f"find {repo_path} -name '{args}' 2>/dev/null | head -50")
        return output[:2000]

    elif action_type == "list_dir":
        path = args if args else repo_path
        if not path.startswith("/"):
            path = f"{repo_path}/{path}"
        output, _ = env.run(f"ls -la {path}")
        return output[:2000]

    elif action_type == "edit":
        lines = args.split("\n")
        if len(lines) < 2:
            return "Error: edit requires filepath, line range, and new content"

        first_line = lines[0].split()
        if len(first_line) < 3:
            return "Error: edit format: edit <filepath> <start> <end>\\n<new_content>"

        filepath = first_line[0]
        if not filepath.startswith("/"):
            filepath = f"{repo_path}/{filepath}"
        start = int(first_line[1])
        end = int(first_line[2])
        new_content = "\n".join(lines[1:])

        # Write new content to temp file
        content_b64 = base64.b64encode(new_content.encode()).decode()
        env.run(f"echo '{content_b64}' | base64 -d > /tmp/edit_content.txt")

        # Use sed to delete old lines and insert new content
        if start == end:
            env.run(f"sed -i '{start}d' {filepath}")
            env.run(f"sed -i '{start-1}r /tmp/edit_content.txt' {filepath}")
        else:
            env.run(f"sed -i '{start},{end}d' {filepath}")
            if start > 1:
                env.run(f"sed -i '{start-1}r /tmp/edit_content.txt' {filepath}")
            else:
                env.run(f"cat /tmp/edit_content.txt {filepath} > /tmp/temp_file && mv /tmp/temp_file {filepath}")

        return f"Successfully edited {filepath}"

    elif action_type == "submit":
        return "SUBMIT"

    elif action_type == "run_test":
        output, _ = env.run(args, timeout=180)
        return output[:5000]

    else:
        return f"Unknown action: {action_type}"


def run_rollout(instance: Dict, max_steps: int = 50) -> Dict:
    """Run a single rollout."""
    instance_id = instance.get("instance_id", "unknown")
    logger.info(f"Starting rollout for {instance_id}")

    env = SimpleDockerEnv(instance)

    try:
        env.start()
        logger.info(f"Docker container started for {instance_id}")

        # Get problem statement
        problem_stmt = instance.get("problem_statement", "Fix the bug in this repository.")

        # Initialize conversation
        system_prompt = SYSTEM_PROMPT.format(max_steps=max_steps)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem Statement:\n{problem_stmt}\n\nThe repository is checked out at /testbed. Start by exploring the codebase to understand the issue."}
        ]

        trajectory = []
        done = False
        step = 0

        while not done and step < max_steps:
            logger.info(f"Step {step + 1}/{max_steps}")

            # Generate response
            response = generate_response(messages)
            logger.info(f"Model response (first 500 chars): {response[:500]}")

            # Parse action
            thought, action_type, action_args = parse_action(response)

            if action_type is None:
                obs = "No valid action found. Please use format: Action: <type> <args>"
                logger.warning(f"No valid action in response")
            elif action_type == "submit":
                obs = "Submitting solution..."
                done = True
            else:
                obs = execute_action(env, action_type, action_args or "")
                logger.info(f"Action: {action_type} -> Observation (first 200): {obs[:200]}...")

            # Update messages
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})

            trajectory.append({
                "step": step,
                "thought": thought,
                "action_type": action_type,
                "action_args": action_args,
                "observation": obs[:1000],
            })
            step += 1

        # Run tests and get reward
        logger.info("Running tests...")
        test_output, test_code = env.run_tests(timeout=300)

        # Simple reward: check if tests pass
        # This is simplified - real evaluation uses swebench grading
        reward = 1.0 if "PASSED" in test_output.upper() or test_code == 0 else 0.0
        logger.info(f"Test result: code={test_code}, reward={reward}")

        # Get patch
        patch = env.get_patch()

        return {
            "instance_id": instance_id,
            "final_reward": reward,
            "test_code": test_code,
            "num_steps": step,
            "trajectory": trajectory,
            "patch": patch[:5000] if patch else "",
            "test_output": test_output[:3000] if test_output else "",
        }

    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        import traceback
        return {
            "instance_id": instance_id,
            "final_reward": 0.0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        env.stop()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./outputs/docker_test")
    args = parser.parse_args()

    # Load dataset
    instances = []
    with open(args.data_path, "r") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))

    logger.info(f"Loaded {len(instances)} instances")

    # Verify inference server
    try:
        test_resp = requests.post(
            INFERENCE_URL,
            json={"messages": [{"role": "user", "content": "test"}], "max_tokens": 5},
            timeout=30,
        )
        logger.info(f"Inference server OK: {test_resp.status_code}")
    except Exception as e:
        logger.error(f"Inference server not available: {e}")
        return

    # Run rollouts
    os.makedirs(args.output_dir, exist_ok=True)

    for i, instance in enumerate(instances):
        logger.info(f"\n{'='*60}")
        logger.info(f"Instance {i+1}/{len(instances)}: {instance.get('instance_id')}")
        logger.info(f"{'='*60}")

        result = run_rollout(instance, max_steps=args.max_steps)

        logger.info(f"Result: reward={result.get('final_reward')}, steps={result.get('num_steps')}")

        # Save result
        output_file = os.path.join(args.output_dir, "docker_test_result.jsonl")
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    logger.info("\nTest complete!")


if __name__ == "__main__":
    main()
