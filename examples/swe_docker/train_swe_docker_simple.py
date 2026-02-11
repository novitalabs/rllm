#!/usr/bin/env python3
"""
SWE-bench Docker Training Test (Simplified)
Uses Docker directly with R2E-Gym XML function format.
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

# R2E-Gym style system prompt
SYSTEM_PROMPT = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

We have access to the following functions:

-- BEGIN FUNCTION #1: file_editor --
Description:
Custom editing tool for viewing, creating and editing files
  - State is persistent across command calls and discussions with the user
  - If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
  - The create command cannot be used if the specified path already exists as a file
  - If a command generates a long output, it will be truncated and marked with <response clipped>
  - The undo_edit command will revert the last edit made to the file at path

Notes for using the str_replace command:
  - The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
  - If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
  - The new_str parameter should contain the edited lines that should replace the old_str

Parameters:
  1. command (string, required)
Allowed values: [view, create, str_replace, insert, undo_edit]
The command to run.
  2. path (string, required)
Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
  3. file_text (string, optional)
Required for the create command. Contains the content of the file to be created.
  4. old_str (string, optional)
Required for the str_replace command. The exact string in path to replace.
  5. new_str (string, optional)
  - Optional for the str_replace command to specify the replacement string.
  - Required for the insert command to specify the string to insert.
  6. insert_line (integer, optional)
Required for the insert command. The new_str will be inserted after the line number specified here.
  7. view_range (array, optional)
  - Optional for the view command (when path is a file).
  - If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
  - [start_line, -1] will show all lines from start_line to the end of file.
  8. concise (boolean, optional)
  - Optional for the view command.
  - Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

-- END FUNCTION #1 --

-- BEGIN FUNCTION #2: execute_bash --
Description:
Execute a bash command in the terminal.

Behavior notes:
  - If a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
  - If the bash command returns exit code -1, it means the process is still running. The assistant may:
  - Call this function again with command as an empty string ("") to retrieve additional logs.
  - Send more input to STDIN of the running process by calling this function again with command set to the text input.
  - Send command="ctrl+c" to interrupt the currently running process.
  - If the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

Parameters:
  1. cmd (string, required)
The bash command (and optional arguments) to execute.
  - Can be empty ("") to retrieve more logs if the process is still running.
  - Can be "ctrl+c" to interrupt the running process.

-- END FUNCTION #2 --

-- BEGIN FUNCTION #3: search --
Description:
Search for a term in a directory or a single file.
  - If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
  - If path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
  - If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
  - If no matches are found, it will inform you as well.

Parameters:
  1. search_term (string, required)
The term or string to search for in files.
  2. path (string, optional)
The file or directory to search in. Defaults to . if not specified.

-- END FUNCTION #3 --

-- BEGIN FUNCTION #4: finish --
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
  - The submit command finalizes your output.

Parameters:
  1. command (string, required)
Currently allowed value: [submit]
  2. result (string, optional)
The result text or final message to submit. Defaults to an empty string if not provided.

-- END FUNCTION #4 --

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
</IMPORTANT>

REPOSITORY: {repo}
BASE_COMMIT: {base_commit}

ISSUE:
{problem_statement}
"""


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

    def run_tests(self, fail_to_pass: List[str], timeout: int = 300) -> Tuple[str, int, int, int]:
        """Run specific tests and return (output, exit_code, passed, failed)."""
        if not fail_to_pass:
            output, code = self.run("/run_tests.sh", timeout=timeout)
            return output, code, 0, 0

        passed = 0
        failed = 0
        all_output = []

        for test in fail_to_pass:
            test_cmd = f"python -m pytest '{test}' -xvs 2>&1"
            output, code = self.run(test_cmd, timeout=180)
            all_output.append(f"=== Test: {test} ===\n{output}")
            if code == 0:
                passed += 1
            else:
                failed += 1

        return "\n".join(all_output), 0 if failed == 0 else 1, passed, failed

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
            proxies={"http": None, "https": None},  # Disable proxy for localhost
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Remove <think>...</think> tags if present (for Qwen models)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def parse_xml_action(response: str) -> Tuple[str, Dict]:
    """Parse R2E-Gym XML-style action from model response."""
    # Look for <function=name>...</function>
    fn_match = re.search(r'<function=([\w_]+)>(.*?)</function>', response, re.DOTALL)
    if not fn_match:
        return "invalid", {"error": "No valid action found"}

    fn_name = fn_match.group(1).strip()
    fn_body = fn_match.group(2).strip()

    # Parse <parameter=key>value</parameter>
    params = {}
    for param_match in re.finditer(r'<parameter=([\w_]+)>(.*?)</parameter>', fn_body, re.DOTALL):
        key = param_match.group(1).strip()
        value = param_match.group(2).strip()
        params[key] = value

    return fn_name, params


def execute_action(env: SimpleDockerEnv, action_type: str, params: Dict) -> str:
    """Execute action in Docker environment using R2E-Gym format."""
    repo_path = DEFAULT_REPO_PATH

    if action_type == "execute_bash":
        cmd = params.get("cmd", params.get("command", ""))
        if not cmd:
            return "Error: No command provided"
        output, _ = env.run(cmd)
        return output[:8000]

    elif action_type in ("file_editor", "str_replace_editor"):
        command = params.get("command", "view")
        path = params.get("path", "")

        if not path:
            return "Error: path is required"

        # Make path absolute if relative
        if not path.startswith("/"):
            path = f"{repo_path}/{path}"

        if command == "view":
            view_range = params.get("view_range", "")
            if view_range:
                # Parse range like [11, 20] or "11, 20"
                range_match = re.search(r'\[?(\d+),\s*(-?\d+)\]?', view_range)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    if end == -1:
                        output, _ = env.run(f"sed -n '{start},$p' {path} | head -500 | cat -n")
                    else:
                        output, _ = env.run(f"sed -n '{start},{end}p' {path} | cat -n")
                else:
                    output, _ = env.run(f"cat -n {path} | head -500")
            else:
                output, _ = env.run(f"cat -n {path} | head -500")
            return output[:10000]

        elif command == "str_replace":
            old_str = params.get("old_str", "")
            new_str = params.get("new_str", "")

            if not old_str:
                return "Error: old_str is required for str_replace"

            # Read file content
            content_out, code = env.run(f"cat {path}")
            if code != 0:
                return f"Error reading file: {content_out}"

            # Check if old_str exists
            if old_str not in content_out:
                return f"Error: old_str not found in {path}. Make sure it matches exactly including whitespace."

            # Count occurrences
            count = content_out.count(old_str)
            if count > 1:
                return f"Error: old_str appears {count} times in {path}. Please make it unique by including more context."

            # Do replacement
            new_content = content_out.replace(old_str, new_str, 1)

            # Write back using base64 encoding
            content_b64 = base64.b64encode(new_content.encode()).decode()
            env.run(f"echo '{content_b64}' | base64 -d > {path}")
            return f"Successfully replaced text in {path}"

        elif command == "create":
            file_text = params.get("file_text", "")
            env.run(f"mkdir -p $(dirname {path})")
            content_b64 = base64.b64encode(file_text.encode()).decode()
            env.run(f"echo '{content_b64}' | base64 -d > {path}")
            return f"Created {path}"

        elif command == "insert":
            new_str = params.get("new_str", "")
            insert_line = params.get("insert_line", "0")
            try:
                line_num = int(insert_line)
            except:
                return f"Error: invalid insert_line: {insert_line}"

            # Read file, insert, write back
            content_out, _ = env.run(f"cat {path}")
            lines = content_out.split("\n")
            lines.insert(line_num, new_str)
            new_content = "\n".join(lines)
            content_b64 = base64.b64encode(new_content.encode()).decode()
            env.run(f"echo '{content_b64}' | base64 -d > {path}")
            return f"Inserted text after line {line_num} in {path}"

        elif command == "undo_edit":
            output, _ = env.run(f"git checkout -- {path}")
            return f"Reverted changes to {path}"

        return f"Unknown file_editor command: {command}"

    elif action_type == "search":
        search_term = params.get("search_term", "")
        path = params.get("path", ".")

        if not search_term:
            return "Error: search_term is required"

        if not path.startswith("/"):
            path = f"{repo_path}/{path}"

        output, _ = env.run(f"grep -rn '{search_term}' {path} 2>/dev/null | head -100")
        return output[:5000] if output.strip() else f"No matches found for '{search_term}' in {path}"

    elif action_type == "finish":
        return "SUBMIT"

    elif action_type == "invalid":
        return params.get("error", "Invalid action")

    else:
        return f"Unknown action: {action_type}"


def run_rollout(instance: Dict, max_steps: int = 50) -> Dict:
    """Run a single rollout."""
    instance_id = instance.get("instance_id", "unknown")
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")[:8]
    problem_statement = instance.get("problem_statement", "Fix the bug in this repository.")

    logger.info(f"Starting rollout for {instance_id}")

    env = SimpleDockerEnv(instance)

    try:
        env.start()
        logger.info(f"Docker container started for {instance_id}")

        # Build system prompt with instance info
        system_prompt = SYSTEM_PROMPT.format(
            repo=repo,
            base_commit=base_commit,
            problem_statement=problem_statement[:8000]
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        trajectory = []
        done = False
        step = 0

        while not done and step < max_steps:
            step += 1
            logger.info(f"Step {step}/{max_steps}")

            # Generate response
            response = generate_response(messages)
            logger.info(f"Model response (first 500 chars): {response[:500]}")

            # Parse XML action
            action_type, params = parse_xml_action(response)
            logger.info(f"Action: {action_type}, params: {list(params.keys())}")

            # Execute action
            if action_type == "finish" or (action_type == "invalid" and "finish" in response.lower()):
                obs = "Submitting solution..."
                done = True
            else:
                obs = execute_action(env, action_type, params)
                logger.info(f"Observation (first 200): {obs[:200]}...")

            # Update messages
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

            trajectory.append({
                "step": step,
                "action_type": action_type,
                "params": {k: v[:200] for k, v in params.items()},
                "observation": obs[:1000],
            })

        # Run tests
        logger.info("Running tests...")
        fail_to_pass = instance.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except:
                fail_to_pass = [fail_to_pass] if fail_to_pass else []

        test_output, test_code, passed, failed = env.run_tests(fail_to_pass, timeout=300)

        # Calculate reward
        if fail_to_pass:
            reward = passed / len(fail_to_pass) if len(fail_to_pass) > 0 else 0.0
            resolved = passed == len(fail_to_pass)
        else:
            reward = 1.0 if "PASSED" in test_output.upper() or test_code == 0 else 0.0
            resolved = reward > 0

        logger.info(f"Test result: passed={passed}, failed={failed}, reward={reward}, resolved={resolved}")

        # Get patch
        patch = env.get_patch()

        return {
            "instance_id": instance_id,
            "resolved": resolved,
            "final_reward": reward,
            "test_passed": passed,
            "test_failed": failed,
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
            "resolved": False,
            "final_reward": 0.0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        env.stop()


def main():
    global INFERENCE_URL
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./outputs/docker_test")
    parser.add_argument("--inference-url", type=str, default=INFERENCE_URL)
    args = parser.parse_args()

    INFERENCE_URL = args.inference_url

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
            proxies={"http": None, "https": None},
        )
        logger.info(f"Inference server OK: {test_resp.status_code}")
    except Exception as e:
        logger.error(f"Inference server not available: {e}")
        return

    # Run rollouts
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    resolved_count = 0

    for i, instance in enumerate(instances):
        logger.info(f"\n{'='*60}")
        logger.info(f"Instance {i+1}/{len(instances)}: {instance.get('instance_id')}")
        logger.info(f"{'='*60}")

        result = run_rollout(instance, max_steps=args.max_steps)
        results.append(result)

        if result.get("resolved"):
            resolved_count += 1

        logger.info(f"Result: resolved={result.get('resolved')}, reward={result.get('final_reward')}, steps={result.get('num_steps')}")
        logger.info(f"Progress: {resolved_count}/{i+1} resolved ({100*resolved_count/(i+1):.1f}%)")

        # Save result incrementally
        output_file = os.path.join(args.output_dir, "docker_test_result.jsonl")
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS: {resolved_count}/{len(results)} resolved ({100*resolved_count/len(results):.1f}%)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
