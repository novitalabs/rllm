#!/usr/bin/env python3
"""
SWE-bench Verified Evaluation Script for DeepSWE on Kubernetes
Adapted from swebench_deepswe_eval_v2.py (196.2) — uses R2E-Gym XML function format
with k8s pods instead of PPIO sandbox.
"""

import json
import logging
import os
import re
import sys
import time
import base64
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock
import threading
import random

import openai
from kubernetes import client, config
from kubernetes.stream import stream as k8s_stream

# SWE-bench official grading
from swebench.harness.grading import (
    get_eval_tests_report,
    get_resolution_status,
    ResolvedStatus,
    get_eval_type,
    MAP_REPO_TO_PARSER,
    MAP_REPO_VERSION_TO_SPECS,
)
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    FAIL_TO_PASS as SWE_FAIL_TO_PASS,
    PASS_TO_PASS as SWE_PASS_TO_PASS,
    KEY_INSTANCE_ID,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32768"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "true").lower() == "true"
TESTBED = "/testbed"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://vllm-qwen3-32b:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "DeepSWE-Step164")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
NUM_EVAL = int(os.environ.get("NUM_EVAL", "500"))
START_INDEX = int(os.environ.get("START_INDEX", "0"))
END_INDEX = int(os.environ.get("END_INDEX", "0"))  # 0 = use NUM_EVAL
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "true").lower() == "true"
NODE_SELECTOR = {"workload": "r2e-eval"}
IMAGE_PULL_SECRET = "dockerhub"

# Copied exactly from R2E-Gym edit_non_fn_calling.yaml (via rllm-origin/rllm/agents/r2egym_agent.py)
SYSTEM_PROMPT = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

We have access to the following functions:

\u2013\u2013 BEGIN FUNCTION #1: file_editor \u2013\u2013
Description:
Custom editing tool for viewing, creating and editing files
\t\u2022\tState is persistent across command calls and discussions with the user
\t\u2022\tIf path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
\t\u2022\tThe create command cannot be used if the specified path already exists as a file
\t\u2022\tIf a command generates a long output, it will be truncated and marked with <response clipped>
\t\u2022\tThe undo_edit command will revert the last edit made to the file at path

Notes for using the str_replace command:
\t\u2022\tThe old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
\t\u2022\tIf the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
\t\u2022\tThe new_str parameter should contain the edited lines that should replace the old_str

Parameters:
\t1.\tcommand (string, required)
Allowed values: [view, create, str_replace, insert, undo_edit]
The command to run.
\t2.\tpath (string, required)
Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
\t3.\tfile_text (string, optional)
Required for the create command. Contains the content of the file to be created.
\t4.\told_str (string, optional)
Required for the str_replace command. The exact string in path to replace.
\t5.\tnew_str (string, optional)
\t\u2022\tOptional for the str_replace command to specify the replacement string.
\t\u2022\tRequired for the insert command to specify the string to insert.
\t6.\tinsert_line (integer, optional)
Required for the insert command. The new_str will be inserted after the line number specified here.
\t7.\tview_range (array, optional)
\t\u2022\tOptional for the view command (when path is a file).
\t\u2022\tIf provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
\t\u2022\t[start_line, -1] will show all lines from start_line to the end of file.
\t8.\tconcise (boolean, optional)
\t\u2022\tOptional for the view command.
\t\u2022\tDefaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

\u2013\u2013 END FUNCTION #1 \u2013\u2013

\u2013\u2013 BEGIN FUNCTION #2: execute_bash \u2013\u2013
Description:
Execute a bash command in the terminal.

Behavior notes:
\t\u2022\tIf a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
\t\u2022\tIf the bash command returns exit code -1, it means the process is still running. The assistant may:
\t\u2022\tCall this function again with command as an empty string (\u201c\u201d) to retrieve additional logs.
\t\u2022\tSend more input to STDIN of the running process by calling this function again with command set to the text input.
\t\u2022\tSend command=\u201cctrl+c\u201d to interrupt the currently running process.
\t\u2022\tIf the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

Parameters:
\t1.\tcmd (string, required)
The bash command (and optional arguments) to execute.
\t\u2022\tCan be empty (\u201c\u201d) to retrieve more logs if the process is still running.
\t\u2022\tCan be \u201cctrl+c\u201d to interrupt the running process.

\u2013\u2013 END FUNCTION #2 \u2013\u2013

\u2013\u2013 BEGIN FUNCTION #3: search \u2013\u2013
Description:
Search for a term in a directory or a single file.
\t\u2022\tIf path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
\t\u2022\tIf path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
\t\u2022\tIf more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
\t\u2022\tIf no matches are found, it will inform you as well.

Parameters:
\t1.\tsearch_term (string, required)
The term or string to search for in files.
\t2.\tpath (string, optional)
The file or directory to search in. Defaults to . if not specified.

\u2013\u2013 END FUNCTION #3 \u2013\u2013

\u2013\u2013 BEGIN FUNCTION #4: finish \u2013\u2013
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
\t\u2022\tThe submit command finalizes your output.

Parameters:
\t1.\tcommand (string, required)
Currently allowed value: [submit]
\t2.\tresult (string, optional)
The result text or final message to submit. Defaults to an empty string if not provided.

\u2013\u2013 END FUNCTION #4 \u2013\u2013

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
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task."""

# Copied exactly from R2E-Gym edit_non_fn_calling.yaml (via rllm-origin/rllm/agents/r2egym_agent.py)
USER_PROMPT = """I have uploaded a python code repository in the /testbed directory.

Now consider the following Github issue:

<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I have already taken care of all changes to any of the test files described in the <github_issue>. This means you DON\u2019T have to modify the testing logic or any of the tests in any way! Your task is to make changes to non-test files in the /testbed directory to ensure the <github_issue> is resolved.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <github_issue>.
  - Use efficient search commands to identify key files and functions (i.e. use `grep` instead of `search`).
  - You should err on the side of caution and look at various relevant files and build your understanding of
    - how the code works
    - what are the expected behaviors and edge cases
    - what are the potential root causes for the given issue

2. Assess whether you can reproduce the issue:
   - Create a script at '/testbed/reproduce_issue.py' that demonstrates the error.
   - Execute this script to confirm the error behavior.
   - You should reproduce the issue before fixing it.
   - Your reproduction script should also assert the expected behavior for the fixed code.

3. Analyze the root cause:
   - Identify the underlying problem based on your code exploration and reproduction results.
   - Critically analyze different potential approaches to fix the issue.
   - You NEED to explicitly reason about multiple approaches to fix the issue. Next, find the most elegant and effective solution among them considering the tradeoffs (correctness, generality, side effects, etc.).
   - You would need to reason about execution paths, edge cases, and other potential issues. You should look at the unit tests to understand the expected behavior of the relevant code.

4. Implement your solution:
   - Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
   - You should be thorough and methodical.

5. Verify your solution:
   - Rerun your reproduction script to confirm the error is fixed.
   - If verification fails, iterate on your solution until successful. If you identify the reproduction script is buggy, adjust it as needed.

6. Run unit tests:
    - Find and run the relevant unit tests relevant to the performed fix.
    - You should run the unit tests to ensure your solution is correct and does not cause any regressions.
    - In cases where the unit tests are do not pass, you should consider whether the unit tests does not reflect the *new* expected behavior of the code. If so, you can test it by writing additional edge test cases.
    - Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
       - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
       - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
       - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`
    - RUN ALL relevant unit tests to ensure your solution is correct and does not cause any regressions.
    - DO NOT MODIFY any of the existing unit tests. You can add new edge test cases in a separate file if needed BUT DO NOT MODIFY THE EXISTING TESTS.

7. Test edge cases:
   - Identify potential edge cases that might challenge your solution.
   - Create additional test cases in a separate file '/testbed/edge_case_tests.py'.
   - Execute these tests to verify your solution\u2019s robustness.
   - You should run multiple rounds of edge cases. When creating edge cases:
      - Consider complex scenarios beyond the original issue description
      - Test for regressions to ensure existing functionality remains intact
      - At each round you should write multiple edge test cases in the same file to be efficient

8. Refine if necessary:
   - If edge case testing reveals issues, refine your solution accordingly.
   - Ensure your final implementation handles all identified scenarios correctly.
   - Document any assumptions or limitations of your solution.

9. Submit your solution:
   - Once you have verified your solution, submit your solution using the `finish` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled


Additional recommendations:
- You should be thorough, methodical, and prioritize quality over speed. Be comprehensive.
- You should think carefully before making the tool call about what should be done. However, each step should only use one tool call. YOU SHOULD NOT USE TOOLS INSIDE YOUR THOUGHT PROCESS. YOU SHOULD PRIMARILY USE THINKING FOR IDENTIFYING THE ROOT CAUSE OF THE ISSUE, MAKING THE CHANGES, AND CREATING TEST CASES (REPRODUCTION OR EDGE CASES).
- Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action (e.g., combine multiple bash commands, use sed/grep for bulk operations).
    - Your grep commands should identify both relevant files and line numbers so you can use the file_editor tool.
    - Use grep with `-A -B -C` flags to quickly identify the relevant code blocks during your exploration.
- When exploring the codebase, use targeted search patterns to minimize unnecessary operations.
- When creating edge cases, you should look at the relevant existing tests to understand existing "regression" test cases. Ensure the fix doesn't break existing functionality."""


@dataclass
class ActionResult:
    output: str
    success: bool
    done: bool = False


class K8sPodManager:
    """Manages k8s pods for SWE-Bench instances — thread-safe with retry logic"""

    def __init__(self):
        config.load_incluster_config()
        self._local = threading.local()
        self.namespace = "default"

    @property
    def v1(self):
        """Thread-local k8s CoreV1Api client to avoid shared websocket issues"""
        if not hasattr(self._local, 'v1'):
            self._local.v1 = client.CoreV1Api()
        return self._local.v1

    def _reset_client(self):
        """Force re-create the thread-local client after errors"""
        self._local.v1 = client.CoreV1Api()

    def create_pod(self, instance_id: str, docker_image: str, max_retries: int = 3) -> str:
        """Create a k8s pod from SWE-Bench Docker image, return pod name"""
        import uuid
        pod_name = str(uuid.uuid4())

        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(name=pod_name),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name=pod_name,
                        image=docker_image,
                        command=["/bin/sh", "-c"],
                        args=["/bin/bash -l"],
                        stdin=True,
                        tty=True,
                    )
                ],
                node_selector=NODE_SELECTOR,
                image_pull_secrets=[client.V1LocalObjectReference(name=IMAGE_PULL_SECRET)],
                restart_policy="Never",
            )
        )

        for attempt in range(max_retries):
            try:
                self.v1.create_namespaced_pod(namespace=self.namespace, body=pod)
                logger.info(f"[{instance_id}] Created pod {pod_name}")
                return pod_name
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.random()
                    logger.warning(f"[{instance_id}] create_pod retry {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
                    self._reset_client()
                    time.sleep(delay)
                else:
                    raise

    def wait_for_pod(self, pod_name: str, timeout: int = 600) -> bool:
        """Wait for pod to be Running"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                pod = self.v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
                if pod.status.phase == "Running":
                    return True
                if pod.status.phase in ("Failed", "Succeeded"):
                    return False
            except Exception:
                self._reset_client()
            time.sleep(2)
        return False

    def exec_command(self, pod_name: str, cmd: str, timeout: int = 60, max_retries: int = 5) -> tuple:
        """Execute command in pod with retry, return (stdout, exit_code)"""
        for attempt in range(max_retries):
            try:
                resp = k8s_stream(
                    self.v1.connect_get_namespaced_pod_exec,
                    pod_name, self.namespace,
                    command=["/bin/bash", "-c", cmd + "; echo __EXIT_CODE__=$?"],
                    stderr=True, stdout=True, stdin=False, tty=False,
                    _preload_content=True,
                    _request_timeout=timeout,
                )
                # Parse exit code from output
                lines = resp.rstrip().split("\n")
                exit_code = 1
                output_lines = []
                for line in lines:
                    if line.startswith("__EXIT_CODE__="):
                        try:
                            exit_code = int(line.split("=")[1])
                        except:
                            exit_code = 1
                    else:
                        output_lines.append(line)
                return "\n".join(output_lines), exit_code
            except Exception as e:
                err_str = str(e)
                if "Handshake status" in err_str and attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.random()
                    logger.warning(f"exec_command retry {attempt+1}/{max_retries} on {pod_name}: {err_str[:80]}")
                    self._reset_client()
                    time.sleep(delay)
                else:
                    return f"Error: {e}", 1
        return "Error: max retries exceeded", 1

    def delete_pod(self, pod_name: str, max_retries: int = 3):
        """Delete a pod with retry"""
        for attempt in range(max_retries):
            try:
                self.v1.delete_namespaced_pod(
                    name=pod_name, namespace=self.namespace,
                    body=client.V1DeleteOptions(grace_period_seconds=0),
                )
                return
            except Exception:
                if attempt < max_retries - 1:
                    self._reset_client()
                    time.sleep(1)


class VLLMCompleter:
    """Completer using vLLM OpenAI-compatible API (text mode, no function calling)"""

    def __init__(self, base_url: str, model: str, max_tokens: int = 32768,
                 temperature: float = 1.0, enable_thinking: bool = True):
        self.client = openai.OpenAI(api_key="not-needed", base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def complete(self, messages: List[Dict[str, str]], max_context: int = 65536) -> tuple:
        """Returns (full_text_for_history, content_for_parsing).
        When thinking is enabled, full_text includes <think>...</think> tags for conversation history,
        while content_for_parsing contains only the action text.
        Dynamically caps max_tokens so input + output <= max_context."""
        # Estimate input tokens (~4 chars per token) and cap max_tokens to fit context
        input_chars = sum(len(m.get("content", "")) for m in messages)
        est_input_tokens = input_chars // 3  # conservative estimate
        effective_max_tokens = min(self.max_tokens, max(1024, max_context - est_input_tokens))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=effective_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )
        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        if reasoning:
            full_text = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            full_text = content

        return full_text, content


class SWEBenchAgent:
    """Agent using R2E-Gym XML function format"""

    def __init__(self, instance: Dict[str, Any], pod_mgr: K8sPodManager,
                 pod_name: str, completer: VLLMCompleter):
        self.instance = instance
        self.pod_mgr = pod_mgr
        self.pod_name = pod_name
        self.completer = completer
        self.current_step = 0
        self.messages: List[Dict[str, str]] = []
        self.has_submitted = False
        self.has_edited = False

    def _init_messages(self):
        # System prompt: exact R2EGYM_SYSTEM_PROMPT (no issue embedded)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # First user message: R2EGYM_USER_PROMPT with 9-step workflow + problem statement
        problem_statement = self.instance.get("problem_statement", "")[:8000]
        user_msg = USER_PROMPT.format(problem_statement=problem_statement)
        remaining = MAX_STEPS - 1
        user_msg += f"\nSteps Remaining: {remaining}"
        self.messages.append({"role": "user", "content": user_msg})

    def _parse_action(self, response: str) -> Dict[str, Any]:
        fn_match = re.search(r'<function=([\w_]+)>(.*?)</function>', response, re.DOTALL)
        if not fn_match:
            return {"type": "invalid", "error": "No action found"}
        fn_name = fn_match.group(1)
        fn_body = fn_match.group(2)
        params = {}
        for param_match in re.finditer(r'<parameter=([\w_]+)>(.*?)</parameter>', fn_body, re.DOTALL):
            params[param_match.group(1)] = param_match.group(2).strip()
        return {"type": fn_name, "params": params}

    def _exec(self, cmd: str, timeout: int = 60) -> tuple:
        return self.pod_mgr.exec_command(self.pod_name, cmd, timeout)

    def _execute_action(self, action: Dict[str, Any]) -> ActionResult:
        action_type = action.get("type", "invalid")
        params = action.get("params", {})

        if action_type == "execute_bash":
            cmd = params.get("cmd", params.get("command", ""))
            if not cmd:
                return ActionResult(output="No command provided", success=False)
            output, exit_code = self._exec(f"cd {TESTBED} && {cmd}", timeout=60)
            return ActionResult(output=output[:4000], success=exit_code == 0)

        elif action_type == "file_editor":
            command = params.get("command", "view")
            path = params.get("path", "")
            if path and not path.startswith("/"):
                path = f"{TESTBED}/{path}"

            if command == "view":
                view_range = params.get("view_range", "")
                if view_range:
                    range_match = re.search(r'\[(\d+),\s*(-?\d+)\]', view_range)
                    if range_match:
                        start = int(range_match.group(1))
                        end = int(range_match.group(2))
                        if end == -1:
                            cmd = f"sed -n '{start},$p' {path} | head -200 | cat -n"
                        else:
                            cmd = f"sed -n '{start},{end}p' {path} | cat -n"
                    else:
                        cmd = f"cat -n {path}"
                else:
                    cmd = f"cat -n {path}"
                output, _ = self._exec(cmd, timeout=10)
                return ActionResult(output=output[:8000], success=True)

            elif command == "str_replace":
                old_str = params.get("old_str", "")
                new_str = params.get("new_str", "")
                if not old_str:
                    return ActionResult(output="old_str is required", success=False)
                # Read file, replace, write back using base64 to avoid shell escaping issues
                content_out, _ = self._exec(f"cat {path}", timeout=10)
                if old_str not in content_out:
                    return ActionResult(output=f"old_str not found in {path}", success=False)
                new_content = content_out.replace(old_str, new_str, 1)
                encoded = base64.b64encode(new_content.encode()).decode()
                self._exec(f"echo '{encoded}' | base64 -d > {path}", timeout=10)
                self.has_edited = True
                return ActionResult(output=f"Successfully edited {path}", success=True)

            elif command == "create":
                file_text = params.get("file_text", "")
                self._exec(f"mkdir -p $(dirname {path})", timeout=5)
                encoded = base64.b64encode(file_text.encode()).decode()
                self._exec(f"echo '{encoded}' | base64 -d > {path}", timeout=10)
                self.has_edited = True
                return ActionResult(output=f"Created {path}", success=True)

            elif command == "insert":
                new_str = params.get("new_str", "")
                insert_line = params.get("insert_line", "0")
                content_out, _ = self._exec(f"cat {path}", timeout=10)
                lines = content_out.split("\n")
                line_num = int(insert_line)
                lines.insert(line_num, new_str)
                new_content = "\n".join(lines)
                encoded = base64.b64encode(new_content.encode()).decode()
                self._exec(f"echo '{encoded}' | base64 -d > {path}", timeout=10)
                self.has_edited = True
                return ActionResult(output=f"Inserted at line {line_num} in {path}", success=True)

            return ActionResult(output=f"Unknown file_editor command: {command}", success=False)

        elif action_type == "search":
            search_term = params.get("search_term", "")
            path = params.get("path", ".")
            if not path.startswith("/"):
                path = f"{TESTBED}/{path}"
            output, _ = self._exec(
                f"cd {TESTBED} && grep -rn '{search_term}' {path} 2>/dev/null | head -50",
                timeout=30
            )
            return ActionResult(output=output[:4000] or "No matches found", success=True)

        elif action_type == "finish":
            self.has_submitted = True
            return ActionResult(output="Submission received.", success=True, done=True)

        elif action_type == "invalid":
            return ActionResult(output=f"Invalid action: {action.get('error', 'unknown')}", success=False)

        return ActionResult(output=f"Unknown action: {action_type}", success=False)

    def run(self) -> Dict[str, Any]:
        instance_id = self.instance.get("instance_id", "unknown")
        self._init_messages()

        while self.current_step < MAX_STEPS and not self.has_submitted:
            self.current_step += 1
            logger.info(f"[{instance_id}] Step {self.current_step}/{MAX_STEPS}")

            try:
                full_text, action_text = self.completer.complete(self.messages)
            except Exception as e:
                logger.error(f"[{instance_id}] LLM call failed: {e}")
                break

            # Store full text (with thinking) in conversation history
            self.messages.append({"role": "assistant", "content": full_text})
            # Parse actions from content only (excluding thinking)
            action = self._parse_action(action_text)
            logger.info(f"[{instance_id}] Action: {action.get('type', 'unknown')}")

            result = self._execute_action(action)
            # Match training scaffold: raw observation + Steps Remaining counter
            observation = result.output
            remaining_steps = MAX_STEPS - self.current_step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."
            self.messages.append({"role": "user", "content": observation})

            if result.done:
                break

        return {"steps": self.current_step, "submitted": self.has_submitted, "edited": self.has_edited}


def get_logs_eval_from_string(test_spec: TestSpec, content: str) -> tuple:
    """Parse test output using swebench's repo-specific log parser.
    Adapted from R2E-Gym DockerRuntime.get_logs_eval (string-based, not file-based).
    """
    repo = test_spec.repo
    version = test_spec.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    bad_codes = [x for x in [APPLY_PATCH_FAIL, "RESET_FAILED", "TESTS_ERROR", "TESTS_TIMEOUT"] if x in content]
    if bad_codes:
        logger.warning(f"Bad codes in test output: {bad_codes}")
        return {}, False

    # Split on test command to get just the test output portion
    content = content.split(test_cmd)[-1]
    return log_parser(content, test_spec), True


def run_tests(pod_mgr: K8sPodManager, pod_name: str, instance: Dict[str, Any],
              test_spec: TestSpec) -> Dict[str, Any]:
    """Run /run_tests.sh (baked into Docker image) and grade using official swebench harness."""
    instance_id = instance.get("instance_id", "unknown")

    # Run the official test script baked into the SWE-bench Docker image
    logger.info(f"[{instance_id}] Running /run_tests.sh (official harness)...")
    test_output, exit_code = pod_mgr.exec_command(
        pod_name, "bash /run_tests.sh", timeout=600
    )
    logger.info(f"[{instance_id}] /run_tests.sh exit_code={exit_code}, output_len={len(test_output)}")

    # Parse test output using swebench's repo-specific log parser
    eval_status_map, found = get_logs_eval_from_string(test_spec, test_output)
    if not found:
        logger.warning(f"[{instance_id}] Could not parse test output")
        return {"resolved": False, "error": "parse_failed", "test_output_tail": test_output[-500:]}

    # Grade using official swebench grading
    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        SWE_FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        SWE_PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=get_eval_type(test_spec))
    resolved = get_resolution_status(report) == ResolvedStatus.FULL.value

    f2p_success = len(report.get("FAIL_TO_PASS", {}).get("success", []))
    f2p_failure = len(report.get("FAIL_TO_PASS", {}).get("failure", []))
    p2p_success = len(report.get("PASS_TO_PASS", {}).get("success", []))
    p2p_failure = len(report.get("PASS_TO_PASS", {}).get("failure", []))

    logger.info(f"[{instance_id}] F2P: {f2p_success}/{f2p_success+f2p_failure}, P2P: {p2p_success}/{p2p_success+p2p_failure}, resolved={resolved}")

    return {
        "resolved": resolved,
        "f2p_passed": f2p_success, "f2p_total": f2p_success + f2p_failure,
        "p2p_passed": p2p_success, "p2p_total": p2p_success + p2p_failure,
    }


# Global results tracking
results_lock = Lock()
results = []
resolved_count = 0


def evaluate_instance(instance: Dict[str, Any], completer: VLLMCompleter,
                      pod_mgr: K8sPodManager, existing_ids: set) -> Dict[str, Any]:
    """Evaluate a single instance"""
    global resolved_count
    instance_id = instance.get("instance_id", "unknown")
    docker_image = instance.get("docker_image", "")

    if SKIP_EXISTING and instance_id in existing_ids:
        logger.info(f"[{instance_id}] Skipping (already evaluated)")
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {instance_id} ({docker_image.split('.')[-1]})")

    pod_name = None
    try:
        pod_name = pod_mgr.create_pod(instance_id, docker_image)
        if not pod_mgr.wait_for_pod(pod_name, timeout=600):
            logger.error(f"[{instance_id}] Pod failed to start")
            return {"instance_id": instance_id, "error": "pod_start_failed", "resolved": False}

        # Create test_spec for official grading
        test_spec = make_test_spec(instance)

        agent = SWEBenchAgent(instance, pod_mgr, pod_name, completer)
        agent_result = agent.run()

        if agent_result["submitted"] or agent_result["edited"]:
            test_result = run_tests(pod_mgr, pod_name, instance, test_spec)
        else:
            test_result = {"resolved": False, "error": "no_edits"}

        result = {
            "instance_id": instance_id,
            "resolved": test_result.get("resolved", False),
            "steps": agent_result["steps"],
            "submitted": agent_result["submitted"],
            "edited": agent_result["edited"],
            "test_result": test_result,
        }

        with results_lock:
            results.append(result)
            if result["resolved"]:
                resolved_count += 1
            total = len(results)
            logger.info(f"Progress: {total}/{NUM_EVAL}, Resolved: {resolved_count}/{total} ({100*resolved_count/total:.1f}%)")

        return result

    except Exception as e:
        logger.error(f"[{instance_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"instance_id": instance_id, "error": str(e), "resolved": False}

    finally:
        if pod_name:
            pod_mgr.delete_pod(pod_name)


def main():
    logger.info(f"Config: model={MODEL_NAME}, steps={MAX_STEPS}, tokens={MAX_TOKENS}, temp={TEMPERATURE}, thinking={ENABLE_THINKING}, workers={MAX_WORKERS}, start={START_INDEX}, end={END_INDEX}")

    completer = VLLMCompleter(VLLM_BASE_URL, MODEL_NAME, MAX_TOKENS, TEMPERATURE, ENABLE_THINKING)

    # Test connection
    try:
        _, test = completer.complete([{"role": "user", "content": "Say hello"}])
        logger.info(f"vLLM OK: {test[:80]}...")
    except Exception as e:
        logger.error(f"vLLM connection failed: {e}")
        sys.exit(1)

    # Load dataset
    import datasets
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/app/hf-cache"
    ds = datasets.load_dataset("R2E-Gym/SWE-Bench-Verified", split="test", cache_dir="/app/hf-cache/datasets")
    # Support dataset sharding via START_INDEX/END_INDEX
    end_idx = END_INDEX if END_INDEX > 0 else NUM_EVAL
    end_idx = min(end_idx, len(ds))
    start_idx = max(START_INDEX, 0)
    instances = [dict(ds[i]) for i in range(start_idx, end_idx)]
    logger.info(f"Loaded {len(instances)} instances (index {start_idx}..{end_idx-1})")

    # Load existing results for skip_existing (only skip non-error entries)
    existing_ids = set()
    output_path = os.environ.get("OUTPUT_FILE", f"{OUTPUT_DIR}/eval_step164_custom.jsonl")
    if SKIP_EXISTING and os.path.exists(output_path):
        # Re-write file keeping only non-error entries
        good_lines = []
        with open(output_path) as f:
            for line in f:
                d = json.loads(line.strip())
                if "error" not in d:
                    existing_ids.add(d.get("instance_id"))
                    good_lines.append(line.strip())
        # Overwrite with only good entries
        with open(output_path, "w") as f:
            for line in good_lines:
                f.write(line + "\n")
        logger.info(f"Found {len(existing_ids)} good existing results (removed error entries), will skip")

    pod_mgr = K8sPodManager()

    # File write lock
    file_lock = Lock()

    # Run evaluations with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for inst in instances:
            fut = executor.submit(evaluate_instance, inst, completer, pod_mgr, existing_ids)
            futures.append(fut)

        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result:
                with file_lock:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result) + "\n")

    logger.info(f"\nFINAL: {resolved_count}/{len(results)} resolved ({100*resolved_count/max(len(results),1):.1f}%)")
    logger.info(f"Results: {output_path}")


if __name__ == "__main__":
    main()
