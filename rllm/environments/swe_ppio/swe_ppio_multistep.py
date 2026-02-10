#!/usr/bin/env python3
"""
Multi-step SWE-bench Environment using PPIO Sandbox

This environment supports r2egym-style tool calls (execute_bash, str_replace_editor, submit)
and executes commands in PPIO sandbox instead of Docker.
"""

import json
import os
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rllm.environments.base.base_env import BaseEnv
from .ppio_reward import (
    PPIOSandboxManager,
    extract_patch_from_response,
    parse_pytest_output,
    setup_proxy_tunnel,
    get_sandbox_pool,
    DEFAULT_WORKDIR,
    REPO_TEMPLATE_MAP,
    normalize_repo_name,
)

# Setup proxy
setup_proxy_tunnel()


# =============================================================================
# Action Parsing
# =============================================================================
@dataclass
class ParsedAction:
    """Represents a parsed action from XML format."""
    function_name: str
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_string(cls, action_str: str) -> "ParsedAction":
        """Parse XML-style action string.

        Format:
        <function=function_name>
        <parameter=param_name>param_value</parameter>
        ...
        </function>
        """
        if not action_str or not action_str.strip():
            return cls(function_name="", parameters={})

        # Extract function name
        func_match = re.search(r"<function=([^>]+)>", action_str)
        if not func_match:
            return cls(function_name="", parameters={})

        function_name = func_match.group(1).strip()

        # Extract parameters
        parameters = {}
        param_pattern = re.compile(
            r"<parameter=([^>]+)>(.*?)</parameter>",
            re.DOTALL
        )
        for match in param_pattern.finditer(action_str):
            param_name = match.group(1).strip()
            param_value = match.group(2)
            # Clean up param value (preserve internal structure but strip outer whitespace)
            parameters[param_name] = param_value.strip()

        return cls(function_name=function_name, parameters=parameters)


def parse_xml_action(response_text: str) -> tuple[str, ParsedAction]:
    """
    Parse model response to extract thought and action.

    Returns:
        Tuple of (thought, ParsedAction)
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action_str = match.group(1)
        thought = response_text[:match.start()]
    else:
        thought = response_text
        action_str = ""

    thought = thought.strip()
    action = ParsedAction.from_string(action_str)

    return thought, action


# =============================================================================
# Sandbox File I/O with Retry
# =============================================================================
def _sandbox_file_write(sandbox, path: str, content: str, max_retries: int = 3):
    """Write file to sandbox with retry on 500 errors."""
    for attempt in range(max_retries):
        try:
            sandbox.files.write(path, content)
            return
        except Exception as e:
            if attempt < max_retries - 1 and "500" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise


def _sandbox_file_read(sandbox, path: str, max_retries: int = 3) -> str:
    """Read file from sandbox with retry on 500 errors."""
    for attempt in range(max_retries):
        try:
            return sandbox.files.read(path)
        except Exception as e:
            if attempt < max_retries - 1 and "500" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise


# =============================================================================
# File Editor State
# =============================================================================
@dataclass
class FileState:
    """Track file state for undo support."""
    content: str
    history: list = field(default_factory=list)


# =============================================================================
# Test Command & Eval Script Generation (using swebench TestSpec)
# =============================================================================

# Fallback test commands (only used if swebench specs unavailable)
REPO_TEST_CMDS_FALLBACK = {
    "django/django": "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    "sympy/sympy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
    "pytest-dev/pytest": "pytest -rA",
    "matplotlib/matplotlib": "pytest -rA",
    "scikit-learn/scikit-learn": "pytest -rA",
    "astropy/astropy": "pytest -rA",
    "sphinx-doc/sphinx": "tox --current-env -epy39 -v --",
    "pylint-dev/pylint": "pytest -rA",
    "pallets/flask": "pytest -rA",
    "psf/requests": "pytest -rA",
    "pydata/xarray": "pytest -rA",
    "mwaskom/seaborn": "pytest --no-header -rA",
}


def build_eval_script(entry: dict) -> tuple[str, str]:
    """Build evaluation script using swebench's TestSpec, matching Standard flow.

    The Standard (Docker/R2E-Gym) implementation:
    1. Extracts test directives from test_patch (NOT from FAIL_TO_PASS/PASS_TO_PASS)
    2. Resets test files to base_commit
    3. Applies test_patch (adds/modifies test files)
    4. Runs test command with proper directives
    5. Resets test files after

    Returns:
        (eval_script, test_cmd) - the full bash eval script and the test command alone
    """
    repo = entry.get("repo", "")
    version = entry.get("version", "")
    base_commit = entry.get("base_commit", "")
    test_patch = entry.get("test_patch", "")
    instance_id = entry.get("instance_id", "")

    try:
        from swebench.harness.test_spec.python import get_test_directives, get_modified_files
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

        # Build instance dict for swebench functions
        instance = {
            "repo": repo,
            "instance_id": instance_id,
            "test_patch": test_patch,
            "version": version,
            "base_commit": base_commit,
        }

        # Get test directives from test_patch (file paths, not test names)
        directives = get_test_directives(instance)

        # Get version-specific test command from swebench specs
        specs = MAP_REPO_VERSION_TO_SPECS.get(repo, {}).get(version, {})
        test_cmd_base = specs.get("test_cmd", REPO_TEST_CMDS_FALLBACK.get(repo, "pytest -rA"))

        # Build full test command: base_cmd + directives
        test_cmd = " ".join([test_cmd_base] + directives)

        # Get eval_commands (env vars needed for some repos like Django)
        eval_commands = specs.get("eval_commands", [])

        # Get install command from specs
        install_cmd = specs.get("install", "")

        # Get modified test files for reset
        test_files = []
        if test_patch:
            try:
                test_files = get_modified_files(test_patch)
            except Exception:
                # Fallback: extract from diff headers
                test_files = re.findall(r"diff --git a/.* b/(.*)", test_patch)

        # Build eval script matching Standard flow
        HEREDOC_DELIMITER = "EOF_114329324912"
        script_lines = ["#!/bin/bash", "set -uxo pipefail"]

        # Add eval commands (env vars)
        for cmd in eval_commands:
            script_lines.append(cmd)

        # cd to workdir (will be replaced with actual workdir)
        script_lines.append("cd __WORKDIR__")

        # Reset test files to base_commit (undo any changes model made to tests)
        if test_files and base_commit:
            reset_cmd = f"git checkout {base_commit} {' '.join(test_files)}"
            script_lines.append(reset_cmd)

        # Apply test_patch (adds the tests that verify the fix)
        if test_patch:
            script_lines.append(
                f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
            )

        # Note: install is skipped here because reset() already ran install_deps().
        # The Standard includes reinstall in eval script, but it adds significant
        # latency for compiled packages (sklearn, matplotlib). For editable installs
        # (pip install -e .), source changes are reflected immediately.

        # Run the actual test command
        script_lines.append(f": '>>>>> Start Test Output'")
        script_lines.append(test_cmd)
        script_lines.append(f": '>>>>> End Test Output'")

        # Reset test files after running
        if test_files and base_commit:
            reset_cmd = f"git checkout {base_commit} {' '.join(test_files)}"
            script_lines.append(reset_cmd)

        eval_script = "\n".join(script_lines)
        print(f"[build_eval_script] {instance_id}: test_cmd='{test_cmd}', directives={directives}, test_files={test_files}")
        return eval_script, test_cmd

    except Exception as e:
        print(f"[build_eval_script] Failed to use swebench TestSpec for {instance_id}: {e}")
        # Fallback: use old approach but still better than FAIL_TO_PASS names
        test_cmd = REPO_TEST_CMDS_FALLBACK.get(repo, "pytest -rA")
        return f"#!/bin/bash\ncd __WORKDIR__\n{test_cmd}", test_cmd


def parse_fail_to_pass(fail_to_pass):
    """Handle JSON string parsing."""
    if isinstance(fail_to_pass, str):
        try:
            return json.loads(fail_to_pass)
        except json.JSONDecodeError:
            return [fail_to_pass]
    return fail_to_pass if fail_to_pass else []


# =============================================================================
# Multi-Step Environment
# =============================================================================
class SWEBenchPPIOMultiStepEnv(BaseEnv):
    """
    Multi-step SWE-bench Environment using PPIO Sandbox.

    Supports r2egym-style tool calls:
    - execute_bash: Run bash commands
    - str_replace_editor: View/edit/create files
    - submit: Submit solution and compute reward
    """

    # Class-level trajectory counter for pool indexing
    _trajectory_counter = 0
    _counter_lock = None

    def __init__(
        self,
        entry: Optional[dict] = None,
        dataset: Optional[Any] = None,
        idx: Optional[int] = None,
        timeout: int = 3600,
        workdir: str = DEFAULT_WORKDIR,
        use_pool: bool = True,
        pool_size: int = 32,
        step_timeout: int = 90,
        max_steps: int = 50,
        reward_timeout: int = 1800,
        sandbox_pause: bool = True,
    ):
        """Initialize the multi-step environment.

        Args:
            entry: Task dictionary with instance info
            dataset: Dataset to select from (optional)
            idx: Index in dataset (optional)
            timeout: Sandbox timeout in seconds
            workdir: Working directory in sandbox
            use_pool: Whether to use sandbox pool
            pool_size: Size of sandbox pool
            step_timeout: Timeout for each tool execution (seconds)
            max_steps: Maximum steps before forced termination
            reward_timeout: Timeout for test execution (seconds)
            sandbox_pause: Whether to pause/resume sandboxes between uses (saves cost but adds latency)
        """
        if SWEBenchPPIOMultiStepEnv._counter_lock is None:
            SWEBenchPPIOMultiStepEnv._counter_lock = threading.Lock()

        self.entry = entry
        self.dataset = dataset
        self._idx = idx
        self.timeout = timeout
        self.workdir = workdir
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.step_timeout = step_timeout
        self.max_steps = max_steps
        self.reward_timeout = reward_timeout
        self.sandbox_pause = sandbox_pause

        # Assign trajectory index from counter
        with SWEBenchPPIOMultiStepEnv._counter_lock:
            self.trajectory_idx = SWEBenchPPIOMultiStepEnv._trajectory_counter
            SWEBenchPPIOMultiStepEnv._trajectory_counter += 1

        self.sandbox_manager: Optional[PPIOSandboxManager] = None
        self.api_key = self._load_api_key()
        self.task_instruction = ""
        self.total_steps = 0

        # Get repo from entry for template selection
        self.repo = entry.get("repo", "") if entry else ""

        # File state tracking for undo support
        self.file_states: dict[str, FileState] = {}

        # Reward from submit (None = not submitted yet)
        self._submit_reward: Optional[float] = None

    @property
    def idx(self) -> Any:
        return self._idx

    @idx.setter
    def idx(self, value: Any):
        self._idx = value

    def _load_api_key(self) -> str:
        """Load PPIO API key."""
        api_key = os.environ.get("PPIO_API_KEY")
        if api_key:
            return api_key

        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("PPIO_API_KEY="):
                        return line.split("=", 1)[1].strip()
        raise ValueError("PPIO_API_KEY not found")

    def reset(self) -> tuple[dict, dict]:
        """Reset the environment to initial state."""
        # Release sandbox back to pool
        if self.sandbox_manager:
            self.sandbox_manager.cleanup(pause=self.sandbox_pause)
            self.sandbox_manager = None

        # Update repo from entry
        if self.entry:
            self.repo = self.entry.get("repo", "")

        # Get sandbox from pool
        self.sandbox_manager = PPIOSandboxManager(
            api_key=self.api_key,
            timeout=self.timeout,
            workdir=self.workdir,
            use_pool=self.use_pool,
            trajectory_idx=self.trajectory_idx,
            pool_size=self.pool_size,
            repo=self.repo,
            sandbox_pause=self.sandbox_pause,
        )
        self.sandbox_manager.create_sandbox()

        # Clone repository
        if self.entry:
            repo = self.entry.get("repo", "")
            base_commit = self.entry.get("base_commit", "HEAD")

            if repo:
                # Normalize R2E-Gym short names to full GitHub paths
                full_repo = normalize_repo_name(repo)
                repo_url = f"https://github.com/{full_repo}.git"
                # Check both short name and full path for template mapping
                using_prebuilt = repo in REPO_TEMPLATE_MAP or full_repo in REPO_TEMPLATE_MAP
                if using_prebuilt:
                    print(f"[reset] Using pre-built template for {repo} ({full_repo}), checking out {base_commit}...")
                else:
                    print(f"[reset] Cloning {repo_url} at {base_commit}...")
                try:
                    success, output = self.sandbox_manager.clone_repo(repo_url, base_commit)
                    if not success:
                        raise RuntimeError(f"Failed to setup repository: {output}")
                    print(f"[reset] Repository setup successful")
                except Exception as e:
                    print(f"[reset] Repository setup failed: {e}")
                    raise

            # Install dependencies
            install_cmd = self.entry.get("install_cmd", "pip install -e . 2>&1")
            self.sandbox_manager.install_deps(install_cmd)

            # Build task instruction (use SWEAGENT format)
            problem_statement = self.entry.get("problem_statement", "")
            self.task_instruction = problem_statement

        self.total_steps = 0
        self.file_states = {}
        self._submit_reward = None

        return {"task_instruction": self.task_instruction}, {
            "instance_id": self.entry.get("instance_id", "unknown") if self.entry else "unknown",
            "repo": self.entry.get("repo", "") if self.entry else "",
            "max_steps": self.max_steps,
        }

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute an action in the environment.

        Args:
            action: Model response containing XML-formatted tool call

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.total_steps += 1

        # Parse action
        thought, parsed_action = parse_xml_action(action)

        # Check for empty action
        if not parsed_action.function_name:
            return "No valid action found in response. Please provide a function call.", 0.0, False, {
                "error": "No action found",
                "step": self.total_steps,
                "max_steps": self.max_steps,
            }

        # Execute action based on function name
        function_name = parsed_action.function_name.lower()

        if function_name == "execute_bash":
            obs, reward, done, info = self._execute_bash(parsed_action)
        elif function_name in ["str_replace_editor", "file_editor"]:
            obs, reward, done, info = self._str_replace_editor(parsed_action)
        elif function_name == "submit":
            obs, reward, done, info = self._submit(parsed_action)
        elif function_name == "search":
            obs, reward, done, info = self._search(parsed_action)
        elif function_name == "finish":
            obs, reward, done, info = self._submit(parsed_action)
        else:
            obs = f"Unknown function: {function_name}. Available functions: execute_bash, str_replace_editor, submit, search"
            reward = 0.0
            done = False
            info = {"error": f"Unknown function: {function_name}"}

        # Add step info
        info["step"] = self.total_steps
        info["max_steps"] = self.max_steps

        # Check if max steps reached
        if self.total_steps >= self.max_steps and not done:
            obs += f"\n\nYou have reached the maximum number of steps ({self.max_steps}). Please submit your answer."

        return obs, reward, done, info

    def _execute_bash(self, action: ParsedAction) -> tuple[str, float, bool, dict]:
        """Execute bash command."""
        command = action.parameters.get("command", action.parameters.get("cmd", ""))

        if not command:
            return "Error: No command provided. Usage: <parameter=command>your_command</parameter>", 0.0, False, {
                "error": "No command provided"
            }

        # Handle special commands
        if command == "ctrl+c":
            return "Interrupted.", 0.0, False, {}

        try:
            exit_code, output = self.sandbox_manager._run_command(
                f"cd {self.workdir} && {command}",
                timeout=self.step_timeout
            )

            # Truncate long output
            max_output_len = 16000
            if len(output) > max_output_len:
                output = output[:max_output_len] + f"\n... (output truncated, {len(output) - max_output_len} chars hidden)"

            if exit_code == 0:
                return output if output else "(No output)", 0.0, False, {"exit_code": exit_code}
            else:
                return f"Command failed (exit code {exit_code}):\n{output}", 0.0, False, {"exit_code": exit_code}

        except Exception as e:
            return f"Error executing command: {str(e)}", 0.0, False, {"error": str(e)}

    def _str_replace_editor(self, action: ParsedAction) -> tuple[str, float, bool, dict]:
        """Handle str_replace_editor / file_editor commands."""
        command = action.parameters.get("command", "view")
        path = action.parameters.get("path", "")

        if not path:
            return "Error: 'path' parameter is required.", 0.0, False, {"error": "No path provided"}

        # Ensure absolute path
        if not path.startswith("/"):
            path = f"{self.workdir}/{path}"

        if command == "view":
            return self._editor_view(path, action.parameters)
        elif command == "create":
            return self._editor_create(path, action.parameters)
        elif command == "str_replace":
            return self._editor_str_replace(path, action.parameters)
        elif command == "insert":
            return self._editor_insert(path, action.parameters)
        elif command == "undo_edit":
            return self._editor_undo(path)
        else:
            return f"Unknown command: {command}. Available: view, create, str_replace, insert, undo_edit", 0.0, False, {
                "error": f"Unknown command: {command}"
            }

    def _editor_view(self, path: str, params: dict) -> tuple[str, float, bool, dict]:
        """View file or directory contents."""
        # Check if path is directory
        exit_code, is_dir = self.sandbox_manager._run_command(
            f"test -d {path} && echo 'DIR' || echo 'FILE'",
            timeout=10
        )

        if "DIR" in is_dir:
            # List directory
            exit_code, output = self.sandbox_manager._run_command(
                f"find {path} -maxdepth 2 -type f -o -type d | head -100",
                timeout=30
            )
            return output, 0.0, False, {}

        # View file
        view_range = params.get("view_range", None)

        if view_range:
            try:
                if isinstance(view_range, str):
                    view_range = json.loads(view_range)
                start, end = view_range
                if end == -1:
                    cmd = f"cat -n {path} | tail -n +{start}"
                else:
                    cmd = f"cat -n {path} | sed -n '{start},{end}p'"
            except:
                cmd = f"cat -n {path}"
        else:
            cmd = f"cat -n {path}"

        exit_code, output = self.sandbox_manager._run_command(cmd, timeout=30)

        if exit_code != 0:
            return f"Error viewing {path}: {output}", 0.0, False, {"error": output}

        # Truncate if too long
        max_lines = 500
        lines = output.split('\n')
        if len(lines) > max_lines:
            output = '\n'.join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

        return output, 0.0, False, {}

    def _editor_create(self, path: str, params: dict) -> tuple[str, float, bool, dict]:
        """Create a new file."""
        file_text = params.get("file_text", "")

        # Check if file exists
        exit_code, _ = self.sandbox_manager._run_command(f"test -f {path}", timeout=10)
        if exit_code == 0:
            return f"Error: File {path} already exists. Use str_replace to modify.", 0.0, False, {
                "error": "File exists"
            }

        # Create directory if needed
        dir_path = os.path.dirname(path)
        self.sandbox_manager._run_command(f"mkdir -p {dir_path}", timeout=10)

        # Write file
        try:
            _sandbox_file_write(self.sandbox_manager.sandbox, path, file_text)
            # Track file state
            self.file_states[path] = FileState(content=file_text, history=[])
            return f"File created successfully at: {path}", 0.0, False, {}
        except Exception as e:
            return f"Error creating file: {str(e)}", 0.0, False, {"error": str(e)}

    def _editor_str_replace(self, path: str, params: dict) -> tuple[str, float, bool, dict]:
        """Replace string in file."""
        old_str = params.get("old_str", "")
        new_str = params.get("new_str", "")

        if not old_str:
            return "Error: 'old_str' parameter is required.", 0.0, False, {"error": "No old_str"}

        # Read current file content
        try:
            content = _sandbox_file_read(self.sandbox_manager.sandbox, path)
        except Exception as e:
            return f"Error reading {path}: {str(e)}", 0.0, False, {"error": str(e)}

        # Check if old_str exists and is unique
        count = content.count(old_str)
        if count == 0:
            return f"Error: '{old_str[:100]}...' not found in {path}.", 0.0, False, {
                "error": "old_str not found"
            }
        if count > 1:
            return f"Error: '{old_str[:100]}...' found {count} times in {path}. Make it unique.", 0.0, False, {
                "error": "old_str not unique"
            }

        # Save history for undo
        if path not in self.file_states:
            self.file_states[path] = FileState(content=content, history=[])
        self.file_states[path].history.append(content)

        # Perform replacement
        new_content = content.replace(old_str, new_str, 1)

        # Write back
        try:
            _sandbox_file_write(self.sandbox_manager.sandbox, path, new_content)
            self.file_states[path].content = new_content
            return f"Successfully replaced text in {path}.", 0.0, False, {}
        except Exception as e:
            return f"Error writing to {path}: {str(e)}", 0.0, False, {"error": str(e)}

    def _editor_insert(self, path: str, params: dict) -> tuple[str, float, bool, dict]:
        """Insert text after a line."""
        new_str = params.get("new_str", "")
        insert_line = params.get("insert_line", None)

        if insert_line is None:
            return "Error: 'insert_line' parameter is required.", 0.0, False, {"error": "No insert_line"}

        try:
            insert_line = int(insert_line)
        except:
            return f"Error: insert_line must be an integer, got {insert_line}", 0.0, False, {
                "error": "Invalid insert_line"
            }

        # Read current file content
        try:
            content = _sandbox_file_read(self.sandbox_manager.sandbox, path)
        except Exception as e:
            return f"Error reading {path}: {str(e)}", 0.0, False, {"error": str(e)}

        # Save history for undo
        if path not in self.file_states:
            self.file_states[path] = FileState(content=content, history=[])
        self.file_states[path].history.append(content)

        # Insert after specified line
        lines = content.split('\n')
        if insert_line < 0 or insert_line > len(lines):
            return f"Error: insert_line {insert_line} out of range (0-{len(lines)})", 0.0, False, {
                "error": "insert_line out of range"
            }

        lines.insert(insert_line, new_str)
        new_content = '\n'.join(lines)

        # Write back
        try:
            _sandbox_file_write(self.sandbox_manager.sandbox, path, new_content)
            self.file_states[path].content = new_content
            return f"Successfully inserted text at line {insert_line} in {path}.", 0.0, False, {}
        except Exception as e:
            return f"Error writing to {path}: {str(e)}", 0.0, False, {"error": str(e)}

    def _editor_undo(self, path: str) -> tuple[str, float, bool, dict]:
        """Undo last edit to file."""
        if path not in self.file_states or not self.file_states[path].history:
            return f"No edit history for {path}.", 0.0, False, {"error": "No history"}

        # Pop last state
        previous_content = self.file_states[path].history.pop()

        # Write back
        try:
            _sandbox_file_write(self.sandbox_manager.sandbox, path, previous_content)
            self.file_states[path].content = previous_content
            return f"Successfully reverted {path} to previous state.", 0.0, False, {}
        except Exception as e:
            return f"Error reverting {path}: {str(e)}", 0.0, False, {"error": str(e)}

    def _search(self, action: ParsedAction) -> tuple[str, float, bool, dict]:
        """Search for term in files."""
        search_term = action.parameters.get("search_term", "")
        path = action.parameters.get("path", self.workdir)

        if not search_term:
            return "Error: 'search_term' parameter is required.", 0.0, False, {"error": "No search_term"}

        if not path.startswith("/"):
            path = f"{self.workdir}/{path}"

        # Use grep for search
        exit_code, output = self.sandbox_manager._run_command(
            f"grep -rn '{search_term}' {path} 2>/dev/null | head -100",
            timeout=60
        )

        if not output.strip():
            return f"No matches found for '{search_term}' in {path}", 0.0, False, {}

        return output, 0.0, False, {}

    def _run_eval_script(self, entry: dict) -> tuple[int, str]:
        """Run the eval script in sandbox, matching Standard (Docker/R2E-Gym) flow.

        Generates eval script using swebench TestSpec:
        1. Sets up env vars (e.g. LANG for Django)
        2. Resets test files to base_commit
        3. Applies test_patch (adds tests that verify the fix)
        4. Runs test command with proper directives from test_patch
        5. Resets test files after

        Returns:
            (exit_code, test_output) - only the test output between markers
        """
        eval_script, test_cmd = build_eval_script(entry)

        # Replace workdir placeholder
        eval_script = eval_script.replace("__WORKDIR__", self.workdir)

        # Write eval script to sandbox
        eval_script_path = f"{self.workdir}/_eval.sh"
        import base64
        script_b64 = base64.b64encode(eval_script.encode()).decode()
        self.sandbox_manager._run_command(
            f"echo '{script_b64}' | base64 -d > {eval_script_path} && chmod +x {eval_script_path}",
            timeout=30
        )

        # Run eval script
        exit_code, output = self.sandbox_manager._run_command(
            f"cd {self.workdir} && bash {eval_script_path} 2>&1",
            timeout=self.reward_timeout
        )

        # Extract test output between markers (matching Standard flow)
        start_marker = ">>>>> Start Test Output"
        end_marker = ">>>>> End Test Output"
        if start_marker in output:
            test_output = output.split(start_marker, 1)[1]
            if end_marker in test_output:
                test_output = test_output.split(end_marker, 1)[0]
        else:
            # No markers found - use full output (fallback)
            test_output = output

        # Clean up
        self.sandbox_manager._run_command(f"rm -f {eval_script_path}", timeout=10)

        return exit_code, test_output

    def _submit(self, action: ParsedAction) -> tuple[str, float, bool, dict]:
        """Submit solution and compute reward."""
        # Get git diff as patch
        exit_code, patch = self.sandbox_manager._run_command(
            f"cd {self.workdir} && git diff HEAD",
            timeout=30
        )

        if exit_code != 0 or not patch.strip():
            # Try to get staged changes too
            exit_code, patch = self.sandbox_manager._run_command(
                f"cd {self.workdir} && git diff",
                timeout=30
            )

        if not patch.strip():
            self._submit_reward = 0.0
            return "No changes detected. Did you make any modifications?", 0.0, True, {
                "resolved": False,
                "error": "No patch"
            }

        # Parse FAIL_TO_PASS / PASS_TO_PASS for grading (NOT for test command)
        fail_to_pass = self.entry.get("FAIL_TO_PASS", []) if self.entry else []
        pass_to_pass = self.entry.get("PASS_TO_PASS", []) if self.entry else []
        fail_to_pass = parse_fail_to_pass(fail_to_pass)
        pass_to_pass = parse_fail_to_pass(pass_to_pass)
        repo = self.entry.get("repo", "") if self.entry else ""

        # Run eval script (uses test_patch directives for test command, NOT FAIL_TO_PASS names)
        print(f"[_submit] Running eval script for {self.entry.get('instance_id', 'unknown')}...")
        exit_code, test_output = self._run_eval_script(self.entry)
        print(f"[_submit] Test exit_code={exit_code}, output_len={len(test_output) if test_output else 0}")
        print(f"[_submit] Test output (last 500 chars): {test_output[-500:] if test_output else 'EMPTY'}")

        # Parse results using repo-specific swebench parser
        result = parse_pytest_output(test_output, fail_to_pass, pass_to_pass, repo=repo)
        print(f"[_submit] Parse result: passed={result.passed}, failed={result.failed}, total={result.total}")
        print(f"[_submit] f2p_success={result.f2p_success}, f2p_failure={result.f2p_failure}")
        print(f"[_submit] p2p_success_count={len(result.p2p_success)}, p2p_failure={result.p2p_failure}")
        print(f"[_submit] resolved={result.resolved}")

        # Calculate reward
        if result.resolved:
            reward = 1.0
        elif len(fail_to_pass) > 0:
            reward = len(result.f2p_success) / len(fail_to_pass)
        else:
            reward = 0.0
        print(f"[_submit] reward={reward}")

        observation = f"""Test Results:
- Tests Passed: {result.passed}
- Tests Failed: {result.failed}
- FAIL_TO_PASS Success: {len(result.f2p_success)}/{len(fail_to_pass)}
- PASS_TO_PASS Success: {len(result.p2p_success)}/{len(pass_to_pass)}
- Resolved: {result.resolved}
- Reward: {reward}
"""

        info = {
            "resolved": result.resolved,
            "f2p_success": result.f2p_success,
            "f2p_failure": result.f2p_failure,
            "p2p_success": result.p2p_success,
            "p2p_failure": result.p2p_failure,
            "test_passed": result.passed,
            "test_failed": result.failed,
            "patch": patch[:1000],
        }

        # Store reward for compute_final_reward()
        self._submit_reward = reward

        return observation, reward, True, info

    def _run_evaluation(self) -> float:
        """Run test evaluation (used when episode ends without explicit submit)."""
        if not self.sandbox_manager or not self.entry:
            return 0.0

        try:
            # Get git diff as patch
            exit_code, patch = self.sandbox_manager._run_command(
                f"cd {self.workdir} && git diff HEAD",
                timeout=30
            )
            if exit_code != 0 or not patch.strip():
                exit_code, patch = self.sandbox_manager._run_command(
                    f"cd {self.workdir} && git diff",
                    timeout=30
                )
            if not patch.strip():
                return 0.0

            # Parse FAIL_TO_PASS / PASS_TO_PASS for grading
            fail_to_pass = self.entry.get("FAIL_TO_PASS", [])
            pass_to_pass = self.entry.get("PASS_TO_PASS", [])
            fail_to_pass = parse_fail_to_pass(fail_to_pass)
            pass_to_pass = parse_fail_to_pass(pass_to_pass)
            repo = self.entry.get("repo", "")

            # Run eval script (uses test_patch directives, matching Standard flow)
            exit_code, test_output = self._run_eval_script(self.entry)

            result = parse_pytest_output(test_output, fail_to_pass, pass_to_pass, repo=repo)

            if result.resolved:
                return 1.0
            elif len(fail_to_pass) > 0:
                return len(result.f2p_success) / len(fail_to_pass)
            else:
                return 0.0
        except Exception as e:
            print(f"[compute_final_reward] Evaluation failed: {e}")
            return 0.0

    def compute_final_reward(self) -> float:
        """Compute final reward (called by execution engine after episode ends).

        If submit was already called, returns the stored reward.
        Otherwise, runs evaluation to compute reward from current state.
        """
        if self._submit_reward is not None:
            print(f"[compute_final_reward] Returning stored submit reward: {self._submit_reward}")
            return self._submit_reward
        # Episode ended without submit (timeout/max_steps) - run evaluation
        print(f"[compute_final_reward] No submit reward, running evaluation...")
        reward = self._run_evaluation()
        print(f"[compute_final_reward] Evaluation reward: {reward}")
        return reward

    def close(self):
        """Clean up resources."""
        if self.sandbox_manager:
            self.sandbox_manager.cleanup(pause=self.sandbox_pause)
            self.sandbox_manager = None

    @classmethod
    def cleanup_pool(cls):
        """Clean up all sandboxes in the pool."""
        pool = get_sandbox_pool()
        pool.cleanup_all()

    @staticmethod
    def from_dict(info: dict | str) -> "SWEBenchPPIOMultiStepEnv":
        """Create environment from dictionary.

        Args:
            info: Dictionary containing task data. The entire dict will be used
                  as 'entry', and any keys matching __init__ parameters will be
                  extracted and passed separately.

        Returns:
            Initialized SWEBenchPPIOMultiStepEnv instance
        """
        import inspect

        if isinstance(info, str):
            info = json.loads(info)

        # Extract init params that match __init__ signature
        sig = inspect.signature(SWEBenchPPIOMultiStepEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in info:
                init_params[param_name] = info[param_name]

        # Use entire info as entry (matches standard SWEEnv behavior)
        init_params["entry"] = info

        return SWEBenchPPIOMultiStepEnv(**init_params)

    @staticmethod
    def is_multithread_safe() -> bool:
        """PPIO sandboxes are isolated, so multithread safe."""
        return True
