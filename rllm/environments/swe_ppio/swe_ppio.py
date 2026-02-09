#!/usr/bin/env python3
"""
SWE-bench Environment using PPIO Sandbox

A minimal environment implementation that follows the rllm BaseEnv protocol,
using PPIO sandbox for code execution instead of Docker.
"""

import os
import json
import re
import socket
from pathlib import Path
from typing import Any, Optional

from rllm.environments.base.base_env import BaseEnv
from .ppio_reward import (
    PPIOSandboxManager,
    RewardOutput,
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

# XML Action Parser and Handlers for SWE-bench Environment

import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


@dataclass
class XMLAction:
    """Parsed XML function call action."""
    function_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_action: str = ""

    @classmethod
    def from_string(cls, action_str: str) -> "XMLAction":
        """Parse XML-format action string like <function=name>params</function>"""
        if not action_str:
            return cls(raw_action=action_str)

        # Parse <function=name>...</function>
        pattern = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
        match = pattern.search(action_str)

        if match:
            function_name = match.group(1).strip()
            params_str = match.group(2).strip()

            # Parse <parameter=key>value</parameter>
            parameters = {}
            param_pattern = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
            for m in param_pattern.finditer(params_str):
                key = m.group(1).strip()
                value = m.group(2).strip()
                parameters[key] = value

            return cls(function_name=function_name, parameters=parameters, raw_action=action_str)

        return cls(raw_action=action_str)


def handle_execute_bash(sandbox_manager, params: Dict[str, Any], workdir: str) -> Tuple[str, int]:
    """Handle execute_bash action."""
    cmd = params.get("cmd", params.get("command", ""))
    if not cmd:
        return "Error: No command provided", 1

    # Execute in workdir
    full_cmd = f"cd {workdir} && {cmd}"
    exit_code, output = sandbox_manager.run_command(full_cmd, timeout=120)
    return output, exit_code


def handle_file_editor(sandbox_manager, params: Dict[str, Any], workdir: str) -> Tuple[str, int]:
    """Handle file_editor action (view, create, str_replace, insert, undo_edit)."""
    command = params.get("command", "")
    path = params.get("path", "")

    if not path:
        return "Error: No path provided", 1

    # Make path absolute if relative
    if not path.startswith("/"):
        path = f"{workdir}/{path}"

    if command == "view":
        view_range = params.get("view_range", "")
        concise = params.get("concise", "true").lower() == "true"

        if view_range:
            # Parse range like [11, 20] or 11,20
            try:
                if "[" in view_range:
                    range_str = view_range.strip("[]")
                else:
                    range_str = view_range
                parts = [int(x.strip()) for x in range_str.split(",")]
                if len(parts) == 2:
                    start, end = parts
                    if end == -1:
                        cmd = f"sed -n '{start},' {path}"
                    else:
                        cmd = f"sed -n '{start},{end}p' {path}"
                else:
                    cmd = f"cat -n {path}"
            except:
                cmd = f"cat -n {path}"
        else:
            cmd = f"cat -n {path}"

        exit_code, output = sandbox_manager.run_command(cmd, timeout=30)
        return output, exit_code

    elif command == "create":
        file_text = params.get("file_text", "")
        # Escape for bash
        escaped_text = file_text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\$")
        cmd = f'mkdir -p "." && echo "{escaped_text}" > {path}'
        exit_code, output = sandbox_manager.run_command(cmd, timeout=30)
        if exit_code == 0:
            return f"File created at {path}", 0
        return output, exit_code

    elif command == "str_replace":
        old_str = params.get("old_str", "")
        new_str = params.get("new_str", "")

        if not old_str:
            return "Error: old_str is required for str_replace", 1

        # Read file, do replacement, write back
        exit_code, content = sandbox_manager.run_command(f"cat {path}", timeout=30)
        if exit_code != 0:
            return f"Error reading file: {content}", exit_code

        # Check if old_str exists
        if old_str not in content:
            return f"Error: old_str not found in {path}. Make sure it matches exactly including whitespace.", 1

        # Count occurrences
        count = content.count(old_str)
        if count > 1:
            return f"Error: old_str appears {count} times in {path}. Please make it unique by including more context.", 1

        # Do replacement
        new_content = content.replace(old_str, new_str, 1)

        # Write back using heredoc
        cmd = f"cat > {path} << 'HEREDOC_END'\n{new_content}\nHEREDOC_END"
        exit_code, output = sandbox_manager.run_command(cmd, timeout=60)
        if exit_code == 0:
            return f"Successfully replaced text in {path}", 0
        return output, exit_code

    elif command == "insert":
        new_str = params.get("new_str", "")
        insert_line = params.get("insert_line", "")

        if not new_str or not insert_line:
            return "Error: new_str and insert_line are required for insert", 1

        try:
            line_num = int(insert_line)
        except:
            return f"Error: invalid insert_line: {insert_line}", 1

        # Use sed to insert after line
        escaped_new = new_str.replace("\\", "\\\\").replace("/", "\/")
        cmd = f"sed -i '{line_num}a\{escaped_new}' {path}"
        exit_code, output = sandbox_manager.run_command(cmd, timeout=30)
        if exit_code == 0:
            return f"Successfully inserted text after line {line_num} in {path}", 0
        return output, exit_code

    elif command == "undo_edit":
        # Use git checkout to undo changes to file
        cmd = f"cd {workdir} && git checkout -- {path}"
        exit_code, output = sandbox_manager.run_command(cmd, timeout=30)
        if exit_code == 0:
            return f"Successfully reverted {path}", 0
        return output, exit_code

    else:
        return f"Unknown file_editor command: {command}", 1


def handle_search(sandbox_manager, params: Dict[str, Any], workdir: str) -> Tuple[str, int]:
    """Handle search action."""
    search_term = params.get("search_term", "")
    path = params.get("path", ".")

    if not search_term:
        return "Error: search_term is required", 1

    # Make path absolute if relative
    if not path.startswith("/"):
        path = f"{workdir}/{path}"

    # Use grep for search
    cmd = f"grep -rn '{search_term}' {path} | head -100"
    exit_code, output = sandbox_manager.run_command(cmd, timeout=60)

    if exit_code == 1 and not output.strip():
        return f"No matches found for '{search_term}' in {path}", 0

    return output, 0


def is_xml_action(action: str) -> bool:
    """Check if action contains XML function call."""
    return "<function=" in action and "</function>" in action


def is_submit_action(action: XMLAction) -> bool:
    """Check if action is a submit/finish action."""
    return action.function_name.lower() in ("finish", "submit")



# Hardcoded test commands for common repos (from SWE-bench harness)
REPO_TEST_CMDS = {
    "django/django": "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    "sympy/sympy": "bin/test -C --verbose",
    "pytest-dev/pytest": "pytest -rA",
    "matplotlib/matplotlib": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "scikit-learn/scikit-learn": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "astropy/astropy": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "sphinx-doc/sphinx": "tox -e py39 --",
    "pylint-dev/pylint": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "pallets/flask": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "psf/requests": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "pydata/xarray": "pytest --no-header -rA --tb=no -p no:cacheprovider",
    "mwaskom/seaborn": "pytest --no-header -rA --tb=no -p no:cacheprovider",
}


def get_test_cmd_for_repo(repo: str) -> str:
    """Get repo-specific test command."""
    return REPO_TEST_CMDS.get(repo, "pytest -xvs")


def convert_test_names_for_django(fail_to_pass):
    """Convert unittest test names to Django module format.

    Django tests use format: test_method(module.TestClass)
    Need to extract module path for runtests.py
    """
    modules = set()
    for test in fail_to_pass:
        match = re.match(r"[^(]+\(([^)]+)\)", test)
        if match:
            full_path = match.group(1)
            parts = full_path.rsplit(".", 1)
            if len(parts) >= 1:
                modules.add(parts[0])
        else:
            modules.add(test)
    return " ".join(sorted(modules))


def parse_fail_to_pass(fail_to_pass):
    """Handle JSON string parsing (some datasets store as string)."""
    if isinstance(fail_to_pass, str):
        try:
            return json.loads(fail_to_pass)
        except json.JSONDecodeError:
            return [fail_to_pass]
    return fail_to_pass if fail_to_pass else []


class SWEBenchPPIOEnv(BaseEnv):
    """SWE-bench Environment using PPIO Sandbox"""

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
    ):
        """Initialize the environment.

        Args:
            entry: Task dictionary with instance info
            dataset: Dataset to select from (optional)
            idx: Index in dataset (optional)
            timeout: Sandbox timeout in seconds
            workdir: Working directory in sandbox (default: /testbed for pre-built templates)
            use_pool: Whether to use sandbox pool (recommended to avoid 429)
            pool_size: Size of sandbox pool for reuse
        """
        import threading
        if SWEBenchPPIOEnv._counter_lock is None:
            SWEBenchPPIOEnv._counter_lock = threading.Lock()

        self.entry = entry
        self.dataset = dataset
        self._idx = idx
        self.timeout = timeout
        self.workdir = workdir
        self.use_pool = use_pool
        self.pool_size = pool_size

        # Assign trajectory index from counter
        with SWEBenchPPIOEnv._counter_lock:
            self.trajectory_idx = SWEBenchPPIOEnv._trajectory_counter
            SWEBenchPPIOEnv._trajectory_counter += 1

        self.sandbox_manager: Optional[PPIOSandboxManager] = None
        self.api_key = self._load_api_key()
        self.task_instruction = ""
        self.total_steps = 0

        # Get repo from entry for template selection
        self.repo = entry.get("repo", "") if entry else ""

    @property
    def idx(self) -> Any:
        return self._idx

    @idx.setter
    def idx(self, value: Any):
        self._idx = value

    def _load_api_key(self) -> str:
        """Load PPIO API key"""
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
        """Reset the environment to initial state.

        Returns:
            Tuple of (observation_dict, info_dict)
        """
        # Release sandbox back to pool (pause, don't destroy)
        if self.sandbox_manager:
            self.sandbox_manager.cleanup(pause=True)
            self.sandbox_manager = None

        # Update repo from entry if available
        if self.entry:
            self.repo = self.entry.get("repo", "")

        # Get sandbox from pool (reuses existing or creates new)
        # Pass repo for pre-built template selection
        self.sandbox_manager = PPIOSandboxManager(
            api_key=self.api_key,
            timeout=self.timeout,
            workdir=self.workdir,
            use_pool=self.use_pool,
            trajectory_idx=self.trajectory_idx,
            pool_size=self.pool_size,
            repo=self.repo,
        )
        self.sandbox_manager.create_sandbox()

        # Clone repository if entry is provided
        if self.entry:
            repo = self.entry.get("repo", "")
            base_commit = self.entry.get("base_commit", "HEAD")

            if repo:
                # Normalize short repo names to full org/repo format
                repo = normalize_repo_name(repo)
                repo_url = f"https://github.com/{repo}.git"
                using_prebuilt = repo in REPO_TEMPLATE_MAP
                if using_prebuilt:
                    print(f"[reset] Using pre-built template for {repo}, checking out {base_commit}...")
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

            # Install dependencies (may be skipped for pre-built templates)
            install_cmd = self.entry.get("install_cmd", "pip install -e . 2>&1")
            self.sandbox_manager.install_deps(install_cmd)

            # Build task instruction
            problem_statement = self.entry.get("problem_statement", "")
            self.task_instruction = f"""You are a software engineer working on a bug fix.

Repository: {repo}
Commit: {base_commit}

Issue Description:
{problem_statement}

Please analyze the issue and provide a fix in the form of a git patch.
Your response should include a patch in unified diff format starting with "diff --git".
"""

        self.total_steps = 0

        return {"task_instruction": self.task_instruction}, {
            "instance_id": self.entry.get("instance_id", "unknown") if self.entry else "unknown",
            "repo": self.entry.get("repo", "") if self.entry else "",
        }

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute an action in the environment.

        Supports both:
        1. XML function calls: <function=name>params</function>
        2. Direct diff patches: diff --git...

        Args:
            action: The model's response containing action or patch

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.total_steps += 1

        # Check if this is an XML function call
        if is_xml_action(action):
            parsed = XMLAction.from_string(action)

            if is_submit_action(parsed):
                # Submit action - run tests and calculate reward
                return self._handle_submit()

            elif parsed.function_name == "execute_bash":
                output, exit_code = handle_execute_bash(
                    self.sandbox_manager, parsed.parameters, self.workdir
                )
                return output, 0.0, False, {"action_type": "execute_bash", "exit_code": exit_code}

            elif parsed.function_name in ("file_editor", "str_replace_editor"):
                output, exit_code = handle_file_editor(
                    self.sandbox_manager, parsed.parameters, self.workdir
                )
                return output, 0.0, False, {"action_type": "file_editor", "exit_code": exit_code}

            elif parsed.function_name == "search":
                output, exit_code = handle_search(
                    self.sandbox_manager, parsed.parameters, self.workdir
                )
                return output, 0.0, False, {"action_type": "search", "exit_code": exit_code}

            else:
                # Unknown function - try as bash command
                output, exit_code = handle_execute_bash(
                    self.sandbox_manager, {"cmd": action}, self.workdir
                )
                return output, 0.0, False, {"action_type": "unknown", "exit_code": exit_code}

        # Legacy: Check for direct diff patch
        patch = extract_patch_from_response(action)
        if patch:
            # Apply patch directly and run tests
            success, output = self.sandbox_manager.apply_patch(patch)
            if not success:
                return f"Failed to apply patch: {output}", 0.0, False, {
                    "error": "Patch application failed",
                    "output": output[:500]
                }
            # Patch applied, now run tests
            return self._handle_submit()

        # No valid action found
        return "No valid action found in response. Use <function=name>params</function> format.", 0.0, False, {
            "error": "No valid action"
        }

    def _handle_submit(self) -> tuple[str, float, bool, dict]:
        """Handle submit action - run tests and calculate reward."""
        # Get test configuration
        fail_to_pass = self.entry.get("FAIL_TO_PASS", []) if self.entry else []
        pass_to_pass = self.entry.get("PASS_TO_PASS", []) if self.entry else []

        # Parse JSON strings if needed
        fail_to_pass = parse_fail_to_pass(fail_to_pass)
        pass_to_pass = parse_fail_to_pass(pass_to_pass)

        # Get repo-specific test command
        repo = self.entry.get("repo", "") if self.entry else ""
        test_cmd = get_test_cmd_for_repo(repo)
        is_django = "runtests.py" in test_cmd

        # Build test command
        if fail_to_pass:
            if is_django:
                tests_to_run = convert_test_names_for_django(fail_to_pass)
            else:
                tests_to_run = " ".join(fail_to_pass)
            full_test_cmd = f"{test_cmd} {tests_to_run}"
        else:
            full_test_cmd = test_cmd

        exit_code, test_output = self.sandbox_manager.run_tests(full_test_cmd, timeout=1800)

        # Parse results
        result = parse_pytest_output(test_output, fail_to_pass, pass_to_pass)

        # Calculate reward
        if result.resolved:
            reward = 1.0
        elif len(fail_to_pass) > 0:
            reward = len(result.f2p_success) / len(fail_to_pass)
        else:
            reward = 0.0

        # Episode is done after submit
        done = True

        observation = f"""Test Results:
- Tests Passed: {result.passed}
- Tests Failed: {result.failed}
- FAIL_TO_PASS Success: {len(result.f2p_success)}/{len(fail_to_pass)}
- PASS_TO_PASS Success: {len(result.p2p_success)}/{len(pass_to_pass)}
- Resolved: {result.resolved}
"""

        info = {
            "resolved": result.resolved,
            "f2p_success": result.f2p_success,
            "f2p_failure": result.f2p_failure,
            "p2p_success": result.p2p_success,
            "p2p_failure": result.p2p_failure,
            "test_passed": result.passed,
            "test_failed": result.failed,
            "reward": reward,
        }

        return observation, reward, done, info


    def compute_final_reward(self) -> float:
        """Compute the final reward for the episode."""
        # For SWE-bench, the step already computes the final reward
        return 0.0

    def close(self):
        """Clean up resources (pause sandbox, keep in pool)."""
        if self.sandbox_manager:
            self.sandbox_manager.cleanup(pause=True)
            self.sandbox_manager = None

    @classmethod
    def cleanup_pool(cls):
        """Clean up all sandboxes in the pool. Call at end of training."""
        pool = get_sandbox_pool()
        pool.cleanup_all()

    @staticmethod
    def from_dict(info: dict) -> "SWEBenchPPIOEnv":
        """Create environment from dictionary."""
        return SWEBenchPPIOEnv(
            entry=info.get("entry"),
            timeout=info.get("timeout", 3600),
            workdir=info.get("workdir", DEFAULT_WORKDIR),
            use_pool=info.get("use_pool", True),
            pool_size=info.get("pool_size", 32),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        """PPIO sandboxes are isolated, so multithread safe."""
        return True
