#!/usr/bin/env python3
"""
R2E-Gym Environment with PPIO Sandbox Backend

Provides gym.Env compatible interface for training and evaluation.
Uses PPIO Sandbox instead of Docker for execution isolation.
Compatible with both R2E-Gym and SWE-bench dataset formats.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
from gym import spaces

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Supported dataset types."""
    SWEBENCH = "swebench"
    R2E_GYM = "r2e_gym"
    AUTO = "auto"


@dataclass
class Action:
    """
    Represents an action in the environment.
    Compatible with R2E-Gym XML function format.
    """
    raw_action: str
    action_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.action_type and self.raw_action:
            self._parse_action()

    def _parse_action(self):
        """Parse R2E-Gym XML function format."""
        # Match <function=name>...</function>
        func_match = re.search(r'<function=(\w+)>(.*?)</function>', self.raw_action, re.DOTALL)
        if func_match:
            self.action_type = func_match.group(1)
            params_str = func_match.group(2)
            # Match <parameter=name>value</parameter>
            param_matches = re.findall(r'<parameter=(\w+)>(.*?)</parameter>', params_str, re.DOTALL)
            self.parameters = {name: value for name, value in param_matches}


@dataclass
class Observation:
    """
    Represents an observation from the environment.
    Compatible with R2E-Gym observation format.
    """
    content: str
    exit_code: int = 0
    action: Optional[Action] = None
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "exit_code": self.exit_code,
            "action_type": self.action.action_type if self.action else None,
            "timestamp": self.timestamp,
        }


# Default constants
DEFAULT_CMD_TIMEOUT = 90
DEFAULT_WORKDIR = "/testbed"  # Changed to match pre-built templates
STATE_FILE = "/tmp/editor_state.json"

# Pre-built PPIO templates for SWE-bench repos
REPO_TEMPLATE_MAP = {
    "pallets/flask": "swebench-flask",
    "psf/requests": "swebench-psf-requests",
    "pytest-dev/pytest": "swebench-pytest-dev-pytest",
    "pylint-dev/pylint": "swebench-pylint-dev-pylint",
    "django/django": "swebench-django-django",
    "sympy/sympy": "swebench-sympy-sympy",
    "sphinx-doc/sphinx": "swebench-sphinx-doc-sphinx",
    "matplotlib/matplotlib": "swebench-matplotlib-matplotlib",
    "scikit-learn/scikit-learn": "swebench-scikit-learn-scikit-learn",
    "astropy/astropy": "swebench-astropy-astropy",
    "pydata/xarray": "swebench-pydata-xarray",
    "mwaskom/seaborn": "swebench-mwaskom-seaborn",
}


class R2EGymPPIOEnvironment(gym.Env):
    """
    R2E-Gym compatible environment using PPIO Sandbox.

    Implements gym.Env interface for reinforcement learning training:
    - reset(): Reset environment to initial state
    - step(action): Execute action and return (observation, reward, done, info)
    - close(): Clean up resources

    Supports both R2E-Gym and SWE-bench dataset formats.
    Uses PPIO Sandbox for isolated execution instead of Docker.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ds: Dict,
        api_key: Optional[str] = None,
        template: Optional[str] = None,
        cmd_timeout: int = DEFAULT_CMD_TIMEOUT,
        workdir: str = DEFAULT_WORKDIR,
        dataset_type: DatasetType = DatasetType.AUTO,
        max_steps: int = 100,
        sandbox_timeout: int = 3600,
        auto_start: bool = False,
    ):
        """
        Initialize the PPIO environment.

        Args:
            ds: Dataset entry dict containing instance_id, repo, etc.
            api_key: PPIO API key (or from env PPIO_API_KEY).
            template: PPIO template ID (or from env PPIO_TEMPLATE_BASE).
            cmd_timeout: Default command timeout in seconds.
            workdir: Working directory in sandbox.
            dataset_type: Type of dataset (SWEBENCH, R2E_GYM, or AUTO).
            max_steps: Maximum steps per episode.
            sandbox_timeout: Sandbox lifetime in seconds.
            auto_start: If True, start sandbox immediately.
        """
        super().__init__()

        self.ds = ds
        self.cmd_timeout = cmd_timeout
        self.workdir = workdir
        self.max_steps = max_steps
        self.sandbox_timeout = sandbox_timeout

        # Load API credentials
        self.api_key = api_key or self._load_api_key()

        # Auto-select template based on repo if available
        repo = ds.get("repo", "")
        if template:
            self.template = template
            self.using_prebuilt_template = False
        elif repo in REPO_TEMPLATE_MAP:
            self.template = REPO_TEMPLATE_MAP[repo]
            self.using_prebuilt_template = True
            logger.info(f"Using pre-built template for {repo}: {self.template}")
        else:
            self.template = os.environ.get("PPIO_TEMPLATE_BASE", "vn9xnp3cm92x6rmqlgwc")
            self.using_prebuilt_template = False

        # Detect dataset type
        self.dataset_type = self._detect_dataset_type(ds) if dataset_type == DatasetType.AUTO else dataset_type

        # gym.Env spaces
        self.action_space = spaces.Text(max_length=10000)
        self.observation_space = spaces.Text(max_length=100000)

        # Episode state
        self._step_count = 0
        self._done = False
        self._observation: Optional[Observation] = None
        self._episode_actions: List[Action] = []
        self._episode_rewards: List[float] = []

        # PPIO Sandbox
        self.sandbox = None
        self._sandbox_ready = False

        logger.info(f"Dataset type: {self.dataset_type.value}, Template: {self.template}")

        if auto_start:
            self.start()

    def _load_api_key(self) -> str:
        """Load PPIO API key from environment or .env file."""
        api_key = os.environ.get("PPIO_API_KEY")
        if api_key:
            return api_key

        # Try loading from .env file
        env_paths = [
            Path(__file__).parent / ".env.local",
            Path(__file__).parent / ".env",
            Path.home() / ".ppio" / ".env",
        ]
        for env_file in env_paths:
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("PPIO_API_KEY="):
                            return line.split("=", 1)[1].strip()

        raise ValueError("PPIO_API_KEY not found in environment or .env files")

    def _detect_dataset_type(self, ds: Dict) -> DatasetType:
        """Auto-detect dataset type based on available fields."""
        # R2E-Gym fields
        if "test_files" in ds or "test_codes" in ds or "commit_hash" in ds:
            return DatasetType.R2E_GYM
        # SWE-bench fields
        if "FAIL_TO_PASS" in ds or "PASS_TO_PASS" in ds or "base_commit" in ds:
            return DatasetType.SWEBENCH
        # Default based on instance_id format
        if "instance_id" in ds and "__" in ds["instance_id"]:
            return DatasetType.SWEBENCH
        return DatasetType.R2E_GYM

    def start(self):
        """Start the PPIO sandbox."""
        try:
            from ppio_sandbox.core import Sandbox

            logger.info(f"Creating PPIO sandbox with template: {self.template}")
            self.sandbox = Sandbox.create(
                api_key=self.api_key,
                template=self.template,
                timeout=self.sandbox_timeout,
            )
            self._setup_env()
            self._sandbox_ready = True
            logger.info(f"Sandbox created: {self.sandbox.id if hasattr(self.sandbox, 'id') else 'unknown'}")
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise

    def _setup_env(self):
        """Setup sandbox environment after creation."""
        if not self.sandbox:
            return

        # Clone repository if specified
        repo = self.ds.get("repo", "")
        commit = self.ds.get("commit_hash") or self.ds.get("base_commit", "HEAD")

        if repo:
            if self.using_prebuilt_template:
                # Pre-built template: repo already cloned at /testbed
                # Just need to fetch and checkout the specific commit
                logger.info(f"Using pre-built template, checking out commit {commit}")

                # Fix permissions and git safe.directory issue (ownership mismatch in container)
                self.run(f"chmod -R 777 {self.workdir} 2>/dev/null || true", timeout=30)
                self.run(f"git config --global --add safe.directory {self.workdir}", timeout=10)

                if commit and commit != "HEAD":
                    # Reset any local changes first (from pip install -e .)
                    self.run(f"cd {self.workdir} && git checkout -- . 2>/dev/null || true", timeout=30)
                    self.run(f"cd {self.workdir} && git clean -fd 2>/dev/null || true", timeout=30)

                    # Fetch to get latest refs
                    self.run(f"cd {self.workdir} && git fetch origin", timeout=300)
                    # Try direct checkout with force
                    checkout_out, checkout_code = self.run(f"cd {self.workdir} && git checkout -f {commit}", timeout=60)
                    if checkout_code != 0:
                        # Fetch the specific commit
                        self.run(f"cd {self.workdir} && git fetch origin {commit}", timeout=120)
                        checkout_out, checkout_code = self.run(f"cd {self.workdir} && git checkout -f {commit}", timeout=60)
                        if checkout_code != 0:
                            logger.warning(f"Checkout failed: {checkout_out}")
            else:
                # Clone repository from scratch
                # Clean up existing testbed
                self.run("rm -rf /home/user/testbed", timeout=10)

                # Clone repository
                if "/" in repo and not repo.startswith("http"):
                    repo_url = f"https://github.com/{repo}.git"
                else:
                    repo_url = repo

                logger.info(f"Cloning {repo_url} at {commit}")
                # Use longer timeout for large repos (5 minutes)
                output, code = self.run(f"git clone --depth 100 {repo_url} {self.workdir}", timeout=300)
                if code != 0:
                    logger.warning(f"Clone failed: {output}")
                    # Try shallow clone without depth limit
                    output, code = self.run(f"git clone {repo_url} {self.workdir}", timeout=600)
                    if code != 0:
                        logger.error(f"Clone failed completely: {output}")
                        return

                # Checkout specific commit
                if commit and commit != "HEAD":
                    # Unshallow and fetch the specific commit
                    self.run(f"cd {self.workdir} && git fetch --unshallow origin 2>/dev/null || true", timeout=300)
                    self.run(f"cd {self.workdir} && git fetch origin {commit}", timeout=120)
                    checkout_out, checkout_code = self.run(f"cd {self.workdir} && git checkout {commit}", timeout=60)
                    if checkout_code != 0:
                        logger.warning(f"Checkout failed: {checkout_out}")

        # Install dependencies if specified
        install_cmd = self.ds.get("install_cmd")
        if install_cmd:
            logger.info(f"Running install: {install_cmd}")
            self.run(f"cd {self.workdir} && {install_cmd}", timeout=600)

        # Initialize state file for undo_edit
        self.run(f"echo '{{}}' > {STATE_FILE}")

    def run(self, cmd: str, timeout: int = None) -> Tuple[str, int]:
        """
        Execute a command in the sandbox.

        Args:
            cmd: Command to execute.
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (output_string, exit_code).
        """
        if not self.sandbox:
            return "Sandbox not started", -1

        timeout = timeout or self.cmd_timeout
        timeout_ms = timeout * 1000

        try:
            result = self.sandbox.commands.run(cmd, timeout=timeout_ms)
            output = result.stdout or ""
            if result.stderr:
                output += "\n" + result.stderr
            return output, result.exit_code
        except Exception as e:
            return f"Error: {e}", -1

    def get_patch(self) -> str:
        """Get git diff of changes made in the sandbox."""
        output, _ = self.run(f"cd {self.workdir} && git add -A && git diff --cached")
        return output

    def calculate_reward(self, timeout: int = 300) -> Tuple[float, str]:
        """
        Run tests and calculate reward.

        Returns:
            Tuple of (reward, test_output).
        """
        if self.dataset_type == DatasetType.R2E_GYM:
            return self._calculate_reward_r2e_gym(timeout)
        else:
            return self._calculate_reward_swebench(timeout)

    def _calculate_reward_r2e_gym(self, timeout: int) -> Tuple[float, str]:
        """Calculate reward for R2E-Gym dataset."""
        test_files = self.ds.get("test_files", [])
        test_codes = self.ds.get("test_codes", [])

        if not test_files:
            return 0.0, "No test files specified"

        # Write test files
        for test_file, test_code in zip(test_files, test_codes):
            self.sandbox.files.write(
                f"{self.workdir}/{test_file}",
                test_code.encode("utf-8")
            )

        # Run tests
        all_passed = True
        test_outputs = []
        for test_file in test_files:
            output, code = self.run(
                f"cd {self.workdir} && python3 -m pytest {test_file} -v",
                timeout=timeout
            )
            test_outputs.append(output)
            if code != 0:
                all_passed = False

        return float(all_passed), "\n".join(test_outputs)

    def _calculate_reward_swebench(self, timeout: int) -> Tuple[float, str]:
        """Calculate reward for SWE-bench dataset."""
        fail_to_pass = self.ds.get("FAIL_TO_PASS", [])
        pass_to_pass = self.ds.get("PASS_TO_PASS", [])
        test_cmd = self.ds.get("test_cmd", "pytest -xvs")

        if fail_to_pass:
            tests = " ".join(fail_to_pass)
            full_cmd = f"cd {self.workdir} && {test_cmd} {tests}"
        else:
            full_cmd = f"cd {self.workdir} && {test_cmd}"

        output, code = self.run(full_cmd, timeout=timeout)

        # Simple pass/fail based on exit code
        # More sophisticated parsing could be added
        if code == 0:
            return 1.0, output
        return 0.0, output

    def stop(self):
        """Stop and cleanup the sandbox."""
        if self.sandbox:
            try:
                self.sandbox.kill()
            except Exception as e:
                logger.warning(f"Error killing sandbox: {e}")
            self.sandbox = None
            self._sandbox_ready = False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    # =========================================================================
    # gym.Env Interface Methods
    # =========================================================================

    def reset(self, **kwargs) -> Union[Observation, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation.
        """
        # Stop existing sandbox
        if self.sandbox:
            self.stop()

        # Reset episode state
        self._step_count = 0
        self._done = False
        self._episode_actions = []
        self._episode_rewards = []

        # Start fresh sandbox
        self.start()

        # Create initial observation
        initial_content = self._get_initial_observation()
        self._observation = Observation(
            content=initial_content,
            exit_code=0,
            action=None,
        )

        if kwargs.get("return_info", False):
            return self._observation, {"instance_id": self.ds.get("instance_id")}
        return self._observation

    def _get_initial_observation(self) -> str:
        """Get initial observation content after reset."""
        parts = []

        instance_id = self.ds.get("instance_id", "unknown")
        parts.append(f"Instance: {instance_id}")
        parts.append(f"Working directory: {self.workdir}")

        # Problem statement
        problem_stmt = self.ds.get("problem_statement") or self.ds.get("issue_text", "")
        if problem_stmt:
            parts.append(f"\nProblem Statement:\n{problem_stmt[:2000]}")

        # Directory listing
        ls_output, _ = self.run(f"ls -la {self.workdir}")
        parts.append(f"\nDirectory contents:\n{ls_output[:1000]}")

        return "\n".join(parts)

    def step(self, action: Union[str, Action], timeout: int = None) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute an action and return the result.

        Args:
            action: Action to execute (string or Action object).
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        start_time = time.time()

        if isinstance(action, str):
            action = Action(raw_action=action)

        self._episode_actions.append(action)

        # Execute action
        output, exit_code, action_time = self._execute_action(action, timeout)

        self._observation = Observation(
            content=output,
            exit_code=exit_code,
            action=action,
        )

        reward = 0.0
        info = {
            "step": self._step_count,
            "action_type": action.action_type,
            "exit_code": exit_code,
            "action_time": action_time,
        }

        # Check if done
        if self._step_count >= self.max_steps:
            self._done = True
            info["truncated"] = True

        # Check for submit action
        if action.action_type == "submit":
            self._done = True
            reward, test_output = self.calculate_reward()
            info["test_output"] = test_output
            info["final_reward"] = reward

        self._episode_rewards.append(reward)

        return self._observation, reward, self._done, info

    def _execute_action(self, action: Action, timeout: int = None) -> Tuple[str, int, float]:
        """Execute a parsed action."""
        start_time = time.time()

        action_type = action.action_type
        params = action.parameters

        if action_type == "execute_bash":
            cmd = params.get("cmd", params.get("command", ""))
            output, exit_code = self.run(cmd, timeout=timeout)

        elif action_type == "str_replace_editor":
            output, exit_code = self._handle_str_replace_editor(params)

        elif action_type == "submit":
            patch = self.get_patch()
            output = f"Patch submitted.\n\n{patch}"
            exit_code = 0

        else:
            # Try to execute as bash command
            output, exit_code = self.run(action.raw_action, timeout=timeout)

        execution_time = time.time() - start_time
        return output, exit_code, execution_time

    def _handle_str_replace_editor(self, params: Dict[str, Any]) -> Tuple[str, int]:
        """Handle str_replace_editor action."""
        command = params.get("command", "view")
        path = params.get("path", "")

        # Resolve relative paths
        if not path.startswith("/"):
            path = f"{self.workdir}/{path}"

        if command == "view":
            view_range = params.get("view_range")
            if view_range:
                try:
                    import ast
                    start, end = ast.literal_eval(view_range)
                    output, code = self.run(f"sed -n '{start},{end}p' '{path}'")
                except:
                    output, code = self.run(f"cat '{path}'")
            else:
                output, code = self.run(f"cat '{path}'")
            return output, code

        elif command == "create":
            file_text = params.get("file_text", "")
            import base64
            encoded = base64.b64encode(file_text.encode()).decode()
            self.run(f"mkdir -p $(dirname '{path}')")
            output, code = self.run(f"echo '{encoded}' | base64 -d > '{path}'")
            return f"File created: {path}", code

        elif command == "str_replace":
            old_str = params.get("old_str", "")
            new_str = params.get("new_str", "")

            # Save for undo
            content_out, _ = self.run(f"cat '{path}'")
            self._save_for_undo(path, content_out)

            # Perform replacement
            import base64
            script = f'''
import sys
try:
    with open("{path}", "r") as f:
        content = f.read()
    old_str = """{base64.b64encode(old_str.encode()).decode()}"""
    new_str = """{base64.b64encode(new_str.encode()).decode()}"""
    import base64
    old_str = base64.b64decode(old_str).decode()
    new_str = base64.b64decode(new_str).decode()
    if old_str not in content:
        print("ERROR: old_str not found in file")
        sys.exit(1)
    new_content = content.replace(old_str, new_str, 1)
    with open("{path}", "w") as f:
        f.write(new_content)
    print("Replacement successful")
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
'''
            script_b64 = base64.b64encode(script.encode()).decode()
            self.run(f"echo '{script_b64}' | base64 -d > /tmp/replace.py")
            output, code = self.run("python3 /tmp/replace.py")
            return output, code

        elif command == "insert":
            insert_line = int(params.get("insert_line", 0))
            new_str = params.get("new_str", "")

            content_out, _ = self.run(f"cat '{path}'")
            self._save_for_undo(path, content_out)

            import base64
            script = f'''
import sys
try:
    with open("{path}", "r") as f:
        lines = f.readlines()
    new_str = """{base64.b64encode(new_str.encode()).decode()}"""
    import base64
    new_str = base64.b64decode(new_str).decode()
    insert_line = {insert_line}
    lines.insert(insert_line, new_str + "\\n")
    with open("{path}", "w") as f:
        f.writelines(lines)
    print("Insert successful")
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
'''
            script_b64 = base64.b64encode(script.encode()).decode()
            self.run(f"echo '{script_b64}' | base64 -d > /tmp/insert.py")
            output, code = self.run("python3 /tmp/insert.py")
            return output, code

        elif command == "undo_edit":
            return self._undo_edit(path)

        return f"Unknown editor command: {command}", 1

    def _save_for_undo(self, path: str, content: str):
        """Save file content for later undo."""
        import base64
        script = f'''
import json
history_file = "{STATE_FILE}"
try:
    with open(history_file, "r") as f:
        history = json.load(f)
except:
    history = {{}}
content = """{base64.b64encode(content.encode()).decode()}"""
import base64
content = base64.b64decode(content).decode()
if "{path}" not in history:
    history["{path}"] = []
history["{path}"].append(content)
with open(history_file, "w") as f:
    json.dump(history, f)
'''
        script_b64 = base64.b64encode(script.encode()).decode()
        self.run(f"echo '{script_b64}' | base64 -d > /tmp/save_undo.py")
        self.run("python3 /tmp/save_undo.py")

    def _undo_edit(self, path: str) -> Tuple[str, int]:
        """Undo the last edit to a file."""
        import base64
        script = f'''
import sys, json
history_file = "{STATE_FILE}"
try:
    with open(history_file, "r") as f:
        history = json.load(f)
except:
    print("ERROR: No edit history found.")
    sys.exit(1)
if "{path}" not in history or not history["{path}"]:
    print(f"ERROR: No previous edits for {path}")
    sys.exit(1)
old_content = history["{path}"].pop()
with open("{path}", "w") as f:
    f.write(old_content)
with open(history_file, "w") as f:
    json.dump(history, f)
print(f"Last edit to {path} undone.")
'''
        script_b64 = base64.b64encode(script.encode()).decode()
        self.run(f"echo '{script_b64}' | base64 -d > /tmp/undo.py")
        output, code = self.run("python3 /tmp/undo.py")
        return output, code

    def close(self):
        """Clean up resources."""
        self.stop()

    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human" and self._observation:
            print(f"Step {self._step_count}:")
            print(f"  Action: {self._observation.action.action_type if self._observation.action else 'None'}")
            print(f"  Exit code: {self._observation.exit_code}")
            print(f"  Output: {self._observation.content[:200]}...")

    @property
    def done(self) -> bool:
        return self._done

    @property
    def observation(self) -> Optional[Observation]:
        return self._observation

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the current episode."""
        return {
            "instance_id": self.ds.get("instance_id"),
            "steps": self._step_count,
            "done": self._done,
            "total_reward": sum(self._episode_rewards),
            "action_types": [a.action_type for a in self._episode_actions],
        }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="R2E-Gym PPIO Environment Test")
    parser.add_argument("--mode", choices=["basic", "gym"], default="basic")
    args = parser.parse_args()

    # Test instance (R2E-Gym format)
    test_instance = {
        "instance_id": "test__requests",
        "repo": "psf/requests",
        "commit_hash": "HEAD",
        "test_files": ["test_example.py"],
        "test_codes": ['''def test_simple():
    assert 1 + 1 == 2
    print("Test passed!")
'''],
    }

    print("=" * 60)
    print(f"R2E-Gym PPIO Environment Test (mode: {args.mode})")
    print("=" * 60)

    if args.mode == "basic":
        with R2EGymPPIOEnvironment(test_instance) as env:
            print("\n1. Testing command execution...")
            output, code = env.run("pwd")
            print(f"   pwd: {output.strip()} (exit: {code})")

            output, code = env.run("ls -la")
            print(f"   ls: {len(output)} chars (exit: {code})")

            print("\n2. Testing patch extraction...")
            patch = env.get_patch()
            print(f"   Patch length: {len(patch)} chars")

    else:
        env = R2EGymPPIOEnvironment(test_instance, max_steps=10)
        try:
            print("\n1. Testing reset()...")
            obs = env.reset()
            print(f"   Initial observation: {str(obs)[:200]}...")

            print("\n2. Testing step() with execute_bash...")
            action = '<function=execute_bash><parameter=cmd>pwd</parameter></function>'
            obs, reward, done, info = env.step(action)
            print(f"   Output: {obs.content.strip()}")
            print(f"   Reward: {reward}, Done: {done}")

            print("\n3. Testing step() with submit...")
            action = '<function=submit></function>'
            obs, reward, done, info = env.step(action)
            print(f"   Done: {done}, Final reward: {info.get('final_reward', reward)}")

            print("\n4. Episode summary:")
            print(f"   {env.get_episode_summary()}")

        finally:
            env.close()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
