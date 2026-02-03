#!/usr/bin/env python3
"""
SWE-bench Evaluation Script using PPIO Sandbox + LiteLLM
Uses R2E-Gym compatible action format with PPIO sandbox backend.
"""

import asyncio
import json
import logging
import os
import re
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

# Clear proxy for PPIO (direct connection needed)
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)

import litellm

# Add rllm to path
import sys
RLLM_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(RLLM_DIR))

# Import PPIO environment from rllm
from rllm.environments.swe_ppio.swe_ppio import SWEBenchPPIOEnv
from rllm.environments.swe_ppio.ppio_reward import REPO_TEMPLATE_MAP, DEFAULT_WORKDIR

# DatasetType enum for compatibility
class DatasetType:
    SWEBENCH = "swebench"
    R2E_GYM = "r2e_gym"
    AUTO = "auto"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_STEPS = 50
MAX_TOKENS = 4096
TEMPERATURE = 1.0
MAX_CONTEXT_TOKENS = 65536  # R2E-Gym style context limit
# R2E-Gym compatible messages
CONTINUE_MSG = """
You forgot to use a function call in your response. 
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""

TRUNCATED_MSG = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with grep -n in order to find the line numbers of what you are looking for.</NOTE>"
NUM_TRUNCATE_LINES = 40


# LLM Configuration
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:30000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/qwen3-32b")
litellm.api_key = "sk-dummy"

# R2E-Gym compatible system prompt
SYSTEM_PROMPT = '''You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

We have access to the following functions:

–– BEGIN FUNCTION #1: file_editor ––
Description:
Custom editing tool for viewing, creating and editing files
  •	State is persistent across command calls and discussions with the user
  •	If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
  •	The create command cannot be used if the specified path already exists as a file
  •	If a command generates a long output, it will be truncated and marked with <response clipped>
  •	The undo_edit command will revert the last edit made to the file at path

Notes for using the str_replace command:
  •	The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
  •	If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
  •	The new_str parameter should contain the edited lines that should replace the old_str

Parameters:
  1.	command (string, required)
Allowed values: [view, create, str_replace, insert, undo_edit]
The command to run.
  2.	path (string, required)
Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
  3.	file_text (string, optional)
Required for the create command. Contains the content of the file to be created.
  4.	old_str (string, optional)
Required for the str_replace command. The exact string in path to replace.
  5.	new_str (string, optional)
  •	Optional for the str_replace command to specify the replacement string.
  •	Required for the insert command to specify the string to insert.
  6.	insert_line (integer, optional)
Required for the insert command. The new_str will be inserted after the line number specified here.
  7.	view_range (array, optional)
  •	Optional for the view command (when path is a file).
  •	If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
  •	[start_line, -1] will show all lines from start_line to the end of file.
  8.	concise (boolean, optional)
  •	Optional for the view command.
  •	Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

–– END FUNCTION #1 ––

–– BEGIN FUNCTION #2: execute_bash ––
Description:
Execute a bash command in the terminal.

Behavior notes:
  •	If a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
  •	If the bash command returns exit code -1, it means the process is still running. The assistant may:
  •	Call this function again with command as an empty string ("") to retrieve additional logs.
  •	Send more input to STDIN of the running process by calling this function again with command set to the text input.
  •	Send command="ctrl+c" to interrupt the currently running process.
  •	If the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

Parameters:
  1.	cmd (string, required)
The bash command (and optional arguments) to execute.
  •	Can be empty ("") to retrieve more logs if the process is still running.
  •	Can be "ctrl+c" to interrupt the running process.

–– END FUNCTION #2 ––

–– BEGIN FUNCTION #3: search ––
Description:
Search for a term in a directory or a single file.
  •	If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
  •	If path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
  •	If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
  •	If no matches are found, it will inform you as well.

Parameters:
  1.	search_term (string, required)
The term or string to search for in files.
  2.	path (string, optional)
The file or directory to search in. Defaults to . if not specified.

–– END FUNCTION #3 ––

–– BEGIN FUNCTION #4: finish ––
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
  •	The submit command finalizes your output.

Parameters:
  1.	command (string, required)
Currently allowed value: [submit]
  2.	result (string, optional)
The result text or final message to submit. Defaults to an empty string if not provided.

–– END FUNCTION #4 ––

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
'''

INSTANCE_PROMPT = '''I have uploaded a python code repository in the /testbed directory.

Now consider the following Github issue:

<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I have already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way! Your task is to make changes to non-test files in the /testbed directory to ensure the <github_issue> is resolved.

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
    - Use the existing test runner to run the unit tests you identify as relevant to the changes you made.

7. Test edge cases:
   - Identify potential edge cases that might challenge your solution.
   - Create additional test cases in a separate file '/testbed/edge_case_tests.py'.
   - Execute these tests to verify your solution's robustness.

8. Refine if necessary:
   - If edge case testing reveals issues, refine your solution accordingly.
   - Ensure your final implementation handles all identified scenarios correctly.

9. Submit your solution:
   - Once you have verified your solution, submit your solution using the `finish` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled

Current step: {current_step}/{max_steps}
'''


class LiteLLMCompleter:
    """Completer using LiteLLM with token tracking"""

    def __init__(self, model: str, base_url: str, max_tokens: int = 4096, temperature: float = 1.0):
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in messages using litellm."""
        try:
            return litellm.token_counter(model=self.model, messages=messages)
        except Exception:
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars // 4

    async def complete(self, messages: List[Dict]) -> Tuple[str, Dict[str, int]]:
        """Complete with token usage tracking. Returns (content, usage_dict)."""
        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                api_base=self.base_url,
            )
            
            usage = {}
            if hasattr(response, "usage"):
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
            
            return response.choices[0].message.content, usage
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "", {}


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


class SWEBenchPPIOAgent:
    """Agent using PPIO sandbox for SWE-bench evaluation."""

    def __init__(self, ds: Dict, env: R2EGymPPIOEnvironment, completer: LiteLLMCompleter):
        self.ds = ds
        self.env = env
        self.completer = completer
        self.current_step = 0
        self.history = []
        self.has_finished = False


    def _truncate_output(self, output: str, num_lines: int = NUM_TRUNCATE_LINES) -> str:
        """Truncate output keeping top and bottom lines (R2E-Gym style)."""
        if not output:
            return output
        lines = output.splitlines()
        if len(lines) <= 2 * num_lines:
            return output
        top_lines = "\n".join(lines[:num_lines])
        bottom_lines = "\n".join(lines[-num_lines:])
        divider = "-" * 50
        return (
            f"{top_lines}\n"
            f"{divider}\n"
            f"<Observation truncated in middle for saving context>\n"
            f"{divider}\n"
            f"{bottom_lines}"
        )

    def _format_observation(self, fn_name: str, output: str, exit_code: int = 0) -> str:
        """Format observation in R2E-Gym style."""
        # Only truncate bash output (R2E-Gym style)
        if fn_name in ("execute_bash", "bash"):
            truncated = self._truncate_output(output)
            return f"Exit code: {exit_code}\nExecution output of [{fn_name}]:\n{truncated}"
        else:
            # Don't truncate non-bash tools
            return f"Execution output of [{fn_name}]:\n{output}"

    def _build_messages(self) -> List[Dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        problem_stmt = self.ds.get("problem_statement", "No problem statement available")
        instance_prompt = INSTANCE_PROMPT.format(
            problem_statement=problem_stmt,
            current_step=self.current_step,
            max_steps=MAX_STEPS,
        )
        messages.append({"role": "user", "content": instance_prompt})

        # Add history (last 15 entries)
        for entry in self.history:  # Full history (R2E-Gym style)
            messages.append({"role": "assistant", "content": entry["action"]})
            messages.append({"role": "user", "content": entry["observation"]})
        
        # Add steps remaining message (R2E-Gym style)
        steps_remaining = MAX_STEPS - self.current_step
        if steps_remaining > 0:
            stepcount_msg = f"\nSteps Remaining: {steps_remaining}"
        else:
            stepcount_msg = "\nYou have reached the maximum number of steps. Please submit your answer NOW."
        
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += stepcount_msg
        
        return messages

    async def run(self) -> Dict[str, Any]:
        """Run the agent loop."""
        logger.info(f"Starting agent for {self.ds.get('instance_id')}")
        self.exit_reason = None

        while self.current_step < MAX_STEPS and not self.has_finished:
            self.current_step += 1
            logger.info(f"Step {self.current_step}/{MAX_STEPS}")

            messages = self._build_messages()
            
            # Token limit check (R2E-Gym style)
            token_count = self.completer.count_tokens(messages)
            logger.info(f"Context tokens: {token_count}")
            
            if token_count > MAX_CONTEXT_TOKENS:
                logger.warning(f"Token limit exceeded: {token_count} >= {MAX_CONTEXT_TOKENS}")
                self.exit_reason = "token_limit"
                break
            
            response, usage = await self.completer.complete(messages)
            
            # Log token usage
            if usage:
                logger.info(f"Tokens - prompt: {usage.get('prompt_tokens', 0)}, "
                           f"completion: {usage.get('completion_tokens', 0)}, "
                           f"total: {usage.get('total_tokens', 0)}")

            if not response:
                logger.warning("Empty response from LLM")
                continue

            fn_name, params = parse_action(response)
            logger.info(f"Action: {fn_name}")

            # Handle empty or invalid actions with CONTINUE_MSG
            if not fn_name:
                observation = CONTINUE_MSG
            elif fn_name in ("submit", "finish"):
                self.has_finished = True
                self.exit_reason = "agent"
                observation = "<<< Finished >>>"  # R2E-Gym style sentinel
            elif fn_name == "execute_bash":
                cmd = params.get("cmd", "")
                output, code = self.env.run(cmd)
                observation = self._format_observation("execute_bash", output, code)
            elif fn_name in ("file_editor", "str_replace_editor"):
                output = self._handle_editor(params)
                observation = self._format_observation(fn_name, output)
            elif fn_name == "search":
                search_term = params.get("search_term", "")
                search_path = params.get("path", ".")
                if not search_path.startswith("/"):
                    search_path = f"{self.env.workdir}/{search_path}"
                # R2E-Gym style search (count matches per file)
                output = self._search_r2egym_style(search_term, search_path)
                observation = self._format_observation("search", output, 0)
            else:
                # Unknown action also gets CONTINUE_MSG
                observation = CONTINUE_MSG

            # Normalize assistant message (R2E-Gym style: thought + action_xml)
            # Extract thought (everything before <function=)
            thought_match = re.search(r'^(.*?)(?=<function=)', response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""

            # Re-serialize action as clean XML with proper escaping
            def escape_xml(s):
                """Escape XML special characters."""
                if s is None:
                    return ""
                s = str(s)
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            if fn_name and params:
                action_xml = f"<function={fn_name}>\n"
                for k, v in params.items():
                    # Don't escape parameter values - they may contain code
                    # But ensure v is string
                    action_xml += f"<parameter={k}>{str(v) if v is not None else ''}</parameter>\n"
                action_xml += "</function>"
            else:
                action_xml = ""

            # If no valid action, store thought only (not raw response)
            if action_xml:
                normalized_action = f"{thought}\n\n{action_xml}"
            else:
                # No valid function call - store thought only (never raw response)
                normalized_action = thought

            self.history.append({
                "action": normalized_action,
                "observation": observation,
                "function": fn_name,
            })

        if not self.exit_reason:
            self.exit_reason = "max_steps" if self.current_step >= MAX_STEPS else "agent"

        return {
            "steps": self.current_step, 
            "finished": self.has_finished,
            "exit_reason": self.exit_reason,
            "total_prompt_tokens": self.completer.total_prompt_tokens,
            "total_completion_tokens": self.completer.total_completion_tokens,
        }

    def _lint_check(self, content: str, file_path: str) -> str:
        """Check Python syntax using ast.parse(). Returns error message or empty string."""
        import ast
        try:
            ast.parse(content, filename=file_path)
            return ""
        except SyntaxError as e:
            return str(e)

    def _search_r2egym_style(self, search_term: str, search_path: str) -> str:
        """R2E-Gym style search: count matches per file instead of raw grep output."""
        # Use grep to get files with matches and count per file
        cmd = f"grep -rl '{search_term}' '{search_path}' 2>/dev/null | head -100"
        files_output, code = self.env.run(cmd)

        if not files_output or not files_output.strip():
            return f'No matches found for "{search_term}" in {search_path}'

        files = [f.strip() for f in files_output.strip().split("\n") if f.strip()]

        # Count matches in each file
        results = []
        total_matches = 0
        for filepath in files[:100]:  # Limit to 100 files
            count_cmd = f"grep -c '{search_term}' '{filepath}' 2>/dev/null"
            count_output, _ = self.env.run(count_cmd)
            try:
                count = int(count_output.strip())
                if count > 0:
                    results.append((filepath, count))
                    total_matches += count
            except:
                pass

        if not results:
            return f'No matches found for "{search_term}" in {search_path}'

        # Sort by match count descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Format output like R2E-Gym
        output_lines = [f'Found {total_matches} matches for "{search_term}" in {search_path}:']
        for filepath, count in results:
            # Make path relative if possible
            rel_path = filepath.replace(search_path, '.') if filepath.startswith(search_path) else filepath
            output_lines.append(f"{rel_path} ({count} matches)")

        return "\n".join(output_lines)

    def _get_concise_view(self, path: str) -> str:
        """Generate AST-based concise view of a Python file (R2E-Gym style)."""
        import ast
        try:
            # Read file content from sandbox
            output, code = self.env.run(f"cat '{path}'")
            if code != 0:
                return f"Error reading file: {output}"

            file_text = output
            try:
                tree = ast.parse(file_text, filename=path)
            except SyntaxError as e:
                # Fall back to grep-based for files with syntax errors
                skeleton_cmd = f"grep -n '^class \|^    def \|^def \|^import \|^from ' '{path}' | head -100"
                skeleton, _ = self.env.run(skeleton_cmd)
                return f"Concise view of {path} (syntax error, grep fallback):\n{skeleton}"

            lines = file_text.splitlines()

            # Find ranges to elide (function bodies >= 5 lines)
            elide_ranges = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    body_start = node.body[0].lineno - 1
                    last_stmt = node.body[-1]
                    if hasattr(last_stmt, 'end_lineno') and last_stmt.end_lineno:
                        body_end = last_stmt.end_lineno - 1
                    else:
                        body_end = body_start + 4  # fallback
                    if (body_end - body_start) >= 3:
                        elide_ranges.append((body_start, body_end))

            # Merge overlapping ranges
            elide_ranges.sort()
            merged = []
            for start, end in elide_ranges:
                if merged and start <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))

            # Build output with elided sections
            result_lines = []
            i = 0
            for start, end in merged:
                while i < start and i < len(lines):
                    result_lines.append(f"{i+1:6}\t{lines[i]}")
                    i += 1
                if i <= end:
                    result_lines.append(f"{'':6}\t... ({end - start + 1} lines elided)")
                    i = end + 1
            while i < len(lines):
                result_lines.append(f"{i+1:6}\t{lines[i]}")
                i += 1

            return f"Here is a condensed view for file: {path};\n[Note: Function bodies >= 5 lines are elided. Use view_range to see specific sections.]\n" + "\n".join(result_lines[:200])
        except Exception as e:
            return f"Error generating concise view: {e}"


    def _handle_editor(self, params: Dict) -> str:
        """Handle str_replace_editor commands."""
        command = params.get("command", "view")
        path = params.get("path", "")

        # Resolve relative paths
        if path and not path.startswith("/"):
            path = f"{self.env.workdir}/{path}"

        if command == "view":
            concise = params.get("concise", "").lower() in ("true", "1", "yes")
            view_range = params.get("view_range")

            # Auto-enable concise view for large Python files (R2E-Gym style)
            if path.endswith(".py") and not view_range and not concise:
                output, code = self.env.run(f"wc -l < '{path}'")
                try:
                    line_count = int(output.strip())
                    if line_count > 110:
                        concise = True
                except:
                    pass

            if concise and path.endswith(".py"):
                return self._get_concise_view(path)
            if view_range:
                try:
                    import ast
                    start, end = ast.literal_eval(view_range)
                    output, _ = self.env.run(f"sed -n '{start},{end}p' '{path}'")
                except:
                    output, _ = self.env.run(f"cat '{path}'")
            else:
                output, _ = self.env.run(f"cat '{path}'")
            return output[:10000]

        elif command == "create":
            file_text = params.get("file_text", "")
            encoded = base64.b64encode(file_text.encode()).decode()
            self.env.run(f"mkdir -p $(dirname '{path}')")
            output, code = self.env.run(f"echo '{encoded}' | base64 -d > '{path}'")
            return f"File created: {path}" if code == 0 else f"Failed to create file: {output}"

        elif command == "str_replace":
            old_str = params.get("old_str", "")
            new_str = params.get("new_str", "")

            # Read original content
            orig_output, orig_code = self.env.run(f"cat '{path}'")
            if orig_code != 0:
                return f"Error reading file: {orig_output}"
            original_content = orig_output

            # Apply replacement
            if old_str not in original_content:
                return "ERROR: old_str not found in file"
            new_content = original_content.replace(old_str, new_str, 1)

            # Lint check for Python files (R2E-Gym style)
            if path.endswith(".py"):
                lint_error = self._lint_check(new_content, path)
                if lint_error:
                    return f"Your proposed edit has introduced new syntax error(s).\nPlease read this error message carefully and then retry editing the file.\nERRORS:\n{lint_error}"

            # Write new content
            script = f"""
import base64
new_content = base64.b64decode("{base64.b64encode(new_content.encode()).decode()}").decode()
with open("{path}", "w") as f:
    f.write(new_content)
print("Replacement successful")
"""
            script_b64 = base64.b64encode(script.encode()).decode()
            self.env.run(f"echo '{script_b64}' | base64 -d > /tmp/replace.py")
            output, code = self.env.run("python3 /tmp/replace.py")

            # R2E-Gym style: show the edited file with context
            # Find line number where replacement occurred
            old_lines = original_content.split("\n")
            new_lines = new_content.split("\n")

            # Find first different line
            start_line = 1
            for i, (old, new) in enumerate(zip(old_lines, new_lines)):
                if old != new:
                    start_line = max(1, i - 2)  # Show 3 lines before
                    break

            # Get snippet around the change
            end_line = min(start_line + len(new_str.split("\n")) + 6, len(new_lines))

            # Get cat -n style output
            snippet_cmd = f"sed -n '{start_line},{end_line}p' '{path}' | cat -n | sed 's/^/    /'"
            snippet_output, _ = self.env.run(snippet_cmd)

            result = f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n{snippet_output}\nReview the changes and make sure they are as expected. Edit the file again if necessary."
            return result

        elif command == "insert":
            insert_line = int(params.get("insert_line", 0))
            new_str = params.get("new_str", "")

            script = f'''
import sys
try:
    with open("{path}", "r") as f:
        lines = f.readlines()
    new_str = """{base64.b64encode(new_str.encode()).decode()}"""
    import base64
    new_str = base64.b64decode(new_str).decode()
    lines.insert({insert_line}, new_str + "\\n")
    with open("{path}", "w") as f:
        f.writelines(lines)
    print("Insert successful")
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
'''
            script_b64 = base64.b64encode(script.encode()).decode()
            self.env.run(f"echo '{script_b64}' | base64 -d > /tmp/insert.py")
            output, code = self.env.run("python3 /tmp/insert.py")
            return output

        return f"Unknown editor command: {command}"


async def evaluate_instance(ds: Dict, completer: LiteLLMCompleter) -> Dict[str, Any]:
    """Evaluate a single SWE-bench instance using PPIO sandbox."""
    instance_id = ds.get("instance_id", "unknown")
    logger.info(f"\n{'='*60}\nEvaluating: {instance_id}\n{'='*60}")

    # Convert SWE-bench format to R2E-Gym format for PPIO env
    ppio_ds = {
        "instance_id": instance_id,
        "repo": ds.get("repo", ""),
        "version": ds.get("version", ""),  # Added for MAP_REPO_VERSION_TO_SPECS lookup
        "commit_hash": ds.get("base_commit", "HEAD"),
        "problem_statement": ds.get("problem_statement", ""),
        "FAIL_TO_PASS": ds.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": ds.get("PASS_TO_PASS", []),
        "test_cmd": ds.get("test_cmd", "pytest -xvs"),
        "install_cmd": ds.get("install_cmd", "pip install -e . 2>&1"),
        "test_patch": ds.get("test_patch", ""),
    }

    try:
        with R2EGymPPIOEnvironment(ppio_ds, dataset_type=DatasetType.SWEBENCH) as env:
            agent = SWEBenchPPIOAgent(ds, env, completer)
            agent_result = await agent.run()

            if agent_result["finished"]:
                reward, test_output = env.calculate_reward(timeout=300)
                patch = env.get_patch()
            else:
                reward, test_output, patch = 0.0, "", ""

            return {
                "instance_id": instance_id,
                "resolved": reward == 1.0,
                "reward": reward,
                "steps": agent_result["steps"],
                "finished": agent_result["finished"],
                "exit_reason": agent_result.get("exit_reason", "unknown"),
                "patch": patch[:5000] if patch else "",
                "total_prompt_tokens": agent_result.get("total_prompt_tokens", 0),
                "total_completion_tokens": agent_result.get("total_completion_tokens", 0),
            }

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"instance_id": instance_id, "error": str(e), "resolved": False}


async def main():
    logger.info(f"LLM Base URL: {LLM_BASE_URL}")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"MAX_CONTEXT_TOKENS: {MAX_CONTEXT_TOKENS}")

    completer = LiteLLMCompleter(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    # Load SWE-bench verified dataset
    # Use Django subset for faster testing (smaller repo)
    data_file = os.environ.get("DATA_FILE", "django_verified.jsonl")
    data_path = Path(__file__).parent / "data" / data_file
    if not data_path.exists():
        # Try alternative path
        data_path = Path(f"/root/work/rft-tinker/data/{data_file}")

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        return

    with open(data_path) as f:
        dataset = [json.loads(line) for line in f]

    # Number of instances to evaluate
    num_eval = int(os.environ.get("NUM_EVAL", 10))
    dataset = dataset[:num_eval]

    logger.info(f"Evaluating {len(dataset)} instances")

    results = []
    resolved_count = 0

    for i, ds in enumerate(dataset):
        logger.info(f"\nProgress: {i+1}/{len(dataset)}")
        result = await evaluate_instance(ds, completer)
        results.append(result)

        if result.get("resolved"):
            resolved_count += 1
            logger.info(f"RESOLVED! Total: {resolved_count}/{i+1}")

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "swebench_ppio_eval_results.jsonl"

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"Final: {resolved_count}/{len(results)} resolved ({100*resolved_count/len(results):.1f}%)")
    logger.info(f"Results: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
