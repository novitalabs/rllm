"""
R2E-Gym Style Agent for SWE tasks.

This agent uses R2E-Gym detailed 9-step workflow prompts and 4-tool configuration.
Prompts are copied exactly from R2E-Gym's edit_non_fn_calling.yaml config.
"""
import logging
import re

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)

TOKEN_WARNING_THRESHOLD = 28000

# Copied exactly from R2E-Gym edit_non_fn_calling.yaml
R2EGYM_SYSTEM_PROMPT = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

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
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task."""

# Copied exactly from R2E-Gym edit_non_fn_calling.yaml
R2EGYM_USER_PROMPT = """I have uploaded a python code repository in the /testbed directory.

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
    - Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
       - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
       - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
       - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`
    - RUN ALL relevant unit tests to ensure your solution is correct and does not cause any regressions.
    - DO NOT MODIFY any of the existing unit tests. You can add new edge test cases in a separate file if needed BUT DO NOT MODIFY THE EXISTING TESTS.

7. Test edge cases:
   - Identify potential edge cases that might challenge your solution.
   - Create additional test cases in a separate file '/testbed/edge_case_tests.py'.
   - Execute these tests to verify your solution's robustness.
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


def parse_xml_response(response_text: str):
    """Extract thought and action from model response."""
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)

    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)
        thought = response_text[: match.start()]
    else:
        thought = response_text
        action = ""

    thought = thought.strip()
    action = action.strip()
    action = SWEAction.from_string(action)

    return thought, action


class R2EGymAgent(BaseAgent):
    """R2E-Gym style agent with detailed 9-step workflow and 4-tool configuration."""

    def __init__(self, format_model_response: bool = False):
        self.format_model_response = format_model_response
        self.system_prompt = R2EGYM_SYSTEM_PROMPT
        self.user_prompt_template = R2EGYM_USER_PROMPT
        self._trajectory = Trajectory()
        self.reset()

    def process_model_response(self, response: str):
        """Process model response to extract thought and action."""
        thought, action = parse_xml_response(response)
        action_str = action.to_xml_string()
        if self.format_model_response:
            response = f"{thought}\n\n{action_str}"
        return action.to_xml_string(), {"thought": thought}

    def update_from_env(self, observation, reward, done, info):
        """Update agent state from environment observation."""
        if self._trajectory.steps:
            observation = str(observation)
        else:
            observation = str(observation)
            observation = self.user_prompt_template.format(problem_statement=observation)

        max_steps = info.get("max_steps", None)
        if max_steps:
            remaining_steps = max_steps - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."

        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        self.messages.append({"role": "user", "content": observation})
        self.cur_step = Step(observation=observation)

    def update_from_model(self, response: str, **kwargs):
        """Update agent state from model response."""
        self._trajectory.steps.append(self.cur_step)
        thought, action = parse_xml_response(response)
        action_str = action.to_xml_string()

        assert self._trajectory.steps, "Trajectory should not be empty."

        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = action_str
        cur_step.model_response = response

        if self.format_model_response:
            self.messages.append({"role": "assistant", "content": f"{thought}\n\n{action_str}"})
        else:
            self.messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=cur_step.action)

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty."
        return self._trajectory.steps[-1]

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages
