"""
R2E-Gym Style Agent for SWE tasks.

This agent uses R2E-Gym detailed 9-step workflow prompts and 4-tool configuration.
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

R2EGYM_SYSTEM_PROMPT = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

We have access to the following functions:

-- BEGIN FUNCTION #1: file_editor --
Description:
Custom editing tool for viewing, creating and editing files
  - State is persistent across command calls and discussions with the user
  - If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
  - The create command cannot be used if the specified path already exists as a file

Notes for using the str_replace command:
  - The old_str parameter should match EXACTLY one or more consecutive lines from the original file
  - If the old_str parameter is not unique in the file, the replacement will not be performed

Parameters:
  1. command (string, required) Allowed values: [view, create, str_replace, insert, undo_edit]
  2. path (string, required) Absolute path to file or directory
  3. file_text (string, optional) Required for create command
  4. old_str (string, optional) Required for str_replace command
  5. new_str (string, optional) Required for insert command
  6. insert_line (integer, optional) Required for insert command
  7. view_range (array, optional) Optional for view command
  8. concise (boolean, optional) Optional for view command

-- END FUNCTION #1 --

-- BEGIN FUNCTION #2: execute_bash --
Description: Execute a bash command in the terminal.
Parameters:
  1. cmd (string, required) The bash command to execute.

-- END FUNCTION #2 --

-- BEGIN FUNCTION #3: search --
Description: Search for a term in a directory or a single file.
Parameters:
  1. search_term (string, required) The term to search for
  2. path (string, optional) The file or directory to search in. Defaults to .

-- END FUNCTION #3 --

-- BEGIN FUNCTION #4: finish --
Description: Finish the interaction once the task is complete.
Parameters:
  1. command (string, required) Currently allowed value: [submit]
  2. result (string, optional) The result text or final message to submit.

-- END FUNCTION #4 --

If you choose to call a function ONLY reply in the following format:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>

<IMPORTANT>
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Each response must include both reasoning and function call
"""

R2EGYM_USER_PROMPT = """I have uploaded a python code repository in the /testbed directory.

Now consider the following Github issue:

<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I have already taken care of all changes to any of the test files. Your task is to make changes to non-test files in the /testbed directory.

Follow these steps to resolve the issue:
1. Explore the codebase to locate and understand the code relevant to the <github_issue>.
2. Create a script at ./reproduce_issue.py that demonstrates the error. Execute it to confirm.
3. Analyze the root cause. Reason about multiple approaches to fix the issue.
4. Implement your solution with targeted changes.
5. Verify your solution by rerunning the reproduction script.
6. Run relevant unit tests to ensure no regressions.
7. Test edge cases in ./edge_case_tests.py.
8. Refine if necessary.
9. Submit your solution using the finish tool.

A successful resolution means:
- The specific error/issue no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled

Recommendations:
- Be thorough and prioritize quality over speed.
- Combine multiple actions where possible.
- Use grep with -A -B -C flags to identify relevant code blocks.
"""


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
