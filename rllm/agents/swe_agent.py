import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict

# Fallback SWEAction class when r2egym is not installed
@dataclass
class SimpleSWEAction:
    """A simple SWEAction compatible class for when r2egym is not installed."""
    function_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_string(cls, action_str: str) -> "SimpleSWEAction":
        """Parse XML-format action string like <function=name>params</function>"""
        if not action_str:
            return cls()

        # Parse <function=name>...</function>
        pattern = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
        match = pattern.search(action_str)

        if match:
            function_name = match.group(1).strip()
            params_str = match.group(2).strip()

            # Try to parse parameters as JSON-like format
            parameters = {}
            if params_str:
                # Parse key="value" or key=value format
                param_pattern = re.compile(r'(\w+)\s*=\s*"?([^"<>]+)"?')
                for m in param_pattern.finditer(params_str):
                    key, value = m.group(1), m.group(2).strip()
                    parameters[key] = value

            return cls(function_name=function_name, parameters=parameters)

        return cls()

    def to_xml_string(self) -> str:
        """Convert action to XML string format."""
        if not self.function_name:
            return ""
        params_list = []
        for k, v in self.parameters.items():
            params_list.append(f'{k}="{v}"')
        params = "\n".join(params_list)
        return f"<function={self.function_name}>{params}</function>"

    def __str__(self):
        params = ", ".join(f"{k}='{v}'" for k, v in self.parameters.items())
        return f"Action({self.function_name}, {{{params}}})"


try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = SimpleSWEAction

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import (
    SWE_SYSTEM_PROMPT,
    SWE_SYSTEM_PROMPT_FN_CALL,
    SWE_USER_PROMPT,
    SWE_USER_PROMPT_FN_CALL,
    SWEAGENT_SYSTEM_PROMPT,
    SWEAGENT_USER_PROMPT,
)

TOKEN_WARNING_THRESHOLD = 28000
logger = logging.getLogger(__name__)


def parse_oai_response(response):
    thought = response.choices[0].message.content
    if not thought:
        thought = ""
    try:
        function_name = response.choices[0].message.tool_calls[0].function.name
        parameters = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        action = SWEAction(function_name, parameters)
    except Exception:
        action = SWEAction(function_name="", parameters={})
    return thought, action


def parse_xml_response(response_text: str) -> tuple[str, SWEAction]:
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


class SWEAgent(BaseAgent):
    """
    SWE Agent that interacts with SWE-bench environments.
    Refactored to follow the BaseAgent abstraction for AgentExecutionEngine compatibility.
    """

    def __init__(
        self,
        use_fn_calling: bool = False,
        format_model_response: bool = False,
        scaffold: str = "r2egym",
    ):
        self.use_fn_calling = use_fn_calling
        self.format_model_response = format_model_response
        self.scaffold = scaffold
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}"

        self.system_prompt = (
            SWE_SYSTEM_PROMPT_FN_CALL if use_fn_calling else SWE_SYSTEM_PROMPT
        )
        if scaffold == "sweagent":
            self.system_prompt = SWEAGENT_SYSTEM_PROMPT

        self.user_prompt_template = (
            SWE_USER_PROMPT_FN_CALL if use_fn_calling else SWE_USER_PROMPT
        )
        if scaffold == "sweagent":
            self.user_prompt_template = SWEAGENT_USER_PROMPT

        # Initialize state for BaseAgent interface
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.current_task = None

    def reset(self):
        """Resets the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = []
        self.current_observation = None
        self.current_task = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory

    def _format_observation_as_messages(self, obs: Any, info: dict) -> list[dict]:
        """Helper to format observation into messages."""
        messages = []

        # Check if this is the initial observation (contains task info)
        if info.get("max_steps") and not self.messages:
            # This is the initial reset, create system + user prompt
            task = info.get("task", {})
            if not task and hasattr(self, "current_task") and self.current_task:
                task = self.current_task

            # Try to extract task details from observation or info
            problem_statement = ""
            repo = ""
            base_commit = ""

            if isinstance(task, dict):
                problem_statement = task.get("problem_statement", "")
                repo = task.get("repo", "")
                base_commit = task.get("base_commit", "")
            elif isinstance(obs, dict):
                problem_statement = obs.get("problem_statement", obs.get("issue", ""))
                repo = obs.get("repo", "")
                base_commit = obs.get("base_commit", "")

            # If we still don't have a problem statement, use observation as is
            if not problem_statement and isinstance(obs, str):
                problem_statement = obs

            # Build initial messages
            messages.append({"role": "system", "content": self.system_prompt})

            user_content = self.user_prompt_template.format(
                issue=problem_statement,
                repo=repo,
                base_commit=base_commit,
            )
            messages.append({"role": "user", "content": user_content})
        else:
            # This is a step observation
            if isinstance(obs, str):
                messages.append({"role": "user", "content": obs})
            elif isinstance(obs, dict):
                content = obs.get("output", obs.get("observation", str(obs)))
                messages.append({"role": "user", "content": content})
            elif obs is not None:
                messages.append({"role": "user", "content": str(obs)})

        return messages

    def update_from_env(
        self, observation: Any, reward: float, done: bool, info: dict, **kwargs
    ):
        """
        Updates the agent's state based on environment feedback.
        Formats observation and updates the trajectory.
        """
        # Store task if provided
        if "task" in info:
            self.current_task = info["task"]

        # Format the observation for the next model call
        obs_messages = self._format_observation_as_messages(observation, info)
        self.messages.extend(obs_messages)
        self.current_observation = observation

        # Update the last step's reward/done if we have steps
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info.update(info)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        """
        # Parse the response
        if self.use_fn_calling:
            # For function calling, response should be an object
            thought, action = parse_oai_response(response)
            model_response = thought
        else:
            # For XML format, response is a string
            thought, action = parse_xml_response(response)
            model_response = response

        # Get action string for environment
        action_str = action.to_xml_string() if hasattr(action, "to_xml_string") else str(action)

        # Append assistant message to chat history
        assistant_message = {"role": "assistant", "content": model_response}
        self.messages.append(assistant_message)

        # Create new step and append to trajectory
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action,
            model_response=model_response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        return Action(action=action_str)

    # Legacy methods for backward compatibility
    def create_initial_prompt(self, task: dict) -> list[dict]:
        """Creates initial prompt messages from task dict."""
        user_content = self.user_prompt_template.format(
            issue=task.get("problem_statement", ""),
            repo=task.get("repo", ""),
            base_commit=task.get("base_commit", ""),
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def parse_response(self, response) -> tuple[str, dict]:
        """Legacy method: Parse response and return action string and metadata."""
        if self.use_fn_calling:
            thought, action = parse_oai_response(response)
        else:
            response_text = (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else response
            )
            if isinstance(response_text, str):
                thought, action = parse_xml_response(response_text)
            else:
                thought = str(response_text) if response_text else ""
                action = SWEAction()

        action_str = action.to_xml_string() if hasattr(action, "to_xml_string") else ""
        return action_str, {"thought": thought, "action": action}

    def step(
        self,
        messages: list[dict],
        response,
        observation: str,
        reward: float = 0.0,
        done: bool = False,
    ) -> Step:
        """Legacy method: Create a Step object."""
        if self.use_fn_calling:
            thought, action = parse_oai_response(response)
        else:
            response_text = (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else response
            )
            if isinstance(response_text, str):
                thought, action = parse_xml_response(response_text)
            else:
                thought = str(response_text) if response_text else ""
                action = SWEAction()

        return Step(
            chat_completions=messages,
            thought=thought,
            action=action,
            observation=observation,
            reward=reward,
            done=done,
        )

    def format_observation(self, observation: str, done: bool = False) -> dict:
        """Legacy method: Format observation as a message."""
        if done:
            return {
                "role": "user",
                "content": f"Task completed. Final observation:\n{observation}",
            }
        return {"role": "user", "content": observation}

    def get_action_from_response(self, response) -> str:
        """Legacy method: Extract action string from response."""
        if self.use_fn_calling:
            thought, action = parse_oai_response(response)
        else:
            response_text = (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else response
            )
            if isinstance(response_text, str):
                thought, action = parse_xml_response(response_text)
            else:
                action = SWEAction()

        return action.to_xml_string() if hasattr(action, "to_xml_string") else ""
