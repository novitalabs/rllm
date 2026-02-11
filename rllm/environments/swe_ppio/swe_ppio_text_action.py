#!/usr/bin/env python3
"""
Text Action Parser for SWE-bench Environment

Supports parsing text format actions like:
    Thought: <reasoning>
    Action: <action_type> <args>

This is the format used by rft-tinker baseline and Docker training test.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


@dataclass
class TextAction:
    """Parsed text-format action."""
    action_type: str = ""
    args: str = ""
    thought: str = ""
    raw_action: str = ""

    @classmethod
    def from_string(cls, action_str: str) -> "TextAction":
        """Parse text-format action string like 'Action: <type> <args>'."""
        if not action_str:
            return cls(raw_action=action_str)

        lines = action_str.strip().split("\n")

        # Extract thought
        thought = ""
        for line in lines:
            if line.strip().lower().startswith("thought:"):
                thought = line.split(":", 1)[1].strip()
                break

        # Find action line
        action_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("action:"):
                action_start_idx = i
                break

        if action_start_idx is None:
            return cls(raw_action=action_str, thought=thought)

        action_line = lines[action_start_idx].split(":", 1)[1].strip()
        parts = action_line.split(None, 1)
        if not parts:
            return cls(raw_action=action_str, thought=thought)

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
            return cls(action_type=action_type, args=full_content.strip(), thought=thought, raw_action=action_str)

        return cls(action_type=action_type, args=first_line_content, thought=thought, raw_action=action_str)


def is_text_action(action: str) -> bool:
    """Check if action contains text-format action (Action: <type>)."""
    lines = action.strip().split("\n")
    for line in lines:
        if line.strip().lower().startswith("action:"):
            return True
    return False


def is_text_submit_action(action: TextAction) -> bool:
    """Check if text action is a submit/finish action."""
    return action.action_type.lower() in ("finish", "submit")
