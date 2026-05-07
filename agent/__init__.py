"""Haiku agent package — inference loop + iterable prompt registries."""

from agent.haiku_agent import run_agent
from agent.prompts import SYSTEM_PROMPTS, TOOL_DESCRIPTIONS

__all__ = ["run_agent", "SYSTEM_PROMPTS", "TOOL_DESCRIPTIONS"]
