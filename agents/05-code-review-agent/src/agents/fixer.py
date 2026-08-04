"""The Fixer agent: applies the Critic's findings to produce improved code."""

from __future__ import annotations

import autogen

from src.config import load_prompt


def create_fixer(llm_config: dict) -> autogen.AssistantAgent:
    """Create the Fixer AssistantAgent, configured with its fix-only system prompt."""
    return autogen.AssistantAgent(
        name="Fixer",
        system_message=load_prompt("fixer.txt"),
        llm_config=llm_config,
    )
