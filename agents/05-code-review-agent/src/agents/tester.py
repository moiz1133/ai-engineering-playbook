"""The Tester agent: writes pytest cases for the Fixer's code and predicts their outcomes."""

from __future__ import annotations

import autogen

from src.config import load_prompt


def create_tester(llm_config: dict) -> autogen.AssistantAgent:
    """Create the Tester AssistantAgent, configured with its test-writing system prompt."""
    return autogen.AssistantAgent(
        name="Tester",
        system_message=load_prompt("tester.txt"),
        llm_config=llm_config,
    )
