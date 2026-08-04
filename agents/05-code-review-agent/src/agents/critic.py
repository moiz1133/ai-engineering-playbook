"""The Critic agent: systematic code reviewer. Finds issues, never fixes them."""

from __future__ import annotations

import autogen

from src.config import load_prompt


def create_critic(llm_config: dict) -> autogen.AssistantAgent:
    """Create the Critic AssistantAgent, configured with its review-only system prompt."""
    return autogen.AssistantAgent(
        name="Critic",
        system_message=load_prompt("critic.txt"),
        llm_config=llm_config,
    )
