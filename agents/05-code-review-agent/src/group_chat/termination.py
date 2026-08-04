"""Termination conditions for the group chat.

Two conditions; the first one triggered ends the conversation:
  1. Approval -- see is_approved() below.
  2. Max rounds -- handled by AutoGen's own GroupChat(max_round=...), not this module.

Note on the real AutoGen API vs. a single-message check: "both Critic and Tester
approve" is a property of the conversation, not of any one message, so the termination
callback needs access to the live message history -- not just the single message
AutoGen is currently evaluating. make_termination_check() below closes over the
GroupChat's `messages` list (which AutoGen mutates in place) to make that possible, and
is what actually gets wired into GroupChatManager(is_termination_msg=...) in manager.py
(GroupChat.__init__ itself has no is_termination_msg parameter in this AutoGen version).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

import autogen

_CRITIC_APPROVE_RE = re.compile(r"Overall Verdict:\s*\*{0,2}APPROVE\b", re.IGNORECASE)
_TESTER_APPROVE_RE = re.compile(r"Recommendation:\s*\*{0,2}APPROVE\b", re.IGNORECASE)


def _message_approves(name: str, content: str) -> bool:
    """Checks one agent's message against the verdict pattern for its own structured format."""
    if name == "Critic":
        return bool(_CRITIC_APPROVE_RE.search(content))
    if name == "Tester":
        return bool(_TESTER_APPROVE_RE.search(content))
    return False


def _last_message_from(messages: list[dict[str, Any]], name: str) -> Optional[str]:
    """Finds the most recent message content sent by the given agent name, or None if it hasn't spoken."""
    for msg in reversed(messages):
        if msg.get("name") == name:
            return msg.get("content", "") or ""
    return None


def is_approved(messages: list[dict[str, Any]]) -> bool:
    """Returns True only if BOTH the most recent Critic message and the most recent
    Tester message in this conversation indicate approval. Critic approval alone is
    never enough -- the Tester must independently confirm the test suite looks clean."""
    last_critic = _last_message_from(messages, "Critic")
    last_tester = _last_message_from(messages, "Tester")
    if last_critic is None or last_tester is None:
        return False
    return _message_approves("Critic", last_critic) and _message_approves("Tester", last_tester)


def make_termination_check(groupchat: autogen.GroupChat) -> Callable[[dict[str, Any]], bool]:
    """Builds an is_termination_msg callback bound to this groupchat's live message
    history, for use as GroupChatManager(is_termination_msg=...)."""

    def _check(_msg: dict[str, Any]) -> bool:
        return is_approved(groupchat.messages)

    return _check
