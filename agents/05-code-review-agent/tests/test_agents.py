"""Tests for the code-review-agent system: agent construction, termination logic, and
output extraction. No real LLM calls are made -- agent construction is pure/local in
this AutoGen version (it doesn't touch the network), and is_approved/extract_output
operate on plain message dicts rather than live agents."""

from __future__ import annotations

import autogen
import pytest
from pydantic import ValidationError

from src.agents.critic import create_critic
from src.agents.fixer import create_fixer
from src.agents.tester import create_tester
from src.group_chat.termination import is_approved
from src.main import _load_code, extract_output
from src.schemas import ReviewOutput, ReviewVerdict

_FAKE_LLM_CONFIG = {"model": "gpt-4o-mini", "api_key": "sk-test-fake-key", "temperature": 0.1}


def _msg(name: str, content: str) -> dict:
    return {"name": name, "role": "assistant", "content": content}


def test_create_critic_returns_assistant_agent() -> None:
    critic = create_critic(_FAKE_LLM_CONFIG)
    assert isinstance(critic, autogen.AssistantAgent)
    assert critic.name == "Critic"


def test_create_fixer_returns_assistant_agent() -> None:
    fixer = create_fixer(_FAKE_LLM_CONFIG)
    assert isinstance(fixer, autogen.AssistantAgent)
    assert fixer.name == "Fixer"


def test_create_tester_returns_assistant_agent() -> None:
    tester = create_tester(_FAKE_LLM_CONFIG)
    assert isinstance(tester, autogen.AssistantAgent)
    assert tester.name == "Tester"


def test_is_approved_detects_approve_when_both_agree() -> None:
    messages = [
        _msg("Critic", "## Code Review -- Round 1\n### Overall Verdict: APPROVE\n### Severity: LOW"),
        _msg("Fixer", "```python\npass\n```\n## Changes Made\n- None"),
        _msg("Tester", "```python\ndef test_x(): pass\n```\n## Recommendation: APPROVE"),
    ]
    assert is_approved(messages) is True


def test_is_approved_returns_false_when_only_critic_approves() -> None:
    messages = [
        _msg("Critic", "## Code Review -- Round 1\n### Overall Verdict: APPROVE\n### Severity: LOW"),
        _msg("Fixer", "```python\npass\n```\n## Changes Made\n- None"),
        _msg("Tester", "```python\ndef test_x(): pass\n```\n## Recommendation: NEEDS ANOTHER ROUND"),
    ]
    assert is_approved(messages) is False


def test_is_approved_returns_false_when_tester_has_not_spoken_yet() -> None:
    messages = [_msg("Critic", "### Overall Verdict: APPROVE\n### Severity: LOW")]
    assert is_approved(messages) is False


def test_extract_output_parses_code_blocks_from_messages() -> None:
    original_code = "def add(a, b): return a + b"
    messages = [
        {"name": None, "role": "user", "content": original_code},
        _msg("Critic", "## Code Review -- Round 1\n### Bugs Found\n- none\n### Overall Verdict: NEEDS WORK\n### Severity: LOW"),
        _msg("Fixer", "```python\ndef add(a, b):\n    return a + b\n```\n## Changes Made\n- None"),
        _msg("Tester", "```python\ndef test_happy_path():\n    assert add(1, 2) == 3\n```\n## Recommendation: APPROVE"),
        _msg("Critic", "## Code Review -- Round 2\n### Overall Verdict: APPROVE\n### Severity: LOW"),
    ]

    review = extract_output(messages, original_code)

    assert "return a + b" in review.final_code
    assert "def test_happy_path" in review.test_suite
    assert review.rounds_completed == 1  # one Tester message == one completed cycle
    assert review.verdict == ReviewVerdict.APPROVED


def test_extract_output_handles_missing_code_blocks_gracefully() -> None:
    """A Fixer/Tester message with no fenced code block must not crash extraction."""
    original_code = "def x(): pass"
    messages = [
        _msg("Critic", "### Overall Verdict: NEEDS WORK\n### Severity: LOW"),
        _msg("Fixer", "I made some changes but forgot to include the code block."),
        _msg("Tester", "I wrote tests but forgot the code block too.\n## Recommendation: NEEDS ANOTHER ROUND"),
    ]

    review = extract_output(messages, original_code)

    assert review.final_code == original_code  # falls back to original when no block found
    assert review.test_suite == ""
    assert review.verdict == ReviewVerdict.MAX_ROUNDS_HIT


def test_reviewoutput_validates_well_formed_dict() -> None:
    review = ReviewOutput(
        original_code="def x(): pass",
        final_code="def x(): return None",
        test_suite="def test_x(): pass",
        all_issues_found=["missing return value"],
        rounds_completed=1,
        verdict=ReviewVerdict.APPROVED,
        conversation_log=[{"name": "Critic", "content": "..."}],
        output_file="outputs/review_x_20260101_000000.md",
        total_tokens_used=1234,
        total_time_ms=5678,
    )
    assert review.verdict == ReviewVerdict.APPROVED
    assert review.rounds_completed == 1


def test_reviewoutput_rejects_malformed_verdict() -> None:
    with pytest.raises(ValidationError):
        ReviewOutput(
            original_code="x",
            final_code="x",
            test_suite="",
            all_issues_found=[],
            rounds_completed=0,
            verdict="NOT_A_REAL_VERDICT",
            conversation_log=[],
            output_file="",
            total_tokens_used=0,
            total_time_ms=0,
        )


def test_cli_handles_missing_file_gracefully() -> None:
    with pytest.raises(FileNotFoundError, match="No such file"):
        _load_code("this/path/does/not/exist.py", None)
