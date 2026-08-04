"""All Pydantic schemas for the code-review-agent system. No schema definitions live elsewhere."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ReviewVerdict(str, Enum):
    APPROVED = "APPROVED"
    MAX_ROUNDS_HIT = "MAX_ROUNDS_HIT"


class IssuesSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReviewOutput(BaseModel):
    """Structured result extracted from the group chat's full conversation history."""

    original_code: str
    final_code: str
    test_suite: str
    all_issues_found: list[str]
    rounds_completed: int
    verdict: ReviewVerdict
    conversation_log: list[dict]
    output_file: str
    total_tokens_used: int
    total_time_ms: int
