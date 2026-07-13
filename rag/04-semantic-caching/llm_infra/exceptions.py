"""Custom exception hierarchy for llm_infra.

WHAT: custom exception hierarchy — callers catch BudgetExceededError specifically
WHY: allows application code to handle budget exhaustion differently from other
     errors, e.g. show "usage limit reached" to a user vs a generic "something
     went wrong"
"""

from __future__ import annotations


class LLMInfraError(Exception):
    """Base exception for all llm_infra errors."""


class BudgetExceededError(LLMInfraError):
    """Raised when a call would exceed, or has already exceeded, the session budget."""

    def __init__(self, spent: float, limit: float):
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Session budget exceeded: spent ${spent:.4f} of ${limit:.4f} limit"
        )


class CacheError(LLMInfraError):
    """Raised on semantic cache failures (e.g. embedding call failures)."""


class RouterError(LLMInfraError):
    """Raised on model routing failures."""
