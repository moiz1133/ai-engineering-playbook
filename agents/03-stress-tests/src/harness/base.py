"""The core harness contract: every scenario runs setup -> execute -> teardown and produces a ScenarioResult.

"Graceful" is defined by six criteria, checked explicitly by every scenario:
no unhandled exception, a typed (specific) error class, a human-readable
message, a structured error response (never None/empty string), consistent
system state afterward, and the ability to call the tool again successfully.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

GRACE_CRITERIA_KEYS: List[str] = [
    "no_unhandled_exception",
    "typed_error",
    "human_readable_message",
    "structured_error_response",
    "consistent_state",
    "recovery_possible",
]


class ScenarioResult(BaseModel):
    """The outcome of running one scenario."""

    scenario_name: str
    tool_name: str
    failure_type: str
    passed: bool
    failure_description: str
    observed_behavior: str
    grace_criteria_met: Dict[str, bool] = Field(default_factory=dict)
    response_time_ms: int
    error_class: Optional[str] = None
    notes: str = ""


class ToolSummary(BaseModel):
    """Pass/fail counts for one tool's suite of scenarios."""

    total: int
    passed: int


class RunReport(BaseModel):
    """The aggregate result of running some set of scenarios."""

    run_timestamp: str
    total_scenarios: int
    passed: int
    failed: int
    by_tool: Dict[str, ToolSummary] = Field(default_factory=dict)
    scenarios: List[ScenarioResult] = Field(default_factory=list)


class Scenario(ABC):
    """Base class every stress-test scenario implements: setup -> execute -> teardown, in that order, always."""

    name: str
    description: str
    tool_name: str
    failure_type: str

    @abstractmethod
    async def setup(self) -> None:
        """Configure the mock (and/or the real tool under test) for this failure condition."""
        raise NotImplementedError

    @abstractmethod
    async def execute(self) -> ScenarioResult:
        """Run the scenario against the tool and produce a ScenarioResult."""
        raise NotImplementedError

    @abstractmethod
    async def teardown(self) -> None:
        """Reset any state the scenario touched. Runs even if execute() raises."""
        raise NotImplementedError

    async def run(self) -> ScenarioResult:
        """Template method: setup -> execute -> teardown. teardown() is guaranteed even if execute() raises."""
        await self.setup()
        try:
            result = await self.execute()
        finally:
            await self.teardown()
        return result
