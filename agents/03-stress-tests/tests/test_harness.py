"""Tests for the stress-test harness itself: the Scenario template method, ScenarioResult/RunReport
schemas, result aggregation, and JSON report writing. Does not re-test the 24 real scenarios --
`src/runner.py` running them for real against production-tools is what actually exercises those."""

import json
from pathlib import Path
from typing import List

import pytest

from src.harness.base import GRACE_CRITERIA_KEYS, Scenario, ScenarioResult, ToolSummary
from src.harness.reporter import build_run_report, save_json_report


def _make_result(name: str, tool: str, passed: bool) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=name,
        tool_name=tool,
        failure_type="test",
        passed=passed,
        failure_description="test failure",
        observed_behavior="test observed",
        grace_criteria_met={k: passed for k in GRACE_CRITERIA_KEYS},
        response_time_ms=1,
    )


def test_scenario_result_validates_correctly() -> None:
    result = _make_result("dummy", "dummy_tool", True)
    assert result.scenario_name == "dummy"
    assert result.passed is True
    assert set(result.grace_criteria_met.keys()) == set(GRACE_CRITERIA_KEYS)


class _RecordingScenario(Scenario):
    """Records the order setup/execute/teardown were called in, optionally raising in execute()."""

    name = "recording"
    description = "records call order for testing the template method"
    tool_name = "test"
    failure_type = "test"

    def __init__(self, raise_in_execute: bool = False) -> None:
        self.calls: List[str] = []
        self.raise_in_execute = raise_in_execute

    async def setup(self) -> None:
        self.calls.append("setup")

    async def execute(self) -> ScenarioResult:
        self.calls.append("execute")
        if self.raise_in_execute:
            raise ValueError("boom")
        return _make_result(self.name, self.tool_name, True)

    async def teardown(self) -> None:
        self.calls.append("teardown")


@pytest.mark.asyncio
async def test_scenario_run_calls_setup_execute_teardown_in_order() -> None:
    scenario = _RecordingScenario()
    result = await scenario.run()
    assert scenario.calls == ["setup", "execute", "teardown"]
    assert result.passed is True


@pytest.mark.asyncio
async def test_scenario_teardown_called_even_if_execute_raises() -> None:
    scenario = _RecordingScenario(raise_in_execute=True)
    with pytest.raises(ValueError):
        await scenario.run()
    assert scenario.calls == ["setup", "execute", "teardown"]


def test_runner_aggregates_results_correctly() -> None:
    results = [
        _make_result("s1", "tool_a", True),
        _make_result("s2", "tool_a", False),
        _make_result("s3", "tool_b", True),
    ]
    report = build_run_report(results)
    assert report.total_scenarios == 3
    assert report.passed == 2
    assert report.failed == 1
    assert report.by_tool["tool_a"] == ToolSummary(total=2, passed=1)
    assert report.by_tool["tool_b"] == ToolSummary(total=1, passed=1)


def test_reporter_writes_valid_json(tmp_path) -> None:
    results = [_make_result("s1", "tool_a", True), _make_result("s2", "tool_a", False)]
    report = build_run_report(results)
    path = save_json_report(report, reports_dir=str(tmp_path))

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["total_scenarios"] == 2
    assert data["passed"] == 1
    assert data["failed"] == 1
    assert len(data["scenarios"]) == 2
    assert data["by_tool"]["tool_a"]["total"] == 2
    assert data["by_tool"]["tool_a"]["passed"] == 1
