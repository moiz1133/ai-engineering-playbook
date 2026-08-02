"""Aggregates ScenarioResults into a RunReport, then renders it as a Rich console table and a saved JSON file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from src.config import REPORTS_DIR
from src.harness.base import RunReport, ScenarioResult, ToolSummary


def build_run_report(results: List[ScenarioResult]) -> RunReport:
    """Aggregate a list of ScenarioResults (from one or more suites) into a single RunReport."""
    by_tool: Dict[str, ToolSummary] = {}
    for r in results:
        summary = by_tool.setdefault(r.tool_name, ToolSummary(total=0, passed=0))
        summary.total += 1
        summary.passed += int(r.passed)

    return RunReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        total_scenarios=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        by_tool=by_tool,
        scenarios=results,
    )


def print_console_report(report: RunReport, console: Console) -> None:
    """Print a Rich table of every scenario, a per-tool summary, and a highlighted list of failures."""
    table = Table(title="Stress Test Results", show_lines=True)
    table.add_column("Scenario")
    table.add_column("Tool")
    table.add_column("Failure Injected")
    table.add_column("Result", justify="center")
    table.add_column("Error Class")
    table.add_column("Time (ms)", justify="right")

    for r in report.scenarios:
        status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
        table.add_row(r.scenario_name, r.tool_name, r.failure_type, status, r.error_class or "-", str(r.response_time_ms))
    console.print(table)

    summary_table = Table(title="Summary by Tool")
    summary_table.add_column("Tool")
    summary_table.add_column("Passed", justify="right")
    summary_table.add_column("Total", justify="right")
    for tool, summary in report.by_tool.items():
        style = "green" if summary.passed == summary.total else "yellow"
        summary_table.add_row(tool, f"[{style}]{summary.passed}[/{style}]", str(summary.total))
    console.print(summary_table)

    color = "green" if report.failed == 0 else "yellow"
    console.print(f"\n[bold {color}]{report.passed}/{report.total_scenarios} scenarios passed[/bold {color}]")

    failed_scenarios = [r for r in report.scenarios if not r.passed]
    if failed_scenarios:
        console.print("\n[bold red]Failures:[/bold red]")
        for r in failed_scenarios:
            console.print(f"  - [bold]{r.scenario_name}[/bold]: {r.observed_behavior}")
            if r.notes:
                console.print(f"    [dim]{r.notes}[/dim]")


def save_json_report(report: RunReport, reports_dir: str = REPORTS_DIR) -> str:
    """Write the report to reports/stress_test_YYYYMMDD_HHMMSS.json and return the path."""
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"stress_test_{timestamp}.json"

    payload = {
        "run_timestamp": report.run_timestamp,
        "total_scenarios": report.total_scenarios,
        "passed": report.passed,
        "failed": report.failed,
        "by_tool": {tool: summary.model_dump() for tool, summary in report.by_tool.items()},
        "scenarios": [s.model_dump() for s in report.scenarios],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)
