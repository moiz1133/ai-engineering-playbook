"""CLI entry point: runs stress-test scenarios and produces a console + JSON report.

python -m src.runner                      # run all 24 scenarios
python -m src.runner --suite web_search    # run one suite: web_search | database | code
python -m src.runner --suite database
python -m src.runner --suite code
python -m src.runner --scenario web_search_timeout  # run exactly one scenario by name
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Dict, List, Type

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

from src.config import FAIL_FAST
from src.harness.base import RunReport, Scenario, ScenarioResult
from src.harness.reporter import build_run_report, print_console_report, save_json_report
from src.scenarios import code_execution, database, web_search

console = Console()

SUITES: Dict[str, List[Type[Scenario]]] = {
    "web_search": web_search.ALL_SCENARIOS,
    "database": database.ALL_SCENARIOS,
    "code": code_execution.ALL_SCENARIOS,
}


def _all_scenario_classes() -> List[Type[Scenario]]:
    all_classes: List[Type[Scenario]] = []
    for classes in SUITES.values():
        all_classes.extend(classes)
    return all_classes


async def _run_scenarios(scenario_classes: List[Type[Scenario]]) -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running scenarios...", total=len(scenario_classes))
        for cls in scenario_classes:
            scenario = cls()
            progress.update(task, description=f"Running {scenario.name}...")
            result = await scenario.run()
            results.append(result)
            status = "[bold green]PASS[/bold green]" if result.passed else "[bold red]FAIL[/bold red]"
            console.print(f"  {status} {scenario.name}")
            progress.advance(task)
            if FAIL_FAST and not result.passed:
                console.print("[yellow]FAIL_FAST is set -- stopping after first failure.[/yellow]")
                break
    return results


async def run_all_scenarios() -> RunReport:
    """Run every scenario across all three suites and produce a report."""
    results = await _run_scenarios(_all_scenario_classes())
    return build_run_report(results)


async def run_suite(suite: str) -> RunReport:
    """Run every scenario for one suite: 'web_search', 'database', or 'code'."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Choose from: {sorted(SUITES)}")
    results = await _run_scenarios(SUITES[suite])
    return build_run_report(results)


async def run_single_scenario(scenario_name: str) -> RunReport:
    """Run exactly one scenario by its `name` attribute."""
    for cls in _all_scenario_classes():
        if cls.name == scenario_name:
            results = await _run_scenarios([cls])
            return build_run_report(results)
    raise ValueError(f"Unknown scenario {scenario_name!r}. Choose from: {sorted(c.name for c in _all_scenario_classes())}")


async def _main_async(args: argparse.Namespace) -> None:
    if args.scenario:
        report = await run_single_scenario(args.scenario)
    elif args.suite:
        report = await run_suite(args.suite)
    else:
        report = await run_all_scenarios()

    console.print()
    print_console_report(report, console)
    path = save_json_report(report)
    console.print(f"\nReport saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI agent tool stress-test scenarios.")
    parser.add_argument("--suite", choices=sorted(SUITES), help="Run only this suite's scenarios.")
    parser.add_argument("--scenario", help="Run exactly one scenario by name.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
