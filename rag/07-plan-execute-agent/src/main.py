"""CLI entry point: wires planner -> executor -> report into the Plan-and-Execute pipeline, with Rich progress output."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from src.executor import execute_plan
from src.planner import generate_plan
from src.report import save_fallback, save_report, synthesize_report
from src.schemas import Plan, PlanStep, StepResult

console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def _print_plan(plan: Plan) -> None:
    table = Table(title=f"Research Plan: {plan.topic}", show_lines=True)
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Sub-question")
    table.add_column("Search query", style="dim")
    table.add_column("Rationale", style="italic")
    for step in plan.steps:
        table.add_row(str(step.step_number), step.sub_question, step.search_query, step.rationale)
    console.print(table)


def _on_step_start(step: PlanStep) -> None:
    console.print(f"[bold cyan]-> Step {step.step_number}[/bold cyan]: {step.sub_question}")


def _on_step_done(step: PlanStep, result: StepResult, elapsed_s: float, verbose: bool) -> None:
    n = len(result.search_results)
    console.print(f"   [green]done[/green] {n} result(s) in {elapsed_s:.1f}s -- {result.summary}")
    if verbose:
        for r in result.search_results:
            console.print(f"     [dim]- {r.title} ({r.url})[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan-and-Execute research agent.")
    parser.add_argument("topic", type=str, help="Research topic to investigate")
    parser.add_argument("--verbose", action="store_true", help="Print full plan and step details")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    console.rule("[bold]Plan-and-Execute Research Agent[/bold]")
    console.print(f"Topic: [bold]{args.topic}[/bold]\n")

    console.print("[bold]Phase 1: Planning...[/bold]")
    plan = generate_plan(args.topic)
    _print_plan(plan)

    console.print("\n[bold]Phase 2: Executing plan...[/bold]")
    start = time.perf_counter()
    step_results = execute_plan(
        plan,
        on_step_start=_on_step_start,
        on_step_done=lambda step, result, elapsed: _on_step_done(step, result, elapsed, args.verbose),
    )
    console.print(f"[bold]Execution finished in {time.perf_counter() - start:.1f}s[/bold]\n")

    console.print("[bold]Phase 3: Synthesizing report...[/bold]")
    try:
        report = synthesize_report(plan, step_results)
        path = save_report(report)
        console.print(Panel(f"[bold green]Report saved:[/bold green] {path}", expand=False))
    except Exception as e:
        logger.exception("Synthesis failed")
        path = save_fallback(plan, step_results)
        console.print(
            Panel(
                f"[bold red]Synthesis failed:[/bold red] {e}\n"
                f"[yellow]Raw research saved instead:[/yellow] {path}",
                expand=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
