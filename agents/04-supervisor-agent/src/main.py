"""CLI entry point.

    python -m src.main "Analyze the business case for building AI triage systems
    in Pakistani private hospitals in 2026"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from src.config import EST_COST_PER_1K_TOKENS, OUTPUT_DIR, SAVE_RUN_LOG
from src.schemas import AssembledOutput, RunLog, SubTask, WorkerOutput, WorkerType, WorkPlan
from src.supervisor.agent import SupervisorAgent

console = Console()

# Wall-clock bounds of the *parallel* (non-critic) worker phase specifically -- used to
# compute an honest "would be slower without parallelism" estimate. The Critic and
# Assembler phases are sequential either way, so they're excluded from this comparison.
_parallel_phase_start: float | None = None
_parallel_phase_end: float | None = None
_assemble_phase_start: float | None = None


def _on_decompose_start() -> None:
    console.print("\n[bold]Phase 1: Decomposition[/bold]")


def _on_decompose_done(plan: WorkPlan) -> None:
    worker_names = ", ".join(st.worker_type.value.capitalize() for st in plan.sub_tasks)
    console.print("  Decomposing task... [green]done[/green]")
    console.print(f"  Workers assigned: {worker_names}")
    excerpt = plan.supervisor_reasoning[:200] + ("..." if len(plan.supervisor_reasoning) > 200 else "")
    console.print(f"  Plan: {excerpt}\n")
    console.print("[bold]Phase 2: Worker Execution[/bold]")


def _on_worker_start(sub_task: SubTask) -> None:
    global _parallel_phase_start
    label = sub_task.worker_type.value.capitalize()
    if sub_task.worker_type == WorkerType.CRITIC:
        console.print(f"  [Critic] Reviewing outputs...")
    else:
        if _parallel_phase_start is None:
            _parallel_phase_start = time.perf_counter()
        console.print(f"  [{label}] Starting...")


def _on_worker_done(output: WorkerOutput) -> None:
    global _parallel_phase_end
    label = output.worker_type.value.capitalize()
    seconds = output.execution_time_ms / 1000
    if output.worker_type != WorkerType.CRITIC:
        now = time.perf_counter()
        _parallel_phase_end = now if _parallel_phase_end is None else max(_parallel_phase_end, now)
    if output.status == "failed":
        console.print(f"  [bold red]FAILED[/bold red] [{label}] ({seconds:.1f}s): {output.error}")
    else:
        console.print(
            f"  [bold green]OK[/bold green] [{label}] Complete "
            f"({seconds:.1f}s, {output.tokens_used} tokens, confidence: {output.confidence:.2f})"
        )


def _on_assemble_start() -> None:
    global _assemble_phase_start
    _assemble_phase_start = time.perf_counter()
    console.print("\n[bold]Phase 3: Assembly[/bold]")


async def _main_async(task: str) -> None:
    console.print(f"\nTask received: [italic]{task}[/italic]")

    agent = SupervisorAgent()
    result = await agent.run(
        task,
        on_decompose_start=_on_decompose_start,
        on_decompose_done=_on_decompose_done,
        on_worker_start=_on_worker_start,
        on_worker_done=_on_worker_done,
        on_assemble_start=_on_assemble_start,
    )
    assemble_elapsed = (time.perf_counter() - _assemble_phase_start) if _assemble_phase_start is not None else 0.0
    console.print(f"  Assembling final output... [green]done[/green] ({assemble_elapsed:.1f}s)")

    console.print(f"\nOutput saved: {result.output_file}")

    log_path = ""
    if SAVE_RUN_LOG and agent.last_plan is not None:
        log_path = _save_run_log(agent.last_plan, result)
        console.print(f"Run log saved: {log_path}")

    actual_seconds = result.total_time_ms / 1000
    est_cost = (result.total_tokens / 1000) * EST_COST_PER_1K_TOKENS
    workers_used = len(result.worker_outputs)

    # Only the non-critic workers actually ran in parallel (asyncio.gather); the
    # decomposer, critic, and assembler phases are sequential regardless of the
    # PARALLEL_WORKERS setting, so they're held constant in this comparison -- only
    # the parallel phase's wall-clock time is replaced with what it would have cost
    # running one worker after another.
    non_critic_outputs = [o for o in result.worker_outputs if o.worker_type != WorkerType.CRITIC]
    sequential_worker_seconds = sum(o.execution_time_ms for o in non_critic_outputs) / 1000
    if _parallel_phase_start is not None and _parallel_phase_end is not None:
        parallel_phase_seconds = _parallel_phase_end - _parallel_phase_start
        time_saved = max(0.0, sequential_worker_seconds - parallel_phase_seconds)
        no_parallelism_seconds = actual_seconds + time_saved
        console.print("\n[bold]Summary:[/bold]")
        console.print(
            f"  Total time: {actual_seconds:.1f}s (would be {no_parallelism_seconds:.1f}s without parallelism)"
        )
    else:
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Total time: {actual_seconds:.1f}s")
    console.print(f"  Total tokens: {result.total_tokens:,}")
    console.print(f"  Est. cost: ${est_cost:.4f}")
    console.print(f"  Workers used: {workers_used}/4")


def _save_run_log(plan: WorkPlan, result: AssembledOutput) -> str:
    run_log = RunLog(
        plan_id=result.plan_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        original_task=result.original_task,
        plan=plan,
        worker_outputs=result.worker_outputs,
        assembled_output=result,
        total_cost_usd=(result.total_tokens / 1000) * EST_COST_PER_1K_TOKENS,
    )
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"log_{timestamp}.json"
    path.write_text(json.dumps(run_log.model_dump(), indent=2, default=str), encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supervisor/worker multi-agent system on a task.")
    parser.add_argument("task", help="The task for the supervisor to decompose and execute.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_main_async(args.task))


if __name__ == "__main__":
    main()
