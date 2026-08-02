"""The SupervisorAgent: orchestrates decomposition, parallel worker execution, and assembly.

Execution order:
    Researcher + Analyst + Writer run in parallel (asyncio.gather), since none of them
    depend on each other's output.
    If a Critic sub_task is in the plan, it runs AFTER the others complete, since it
    reviews their actual outputs -- this is the one place execution is not parallel.
    The Assembler then synthesizes every worker's output (including the Critic's, if
    present) into the final result.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.assembler.assembler import Assembler
from src.config import MAX_WORKER_TIMEOUT_SECONDS, OUTPUT_DIR, PARALLEL_WORKERS
from src.schemas import AssembledOutput, SubTask, WorkerOutput, WorkerType, WorkPlan
from src.supervisor.decomposer import Decomposer
from src.workers.analyst import AnalystWorker
from src.workers.base import BaseWorker
from src.workers.critic import CriticWorker
from src.workers.researcher import ResearcherWorker
from src.workers.writer import WriterWorker

logger = logging.getLogger(__name__)


def _format_outputs_for_critic(outputs: list[WorkerOutput]) -> str:
    """Renders completed worker outputs as labeled sections for the Critic's context."""
    sections = []
    for output in outputs:
        label = output.worker_type.value.capitalize()
        body = output.content if output.status != "failed" else f"(this worker FAILED: {output.error})"
        sections.append(f"### {label} Output\n{body}")
    return "\n\n".join(sections)


class SupervisorAgent:
    """Top-level orchestrator: decompose -> execute workers -> assemble -> save."""

    def __init__(self) -> None:
        self.decomposer = Decomposer()
        self.workers: dict[WorkerType, BaseWorker] = {
            WorkerType.RESEARCHER: ResearcherWorker(),
            WorkerType.ANALYST: AnalystWorker(),
            WorkerType.WRITER: WriterWorker(),
            WorkerType.CRITIC: CriticWorker(),
        }
        self.assembler = Assembler()
        self.last_plan: Optional[WorkPlan] = None  # set after run(); lets callers build a RunLog

    async def _run_worker(
        self,
        sub_task: SubTask,
        on_worker_start: Optional[Callable[[SubTask], None]] = None,
        on_worker_done: Optional[Callable[[WorkerOutput], None]] = None,
    ) -> WorkerOutput:
        """Runs one worker with a hard timeout; a timeout becomes a 'failed' WorkerOutput
        rather than propagating, since a single stuck worker must never crash the run."""
        worker = self.workers[sub_task.worker_type]
        if on_worker_start:
            on_worker_start(sub_task)
        try:
            output = await asyncio.wait_for(worker.execute(sub_task), timeout=MAX_WORKER_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Worker %s timed out on task_id=%s", sub_task.worker_type.value, sub_task.task_id)
            output = WorkerOutput(
                task_id=sub_task.task_id,
                worker_type=sub_task.worker_type,
                content="",
                confidence=0.0,
                sources=[],
                execution_time_ms=MAX_WORKER_TIMEOUT_SECONDS * 1000,
                tokens_used=0,
                status="failed",
                error=f"Worker timed out after {MAX_WORKER_TIMEOUT_SECONDS}s",
            )
        if on_worker_done:
            on_worker_done(output)
        return output

    async def run(
        self,
        task: str,
        on_decompose_start: Optional[Callable[[], None]] = None,
        on_decompose_done: Optional[Callable[[WorkPlan], None]] = None,
        on_worker_start: Optional[Callable[[SubTask], None]] = None,
        on_worker_done: Optional[Callable[[WorkerOutput], None]] = None,
        on_assemble_start: Optional[Callable[[], None]] = None,
    ) -> AssembledOutput:
        """Decompose the task, run workers (parallel non-critic, then critic), assemble, and save.

        The optional `on_*` callbacks exist purely so a CLI (or any other caller) can
        report live progress without this orchestrator depending on any UI library --
        every callback is a plain synchronous function, called at the moment each
        phase actually starts or finishes.
        """
        start = time.perf_counter()

        if on_decompose_start:
            on_decompose_start()
        plan: WorkPlan = await self.decomposer.decompose(task)
        self.last_plan = plan
        if on_decompose_done:
            on_decompose_done(plan)

        non_critic_tasks = [st for st in plan.sub_tasks if st.worker_type != WorkerType.CRITIC]
        critic_tasks = [st for st in plan.sub_tasks if st.worker_type == WorkerType.CRITIC]

        if PARALLEL_WORKERS:
            non_critic_outputs = list(
                await asyncio.gather(
                    *(self._run_worker(st, on_worker_start, on_worker_done) for st in non_critic_tasks)
                )
            )
        else:
            non_critic_outputs = [
                await self._run_worker(st, on_worker_start, on_worker_done) for st in non_critic_tasks
            ]

        worker_outputs: list[WorkerOutput] = list(non_critic_outputs)

        for critic_task in critic_tasks:
            formatted = _format_outputs_for_critic(non_critic_outputs)
            enriched_context = (
                f"{critic_task.context}\n\nOriginal task: {plan.original_task}\n\n{formatted}"
            )
            critic_sub_task = critic_task.model_copy(update={"context": enriched_context})
            critic_output = await self._run_worker(critic_sub_task, on_worker_start, on_worker_done)
            worker_outputs.append(critic_output)

        if on_assemble_start:
            on_assemble_start()
        final_content = await self.assembler.assemble(
            original_task=plan.original_task,
            plan=plan,
            worker_outputs=worker_outputs,
        )

        output_file = self._save_output(plan.plan_id, final_content)

        total_time_ms = int((time.perf_counter() - start) * 1000)
        total_tokens = sum(o.tokens_used for o in worker_outputs)

        return AssembledOutput(
            plan_id=plan.plan_id,
            original_task=plan.original_task,
            final_content=final_content,
            worker_outputs=worker_outputs,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            output_file=output_file,
        )

    @staticmethod
    def _save_output(plan_id: str, final_content: str) -> str:
        """Writes the assembled markdown to outputs/task_<timestamp>.md and returns the path."""
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"task_{timestamp}.md"
        path.write_text(final_content, encoding="utf-8")
        logger.info("Saved assembled output for plan_id=%s to %s", plan_id, path)
        return str(path)
