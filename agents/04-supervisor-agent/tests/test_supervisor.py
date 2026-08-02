"""Tests for the supervisor-agent system: schema validation and orchestration behavior.

All LLM and search calls are mocked -- these tests verify the orchestration logic
(routing, parallel-then-critic execution order, graceful handling of a failed worker),
not the quality of any real model output.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.assembler.assembler import Assembler
from src.schemas import SubTask, WorkerOutput, WorkerType, WorkPlan
from src.supervisor.agent import SupervisorAgent


def _make_sub_task(worker_type: WorkerType, task_id: str, depends_on: list[str] | None = None) -> SubTask:
    return SubTask(
        task_id=task_id,
        worker_type=worker_type,
        instruction=f"do the {worker_type.value} thing",
        context="some context",
        priority=1,
        depends_on=depends_on or [],
    )


def _make_output(worker_type: WorkerType, task_id: str, status: str = "success") -> WorkerOutput:
    return WorkerOutput(
        task_id=task_id,
        worker_type=worker_type,
        content="" if status == "failed" else f"{worker_type.value} output content",
        confidence=0.0 if status == "failed" else 0.8,
        sources=[],
        execution_time_ms=10,
        tokens_used=0 if status == "failed" else 100,
        status=status,
        error="simulated failure" if status == "failed" else None,
    )


def test_workplan_validates_well_formed_dict() -> None:
    plan = WorkPlan(
        original_task="do a thing",
        plan_id="plan-1",
        sub_tasks=[_make_sub_task(WorkerType.RESEARCHER, "t1")],
        supervisor_reasoning="because it needs research",
        estimated_complexity="simple",
    )
    assert plan.original_task == "do a thing"
    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].worker_type == WorkerType.RESEARCHER


def test_subtask_requires_mandatory_fields() -> None:
    with pytest.raises(ValidationError):
        SubTask(worker_type=WorkerType.WRITER, instruction="write it")  # missing task_id, context, priority


@pytest.mark.asyncio
async def test_supervisor_routes_workers_based_on_workplan(monkeypatch, tmp_path) -> None:
    """Only the workers present in the WorkPlan should have their execute() called."""
    monkeypatch.setattr("src.supervisor.agent.OUTPUT_DIR", str(tmp_path))
    agent = SupervisorAgent()
    plan = WorkPlan(
        original_task="a simple factual question",
        plan_id="plan-2",
        sub_tasks=[
            _make_sub_task(WorkerType.RESEARCHER, "t1"),
            _make_sub_task(WorkerType.WRITER, "t2"),
        ],
        supervisor_reasoning="simple factual task needs only researcher + writer",
        estimated_complexity="simple",
    )
    agent.decomposer.decompose = AsyncMock(return_value=plan)
    agent.assembler.assemble = AsyncMock(return_value="# assembled")

    agent.workers[WorkerType.RESEARCHER].execute = AsyncMock(
        return_value=_make_output(WorkerType.RESEARCHER, "t1")
    )
    agent.workers[WorkerType.WRITER].execute = AsyncMock(return_value=_make_output(WorkerType.WRITER, "t2"))
    agent.workers[WorkerType.ANALYST].execute = AsyncMock(side_effect=AssertionError("analyst should not run"))
    agent.workers[WorkerType.CRITIC].execute = AsyncMock(side_effect=AssertionError("critic should not run"))

    result = await agent.run("a simple factual question")

    agent.workers[WorkerType.RESEARCHER].execute.assert_awaited_once()
    agent.workers[WorkerType.WRITER].execute.assert_awaited_once()
    agent.workers[WorkerType.ANALYST].execute.assert_not_awaited()
    agent.workers[WorkerType.CRITIC].execute.assert_not_awaited()
    assert {o.worker_type for o in result.worker_outputs} == {WorkerType.RESEARCHER, WorkerType.WRITER}


@pytest.mark.asyncio
async def test_critic_runs_only_after_other_workers_complete(monkeypatch, tmp_path) -> None:
    """Researcher/Analyst/Writer run in parallel; the Critic must not start until all three finish."""
    monkeypatch.setattr("src.supervisor.agent.OUTPUT_DIR", str(tmp_path))
    agent = SupervisorAgent()
    plan = WorkPlan(
        original_task="a complex high-stakes task",
        plan_id="plan-3",
        sub_tasks=[
            _make_sub_task(WorkerType.RESEARCHER, "t1"),
            _make_sub_task(WorkerType.ANALYST, "t2"),
            _make_sub_task(WorkerType.WRITER, "t3"),
            _make_sub_task(WorkerType.CRITIC, "t4", depends_on=["t1", "t2", "t3"]),
        ],
        supervisor_reasoning="complex task needs all four with a critic pass",
        estimated_complexity="complex",
    )
    agent.decomposer.decompose = AsyncMock(return_value=plan)
    agent.assembler.assemble = AsyncMock(return_value="# assembled")

    completed_non_critic: list[str] = []

    async def slow_researcher(sub_task: SubTask) -> WorkerOutput:
        await asyncio.sleep(0.05)  # deliberately the slowest, to prove critic waits for it
        completed_non_critic.append("researcher")
        return _make_output(WorkerType.RESEARCHER, sub_task.task_id)

    def make_fast_worker(worker_type: WorkerType):
        async def _inner(sub_task: SubTask) -> WorkerOutput:
            completed_non_critic.append(worker_type.value)
            return _make_output(worker_type, sub_task.task_id)

        return _inner

    agent.workers[WorkerType.RESEARCHER].execute = slow_researcher
    agent.workers[WorkerType.ANALYST].execute = make_fast_worker(WorkerType.ANALYST)
    agent.workers[WorkerType.WRITER].execute = make_fast_worker(WorkerType.WRITER)

    async def critic_execute(sub_task: SubTask) -> WorkerOutput:
        # By the time the critic runs, all three non-critic workers must already be done.
        assert set(completed_non_critic) == {"researcher", "analyst", "writer"}
        assert "researcher" in sub_task.context  # supervisor injects other outputs into context
        return _make_output(WorkerType.CRITIC, sub_task.task_id)

    agent.workers[WorkerType.CRITIC].execute = critic_execute

    result = await agent.run("a complex high-stakes task")

    assert len(result.worker_outputs) == 4
    assert any(o.worker_type == WorkerType.CRITIC for o in result.worker_outputs)


@pytest.mark.asyncio
async def test_assembler_handles_a_failed_worker_output_without_crashing() -> None:
    """A failed WorkerOutput mixed in with successful ones must not raise -- the assembler
    should skip it and synthesize from whatever succeeded."""
    assembler = Assembler()
    assembler._call_llm = AsyncMock(return_value="# synthesized despite one failure")

    plan = WorkPlan(
        original_task="task with one failure",
        plan_id="plan-4",
        sub_tasks=[_make_sub_task(WorkerType.RESEARCHER, "t1"), _make_sub_task(WorkerType.WRITER, "t2")],
        supervisor_reasoning="reasoning",
        estimated_complexity="simple",
    )
    outputs = [
        _make_output(WorkerType.RESEARCHER, "t1", status="failed"),
        _make_output(WorkerType.WRITER, "t2", status="success"),
    ]

    final_content = await assembler.assemble("task with one failure", plan, outputs)

    assert final_content == "# synthesized despite one failure"
    assembler._call_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_assembler_handles_every_worker_failing_without_crashing() -> None:
    """If ALL workers failed, the assembler must return a graceful failure notice, not raise
    or call the LLM with nothing useful to synthesize."""
    assembler = Assembler()
    assembler._call_llm = AsyncMock(side_effect=AssertionError("should not call the LLM with no successful outputs"))

    plan = WorkPlan(
        original_task="task where everything fails",
        plan_id="plan-5",
        sub_tasks=[_make_sub_task(WorkerType.RESEARCHER, "t1")],
        supervisor_reasoning="reasoning",
        estimated_complexity="simple",
    )
    outputs = [_make_output(WorkerType.RESEARCHER, "t1", status="failed")]

    final_content = await assembler.assemble("task where everything fails", plan, outputs)

    assert "task where everything fails" in final_content
    assembler._call_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_assembled_output_is_valid_when_one_worker_partially_fails(monkeypatch, tmp_path) -> None:
    """A full supervisor.run() with one failed worker must still produce a valid AssembledOutput."""
    monkeypatch.setattr("src.supervisor.agent.OUTPUT_DIR", str(tmp_path))
    agent = SupervisorAgent()
    plan = WorkPlan(
        original_task="task with a partial failure",
        plan_id="plan-6",
        sub_tasks=[_make_sub_task(WorkerType.RESEARCHER, "t1"), _make_sub_task(WorkerType.WRITER, "t2")],
        supervisor_reasoning="reasoning",
        estimated_complexity="simple",
    )
    agent.decomposer.decompose = AsyncMock(return_value=plan)
    agent.assembler.assemble = AsyncMock(return_value="# assembled despite failure")
    agent.workers[WorkerType.RESEARCHER].execute = AsyncMock(
        return_value=_make_output(WorkerType.RESEARCHER, "t1", status="failed")
    )
    agent.workers[WorkerType.WRITER].execute = AsyncMock(return_value=_make_output(WorkerType.WRITER, "t2"))

    result = await agent.run("task with a partial failure")

    assert result.final_content == "# assembled despite failure"
    statuses = {o.worker_type: o.status for o in result.worker_outputs}
    assert statuses[WorkerType.RESEARCHER] == "failed"
    assert statuses[WorkerType.WRITER] == "success"
    assert result.output_file  # a file path was still produced
