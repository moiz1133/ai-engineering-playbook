"""All Pydantic schemas for the supervisor-agent system. No schema definitions live elsewhere."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WorkerType(str, Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


class SubTask(BaseModel):
    """One unit of work assigned by the supervisor to a specific worker."""

    task_id: str
    worker_type: WorkerType
    instruction: str
    context: str
    priority: int
    depends_on: list[str] = Field(default_factory=list)


class WorkPlan(BaseModel):
    """The supervisor's decomposition of a task into sub-tasks."""

    original_task: str
    plan_id: str
    sub_tasks: list[SubTask]
    supervisor_reasoning: str
    estimated_complexity: str  # "simple" | "moderate" | "complex"


class WorkerOutput(BaseModel):
    """Structured result of one worker executing its assigned SubTask."""

    task_id: str
    worker_type: WorkerType
    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    execution_time_ms: int
    tokens_used: int
    status: str  # "success" | "partial" | "failed"
    error: Optional[str] = None


class AssembledOutput(BaseModel):
    """The final synthesized result of a full supervisor run."""

    plan_id: str
    original_task: str
    final_content: str
    worker_outputs: list[WorkerOutput]
    total_tokens: int
    total_time_ms: int
    output_file: str


class RunLog(BaseModel):
    """Full record of one run, saved alongside the final output for auditability."""

    plan_id: str
    timestamp: str
    original_task: str
    plan: WorkPlan
    worker_outputs: list[WorkerOutput]
    assembled_output: AssembledOutput
    total_cost_usd: float
