"""Analyst worker: pure reasoning over the task -- patterns, implications, trade-offs, risks."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.schemas import SubTask, WorkerOutput, WorkerType
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.txt"
_CONFIDENCE_RE = re.compile(r"##\s*Confidence:\s*([01](?:\.\d+)?)", re.IGNORECASE)
_DEFAULT_CONFIDENCE = 0.6


def _parse_confidence(content: str) -> float:
    match = _CONFIDENCE_RE.search(content)
    if not match:
        return _DEFAULT_CONFIDENCE
    try:
        value = float(match.group(1))
    except ValueError:
        return _DEFAULT_CONFIDENCE
    return max(0.0, min(1.0, value))


class AnalystWorker(BaseWorker):
    """Reasons about the task and produces structured insights, trade-offs, and risks."""

    worker_type = WorkerType.ANALYST

    async def execute(self, sub_task: SubTask) -> WorkerOutput:
        """Call the LLM with the analyst prompt and parse its self-reported confidence."""
        start = time.perf_counter()
        try:
            prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
                instruction=sub_task.instruction,
                context=sub_task.context,
            )
            content, tokens = await self._call_llm(prompt, system="You are a sharp, structured analytical reasoner.")
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            return WorkerOutput(
                task_id=sub_task.task_id,
                worker_type=self.worker_type,
                content=content,
                confidence=_parse_confidence(content),
                sources=["LLM reasoning -- no external tools used"],
                execution_time_ms=elapsed_ms,
                tokens_used=tokens,
                status="success",
                error=None,
            )
        except Exception as e:
            logger.warning("AnalystWorker failed for task_id=%s: %s", sub_task.task_id, e)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return WorkerOutput(
                task_id=sub_task.task_id,
                worker_type=self.worker_type,
                content="",
                confidence=0.0,
                sources=[],
                execution_time_ms=elapsed_ms,
                tokens_used=0,
                status="failed",
                error=str(e),
            )
