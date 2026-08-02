"""Critic worker: reviews the other workers' outputs for quality, accuracy, and gaps.

Unlike the other three workers, the Critic depends on their outputs -- the supervisor
runs it only after Researcher/Analyst/Writer complete, and formats their outputs into
this sub-task's `context` field before calling execute().
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.schemas import SubTask, WorkerOutput, WorkerType
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic.txt"
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


class CriticWorker(BaseWorker):
    """Reviews the other workers' outputs, flags gaps, and suggests specific fixes."""

    worker_type = WorkerType.CRITIC

    async def execute(self, sub_task: SubTask) -> WorkerOutput:
        """Call the LLM with the critic prompt against the other workers' formatted outputs."""
        start = time.perf_counter()
        try:
            prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
                instruction=sub_task.instruction,
                context=sub_task.context,
            )
            content, tokens = await self._call_llm(
                prompt, system="You are a rigorous, honest quality reviewer. Do not be agreeable for its own sake."
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            return WorkerOutput(
                task_id=sub_task.task_id,
                worker_type=self.worker_type,
                content=content,
                confidence=_parse_confidence(content),
                sources=["Review of other workers' outputs -- no external tools used"],
                execution_time_ms=elapsed_ms,
                tokens_used=tokens,
                status="success",
                error=None,
            )
        except Exception as e:
            logger.warning("CriticWorker failed for task_id=%s: %s", sub_task.task_id, e)
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
