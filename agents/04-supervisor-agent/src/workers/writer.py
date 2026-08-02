"""Writer worker: produces well-structured, readable markdown output on a topic."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.schemas import SubTask, WorkerOutput, WorkerType
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "writer.txt"
_FIXED_CONFIDENCE = 0.7  # writing quality is subjective; use a fixed moderate value


class WriterWorker(BaseWorker):
    """Drafts structured, readable markdown sections based on its instruction and context."""

    worker_type = WorkerType.WRITER

    async def execute(self, sub_task: SubTask) -> WorkerOutput:
        """Call the LLM with the writer prompt and return the drafted markdown."""
        start = time.perf_counter()
        try:
            prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
                instruction=sub_task.instruction,
                context=sub_task.context,
            )
            content, tokens = await self._call_llm(prompt, system="You are a clear, structured technical writer.")
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            return WorkerOutput(
                task_id=sub_task.task_id,
                worker_type=self.worker_type,
                content=content,
                confidence=_FIXED_CONFIDENCE,
                sources=[],
                execution_time_ms=elapsed_ms,
                tokens_used=tokens,
                status="success",
                error=None,
            )
        except Exception as e:
            logger.warning("WriterWorker failed for task_id=%s: %s", sub_task.task_id, e)
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
