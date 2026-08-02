"""Abstract base class every worker inherits from, plus the shared retrying LLM call."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import MAX_RETRIES, OPENAI_API_KEY, WORKER_MODEL
from src.schemas import SubTask, WorkerOutput, WorkerType

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


class BaseWorker(ABC):
    """Common interface and shared LLM plumbing for all specialist workers."""

    worker_type: WorkerType

    @abstractmethod
    async def execute(self, sub_task: SubTask) -> WorkerOutput:
        """Execute the sub-task and return structured output. Must never raise --
        failures are captured and returned as a WorkerOutput with status='failed'."""
        raise NotImplementedError

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_llm(self, prompt: str, system: str) -> tuple[str, int]:
        """Shared LLM call with retry logic. Returns (content, tokens_used)."""
        response = await _client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        return content, tokens_used
