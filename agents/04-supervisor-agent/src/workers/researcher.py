"""Researcher worker: runs web search queries via Tavily and synthesizes a citable summary."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import MAX_RETRIES, TAVILY_API_KEY
from src.schemas import SubTask, WorkerOutput, WorkerType
from src.workers.base import BaseWorker

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher.txt"
_RESULTS_PER_QUERY = 4
_MAX_QUERIES = 3


def _extract_queries(instruction: str) -> list[str]:
    """Derive 1-3 search queries from the supervisor's instruction.

    The instruction is written by the decomposer to already be specific, so the
    simplest reliable approach is: one query per non-empty line if the instruction
    is multi-line, otherwise the whole instruction as a single query.
    """
    lines = [line.strip(" -\t") for line in instruction.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines[:_MAX_QUERIES]
    return [instruction.strip()[:400]]


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
)
async def _tavily_search(query: str) -> list[dict[str, Any]]:
    """One search call to Tavily. Returns a list of {title, url, content} dicts."""
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not configured")

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": _RESULTS_PER_QUERY,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(_TAVILY_URL, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Tavily API error {response.status_code}: {response.text[:200]}")
    data = response.json()
    return data.get("results") or []


def _format_results(all_results: list[dict[str, Any]]) -> str:
    if not all_results:
        return "(no search results returned)"
    lines = []
    for i, r in enumerate(all_results, start=1):
        lines.append(f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{r.get('content', '')[:500]}")
    return "\n\n".join(lines)


def _estimate_confidence(all_results: list[dict[str, Any]]) -> float:
    count = len(all_results)
    if count == 0:
        return 0.2
    if count < 3:
        return 0.5
    if count < 6:
        return 0.7
    return 0.9


class ResearcherWorker(BaseWorker):
    """Finds relevant, current information on a topic via web search."""

    worker_type = WorkerType.RESEARCHER

    async def execute(self, sub_task: SubTask) -> WorkerOutput:
        """Search the web for the sub-task's instruction and synthesize a structured summary."""
        start = time.perf_counter()
        queries = _extract_queries(sub_task.instruction)

        try:
            all_results: list[dict[str, Any]] = []
            for query in queries:
                results = await _tavily_search(query)
                all_results.extend(results)

            prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
                instruction=sub_task.instruction,
                context=sub_task.context,
                search_results=_format_results(all_results),
            )
            content, tokens = await self._call_llm(prompt, system="You are a meticulous research assistant.")

            sources = [r.get("url", "") for r in all_results if r.get("url")]
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            return WorkerOutput(
                task_id=sub_task.task_id,
                worker_type=self.worker_type,
                content=content,
                confidence=_estimate_confidence(all_results),
                sources=sources,
                execution_time_ms=elapsed_ms,
                tokens_used=tokens,
                status="success" if all_results else "partial",
                error=None,
            )
        except Exception as e:
            logger.warning("ResearcherWorker failed for task_id=%s: %s", sub_task.task_id, e)
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
