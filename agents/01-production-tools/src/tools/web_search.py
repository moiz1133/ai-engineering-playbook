"""Web search tool: queries the Tavily API (via async httpx, not the Tavily SDK) and returns structured results."""

from __future__ import annotations

import logging
import time
from typing import List, Literal

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.base import BaseTool
from src.config import TAVILY_API_KEY
from src.errors import ToolExecutionError, ToolInputError, ToolTimeoutError

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_TIMEOUT_SECONDS = 15.0


class _RetryableTavilyError(Exception):
    """Internal marker for Tavily responses worth retrying (429 / 5xx)."""


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)
    search_depth: Literal["basic", "advanced"] = "basic"


class WebSearchOutput(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, _RetryableTavilyError)),
)
async def _call_tavily(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        response = await client.post(_TAVILY_URL, json=payload)

    if response.status_code == 429 or response.status_code >= 500:
        raise _RetryableTavilyError(f"Tavily returned retryable status {response.status_code}")
    if response.status_code != 200:
        raise ToolExecutionError(f"Tavily API error {response.status_code}: {response.text[:200]}")

    return response.json()


class WebSearchTool(BaseTool):
    """Searches the web via Tavily and returns structured, scored results (empty list on zero matches, never an error)."""

    name = "web_search"
    description = "Search the web for a query and return titles, URLs, snippets, and relevance scores."
    input_schema = WebSearchInput
    output_schema = WebSearchOutput

    async def execute(self, inputs: WebSearchInput) -> WebSearchOutput:
        """Call Tavily with retries/backoff on rate limits and transient errors.

        Example:
            result = await WebSearchTool().run(query="HNSW algorithm", max_results=5)
        """
        query = inputs.query.strip()
        if not query:
            raise ToolInputError("Query must not be empty after stripping whitespace")

        if not TAVILY_API_KEY:
            raise ToolExecutionError("TAVILY_API_KEY is not configured")

        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": inputs.search_depth,
            "max_results": inputs.max_results,
        }

        start = time.perf_counter()
        try:
            data = await _call_tavily(payload)
        except _RetryableTavilyError as e:
            raise ToolExecutionError(f"Tavily API unavailable after retries: {e}") from e
        except httpx.TimeoutException as e:
            raise ToolTimeoutError(f"Tavily search timed out after {_TIMEOUT_SECONDS}s: {e}") from e
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"Tavily request failed: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        raw_results = data.get("results") or []
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                relevance_score=max(0.0, min(1.0, float(r.get("score", 0.0) or 0.0))),
            )
            for r in raw_results
        ]

        logger.info(
            "web_search: query=%r max_results=%d result_count=%d latency_ms=%.1f",
            query, inputs.max_results, len(results), latency_ms,
        )

        return WebSearchOutput(results=results, query=inputs.query, total_results=len(results))
