"""Web search tool: dispatches to DuckDuckGo (default, no API key) or Tavily, based on config.SEARCH_PROVIDER."""

from __future__ import annotations

import logging
from typing import List

from src.config import SEARCH_PROVIDER, SEARCH_RESULTS_PER_STEP, TAVILY_API_KEY
from src.schemas import SearchResult

logger = logging.getLogger(__name__)


def _search_duckduckgo(query: str, max_results: int) -> List[SearchResult]:
    from ddgs import DDGS

    raw_results = DDGS().text(query, max_results=max_results)
    return [
        SearchResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
        for r in raw_results
    ]


def _search_tavily(query: str, max_results: int) -> List[SearchResult]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=query, max_results=max_results)
    return [
        SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
        for r in response.get("results", [])
    ]


def search_web(query: str, max_results: int = SEARCH_RESULTS_PER_STEP) -> List[SearchResult]:
    """Run `query` through the configured search provider. Returns an empty list (never raises) if the search fails."""
    try:
        if SEARCH_PROVIDER == "tavily":
            return _search_tavily(query, max_results)
        return _search_duckduckgo(query, max_results)
    except Exception as e:
        logger.warning("Web search failed for query %r via %s: %s", query, SEARCH_PROVIDER, e)
        return []
