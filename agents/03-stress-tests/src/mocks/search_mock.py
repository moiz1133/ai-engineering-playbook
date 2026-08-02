"""Deterministic mock standing in for the HTTP client the search tool talks to.

production-tools' WebSearchTool has no constructor injection point for its
HTTP client -- it constructs `httpx.AsyncClient(timeout=...)` fresh inside
`_call_tavily()` on every call. So rather than a "TavilyClient" object to
inject, `MockSearchClient` implements the same surface httpx.AsyncClient
does (async context manager + `.post()`) and gets monkeypatched in for
`httpx.AsyncClient` itself in the web_search scenarios. Standalone (no
production-tools available), scenarios call `.post()` directly instead.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureMode(Enum):
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    EMPTY_RESULTS = "empty_results"
    MALFORMED_RESPONSE = "malformed_response"
    PARTIAL_RESULTS = "partial_results"
    AUTH_FAILURE = "auth_failure"
    SUCCESS = "success"


class MockResponse:
    """Shaped like an httpx.Response -- just enough surface for WebSearchTool's _call_tavily() to use it."""

    def __init__(
        self,
        status_code: int,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.headers = headers or {}
        self.text = text or str(self._json_data)

    def json(self) -> Dict[str, Any]:
        return self._json_data


class MockSearchClient:
    """Drop-in replacement for httpx.AsyncClient, deterministic per FailureMode.

    Configure once via `MockSearchClient.configure(...)`, then use anywhere
    real code constructs `httpx.AsyncClient(timeout=...)` -- call state
    (call_count, which failure mode is "active") is tracked at the class
    level so it persists correctly across retries, where the real tool
    constructs a brand-new client instance on every attempt.
    """

    _failure_mode: FailureMode = FailureMode.SUCCESS
    _call_sequence: Optional[List[FailureMode]] = None
    _call_count: int = 0
    _timeout_sleep_seconds: float = 20.0
    _timeout_exception_class: type = TimeoutError
    _network_error_class: type = ConnectionError

    def __init__(self, timeout: Optional[float] = None) -> None:
        self._client_timeout = timeout

    @classmethod
    def configure(
        cls,
        failure_mode: FailureMode = FailureMode.SUCCESS,
        call_sequence: Optional[List[FailureMode]] = None,
        timeout_sleep_seconds: float = 20.0,
        timeout_exception_class: Optional[type] = None,
        network_error_class: Optional[type] = None,
    ) -> None:
        """Set failure behavior for every instance constructed after this call, and reset the call counter.

        failure_mode: the failure to simulate on every call.
        call_sequence: overrides failure_mode -- one FailureMode consumed per call (for cascading/retry
                       scenarios); the last entry repeats if more calls happen than entries given.
        timeout_exception_class / network_error_class: standalone use raises plain built-in TimeoutError /
            ConnectionError by default. When wired against the real WebSearchTool, its except clauses match
            httpx's own exception classes (httpx.TimeoutException, httpx.ConnectError, a HTTPError subclass)
            specifically, not the built-ins -- pass those in so the failure is actually caught as intended
            instead of escaping as an unhandled exception the real code never matches.
        """
        cls._failure_mode = failure_mode
        cls._call_sequence = list(call_sequence) if call_sequence else None
        cls._call_count = 0
        cls._timeout_sleep_seconds = timeout_sleep_seconds
        cls._timeout_exception_class = timeout_exception_class or TimeoutError
        cls._network_error_class = network_error_class or ConnectionError

    @classmethod
    def reset(cls) -> None:
        """Return to default (SUCCESS, no sequence, zero calls) -- call in teardown()."""
        cls.configure(failure_mode=FailureMode.SUCCESS)

    def _next_mode(self) -> FailureMode:
        if self._call_sequence:
            idx = min(type(self)._call_count, len(self._call_sequence) - 1)
            return self._call_sequence[idx]
        return self._failure_mode

    async def __aenter__(self) -> "MockSearchClient":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def post(self, url: str, json: Optional[Dict[str, Any]] = None) -> MockResponse:
        """Simulates one POST to the search API. Deterministic: same configured mode -> same response shape."""
        mode = self._next_mode()
        type(self)._call_count += 1
        max_results = (json or {}).get("max_results", 5)

        if mode == FailureMode.TIMEOUT:
            if self._client_timeout is not None and self._timeout_sleep_seconds > self._client_timeout:
                # Simulate the client's own timeout firing before the slow response arrives, without
                # actually waiting out the full delay -- mirrors what a real HTTP client does internally.
                raise type(self)._timeout_exception_class("The read operation timed out")
            await asyncio.sleep(self._timeout_sleep_seconds)
            return MockResponse(200, {"results": []})

        if mode == FailureMode.NETWORK_ERROR:
            raise type(self)._network_error_class("Name resolution failed")

        if mode == FailureMode.RATE_LIMIT:
            return MockResponse(429, {}, headers={"Retry-After": "60"}, text="rate limited")

        if mode == FailureMode.EMPTY_RESULTS:
            return MockResponse(200, {"results": []})

        if mode == FailureMode.MALFORMED_RESPONSE:
            return MockResponse(200, {"garbage": True, "not_what_we": "expected"})

        if mode == FailureMode.PARTIAL_RESULTS:
            results = []
            for i in range(5):
                item: Dict[str, Any] = {"title": f"Result {i}", "content": f"snippet {i}", "score": 0.5}
                item["url"] = None if i >= 2 else f"http://example.com/{i}"  # 3 of 5 missing url
                results.append(item)
            return MockResponse(200, {"results": results})

        if mode == FailureMode.AUTH_FAILURE:
            return MockResponse(401, {}, text="Unauthorized: invalid API key")

        # SUCCESS
        results = [
            {"title": f"Result {i}", "url": f"http://example.com/{i}", "content": f"snippet {i}", "score": 0.9}
            for i in range(max_results)
        ]
        return MockResponse(200, {"results": results})
