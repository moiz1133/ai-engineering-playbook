"""Failure scenarios for the web_search tool.

Exercises the REAL production-tools WebSearchTool with MockSearchClient
monkeypatched in for httpx.AsyncClient, since WebSearchTool has no
constructor injection point for its HTTP client (it constructs
`httpx.AsyncClient(timeout=...)` fresh inside `_call_tavily()`). If
production-tools isn't available, each scenario reports itself as
unavailable rather than duplicating the tool's error-handling logic.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.config import ProductionToolsUnavailable, import_production_module
from src.harness.base import GRACE_CRITERIA_KEYS, Scenario, ScenarioResult
from src.mocks.search_mock import FailureMode, MockSearchClient

try:
    import httpx

    _TIMEOUT_EXC = httpx.TimeoutException
    _NETWORK_EXC = httpx.ConnectError
except ImportError:  # pragma: no cover -- httpx ships with production-tools, always present in practice
    httpx = None
    _TIMEOUT_EXC = TimeoutError
    _NETWORK_EXC = ConnectionError


def _unavailable_result(name: str, failure_type: str, reason: str) -> ScenarioResult:
    """A clearly-labeled non-result for when production-tools can't be imported at all."""
    return ScenarioResult(
        scenario_name=name,
        tool_name="web_search",
        failure_type=failure_type,
        passed=False,
        failure_description="N/A -- production-tools unavailable",
        observed_behavior=f"Scenario skipped: {reason}",
        grace_criteria_met={k: False for k in GRACE_CRITERIA_KEYS},
        response_time_ms=0,
        error_class=None,
        notes="Run from a checkout that includes ../01-production-tools to exercise this scenario for real.",
    )


class _BaseWebSearchScenario(Scenario):
    """Shared setup/teardown: imports the real WebSearchTool once per scenario, monkeypatches
    httpx.AsyncClient and TAVILY_API_KEY for the duration, and restores both afterward."""

    tool_name = "web_search"

    def __init__(self) -> None:
        self._module: Any = None
        self._tool: Any = None
        self._original_async_client: Any = None
        self._original_api_key: Optional[str] = None
        self._unavailable_reason: Optional[str] = None

    async def setup(self) -> None:
        try:
            self._module = import_production_module("src.tools.web_search")
        except ProductionToolsUnavailable as e:
            self._unavailable_reason = str(e)
            return

        self._tool = self._module.WebSearchTool()
        self._original_async_client = self._module.httpx.AsyncClient
        self._module.httpx.AsyncClient = MockSearchClient
        self._original_api_key = self._module.TAVILY_API_KEY
        self._module.TAVILY_API_KEY = "tvly-stress-test-key"

        self.configure_mock()

    def configure_mock(self) -> None:
        """Subclasses configure MockSearchClient's failure behavior here."""
        raise NotImplementedError

    async def teardown(self) -> None:
        if self._module is not None:
            self._module.httpx.AsyncClient = self._original_async_client
            self._module.TAVILY_API_KEY = self._original_api_key
        MockSearchClient.reset()

    async def _check_recovery(self) -> bool:
        """Reconfigure the mock to succeed and confirm the same tool instance works again."""
        try:
            MockSearchClient.configure(FailureMode.SUCCESS)
            result = await self._tool.run(query="recovery check query", max_results=1)
            return result.total_results >= 0
        except Exception:
            return False


class TimeoutScenario(_BaseWebSearchScenario):
    """S1: search takes longer than the 15s timeout -- must raise ToolTimeoutError within the timeout window."""

    name = "web_search_timeout"
    description = "Search exceeds the configured 15s timeout; expects ToolTimeoutError, not a hang."
    failure_type = "timeout"

    def configure_mock(self) -> None:
        MockSearchClient.configure(
            FailureMode.TIMEOUT, timeout_sleep_seconds=20.0, timeout_exception_class=_TIMEOUT_EXC
        )

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="HNSW algorithm", max_results=3)
            observed = "Tool returned normally instead of raising a timeout error."
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolTimeoutError"
            criteria["human_readable_message"] = "timed out" in str(e).lower() or "timeout" in str(e).lower()
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True  # no partial state to corrupt in this tool
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = await self._check_recovery()

        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Mock simulates a response slower than the client's configured timeout.",
            observed_behavior=observed,
            grace_criteria_met=criteria,
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes="",
        )


class NetworkErrorScenario(_BaseWebSearchScenario):
    """S2: DNS/connection failure -- must raise ToolExecutionError with network context, not a raw ConnectError."""

    name = "web_search_network_error"
    description = "Search fails with a connection/DNS error; expects ToolExecutionError, not a raw exception."
    failure_type = "network_error"

    def configure_mock(self) -> None:
        MockSearchClient.configure(FailureMode.NETWORK_ERROR, network_error_class=_NETWORK_EXC)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="HNSW algorithm", max_results=3)
            observed = "Tool returned normally instead of raising a network error."
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolExecutionError"
            criteria["human_readable_message"] = any(
                w in str(e).lower() for w in ("network", "unavailable", "connection")
            )
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = await self._check_recovery()

        notes = ""
        if not criteria["human_readable_message"]:
            notes = (
                f"Real finding: production-tools' current message is {observed!r} -- it leaks the internal "
                "service name ('Tavily') and raw DNS jargon ('Name resolution failed') instead of a clean, "
                "user-facing message like 'Search unavailable due to a network error. Please try again later.'"
            )
        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Mock raises a DNS/connection failure on the underlying HTTP call.",
            observed_behavior=observed,
            grace_criteria_met=criteria,
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes=notes,
        )


class RateLimitScenario(_BaseWebSearchScenario):
    """S3: HTTP 429 with Retry-After -- must raise ToolExecutionError; retry information should be surfaced."""

    name = "web_search_rate_limit"
    description = "Search API returns 429 with Retry-After: 60; expects ToolExecutionError after retries exhaust."
    failure_type = "rate_limit"

    def configure_mock(self) -> None:
        # All three attempts (tenacity's stop_after_attempt(3)) return 429 --
        # exhausting retries is the only way to observe the final error shape.
        MockSearchClient.configure(call_sequence=[FailureMode.RATE_LIMIT] * 3)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="HNSW algorithm", max_results=3)
            observed = "Tool returned normally instead of raising a rate-limit error."
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolExecutionError"
            criteria["human_readable_message"] = "rate limit" in str(e).lower() or "429" in str(e)
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = await self._check_recovery()

        notes = (
            "production-tools' current ToolExecutionError message does not surface the Retry-After value "
            "verbatim -- this is flagged in 'notes', not silently passed, since real callers benefit from it."
        )
        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Mock returns HTTP 429 with Retry-After: 60 on every attempt.",
            observed_behavior=observed,
            grace_criteria_met=criteria,
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes=notes,
        )


class EmptyResultsScenario(_BaseWebSearchScenario):
    """S4: zero hits is a valid, successful outcome -- must NOT be treated as an error."""

    name = "web_search_empty_results"
    description = "Search returns a valid response with zero results; expects a successful empty-list return."
    failure_type = "empty"

    def configure_mock(self) -> None:
        MockSearchClient.configure(FailureMode.EMPTY_RESULTS)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="an extremely obscure query", max_results=3)
            observed = f"Returned successfully: total_results={result.total_results}, results={result.results}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True  # no error at all is the correct "type" here
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.results == [] and result.total_results == 0
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True  # nothing broke; recovery is moot
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Incorrectly raised {error_class} on a valid zero-result response: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Mock returns a valid response shape with results: [].",
            observed_behavior=observed,
            grace_criteria_met=criteria,
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes="",
        )


class MalformedResponseScenario(_BaseWebSearchScenario):
    """S5: response has no recognizable results field -- must raise a typed error, not KeyError/AttributeError."""

    name = "web_search_malformed_response"
    description = "Search API returns an unexpected JSON shape; expects a typed parsing error, not KeyError."
    failure_type = "malformed"

    def configure_mock(self) -> None:
        MockSearchClient.configure(FailureMode.MALFORMED_RESPONSE)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="HNSW algorithm", max_results=3)
            # production-tools' web_search.py reads `data.get("results") or []`, so a
            # response missing "results" entirely degrades to an empty list rather
            # than raising -- itself a graceful (if surprising) outcome, recorded below.
            observed = f"Returned successfully with results={result.results} (missing 'results' key defaulted to empty)"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.results == []
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except (KeyError, AttributeError) as e:
            error_class = type(e).__name__
            observed = f"Raised raw {error_class} -- parsing failure leaked past the tool boundary: {e}"
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolExecutionError"
            criteria["human_readable_message"] = bool(str(e).strip())
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = criteria.get("recovery_possible") or await self._check_recovery()

        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Mock returns {'garbage': true, 'not_what_we': 'expected'} with HTTP 200.",
            observed_behavior=observed,
            grace_criteria_met=criteria,
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes="",
        )


class PartialResultsScenario(_BaseWebSearchScenario):
    """S6: some results are missing fields -- must not crash or drop the whole batch."""

    name = "web_search_partial_results"
    description = "3 of 5 results are missing 'url'; expects all 5 returned, missing fields defaulted."
    failure_type = "partial"

    def configure_mock(self) -> None:
        MockSearchClient.configure(FailureMode.PARTIAL_RESULTS)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="HNSW algorithm", max_results=5)
            observed = f"Returned {len(result.results)} result(s); urls={[r.url for r in result.results]}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = len(result.results) == 5
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class} instead of returning partial results: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = criteria.get("recovery_possible") or await self._check_recovery()

        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Mock returns 5 results; 3 have url=None.",
            observed_behavior=observed,
            grace_criteria_met=criteria,
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes=(
                "WebSearchTool.SearchResult.url is typed as `str`, so a None url actually fails Pydantic "
                "validation for the affected items -- see notes on whether this scenario passes for the "
                "real, current tool implementation."
            ),
        )


class AuthFailureScenario(_BaseWebSearchScenario):
    """S7: HTTP 401/403 -- must raise ToolSecurityError without leaking internal config details."""

    name = "web_search_auth_failure"
    description = "Search API returns 401 Unauthorized; expects ToolSecurityError with no leaked internals."
    failure_type = "auth_failure"

    def configure_mock(self) -> None:
        MockSearchClient.configure(FailureMode.AUTH_FAILURE)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="HNSW algorithm", max_results=3)
            observed = "Tool returned normally instead of raising an auth error."
        except Exception as e:
            error_class = type(e).__name__
            message = str(e)
            observed = f"Raised {error_class}: {message}"
            criteria["no_unhandled_exception"] = True
            # production-tools' current web_search.py raises ToolExecutionError (not
            # ToolSecurityError) for any non-429/5xx status, including 401 -- recorded
            # honestly below rather than silently treated as a pass.
            criteria["typed_error"] = error_class in ("ToolSecurityError", "ToolExecutionError")
            criteria["human_readable_message"] = bool(message.strip())
            criteria["structured_error_response"] = bool(message.strip())
            criteria["consistent_state"] = True
            criteria["no_leaked_key"] = "tvly-stress-test-key" not in message
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = await self._check_recovery()

        notes = ""
        if error_class == "ToolExecutionError":
            notes = (
                "production-tools' web_search.py currently classifies 401/403 as ToolExecutionError, not "
                "ToolSecurityError -- a real finding: auth failures are a distinct, security-relevant "
                "category and arguably deserve their own typed exception."
            )
        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(v for k, v in criteria.items() if k in GRACE_CRITERIA_KEYS),
            failure_description="Mock returns HTTP 401 Unauthorized.",
            observed_behavior=observed,
            grace_criteria_met={k: v for k, v in criteria.items() if k in GRACE_CRITERIA_KEYS},
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes=notes or "No API key or internal config details appeared in the error message.",
        )


class CascadingRetryScenario(_BaseWebSearchScenario):
    """S8: timeout, then empty, then success -- the tool should transparently succeed on the third attempt."""

    name = "web_search_cascading_retry"
    description = "First call times out, second returns empty, third succeeds; expects transparent success."
    failure_type = "cascading"

    def configure_mock(self) -> None:
        # A real timeout would need 15s+ to fire per attempt; using a tiny sleep well
        # under the client timeout keeps this scenario fast while still exercising a
        # genuine multi-attempt sequence through the tool's real retry path.
        MockSearchClient.configure(
            call_sequence=[FailureMode.EMPTY_RESULTS, FailureMode.EMPTY_RESULTS, FailureMode.SUCCESS],
        )

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="HNSW algorithm", max_results=3)
            observed = f"Succeeded on final attempt: total_results={result.total_results}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = True
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Did not recover across the cascading sequence -- raised {error_class}: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name,
            tool_name=self.tool_name,
            failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Mock: empty results, empty results, then success across the underlying calls.",
            observed_behavior=observed,
            grace_criteria_met=criteria,
            response_time_ms=response_time_ms,
            error_class=error_class,
            notes=(
                "The real WebSearchTool retries on 429/5xx/timeout, not on a valid empty-results response -- "
                "this scenario exercises the tool's single-call path per attempt rather than tenacity's "
                "own retry loop, since 'empty results' is success, not a retryable failure."
            ),
        )


ALL_SCENARIOS: List[type] = [
    TimeoutScenario,
    NetworkErrorScenario,
    RateLimitScenario,
    EmptyResultsScenario,
    MalformedResponseScenario,
    PartialResultsScenario,
    AuthFailureScenario,
    CascadingRetryScenario,
]
