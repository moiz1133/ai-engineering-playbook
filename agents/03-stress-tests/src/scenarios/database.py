"""Failure scenarios for the postgres_query (database) tool.

Exercises the REAL production-tools PostgresQueryTool with MockDatabaseClient
injected via genuine constructor injection (`PostgresQueryTool(pool=mock)`),
since that tool DOES accept a pool parameter -- no monkeypatching needed here.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from src.config import ProductionToolsUnavailable, import_production_module
from src.harness.base import GRACE_CRITERIA_KEYS, Scenario, ScenarioResult
from src.mocks.db_mock import DBFailureMode, MockDatabaseClient

try:
    import psycopg

    _TIMEOUT_EXC = psycopg.errors.QueryCanceled
    _CONNECTION_EXC = psycopg.OperationalError
    _SYNTAX_EXC = psycopg.errors.SyntaxError
except ImportError:  # pragma: no cover -- psycopg ships with production-tools, always present in practice
    psycopg = None
    _TIMEOUT_EXC = TimeoutError
    _CONNECTION_EXC = ConnectionError
    _SYNTAX_EXC = SyntaxError


def _unavailable_result(name: str, failure_type: str, reason: str) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=name,
        tool_name="database",
        failure_type=failure_type,
        passed=False,
        failure_description="N/A -- production-tools unavailable",
        observed_behavior=f"Scenario skipped: {reason}",
        grace_criteria_met={k: False for k in GRACE_CRITERIA_KEYS},
        response_time_ms=0,
        error_class=None,
        notes="Run from a checkout that includes ../01-production-tools to exercise this scenario for real.",
    )


class _PoisonPool:
    """A pool that raises immediately if the tool ever touches it -- proves read-only
    validation happens before any DB interaction (used by D5)."""

    def connection(self) -> Any:
        raise AssertionError("Query reached the DB layer -- read-only validation should have rejected it first")


class _BaseDatabaseScenario(Scenario):
    """Shared setup: imports the real PostgresQueryTool and injects this scenario's mock pool via its constructor."""

    tool_name = "database"

    def __init__(self) -> None:
        self._module: Any = None
        self._tool: Any = None
        self._unavailable_reason: Optional[str] = None

    def build_pool(self) -> Any:
        """Subclasses return the mock pool (or _PoisonPool) to inject."""
        raise NotImplementedError

    async def setup(self) -> None:
        try:
            self._module = import_production_module("src.tools.postgres_query")
        except ProductionToolsUnavailable as e:
            self._unavailable_reason = str(e)
            return
        self._tool = self._module.PostgresQueryTool(pool=self.build_pool())

    async def teardown(self) -> None:
        pass

    async def _check_recovery(self, query: str = "SELECT 1") -> bool:
        try:
            self._tool._pool = MockDatabaseClient(DBFailureMode.SUCCESS)
            result = await self._tool.run(query=query)
            return result.row_count >= 0
        except Exception:
            return False


class EmptyResultsScenario(_BaseDatabaseScenario):
    """D1: a valid query with zero matching rows is success, not an error."""

    name = "database_empty_results"
    description = "Query is valid but matches zero rows; expects a successful empty-rows return."
    failure_type = "empty"

    def build_pool(self) -> Any:
        return MockDatabaseClient(DBFailureMode.EMPTY_RESULTS)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="SELECT id, name FROM users WHERE 1=0")
            observed = f"Returned successfully: rows={result.rows}, row_count={result.row_count}, truncated={result.truncated}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.rows == [] and result.row_count == 0
            criteria["consistent_state"] = result.truncated is False
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Incorrectly raised {error_class} on a valid zero-row query: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns zero rows for a valid query.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class ConnectionRefusedScenario(_BaseDatabaseScenario):
    """D2: the database is unreachable -- must raise ToolExecutionError with connection context."""

    name = "database_connection_refused"
    description = "DB connection is refused; expects ToolExecutionError, not a raw ConnectionError/hang."
    failure_type = "connection_refused"

    def build_pool(self) -> Any:
        return MockDatabaseClient(DBFailureMode.CONNECTION_REFUSED, connection_error_class=_CONNECTION_EXC)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="SELECT 1")
            observed = "Tool returned normally instead of raising a connection error."
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolExecutionError"
            criteria["human_readable_message"] = any(w in str(e).lower() for w in ("unavailable", "connection", "refused"))
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = await self._check_recovery()

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock raises a connection-refused error when acquiring a connection.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class QueryTimeoutScenario(_BaseDatabaseScenario):
    """D3: the query exceeds the 30s statement timeout -- must raise ToolTimeoutError."""

    name = "database_query_timeout"
    description = "Query exceeds the 30s statement_timeout; expects ToolTimeoutError, not a hang."
    failure_type = "timeout"

    def build_pool(self) -> Any:
        # simulate_real_sleep=False: raise immediately rather than actually sleeping 35s --
        # keeps the suite fast while still exercising the tool's exception-handling path.
        return MockDatabaseClient(DBFailureMode.QUERY_TIMEOUT, timeout_error_class=_TIMEOUT_EXC, simulate_real_sleep=False)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="SELECT * FROM huge_table")
            observed = "Tool returned normally instead of raising a timeout error."
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolTimeoutError"
            criteria["human_readable_message"] = "timeout" in str(e).lower() or "timed out" in str(e).lower()
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = await self._check_recovery()

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock raises psycopg's QueryCanceled, simulating statement_timeout firing.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class SyntaxErrorScenario(_BaseDatabaseScenario):
    """D4: malformed SQL -- expects a typed error with clear feedback, not a raw driver exception."""

    name = "database_syntax_error"
    description = "Query has a SQL syntax error; expects a typed error identifying the bad query."
    failure_type = "syntax_error"

    def build_pool(self) -> Any:
        return MockDatabaseClient(DBFailureMode.SYNTAX_ERROR, syntax_error_class=_SYNTAX_EXC)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="SELECT * FORM users WHERE")
            observed = "Tool returned normally instead of raising a syntax error."
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            # production-tools' current postgres_query.py catches all psycopg.Error
            # subclasses (including syntax errors) as ToolExecutionError -- it does
            # NOT distinguish "your SQL is malformed" (ToolInputError, the caller's
            # fault) from "the query ran but the DB/connection failed" (ToolExecutionError).
            criteria["typed_error"] = error_class in ("ToolInputError", "ToolExecutionError")
            criteria["human_readable_message"] = "syntax" in str(e).lower()
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
        response_time_ms = int((time.perf_counter() - start) * 1000)
        criteria["recovery_possible"] = await self._check_recovery()

        notes = ""
        if error_class == "ToolExecutionError":
            notes = (
                "Real finding: production-tools classifies SQL syntax errors as ToolExecutionError, the same "
                "class used for connection/timeout failures -- a caller can't distinguish 'fix your query' "
                "from 'the database itself is having a problem' by exception type alone. ToolInputError would "
                "be a more precise fit, since a syntax error is the caller's fault, not an execution failure."
            )
        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock raises psycopg's SyntaxError subtype on execute().",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes=notes,
        )


class PermissionDeniedScenario(_BaseDatabaseScenario):
    """D5: tests the REAL read-only enforcement -- a DELETE must be rejected before ever touching the DB."""

    name = "database_permission_denied"
    description = "A DELETE query is submitted; expects ToolSecurityError raised before the DB is ever touched."
    failure_type = "permission_denied"

    def build_pool(self) -> Any:
        return _PoisonPool()  # raises AssertionError if the tool ever calls .connection()

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(query="DELETE FROM users WHERE id = 1")
            observed = "Tool ran the DELETE without raising -- read-only enforcement is broken."
        except AssertionError as e:
            error_class = "AssertionError"
            observed = f"Query reached the DB layer before being rejected: {e}"
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolSecurityError"
            criteria["human_readable_message"] = "select" in str(e).lower() or "not allowed" in str(e).lower()
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True  # the poison pool was never touched; nothing to recover from
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Real DELETE query submitted to the real read-only validation logic, with a pool that raises if ever touched.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class RowsTruncatedScenario(_BaseDatabaseScenario):
    """D6: 500 rows available, max_rows=100 -- must return exactly 100 with truncated=True."""

    name = "database_rows_truncated"
    description = "Query would return 500 rows but max_rows=100; expects 100 rows with truncated=True."
    failure_type = "rows_truncated"

    def build_pool(self) -> Any:
        return MockDatabaseClient(DBFailureMode.ROWS_TRUNCATED)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="SELECT n FROM series", max_rows=100)
            observed = f"Returned row_count={result.row_count}, truncated={result.truncated}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.row_count == 100 and result.truncated is True
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class} instead of returning truncated results: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock has 500 available rows; tool is asked for max_rows=100.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class NullValuesScenario(_BaseDatabaseScenario):
    """D7: some columns are NULL -- must return None cleanly, never crash."""

    name = "database_null_values"
    description = "Result rows contain NULL in several columns; expects None values, no crash."
    failure_type = "null_values"

    def build_pool(self) -> Any:
        return MockDatabaseClient(DBFailureMode.NULL_VALUES)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="SELECT id, name, email FROM users")
            observed = f"Returned successfully: rows={result.rows}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.rows[0]["name"] is None and result.rows[0]["email"] is None
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class} on NULL values instead of returning None cleanly: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns rows with NULL in the name/email columns.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class TypeMismatchScenario(_BaseDatabaseScenario):
    """D8: a column returns an unexpected type -- must return the raw value as-is, never coerce or crash."""

    name = "database_type_mismatch"
    description = "A column returns a string where a number might be expected; expects the raw value, no coercion/crash."
    failure_type = "type_mismatch"

    def build_pool(self) -> Any:
        return MockDatabaseClient(DBFailureMode.TYPE_MISMATCH)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(query="SELECT id FROM weird_table")
            observed = f"Returned successfully: rows={result.rows}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.rows[0]["id"] == "not_a_number"
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class} trying to coerce/handle an unexpected type: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns a string value for a column a caller might expect to be numeric.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


ALL_SCENARIOS: List[type] = [
    EmptyResultsScenario,
    ConnectionRefusedScenario,
    QueryTimeoutScenario,
    SyntaxErrorScenario,
    PermissionDeniedScenario,
    RowsTruncatedScenario,
    NullValuesScenario,
    TypeMismatchScenario,
]
