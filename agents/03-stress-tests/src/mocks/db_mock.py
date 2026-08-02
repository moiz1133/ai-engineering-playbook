"""Deterministic mock standing in for the psycopg async connection pool the postgres_query tool uses.

Unlike the search tool, production-tools' PostgresQueryTool DOES accept a
`pool` via constructor injection (`PostgresQueryTool(pool=...)`), so this
mock genuinely implements that pool/connection/cursor protocol and gets
passed straight into the real tool's constructor -- no monkeypatching.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, List, Optional, Tuple


class DBFailureMode(Enum):
    EMPTY_RESULTS = "empty_results"
    CONNECTION_REFUSED = "connection_refused"
    QUERY_TIMEOUT = "query_timeout"
    SYNTAX_ERROR = "syntax_error"
    PERMISSION_DENIED = "permission_denied"
    ROWS_TRUNCATED = "rows_truncated"
    NULL_VALUES = "null_values"
    TYPE_MISMATCH = "type_mismatch"
    SUCCESS = "success"


class _Col:
    """Minimal stand-in for a psycopg cursor.description entry -- only needs a `.name` attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


class MockCursor:
    def __init__(self, client: "MockDatabaseClient") -> None:
        self._client = client
        self.description: Optional[List[_Col]] = None
        self._rows: List[Tuple[Any, ...]] = []

    async def __aenter__(self) -> "MockCursor":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def execute(self, query: str, params: Optional[dict] = None) -> None:
        mode = self._client.failure_mode

        if mode == DBFailureMode.QUERY_TIMEOUT:
            if self._client.simulate_real_sleep:
                await asyncio.sleep(self._client.query_sleep_seconds)
            raise self._client.timeout_error_class("canceling statement due to statement timeout")
        if mode == DBFailureMode.SYNTAX_ERROR:
            raise self._client.syntax_error_class('syntax error at or near "WHERE"')

        if mode == DBFailureMode.EMPTY_RESULTS:
            self.description = [_Col("id"), _Col("name")]
            self._rows = []
        elif mode == DBFailureMode.ROWS_TRUNCATED:
            self.description = [_Col("n")]
            self._rows = [(i,) for i in range(500)]
        elif mode == DBFailureMode.NULL_VALUES:
            self.description = [_Col("id"), _Col("name"), _Col("email")]
            self._rows = [(1, None, None), (2, "bob", None)]
        elif mode == DBFailureMode.TYPE_MISMATCH:
            self.description = [_Col("id")]
            self._rows = [("not_a_number",)]
        else:  # SUCCESS
            self.description = [_Col("id"), _Col("name")]
            self._rows = [(1, "alice"), (2, "bob")]

    async def fetchmany(self, n: int) -> List[Tuple[Any, ...]]:
        return self._rows[:n]


class MockConnection:
    def __init__(self, client: "MockDatabaseClient") -> None:
        self._client = client

    async def __aenter__(self) -> "MockConnection":
        # A real pool raises a connection error when *acquiring* a connection,
        # before any query is ever sent -- so this fires here, not in execute().
        if self._client.failure_mode == DBFailureMode.CONNECTION_REFUSED:
            raise self._client.connection_error_class("Connection refused: 5432")
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def execute(self, sql: str) -> None:
        pass  # the tool's own "SET statement_timeout = ..." call -- always a no-op here

    def cursor(self) -> MockCursor:
        return MockCursor(self._client)


class MockDatabaseClient:
    """Drop-in replacement for a psycopg_pool.AsyncConnectionPool, deterministic per DBFailureMode.

    Pass an instance directly to `PostgresQueryTool(pool=mock_client)`.
    """

    def __init__(
        self,
        failure_mode: DBFailureMode = DBFailureMode.SUCCESS,
        connection_error_class: type = ConnectionError,
        timeout_error_class: type = TimeoutError,
        syntax_error_class: type = SyntaxError,
        simulate_real_sleep: bool = False,
        query_sleep_seconds: float = 35.0,
    ) -> None:
        """
        connection_error_class / timeout_error_class / syntax_error_class: standalone use raises plain
            built-ins by default. Wired against the real PostgresQueryTool, its except clauses match
            psycopg's own exception classes specifically (psycopg.errors.QueryCanceled, psycopg.Error) --
            pass those in so failures are actually caught as intended.
        simulate_real_sleep: if True, actually asyncio.sleep(query_sleep_seconds) before raising the
            timeout error (exercises real elapsed-time behavior); if False (default), raises immediately
            so the scenario stays fast while still verifying the tool's timeout *handling* path.
        """
        self.failure_mode = failure_mode
        self.connection_error_class = connection_error_class
        self.timeout_error_class = timeout_error_class
        self.syntax_error_class = syntax_error_class
        self.simulate_real_sleep = simulate_real_sleep
        self.query_sleep_seconds = query_sleep_seconds

    def connection(self) -> MockConnection:
        return MockConnection(self)

    async def open(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Matches psycopg_pool.AsyncConnectionPool.open() so PostgresQueryTool's lazy-open call is a no-op here."""
        return None

    async def close(self) -> None:
        return None
