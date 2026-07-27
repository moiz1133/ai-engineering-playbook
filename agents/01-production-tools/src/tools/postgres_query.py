"""Postgres query tool: executes read-only SELECT queries via a psycopg 3 async connection pool.

Read-only enforcement is layered on purpose: sqlparse's statement-type check catches
the obvious cases, a keyword denylist scanned over the raw SQL text catches CTE-smuggled
writes that get_type() alone would misreport as SELECT (e.g. a WITH-clause containing a
DELETE), and stacked multi-statement queries are rejected outright.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

import psycopg
import sqlparse
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from src.config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from src.base import BaseTool
from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError, ToolTimeoutError

logger = logging.getLogger(__name__)

_FORBIDDEN_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "GRANT", "REVOKE")
_QUERY_TIMEOUT_SECONDS = 30


class PostgresQueryInput(BaseModel):
    query: str = Field(..., min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    max_rows: int = Field(default=100, ge=1, le=1000)


class PostgresQueryOutput(BaseModel):
    rows: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time_ms: int
    truncated: bool


def _validate_read_only(query: str) -> None:
    """Reject anything but a single, plain SELECT statement."""
    statements = [s for s in sqlparse.parse(query) if str(s).strip()]
    if not statements:
        raise ToolInputError("Query is empty")
    if len(statements) > 1:
        raise ToolSecurityError("Multiple SQL statements in a single query are not allowed")

    stmt_type = statements[0].get_type()
    if stmt_type != "SELECT":
        raise ToolSecurityError(f"Only SELECT queries are allowed, got statement type: {stmt_type}")

    upper_query = query.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_query):
            raise ToolSecurityError(f"Query contains a forbidden keyword: {keyword}")


def _ensure_limit(query: str, max_rows: int) -> str:
    """Append a LIMIT clause if the query doesn't already have one -- fetches one extra row to detect truncation."""
    if re.search(r"\bLIMIT\s+\d+", query, re.IGNORECASE):
        return query
    return query.rstrip().rstrip(";") + f" LIMIT {max_rows + 1}"


class PostgresQueryTool(BaseTool):
    """Executes a read-only SELECT query and returns structured rows, columns, and truncation status."""

    name = "postgres_query"
    description = "Execute a read-only SELECT query against Postgres and return structured rows."
    input_schema = PostgresQueryInput
    output_schema = PostgresQueryOutput

    def __init__(self, pool: Optional[AsyncConnectionPool] = None) -> None:
        self._pool = pool
        self._owns_pool = pool is None
        # Injected pools (tests, or a caller managing their own pool) are
        # assumed already open; a pool we create ourselves is opened lazily
        # on first use, since AsyncConnectionPool.open() is itself async and
        # can't be awaited from this synchronous constructor.
        self._pool_opened = pool is not None

    def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            conninfo = (
                f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} "
                f"user={POSTGRES_USER} password={POSTGRES_PASSWORD}"
            )
            self._pool = AsyncConnectionPool(conninfo, open=False)
        return self._pool

    async def execute(self, inputs: PostgresQueryInput) -> PostgresQueryOutput:
        """Validate the query is read-only, run it with a server-side timeout, and cap returned rows at max_rows.

        Example:
            result = await PostgresQueryTool().run(
                query="SELECT id, name FROM users WHERE active = %(active)s",
                parameters={"active": True},
            )
        """
        _validate_read_only(inputs.query)
        query = _ensure_limit(inputs.query, inputs.max_rows)

        pool = self._get_pool()
        if self._owns_pool and not self._pool_opened:
            try:
                await pool.open(wait=True)
            except Exception as e:
                raise ToolExecutionError(f"Failed to open Postgres connection pool: {e}") from e
            self._pool_opened = True

        start = time.perf_counter()
        try:
            async with pool.connection() as conn:
                await conn.execute(f"SET statement_timeout = {_QUERY_TIMEOUT_SECONDS * 1000}")
                async with conn.cursor() as cur:
                    await cur.execute(query, inputs.parameters)
                    columns = [desc.name for desc in cur.description] if cur.description else []
                    fetched = await cur.fetchmany(inputs.max_rows + 1)
        except psycopg.errors.QueryCanceled as e:
            raise ToolTimeoutError(f"Query exceeded the {_QUERY_TIMEOUT_SECONDS}s timeout") from e
        except psycopg.Error as e:
            # psycopg's rich exception hierarchy (OperationalError for connection
            # failures, ProgrammingError for SQL syntax errors, etc.) all surface
            # here as one clearly-scoped execution failure.
            raise ToolExecutionError(f"Query execution failed: {e}") from e
        execution_time_ms = int((time.perf_counter() - start) * 1000)

        truncated = len(fetched) > inputs.max_rows
        rows_to_return = fetched[: inputs.max_rows]
        rows = [dict(zip(columns, row)) for row in rows_to_return]

        # Redact parameter *values* from logs by default -- only their names,
        # since values may carry sensitive data (emails, tokens, etc.).
        logger.info(
            "postgres_query: param_keys=%s row_count=%d truncated=%s execution_time_ms=%d",
            sorted(inputs.parameters.keys()), len(rows), truncated, execution_time_ms,
        )

        return PostgresQueryOutput(
            rows=rows,
            columns=columns,
            row_count=len(rows),
            execution_time_ms=execution_time_ms,
            truncated=truncated,
        )

    async def close(self) -> None:
        """Close the connection pool. Call on shutdown if this tool instance created its own pool.

        Example:
            tool = PostgresQueryTool()
            try:
                await tool.run(query="SELECT 1")
            finally:
                await tool.close()
        """
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool_opened = False
