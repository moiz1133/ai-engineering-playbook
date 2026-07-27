"""Tests for PostgresQueryTool: happy path, input validation, error handling, and read-only enforcement.
No real Postgres connection is used -- the pool/connection/cursor chain is fully faked."""

from collections import namedtuple

import psycopg
import pytest
from pydantic import ValidationError

from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError, ToolTimeoutError
from src.tools import postgres_query as postgres_query_module
from src.tools.postgres_query import PostgresQueryTool

_Col = namedtuple("Col", ["name"])


class _FakeCursor:
    def __init__(self, description, rows, raise_exc=None) -> None:
        self.description = description
        self._rows = rows
        self._raise_exc = raise_exc

    async def __aenter__(self) -> "_FakeCursor":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def execute(self, query, params=None) -> None:
        if self._raise_exc:
            raise self._raise_exc

    async def fetchmany(self, n: int):
        return self._rows[:n]


class _FakeConnection:
    def __init__(self, description, rows, raise_exc=None) -> None:
        self._description = description
        self._rows = rows
        self._raise_exc = raise_exc

    async def __aenter__(self) -> "_FakeConnection":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def execute(self, sql) -> None:
        pass  # the SET statement_timeout call

    def cursor(self):
        return _FakeCursor(self._description, self._rows, self._raise_exc)


class _FakePool:
    def __init__(self, description=None, rows=None, raise_exc=None) -> None:
        self._description = description or []
        self._rows = rows or []
        self._raise_exc = raise_exc

    def connection(self):
        return _FakeConnection(self._description, self._rows, self._raise_exc)


def _tool_with(description=None, rows=None, raise_exc=None) -> PostgresQueryTool:
    return PostgresQueryTool(pool=_FakePool(description=description, rows=rows, raise_exc=raise_exc))


async def test_happy_path_select(monkeypatch) -> None:
    tool = _tool_with(description=[_Col("id"), _Col("name")], rows=[(1, "alice"), (2, "bob")])
    result = await tool.run(query="SELECT id, name FROM users")
    assert result.row_count == 2
    assert result.columns == ["id", "name"]
    assert result.rows == [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
    assert result.truncated is False


async def test_truncation_flag_set_when_more_rows_exist() -> None:
    rows = [(i,) for i in range(5)]
    tool = _tool_with(description=[_Col("n")], rows=rows)
    result = await tool.run(query="SELECT n FROM series", max_rows=2)
    assert result.row_count == 2
    assert result.truncated is True


async def test_empty_query_raises_validation_error() -> None:
    tool = _tool_with()
    with pytest.raises(ValidationError):
        await tool.run(query="")


async def test_delete_query_raises_tool_security_error() -> None:
    tool = _tool_with()
    with pytest.raises(ToolSecurityError):
        await tool.run(query="DELETE FROM users WHERE id = 1")


async def test_update_query_raises_tool_security_error() -> None:
    tool = _tool_with()
    with pytest.raises(ToolSecurityError):
        await tool.run(query="UPDATE users SET active = false")


async def test_drop_table_raises_tool_security_error() -> None:
    tool = _tool_with()
    with pytest.raises(ToolSecurityError):
        await tool.run(query="DROP TABLE users")


async def test_cte_smuggled_delete_raises_tool_security_error() -> None:
    """sqlparse's get_type() reports this as SELECT -- the keyword-scan defense-in-depth must still catch it."""
    tool = _tool_with()
    with pytest.raises(ToolSecurityError):
        await tool.run(query="WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x")


async def test_stacked_statements_raise_tool_security_error() -> None:
    tool = _tool_with()
    with pytest.raises(ToolSecurityError):
        await tool.run(query="SELECT 1; SELECT 2;")


async def test_query_timeout_raises_tool_timeout_error() -> None:
    tool = _tool_with(raise_exc=psycopg.errors.QueryCanceled("canceling statement due to statement timeout"))
    with pytest.raises(ToolTimeoutError):
        await tool.run(query="SELECT * FROM huge_table")


async def test_db_error_raises_tool_execution_error() -> None:
    tool = _tool_with(raise_exc=psycopg.errors.UndefinedTable('relation "nope" does not exist'))
    with pytest.raises(ToolExecutionError):
        await tool.run(query="SELECT * FROM nope")


async def test_max_rows_validation() -> None:
    tool = _tool_with()
    with pytest.raises(ValidationError):
        await tool.run(query="SELECT 1", max_rows=5000)


class _FakeSelfCreatedPool:
    """Stands in for AsyncConnectionPool when the tool creates its own pool (no pool injected)."""

    def __init__(self, conninfo, open=False) -> None:
        self.opened = False

    async def open(self, wait=True) -> None:
        self.opened = True

    def connection(self):
        return _FakeConnection([_Col("n")], [(1,)])

    async def close(self) -> None:
        self.opened = False


async def test_lazily_opens_a_self_created_pool_exactly_once(monkeypatch) -> None:
    """Regression test: a self-created pool must be explicitly opened before use (psycopg_pool requires
    an explicit `await pool.open()` when constructed with open=False) -- this was a real bug caught by
    actually running examples/basic_usage.py, which failed with "the pool 'pool-1' is not open yet"."""
    monkeypatch.setattr(postgres_query_module, "AsyncConnectionPool", _FakeSelfCreatedPool)

    tool = PostgresQueryTool()  # no injected pool -> owns and must open its own
    assert tool._pool_opened is False

    await tool.run(query="SELECT 1")

    assert tool._pool_opened is True
    assert tool._pool.opened is True
