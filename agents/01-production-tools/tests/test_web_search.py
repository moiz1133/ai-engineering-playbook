"""Tests for WebSearchTool: happy path, input validation, error handling, and retry behavior. Tavily is always mocked."""

import httpx
import pytest
from pydantic import ValidationError

from src.errors import ToolExecutionError, ToolInputError, ToolTimeoutError
from src.tools import web_search as web_search_module
from src.tools.web_search import WebSearchTool


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self) -> dict:
        return self._json_data


class _FakeAsyncClient:
    """Replaces httpx.AsyncClient for tests. `responses` is a list of _FakeResponse/Exception, consumed in order."""

    def __init__(self, timeout: float | None = None) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def post(self, url: str, json: dict | None = None):
        item = _FakeAsyncClient.responses[_FakeAsyncClient.call_count]
        _FakeAsyncClient.call_count = min(_FakeAsyncClient.call_count + 1, len(_FakeAsyncClient.responses) - 1)
        if isinstance(item, Exception):
            raise item
        return item

    responses: list = []
    call_count: int = 0


def _install_fake_client(monkeypatch, responses: list) -> None:
    _FakeAsyncClient.responses = responses
    _FakeAsyncClient.call_count = 0
    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", _FakeAsyncClient)


@pytest.fixture
def tool() -> WebSearchTool:
    return WebSearchTool()


@pytest.fixture(autouse=True)
def _has_api_key(monkeypatch):
    monkeypatch.setattr(web_search_module, "TAVILY_API_KEY", "tvly-test-key")


async def test_happy_path_returns_results(tool: WebSearchTool, monkeypatch) -> None:
    _install_fake_client(
        monkeypatch,
        [
            _FakeResponse(
                200,
                {
                    "results": [
                        {"title": "HNSW Wiki", "url": "http://a.com", "content": "snippet a", "score": 0.9},
                        {"title": "HNSW Paper", "url": "http://b.com", "content": "snippet b", "score": 0.8},
                    ]
                },
            )
        ],
    )
    result = await tool.run(query="HNSW algorithm")
    assert result.total_results == 2
    assert result.results[0].title == "HNSW Wiki"
    assert result.query == "HNSW algorithm"


async def test_no_results_returns_empty_list_not_error(tool: WebSearchTool, monkeypatch) -> None:
    _install_fake_client(monkeypatch, [_FakeResponse(200, {"results": []})])
    result = await tool.run(query="an extremely obscure query")
    assert result.results == []
    assert result.total_results == 0


async def test_query_too_short_raises_validation_error(tool: WebSearchTool) -> None:
    with pytest.raises(ValidationError):
        await tool.run(query="ab")


async def test_missing_api_key_raises_tool_execution_error(tool: WebSearchTool, monkeypatch) -> None:
    monkeypatch.setattr(web_search_module, "TAVILY_API_KEY", "")
    with pytest.raises(ToolExecutionError):
        await tool.run(query="HNSW algorithm")


async def test_persistent_rate_limit_raises_tool_execution_error(tool: WebSearchTool, monkeypatch) -> None:
    _install_fake_client(monkeypatch, [_FakeResponse(429), _FakeResponse(429), _FakeResponse(429)])
    with pytest.raises(ToolExecutionError):
        await tool.run(query="HNSW algorithm")


async def test_recovers_after_one_rate_limit_retry(tool: WebSearchTool, monkeypatch) -> None:
    _install_fake_client(
        monkeypatch,
        [_FakeResponse(429), _FakeResponse(200, {"results": [{"title": "T", "url": "http://a.com", "content": "s", "score": 0.5}]})],
    )
    result = await tool.run(query="HNSW algorithm")
    assert result.total_results == 1


async def test_persistent_timeout_raises_tool_timeout_error(tool: WebSearchTool, monkeypatch) -> None:
    _install_fake_client(monkeypatch, [httpx.TimeoutException("timed out")] * 3)
    with pytest.raises(ToolTimeoutError):
        await tool.run(query="HNSW algorithm")


async def test_client_error_status_raises_tool_execution_error(tool: WebSearchTool, monkeypatch) -> None:
    _install_fake_client(monkeypatch, [_FakeResponse(401, text="unauthorized")])
    with pytest.raises(ToolExecutionError):
        await tool.run(query="HNSW algorithm")
