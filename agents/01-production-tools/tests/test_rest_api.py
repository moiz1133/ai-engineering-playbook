"""Tests for RestApiTool: happy path, input validation, error handling, and SSRF protection (including DNS-based and redirect-based SSRF). All network calls are mocked."""

import socket

import httpx
import pytest
from pydantic import ValidationError

from src.errors import ToolExecutionError, ToolSecurityError, ToolTimeoutError
from src.tools import rest_api as rest_api_module
from src.tools.rest_api import RestApiTool


class _FakeStreamResponse:
    def __init__(self, status_code: int, headers: dict | None = None, body: bytes = b"") -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def aiter_bytes(self):
        yield self._body


class _FakeStreamRaises:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *args) -> bool:
        return False


class _FakeAsyncClient:
    responses: list = []
    call_count: int = 0

    def __init__(self, timeout=None) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    def stream(self, method, url, headers=None, params=None, json=None, follow_redirects=False):
        item = _FakeAsyncClient.responses[_FakeAsyncClient.call_count]
        _FakeAsyncClient.call_count = min(_FakeAsyncClient.call_count + 1, len(_FakeAsyncClient.responses) - 1)
        return item


def _install_fake_client(monkeypatch, responses: list) -> None:
    _FakeAsyncClient.responses = responses
    _FakeAsyncClient.call_count = 0
    monkeypatch.setattr(rest_api_module.httpx, "AsyncClient", _FakeAsyncClient)


def _install_fake_dns(monkeypatch, mapping: dict) -> None:
    """mapping: hostname -> ip string. Any hostname not present falls back to a public IP."""

    def fake_getaddrinfo(host, *_args, **_kwargs):
        ip = mapping.get(host, "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(rest_api_module.socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def tool() -> RestApiTool:
    return RestApiTool()


async def test_happy_path_get_json(tool: RestApiTool, monkeypatch) -> None:
    _install_fake_dns(monkeypatch, {})
    _install_fake_client(
        monkeypatch,
        [_FakeStreamResponse(200, headers={"content-type": "application/json"}, body=b'{"ok": true}')],
    )
    result = await tool.run(url="https://api.example.com/data")
    assert result.status_code == 200
    assert result.body == {"ok": True}
    assert result.content_type == "application/json"


async def test_non_json_response_returns_raw_text(tool: RestApiTool, monkeypatch) -> None:
    _install_fake_dns(monkeypatch, {})
    _install_fake_client(monkeypatch, [_FakeStreamResponse(200, headers={"content-type": "text/plain"}, body=b"hello")])
    result = await tool.run(url="https://api.example.com/data")
    assert result.body == "hello"


async def test_invalid_scheme_raises_validation_error(tool: RestApiTool) -> None:
    with pytest.raises(ValidationError):
        await tool.run(url="ftp://example.com/file")


async def test_missing_host_raises_validation_error(tool: RestApiTool) -> None:
    with pytest.raises(ValidationError):
        await tool.run(url="http://")


async def test_localhost_raises_tool_security_error(tool: RestApiTool) -> None:
    with pytest.raises(ToolSecurityError):
        await tool.run(url="http://localhost/admin")


async def test_loopback_ip_literal_raises_tool_security_error(tool: RestApiTool) -> None:
    with pytest.raises(ToolSecurityError):
        await tool.run(url="http://127.0.0.1/admin")


async def test_metadata_endpoint_raises_tool_security_error(tool: RestApiTool) -> None:
    with pytest.raises(ToolSecurityError):
        await tool.run(url="http://169.254.169.254/latest/meta-data/")


async def test_dns_rebinding_to_private_ip_raises_tool_security_error(tool: RestApiTool, monkeypatch) -> None:
    """A hostname that isn't in the denylist by name, but resolves to a private IP, must still be blocked."""
    _install_fake_dns(monkeypatch, {"sneaky.example.com": "10.0.0.5"})
    with pytest.raises(ToolSecurityError):
        await tool.run(url="http://sneaky.example.com/")


async def test_redirect_to_metadata_endpoint_raises_tool_security_error(tool: RestApiTool, monkeypatch) -> None:
    """SSRF via redirect: the initial URL is fine, but it 302s to a blocked internal address -- must be caught."""
    _install_fake_dns(monkeypatch, {})
    _install_fake_client(
        monkeypatch,
        [_FakeStreamResponse(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})],
    )
    with pytest.raises(ToolSecurityError):
        await tool.run(url="https://public.example.com/redirect-me")


async def test_allowed_domains_enforced(tool: RestApiTool, monkeypatch) -> None:
    _install_fake_dns(monkeypatch, {})
    monkeypatch.setattr(rest_api_module, "ALLOWED_DOMAINS", {"good.example.com"})
    with pytest.raises(ToolSecurityError):
        await tool.run(url="https://not-allowed.example.com/data")


async def test_response_size_limit_enforced(tool: RestApiTool, monkeypatch) -> None:
    _install_fake_dns(monkeypatch, {})
    monkeypatch.setattr(rest_api_module, "MAX_RESPONSE_SIZE_MB", 0)
    _install_fake_client(monkeypatch, [_FakeStreamResponse(200, headers={"content-type": "text/plain"}, body=b"x")])
    with pytest.raises(ToolExecutionError):
        await tool.run(url="https://api.example.com/data")


async def test_timeout_raises_tool_timeout_error(tool: RestApiTool, monkeypatch) -> None:
    _install_fake_dns(monkeypatch, {})
    _install_fake_client(monkeypatch, [_FakeStreamRaises(httpx.TimeoutException("timed out"))])
    with pytest.raises(ToolTimeoutError):
        await tool.run(url="https://api.example.com/data")


async def test_connect_error_raises_tool_execution_error(tool: RestApiTool, monkeypatch) -> None:
    _install_fake_dns(monkeypatch, {})
    _install_fake_client(monkeypatch, [_FakeStreamRaises(httpx.ConnectError("connection refused"))])
    with pytest.raises(ToolExecutionError):
        await tool.run(url="https://api.example.com/data")


async def test_dns_failure_raises_tool_execution_error(tool: RestApiTool, monkeypatch) -> None:
    def fake_getaddrinfo(host, *_args, **_kwargs):
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr(rest_api_module.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ToolExecutionError):
        await tool.run(url="https://does-not-resolve.example.invalid/data")
