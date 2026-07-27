"""Tests for CodeExecutorTool: happy path, input validation, error handling, timeout, and the denylist security filter.
Docker is always fully mocked -- no real container ever runs."""

import requests
from docker.errors import DockerException, ImageNotFound
import pytest
from pydantic import ValidationError

from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError
from src.tools import code_executor as code_executor_module
from src.tools.code_executor import CodeExecutorTool


class _FakeContainer:
    def __init__(self, status_code: int = 0, stdout: bytes = b"", stderr: bytes = b"", wait_exc=None) -> None:
        self._status_code = status_code
        self._stdout = stdout
        self._stderr = stderr
        self._wait_exc = wait_exc
        self.killed = False
        self.removed = False

    def wait(self, timeout=None):
        if self._wait_exc:
            raise self._wait_exc
        return {"StatusCode": self._status_code}

    def logs(self, stdout=True, stderr=True):
        if stdout and not stderr:
            return self._stdout
        if stderr and not stdout:
            return self._stderr
        return self._stdout + self._stderr

    def kill(self) -> None:
        self.killed = True

    def remove(self, force=True) -> None:
        self.removed = True


class _FakeContainers:
    def __init__(self, container=None, raise_exc=None) -> None:
        self._container = container
        self._raise_exc = raise_exc

    def run(self, image, **kwargs):
        if self._raise_exc:
            raise self._raise_exc
        return self._container


class _FakeDockerClient:
    def __init__(self, container=None, raise_exc=None) -> None:
        self.containers = _FakeContainers(container=container, raise_exc=raise_exc)


def _tool_with(container=None, raise_exc=None) -> CodeExecutorTool:
    return CodeExecutorTool(docker_client=_FakeDockerClient(container=container, raise_exc=raise_exc))


async def test_happy_path_returns_output_and_removes_container() -> None:
    container = _FakeContainer(status_code=0, stdout=b"4\n")
    tool = _tool_with(container=container)
    result = await tool.run(code="print(2 + 2)")
    assert result.exit_code == 0
    assert result.stdout == "4\n"
    assert result.timed_out is False
    assert container.removed is True


async def test_nonzero_exit_code_returned_not_raised() -> None:
    container = _FakeContainer(status_code=1, stderr=b"Traceback...\n")
    tool = _tool_with(container=container)
    result = await tool.run(code="raise ValueError('boom')")
    assert result.exit_code == 1
    assert "Traceback" in result.stderr


async def test_code_exceeding_max_length_raises_validation_error() -> None:
    tool = _tool_with()
    with pytest.raises(ValidationError):
        await tool.run(code="x" * 10001)


async def test_empty_code_raises_validation_error() -> None:
    tool = _tool_with()
    with pytest.raises(ValidationError):
        await tool.run(code="")


async def test_allowed_packages_raises_tool_input_error() -> None:
    tool = _tool_with()
    with pytest.raises(ToolInputError):
        await tool.run(code="print(1)", allowed_packages=["numpy"])


@pytest.mark.parametrize(
    "malicious_code",
    [
        "import subprocess\nsubprocess.run(['ls'])",
        "import os\nos.system('cat /etc/passwd')",
        "__import__('os').system('ls')",
        "import socket\nsocket.socket()",
        "import ctypes\nctypes.CDLL(None)",
    ],
)
async def test_denylist_catches_escape_attempts(malicious_code: str) -> None:
    # No docker client is even needed here -- the denylist must reject before Docker is touched.
    tool = CodeExecutorTool(docker_client=object())
    with pytest.raises(ToolSecurityError):
        await tool.run(code=malicious_code)


async def test_docker_daemon_not_running_raises_tool_execution_error(monkeypatch) -> None:
    def fake_from_env():
        raise DockerException("Cannot connect to the Docker daemon")

    monkeypatch.setattr(code_executor_module.docker, "from_env", fake_from_env)
    tool = CodeExecutorTool()  # no injected client -> triggers docker.from_env()
    with pytest.raises(ToolExecutionError):
        await tool.run(code="print(1)")


async def test_timeout_kills_container_and_reports_timed_out_true() -> None:
    container = _FakeContainer(wait_exc=requests.exceptions.ReadTimeout("timed out"))
    tool = _tool_with(container=container)
    result = await tool.run(code="while True: pass", timeout_seconds=1)
    assert result.timed_out is True
    assert container.killed is True
    assert container.removed is True


async def test_image_not_found_raises_tool_execution_error() -> None:
    tool = _tool_with(raise_exc=ImageNotFound("no such image: production-tools-sandbox:latest"))
    with pytest.raises(ToolExecutionError):
        await tool.run(code="print(1)")


async def test_generic_docker_error_raises_tool_execution_error() -> None:
    tool = _tool_with(raise_exc=DockerException("something went wrong"))
    with pytest.raises(ToolExecutionError):
        await tool.run(code="print(1)")
