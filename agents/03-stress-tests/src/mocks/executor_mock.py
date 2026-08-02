"""Deterministic mock standing in for the docker-py client the code_executor tool uses.

Like PostgresQueryTool, production-tools' CodeExecutorTool accepts a
`docker_client` via constructor injection, so this mock genuinely implements
the containers.run() -> Container.wait()/.logs()/.kill()/.remove() surface
it needs -- no monkeypatching.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class CodeFailureMode(Enum):
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    INFINITE_LOOP = "infinite_loop"
    IMPORT_BLOCKED = "import_blocked"
    STDERR_ONLY = "stderr_only"
    EMPTY_OUTPUT = "empty_output"
    SUCCESS = "success"


_MODE_RESULTS: Dict[CodeFailureMode, Dict[str, Any]] = {
    CodeFailureMode.SYNTAX_ERROR: {"status_code": 1, "stdout": b"", "stderr": b"SyntaxError: invalid syntax"},
    CodeFailureMode.RUNTIME_ERROR: {"status_code": 1, "stdout": b"", "stderr": b"ZeroDivisionError: division by zero"},
    CodeFailureMode.MEMORY_EXCEEDED: {"status_code": 137, "stdout": b"", "stderr": b"Killed: 9"},
    CodeFailureMode.STDERR_ONLY: {"status_code": 0, "stdout": b"", "stderr": b"warning: something's wrong"},
    CodeFailureMode.EMPTY_OUTPUT: {"status_code": 0, "stdout": b"", "stderr": b""},
    CodeFailureMode.SUCCESS: {"status_code": 0, "stdout": b"4\n", "stderr": b""},
}

# Modes where the container never reaches wait()'s normal return -- it's killed for
# taking too long, and only whatever was printed beforehand is available afterward.
_TIMEOUT_LIKE_MODES = (CodeFailureMode.TIMEOUT, CodeFailureMode.INFINITE_LOOP)


class MockContainer:
    def __init__(self, mode: CodeFailureMode, timeout_exception_class: type) -> None:
        self._mode = mode
        self._timeout_exception_class = timeout_exception_class
        self.killed = False
        self.removed = False
        # INFINITE_LOOP: partial output exists even though the run times out --
        # this is what C5 checks for (partial stdout preserved through a kill).
        self._partial_stdout = b"counting: 1\ncounting: 2\ncounting: 3\n" if mode == CodeFailureMode.INFINITE_LOOP else b""

    def wait(self, timeout: Optional[float] = None) -> Dict[str, int]:
        if self._mode in _TIMEOUT_LIKE_MODES:
            raise self._timeout_exception_class("timed out waiting for container")
        return {"StatusCode": _MODE_RESULTS[self._mode]["status_code"]}

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        if self._mode in _TIMEOUT_LIKE_MODES:
            if stdout and not stderr:
                return self._partial_stdout
            if stderr and not stdout:
                return b""
            return self._partial_stdout
        result = _MODE_RESULTS[self._mode]
        if stdout and not stderr:
            return result["stdout"]
        if stderr and not stdout:
            return result["stderr"]
        return result["stdout"] + result["stderr"]

    def kill(self) -> None:
        self.killed = True

    def remove(self, force: bool = True) -> None:
        self.removed = True


class _MockContainers:
    def __init__(self, mode: CodeFailureMode, timeout_exception_class: type) -> None:
        self._mode = mode
        self._timeout_exception_class = timeout_exception_class
        self.last_container: Optional[MockContainer] = None

    def run(self, image: str, **kwargs: Any) -> MockContainer:
        container = MockContainer(self._mode, self._timeout_exception_class)
        self.last_container = container
        return container


class MockCodeExecutor:
    """Drop-in replacement for a docker.DockerClient, deterministic per CodeFailureMode.

    Pass an instance directly to `CodeExecutorTool(docker_client=mock_executor)`.
    """

    def __init__(
        self,
        failure_mode: CodeFailureMode = CodeFailureMode.SUCCESS,
        timeout_exception_class: type = TimeoutError,
    ) -> None:
        """timeout_exception_class: standalone use raises plain built-in TimeoutError by default. Wired
        against the real CodeExecutorTool, its timeout handling specifically catches
        `requests.exceptions.ReadTimeout` (what container.wait(timeout=N) actually raises) -- pass
        that in so the failure is caught as intended instead of escaping unhandled."""
        self.failure_mode = failure_mode
        self.containers = _MockContainers(failure_mode, timeout_exception_class)
