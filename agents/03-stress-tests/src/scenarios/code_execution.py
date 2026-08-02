"""Failure scenarios for the code_executor tool.

Exercises the REAL production-tools CodeExecutorTool with MockCodeExecutor
injected via genuine constructor injection (`CodeExecutorTool(docker_client=mock)`),
since that tool DOES accept a docker_client parameter -- no monkeypatching.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from src.config import ProductionToolsUnavailable, import_production_module
from src.harness.base import GRACE_CRITERIA_KEYS, Scenario, ScenarioResult
from src.mocks.executor_mock import CodeFailureMode, MockCodeExecutor

try:
    import requests

    _TIMEOUT_EXC = requests.exceptions.ReadTimeout
except ImportError:  # pragma: no cover -- requests ships with docker-py, always present in practice
    requests = None
    _TIMEOUT_EXC = TimeoutError


def _unavailable_result(name: str, failure_type: str, reason: str) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=name,
        tool_name="code_execution",
        failure_type=failure_type,
        passed=False,
        failure_description="N/A -- production-tools unavailable",
        observed_behavior=f"Scenario skipped: {reason}",
        grace_criteria_met={k: False for k in GRACE_CRITERIA_KEYS},
        response_time_ms=0,
        error_class=None,
        notes="Run from a checkout that includes ../01-production-tools to exercise this scenario for real.",
    )


class _PoisonDockerClient:
    """A docker client that raises immediately if the tool ever tries to run a container --
    proves the denylist rejects blocked code before Docker is ever touched (used by C6)."""

    class _PoisonContainers:
        def run(self, image: str, **kwargs: Any) -> Any:
            raise AssertionError("Code reached Docker -- the denylist should have rejected it first")

    def __init__(self) -> None:
        self.containers = self._PoisonContainers()


class _BaseCodeExecutionScenario(Scenario):
    """Shared setup: imports the real CodeExecutorTool and injects this scenario's mock docker client."""

    tool_name = "code_execution"

    def __init__(self) -> None:
        self._module: Any = None
        self._tool: Any = None
        self._unavailable_reason: Optional[str] = None

    def build_docker_client(self) -> Any:
        """Subclasses return the mock docker client (or _PoisonDockerClient) to inject."""
        raise NotImplementedError

    async def setup(self) -> None:
        try:
            self._module = import_production_module("src.tools.code_executor")
        except ProductionToolsUnavailable as e:
            self._unavailable_reason = str(e)
            return
        self._tool = self._module.CodeExecutorTool(docker_client=self.build_docker_client())

    async def teardown(self) -> None:
        pass

    async def _check_recovery(self, code: str = "print(1)") -> bool:
        try:
            self._tool._docker_client = MockCodeExecutor(CodeFailureMode.SUCCESS)
            result = await self._tool.run(code=code)
            return result.exit_code == 0
        except Exception:
            return False


class SyntaxErrorScenario(_BaseCodeExecutionScenario):
    """C1: user code has a syntax error -- returned as a result (non-zero exit), never raised as an exception."""

    name = "code_syntax_error"
    description = "User code has invalid syntax; expects a returned ExecutionResult with exit_code!=0, not a raised exception."
    failure_type = "syntax_error"

    def build_docker_client(self) -> Any:
        return MockCodeExecutor(CodeFailureMode.SYNTAX_ERROR)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(code='def foo(: print("broken")')
            observed = f"Returned exit_code={result.exit_code}, stderr={result.stderr!r}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True  # a structured result, not an exception, is the correct "type" here
            criteria["human_readable_message"] = "SyntaxError" in result.stderr
            criteria["structured_error_response"] = result.exit_code != 0 and result.stderr != ""
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Incorrectly raised {error_class} instead of returning a non-zero exit result: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns exit_code=1 with a SyntaxError in stderr.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class RuntimeErrorScenario(_BaseCodeExecutionScenario):
    """C2: user code raises an uncaught exception -- same pattern as C1: returned, not raised."""

    name = "code_runtime_error"
    description = "User code raises an uncaught exception (1/0); expects a returned result, not a raised exception."
    failure_type = "runtime_error"

    def build_docker_client(self) -> Any:
        return MockCodeExecutor(CodeFailureMode.RUNTIME_ERROR)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(code="print(1/0)")
            observed = f"Returned exit_code={result.exit_code}, stderr={result.stderr!r}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = "ZeroDivisionError" in result.stderr
            criteria["structured_error_response"] = result.exit_code != 0 and result.stderr != ""
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Incorrectly raised {error_class} instead of returning a non-zero exit result: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns exit_code=1 with a ZeroDivisionError in stderr.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class ExecutionTimeoutScenario(_BaseCodeExecutionScenario):
    """C3: code runs past the timeout -- container is killed, timed_out=True, no lingering resources."""

    name = "code_execution_timeout"
    description = "Code exceeds its timeout (while True: pass); expects timed_out=True and the container killed cleanly."
    failure_type = "timeout"

    def build_docker_client(self) -> Any:
        return MockCodeExecutor(CodeFailureMode.TIMEOUT, timeout_exception_class=_TIMEOUT_EXC)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        docker_client = self._tool._docker_client
        start = time.perf_counter()
        try:
            result = await self._tool.run(code="while True:\n    pass", timeout_seconds=1)
            observed = f"Returned timed_out={result.timed_out}, exit_code={result.exit_code}"
            container = docker_client.containers.last_container
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.timed_out is True
            criteria["consistent_state"] = container is not None and container.killed and container.removed
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class} instead of returning a timed_out result: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock's container.wait() raises a client-side read-timeout.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class MemoryExceededScenario(_BaseCodeExecutionScenario):
    """C4: OOM-killed container -- returned as a result (exit_code=137), not a tool crash."""

    name = "code_memory_exceeded"
    description = "Code allocates too much memory and is OOM-killed; expects exit_code=137, no tool crash."
    failure_type = "memory_exceeded"

    def build_docker_client(self) -> Any:
        return MockCodeExecutor(CodeFailureMode.MEMORY_EXCEEDED)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(code="x = [0] * (10**9)")
            observed = f"Returned exit_code={result.exit_code}, stderr={result.stderr!r}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.exit_code == 137
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Tool crashed trying to capture an OOM result instead of returning exit_code=137: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns exit_code=137 (SIGKILL from an OOM kill).",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class InfiniteLoopScenario(_BaseCodeExecutionScenario):
    """C5: infinite loop caught by timeout -- partial stdout printed before the kill must be preserved."""

    name = "code_infinite_loop"
    description = "Code loops forever after printing some output; expects the partial stdout preserved, timed_out=True."
    failure_type = "infinite_loop"

    def build_docker_client(self) -> Any:
        return MockCodeExecutor(CodeFailureMode.INFINITE_LOOP, timeout_exception_class=_TIMEOUT_EXC)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        docker_client = self._tool._docker_client
        start = time.perf_counter()
        try:
            result = await self._tool.run(code="i = 0\nwhile True:\n    print(f'counting: {i}')\n    i += 1", timeout_seconds=1)
            observed = f"Returned timed_out={result.timed_out}, stdout={result.stdout!r}"
            container = docker_client.containers.last_container
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.timed_out is True and result.stdout != ""
            criteria["consistent_state"] = container is not None and container.killed and container.removed
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class} -- lost partial output instead of preserving it: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock preserves partial stdout printed before the timeout kill fires.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class BlockedImportScenario(_BaseCodeExecutionScenario):
    """C6: tests the REAL denylist -- blocked code must be rejected before Docker is ever touched."""

    name = "code_blocked_import"
    description = "Code imports subprocess; expects ToolSecurityError raised before Docker is ever touched."
    failure_type = "import_blocked"

    def build_docker_client(self) -> Any:
        return _PoisonDockerClient()

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            await self._tool.run(code="import subprocess\nsubprocess.run(['ls'])")
            observed = "Tool ran the code without raising -- the denylist is broken."
        except AssertionError as e:
            error_class = "AssertionError"
            observed = f"Code reached Docker before being rejected: {e}"
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Raised {error_class}: {e}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = error_class == "ToolSecurityError"
            criteria["human_readable_message"] = "subprocess" in str(e).lower() or "blocked" in str(e).lower() or "denylist" in str(e).lower()
            criteria["structured_error_response"] = bool(str(e).strip())
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True  # the poison client was never touched; nothing to recover from
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()),
            failure_description="Code containing 'import subprocess' submitted to the real denylist, with a docker client that raises if ever touched.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class StderrOnlyScenario(_BaseCodeExecutionScenario):
    """C7: exit_code=0 with only stderr populated -- must be treated as success, not failure."""

    name = "code_stderr_only"
    description = "Code exits 0 but writes only to stderr (a warning); expects success, stderr surfaced as info, not error."
    failure_type = "stderr_only"

    def build_docker_client(self) -> Any:
        return MockCodeExecutor(CodeFailureMode.STDERR_ONLY)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(code="import sys; sys.stderr.write(\"warning: something's wrong\\n\")")
            observed = f"Returned exit_code={result.exit_code}, stdout={result.stdout!r}, stderr={result.stderr!r}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.exit_code == 0 and result.stderr != ""
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Incorrectly treated stderr output as a failure: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns exit_code=0 with only stderr populated.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


class EmptyOutputScenario(_BaseCodeExecutionScenario):
    """C8: code runs successfully but prints nothing -- a clean, valid empty result."""

    name = "code_empty_output"
    description = "Code runs successfully but produces no output; expects a clean success result."
    failure_type = "empty_output"

    def build_docker_client(self) -> Any:
        return MockCodeExecutor(CodeFailureMode.EMPTY_OUTPUT)

    async def execute(self) -> ScenarioResult:
        if self._unavailable_reason:
            return _unavailable_result(self.name, self.failure_type, self._unavailable_reason)

        criteria = {k: False for k in GRACE_CRITERIA_KEYS}
        error_class = None
        start = time.perf_counter()
        try:
            result = await self._tool.run(code="x = 1 + 1")
            observed = f"Returned exit_code={result.exit_code}, stdout={result.stdout!r}, stderr={result.stderr!r}"
            criteria["no_unhandled_exception"] = True
            criteria["typed_error"] = True
            criteria["human_readable_message"] = True
            criteria["structured_error_response"] = result.exit_code == 0
            criteria["consistent_state"] = True
            criteria["recovery_possible"] = True
        except Exception as e:
            error_class = type(e).__name__
            observed = f"Incorrectly errored on empty output: {e}"
        response_time_ms = int((time.perf_counter() - start) * 1000)

        return ScenarioResult(
            scenario_name=self.name, tool_name=self.tool_name, failure_type=self.failure_type,
            passed=all(criteria.values()), failure_description="Mock returns exit_code=0 with empty stdout and stderr.",
            observed_behavior=observed, grace_criteria_met=criteria, response_time_ms=response_time_ms,
            error_class=error_class, notes="",
        )


ALL_SCENARIOS: List[type] = [
    SyntaxErrorScenario,
    RuntimeErrorScenario,
    ExecutionTimeoutScenario,
    MemoryExceededScenario,
    InfiniteLoopScenario,
    BlockedImportScenario,
    StderrOnlyScenario,
    EmptyOutputScenario,
]
