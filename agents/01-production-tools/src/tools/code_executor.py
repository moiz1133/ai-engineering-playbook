"""Sandboxed code executor tool: runs Python code inside an isolated, non-root, network-disabled Docker container.

This is the highest-risk tool in the package. The denylist below is a first-pass
filter only -- it's trivially bypassable with enough cleverness (string
concatenation, encoding tricks, etc.). The *real* security boundary is the
container itself: no network, a read-only filesystem except /tmp, memory/CPU
caps, and a non-root user, all enforced via docker-py's container.run()
arguments and destroyed after every single execution.
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

import docker
import requests
from docker.errors import APIError, DockerException, ImageNotFound
from pydantic import BaseModel, Field

from src.base import BaseTool
from src.config import CODE_EXECUTOR_CPU_LIMIT, CODE_EXECUTOR_MEMORY_LIMIT, DOCKER_SANDBOX_IMAGE
from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError

logger = logging.getLogger(__name__)

_DENYLIST_PATTERNS = [
    re.compile(r"\bimport\s+subprocess\b"),
    re.compile(r"\bimport\s+socket\b"),
    re.compile(r"\bimport\s+ctypes\b"),
    re.compile(r"\bos\.(system|popen|exec\w*|fork|kill)\s*\("),
    re.compile(r"__import__\s*\(\s*[\"']os[\"']\s*\)"),
    re.compile(r"__import__\s*\(\s*[\"']subprocess[\"']\s*\)"),
]


class CodeExecutorInput(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    timeout_seconds: int = Field(default=10, ge=1, le=30)
    allowed_packages: List[str] = Field(default_factory=list)


class CodeExecutorOutput(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    timed_out: bool


def _check_denylist(code: str) -> None:
    """First-pass filter for obvious sandbox-escape attempts. Not the real defense -- container isolation is."""
    for pattern in _DENYLIST_PATTERNS:
        if pattern.search(code):
            raise ToolSecurityError(f"Code contains a denylisted pattern: {pattern.pattern}")


class CodeExecutorTool(BaseTool):
    """Runs a Python snippet in a locked-down Docker container and returns stdout/stderr/exit code."""

    name = "code_executor"
    description = "Execute a Python code snippet in an isolated sandbox (no network, no filesystem writes outside /tmp)."
    input_schema = CodeExecutorInput
    output_schema = CodeExecutorOutput

    def __init__(self, docker_client: Optional[Any] = None) -> None:
        self._docker_client = docker_client

    def _get_client(self) -> Any:
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except DockerException as e:
                raise ToolExecutionError(f"Docker daemon is not available: {e}") from e
        return self._docker_client

    async def execute(self, inputs: CodeExecutorInput) -> CodeExecutorOutput:
        """Write code to a temp file, run it in a locked-down container, and capture stdout/stderr/exit code.

        Example:
            result = await CodeExecutorTool().run(code="print(2 + 2)")
        """
        _check_denylist(inputs.code)

        if inputs.allowed_packages:
            # The sandbox has no network access (non-negotiable), so pip
            # installs cannot run inside it -- reject rather than silently
            # ignoring a field the caller expects to do something.
            raise ToolInputError(
                "allowed_packages is not supported: the sandbox has no network access, "
                "so packages cannot be installed inside it."
            )

        client = self._get_client()
        code_hash = hashlib.sha256(inputs.code.encode("utf-8")).hexdigest()[:12]

        with tempfile.TemporaryDirectory() as tmp_dir:
            code_path = Path(tmp_dir) / "code.py"
            code_path.write_text(inputs.code, encoding="utf-8")

            container = None
            timed_out = False
            exit_code = -1
            start = time.perf_counter()
            try:
                container = client.containers.run(
                    DOCKER_SANDBOX_IMAGE,
                    volumes={str(code_path.resolve()): {"bind": "/code.py", "mode": "ro"}},
                    network_disabled=True,
                    mem_limit=CODE_EXECUTOR_MEMORY_LIMIT,
                    nano_cpus=int(CODE_EXECUTOR_CPU_LIMIT * 1_000_000_000),
                    read_only=True,
                    tmpfs={"/tmp": ""},
                    detach=True,
                    stdout=True,
                    stderr=True,
                )

                try:
                    wait_result = container.wait(timeout=inputs.timeout_seconds)
                    exit_code = wait_result.get("StatusCode", -1)
                except requests.exceptions.ReadTimeout:
                    # wait() timing out client-side does NOT stop the container --
                    # it may still be running, so it must be killed explicitly.
                    timed_out = True
                    try:
                        container.kill()
                    except APIError:
                        pass  # already stopped/removed between the timeout and the kill

                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            except ImageNotFound as e:
                raise ToolExecutionError(
                    f"Sandbox image {DOCKER_SANDBOX_IMAGE!r} not found -- "
                    "build it from docker/sandbox.Dockerfile first"
                ) from e
            except DockerException as e:
                raise ToolExecutionError(f"Container execution failed: {e}") from e
            finally:
                if container is not None:
                    try:
                        container.remove(force=True)
                    except APIError:
                        pass  # no state persists either way -- best-effort cleanup

        execution_time_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "code_executor: code_hash=%s exit_code=%d timed_out=%s execution_time_ms=%d",
            code_hash, exit_code, timed_out, execution_time_ms,
        )

        return CodeExecutorOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_ms=execution_time_ms,
            timed_out=timed_out,
        )
