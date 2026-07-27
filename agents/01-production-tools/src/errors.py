"""Custom exception hierarchy every tool raises instead of bare Exception, so callers can handle failure modes precisely."""

from __future__ import annotations


class ToolError(Exception):
    """Base class for every exception raised by a tool in this package."""


class ToolInputError(ToolError):
    """Raised when tool input fails validation beyond what Pydantic alone catches (e.g. a semantically invalid value)."""


class ToolExecutionError(ToolError):
    """Raised when a tool's underlying operation fails to run (network error, subprocess failure, DB error, etc.)."""


class ToolSecurityError(ToolError):
    """Raised when a tool refuses to run because the request was blocked by a security control (SSRF, path traversal, denylist, read-only enforcement)."""


class ToolTimeoutError(ToolError):
    """Raised when a tool's execution exceeds its allotted time budget."""
