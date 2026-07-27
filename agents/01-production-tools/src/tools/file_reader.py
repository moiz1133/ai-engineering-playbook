"""File reader tool: reads text files from within a single bounded base directory.

Path traversal protection is the non-negotiable core of this tool -- every path is
resolved and checked against the base directory (which also catches symlink escapes,
since resolving a path follows symlinks) before any file I/O happens.
"""

from __future__ import annotations

import codecs
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.base import BaseTool
from src.config import FILE_READER_BASE_DIR
from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {"txt", "md", "json", "csv", "py", "yaml", "yml", "xml", "log", "html"}


class FileReaderInput(BaseModel):
    file_path: str = Field(..., min_length=1)
    encoding: str = "utf-8"
    max_size_mb: int = Field(default=10, ge=1, le=50)

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, v: str) -> str:
        try:
            codecs.lookup(v)
        except LookupError as e:
            raise ValueError(f"Unknown encoding: {v!r}") from e
        return v


class FileReaderOutput(BaseModel):
    content: str
    file_path: str
    size_bytes: int
    encoding: str
    line_count: int
    file_type: str
    truncated: bool = False


class FileReaderTool(BaseTool):
    """Reads a file from a bounded base directory. Rejects absolute paths, '..' traversal, and symlink escapes."""

    name = "file_reader"
    description = "Read a text file (txt/md/json/csv/py/yaml/xml/log/html) from within a bounded base directory."
    input_schema = FileReaderInput
    output_schema = FileReaderOutput

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = (base_dir or FILE_READER_BASE_DIR).resolve()

    def _resolve_safe_path(self, file_path: str) -> Path:
        """Resolve file_path against base_dir, rejecting anything that escapes it."""
        raw = Path(file_path)
        if raw.is_absolute():
            raise ToolSecurityError(f"Absolute paths are not allowed: {file_path!r}")
        if ".." in raw.parts:
            raise ToolSecurityError(f"Path traversal ('..') is not allowed: {file_path!r}")

        candidate = (self.base_dir / raw).resolve()
        if not candidate.is_relative_to(self.base_dir):
            # Catches symlink escapes too: resolve() follows symlinks, so a symlink
            # pointing outside base_dir ends up outside base_dir after resolution.
            raise ToolSecurityError(f"Resolved path escapes the base directory: {file_path!r}")

        return candidate

    async def execute(self, inputs: FileReaderInput) -> FileReaderOutput:
        """Validate the path, then read up to max_size_mb of the file's content.

        Example:
            result = await FileReaderTool().run(file_path="notes.md")
        """
        full_path = self._resolve_safe_path(inputs.file_path)

        file_type = full_path.suffix.lstrip(".").lower()
        if file_type not in _ALLOWED_EXTENSIONS:
            raise ToolInputError(
                f"Unsupported file type {file_type!r}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
            )

        if not full_path.exists():
            raise ToolExecutionError(f"File not found: {inputs.file_path}")
        if not full_path.is_file():
            raise ToolExecutionError(f"Not a regular file: {inputs.file_path}")

        try:
            size_bytes = full_path.stat().st_size
        except PermissionError as e:
            raise ToolExecutionError(f"Permission denied reading file metadata: {inputs.file_path}") from e

        max_bytes = inputs.max_size_mb * 1024 * 1024
        try:
            with full_path.open("rb") as f:
                raw = f.read(max_bytes + 1)
        except PermissionError as e:
            raise ToolExecutionError(f"Permission denied: {inputs.file_path}") from e

        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]

        try:
            content = raw.decode(inputs.encoding, errors="replace" if truncated else "strict")
        except UnicodeDecodeError as e:
            raise ToolExecutionError(f"Failed to decode file with encoding {inputs.encoding!r}: {e}") from e

        logger.info(
            "file_reader: file_path=%r size_bytes=%d truncated=%s", inputs.file_path, size_bytes, truncated
        )

        return FileReaderOutput(
            content=content,
            file_path=inputs.file_path,
            size_bytes=size_bytes,
            encoding=inputs.encoding,
            line_count=len(content.splitlines()),
            file_type=file_type,
            truncated=truncated,
        )
