"""Tests for FileReaderTool: happy path, input validation, error handling, and path-traversal security."""

import os

import pytest
from pydantic import ValidationError

from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError
from src.tools.file_reader import FileReaderTool


@pytest.fixture
def base_dir(tmp_path):
    # write_bytes (not write_text) so exact content is on disk -- write_text
    # performs platform newline translation (\n -> \r\n) on Windows, which
    # would make this fixture's expected content platform-dependent.
    (tmp_path / "sample.txt").write_bytes(b"line one\nline two\nline three\n")
    (tmp_path / "notes.exe").write_bytes(b"\x00\x01binary")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "nested.md").write_bytes(b"# Nested\n")
    return tmp_path


@pytest.fixture
def tool(base_dir) -> FileReaderTool:
    return FileReaderTool(base_dir=base_dir)


async def test_happy_path_reads_file(tool: FileReaderTool) -> None:
    result = await tool.run(file_path="sample.txt")
    assert result.content == "line one\nline two\nline three\n"
    assert result.line_count == 3
    assert result.file_type == "txt"
    assert result.truncated is False


async def test_happy_path_nested_file(tool: FileReaderTool) -> None:
    result = await tool.run(file_path="sub/nested.md")
    assert "# Nested" in result.content
    assert result.file_type == "md"


async def test_unknown_encoding_raises_validation_error(tool: FileReaderTool) -> None:
    with pytest.raises(ValidationError):
        await tool.run(file_path="sample.txt", encoding="not-a-real-encoding")


async def test_missing_file_raises_tool_execution_error(tool: FileReaderTool) -> None:
    with pytest.raises(ToolExecutionError):
        await tool.run(file_path="does_not_exist.txt")


async def test_unsupported_file_type_raises_tool_input_error(tool: FileReaderTool) -> None:
    with pytest.raises(ToolInputError):
        await tool.run(file_path="notes.exe")


async def test_absolute_path_raises_tool_security_error(tool: FileReaderTool, base_dir) -> None:
    absolute = str(base_dir / "sample.txt")
    with pytest.raises(ToolSecurityError):
        await tool.run(file_path=absolute)


async def test_path_traversal_raises_tool_security_error(tool: FileReaderTool) -> None:
    with pytest.raises(ToolSecurityError):
        await tool.run(file_path="../outside.txt")


async def test_symlink_escape_raises_tool_security_error(tool: FileReaderTool, base_dir, tmp_path_factory) -> None:
    outside_dir = tmp_path_factory.mktemp("outside")
    secret = outside_dir / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    link = base_dir / "escape_link.txt"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    with pytest.raises(ToolSecurityError):
        await tool.run(file_path="escape_link.txt")


async def test_truncates_large_file_and_sets_truncated_flag(base_dir) -> None:
    big_file = base_dir / "big.txt"
    big_file.write_text("x" * (2 * 1024 * 1024), encoding="utf-8")  # 2MB
    tool = FileReaderTool(base_dir=base_dir)

    result = await tool.run(file_path="big.txt", max_size_mb=1)
    assert result.truncated is True
    assert len(result.content.encode("utf-8")) <= 1 * 1024 * 1024
