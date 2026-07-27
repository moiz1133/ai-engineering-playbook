"""Tests for CalculatorTool: happy path, input validation, error handling, and the AST-whitelist security model."""

import pytest
from pydantic import ValidationError

from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError
from src.tools.calculator import CalculatorTool


@pytest.fixture
def tool() -> CalculatorTool:
    return CalculatorTool()


async def test_happy_path_arithmetic(tool: CalculatorTool) -> None:
    result = await tool.run(expression="2 + 2")
    assert result.result == 4
    assert isinstance(result.result, int)
    assert result.expression == "2 + 2"


async def test_happy_path_functions_and_constants(tool: CalculatorTool) -> None:
    result = await tool.run(expression="sqrt(16) + 2")
    assert result.result == 6


async def test_returns_float_when_not_exact_integer(tool: CalculatorTool) -> None:
    result = await tool.run(expression="1 / 3")
    assert isinstance(result.result, float)
    assert round(result.result, 4) == round(1 / 3, 4)


async def test_exact_integer_division_returns_int(tool: CalculatorTool) -> None:
    result = await tool.run(expression="4 / 2")
    assert result.result == 2
    assert isinstance(result.result, int)


async def test_max_length_validation_raises_pydantic_error(tool: CalculatorTool) -> None:
    with pytest.raises(ValidationError):
        await tool.run(expression="1" * 501)


async def test_empty_expression_raises_tool_input_error(tool: CalculatorTool) -> None:
    with pytest.raises(ToolInputError):
        await tool.run(expression="   ")


async def test_invalid_syntax_raises_tool_input_error(tool: CalculatorTool) -> None:
    with pytest.raises(ToolInputError):
        await tool.run(expression="2 +")


async def test_division_by_zero_raises_tool_execution_error(tool: CalculatorTool) -> None:
    with pytest.raises(ToolExecutionError):
        await tool.run(expression="1 / 0")


async def test_disallowed_function_raises_tool_security_error(tool: CalculatorTool) -> None:
    with pytest.raises(ToolSecurityError):
        await tool.run(expression="__import__('os')")


async def test_disallowed_identifier_raises_tool_security_error(tool: CalculatorTool) -> None:
    with pytest.raises(ToolSecurityError):
        await tool.run(expression="os")
