"""Calculator tool: evaluates a mathematical expression safely via an AST whitelist.

Deliberately does NOT use eval() -- even a "just do math" tool has real code-execution
risk if built on eval(), which is exactly the production-quality lesson this tool exists
to demonstrate. Every AST node type, operator, and function name is checked against an
explicit whitelist; anything else raises ToolSecurityError rather than running.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Union

from pydantic import BaseModel, Field

from src.base import BaseTool
from src.errors import ToolExecutionError, ToolInputError, ToolSecurityError

logger = logging.getLogger(__name__)

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
}
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


class CalculatorInput(BaseModel):
    expression: str = Field(..., min_length=1, max_length=500)


class CalculatorOutput(BaseModel):
    result: Union[float, int]
    expression: str


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted AST node. Raises ToolSecurityError for anything not explicitly allowed."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolSecurityError(f"Disallowed constant type: {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ToolSecurityError(f"Disallowed operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return op_fn(left, right)
        except ZeroDivisionError as e:
            raise ToolExecutionError("Division by zero") from e
        except OverflowError as e:
            raise ToolExecutionError("Numeric overflow") from e

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise ToolSecurityError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolSecurityError("Disallowed function call in expression")
        if node.keywords:
            raise ToolSecurityError("Keyword arguments are not allowed in expressions")
        args = [_eval_node(a) for a in node.args]
        try:
            return _FUNCTIONS[node.func.id](*args)
        except ValueError as e:
            raise ToolExecutionError(f"Invalid input to {node.func.id}(): {e}") from e
        except OverflowError as e:
            raise ToolExecutionError("Numeric overflow") from e

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ToolSecurityError(f"Disallowed identifier: {node.id}")
        return _CONSTANTS[node.id]

    raise ToolSecurityError(f"Disallowed expression element: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Evaluates arithmetic expressions (+, -, *, /, %, **), parentheses, sqrt/sin/cos/tan/log/exp/abs, and pi/e."""

    name = "calculator"
    description = (
        "Evaluate a mathematical expression: arithmetic operators, parentheses, "
        "sqrt/sin/cos/tan/log/exp/abs, and the constants pi/e."
    )
    input_schema = CalculatorInput
    output_schema = CalculatorOutput

    async def execute(self, inputs: CalculatorInput) -> CalculatorOutput:
        """Parse and evaluate inputs.expression via a restricted AST walk -- never eval().

        Example:
            result = await CalculatorTool().run(expression="sqrt(16) + 2")
        """
        expr = inputs.expression.strip()
        if not expr:
            raise ToolInputError("Expression must not be empty")

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ToolInputError(f"Invalid expression syntax: {e}") from e

        value: Any = _eval_node(tree)

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        logger.info("calculator: expression=%r result=%r", expr, value)
        return CalculatorOutput(result=value, expression=inputs.expression)
