"""Minimal OpenAI function-calling loop wired to all six tools -- no framework, just the SDK.

Demonstrates that BaseTool.to_openai_schema() plugs directly into OpenAI's
`tools=` parameter, and that a full tool-call round trip -- the model picks a
tool, we execute it, we feed the structured result back, the model answers --
works end to end with real tools. The orchestration here is deliberately
minimal (a single-round dispatch loop); there's no framework and no planning.

Run with: python examples/with_llm.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from src.base import BaseTool
from src.tools.calculator import CalculatorTool
from src.tools.code_executor import CodeExecutorTool
from src.tools.file_reader import FileReaderTool
from src.tools.postgres_query import PostgresQueryTool
from src.tools.rest_api import RestApiTool
from src.tools.web_search import WebSearchTool

_MODEL = "gpt-4o-mini"

TOOLS: Dict[str, BaseTool] = {
    t.name: t
    for t in [
        WebSearchTool(),
        CodeExecutorTool(),
        FileReaderTool(),
        PostgresQueryTool(),
        RestApiTool(),
        CalculatorTool(),
    ]
}

console = Console()


async def _dispatch_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Run the named tool with LLM-supplied arguments, returning a JSON string either way -- result or error."""
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = await tool.run(**arguments)
        return result.model_dump_json()
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


async def run_conversation(user_query: str) -> None:
    """Send user_query to gpt-4o-mini with all six tools available, dispatch any tool calls, and print the full trace."""
    client = OpenAI()
    tool_schemas = [t.to_openai_schema() for t in TOOLS.values()]
    messages = [{"role": "user", "content": user_query}]

    console.rule("[bold]Conversation Trace[/bold]")
    console.print(Panel(user_query, title="User"))

    response = client.chat.completions.create(model=_MODEL, messages=messages, tools=tool_schemas)
    message = response.choices[0].message

    if not message.tool_calls:
        console.print(Panel(message.content or "", title="Assistant (no tool call)"))
        return

    messages.append(message.model_dump(exclude_unset=True))

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments or "{}")
        console.print(Panel(f"{name}({arguments})", title="Tool Call"))

        tool_result_json = await _dispatch_tool_call(name, arguments)
        console.print(Panel(tool_result_json, title=f"Tool Result: {name}"))

        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result_json})

    final_response = client.chat.completions.create(model=_MODEL, messages=messages, tools=tool_schemas)
    final_message = final_response.choices[0].message
    console.print(Panel(final_message.content or "", title="Assistant (final answer)"))


async def main() -> None:
    await run_conversation("What is the square root of 144, plus 8?")


if __name__ == "__main__":
    asyncio.run(main())
