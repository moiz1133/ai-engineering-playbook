"""Demonstrates each of the six tools called directly, with Rich-formatted structured output.

Not every tool can hit a real external dependency in every environment (Tavily
needs an API key, code_executor needs a running Docker daemon, postgres_query
needs a real database). Where that dependency is missing, the tool's own error
handling kicks in and this script reports it clearly -- it never fakes a result.

Run with: python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    # psycopg's async mode refuses to run under asyncio's default Windows
    # event loop (ProactorEventLoop) -- the selector-based policy is required.
    # This is an application-entry-point concern, not something the
    # postgres_query tool should impose on its embedder, so it's set here.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from src.config import FILE_READER_BASE_DIR
from src.errors import ToolError
from src.tools.calculator import CalculatorTool
from src.tools.code_executor import CodeExecutorTool
from src.tools.file_reader import FileReaderTool
from src.tools.postgres_query import PostgresQueryTool
from src.tools.rest_api import RestApiTool
from src.tools.web_search import WebSearchTool

console = Console()


async def _run_demo(title: str, coro) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    try:
        result = await coro
        console.print(Pretty(result.model_dump()))
    except ToolError as e:
        console.print(Panel(f"[yellow]{type(e).__name__}:[/yellow] {e}", title="Handled gracefully", expand=False))
    console.print()


async def main() -> None:
    await _run_demo("1. Web Search", WebSearchTool().run(query="HNSW algorithm", max_results=3))

    await _run_demo("2. Code Executor", CodeExecutorTool().run(code="print(sum(range(1, 11)))"))

    FILE_READER_BASE_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = FILE_READER_BASE_DIR / "sample.txt"
    if not sample_path.exists():
        sample_path.write_text(
            "This is a sample file used by the file_reader tool demo.\nSecond line.\n", encoding="utf-8"
        )
    await _run_demo("3. File Reader", FileReaderTool().run(file_path="sample.txt"))

    # Takes up to ~30s to fail gracefully if no Postgres is configured/reachable
    # (that's the connection pool's own open timeout, not a hang).
    await _run_demo("4. Postgres Query", PostgresQueryTool().run(query="SELECT 1 AS answer"))

    await _run_demo(
        "5. REST API",
        RestApiTool().run(url="https://api.github.com/repos/anthropics/anthropic-sdk-python"),
    )

    await _run_demo("6. Calculator", CalculatorTool().run(expression="sqrt(16) + 2 * 3"))


if __name__ == "__main__":
    asyncio.run(main())
