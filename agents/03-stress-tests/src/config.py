"""Centralized configuration, plus the helper that imports real tool classes from ../01-production-tools.

Both this project and production-tools name their own top-level package
`src`, so they can never both be resolvable under that name at the same
time. `production_tools_import()` swaps `sys.modules`/`sys.path` just long
enough to import what's needed from production-tools, then restores this
project's own `src.*` modules -- see its docstring for the full reasoning.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Type

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SCENARIO_TIMEOUT_SECONDS = 30
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
REPORTS_DIR = "./reports"

# Timeout settings per tool (mirrors production-tools config)
SEARCH_TIMEOUT_SECONDS = 15
DB_QUERY_TIMEOUT_SECONDS = 30
CODE_EXEC_TIMEOUT_SECONDS = 10

# Fail-fast mode: stop on first failure
FAIL_FAST = False

# Where to find production-tools, for scenarios that exercise the real tool classes.
PRODUCTION_TOOLS_DIR = Path(
    os.getenv("PRODUCTION_TOOLS_DIR", str(Path(__file__).resolve().parent.parent.parent / "01-production-tools"))
).resolve()


class ProductionToolsUnavailable(Exception):
    """Raised when production-tools can't be found or imported. Scenarios should fall back to mock-only mode."""


@contextmanager
def _swapped_src_namespace() -> Iterator[None]:
    """Temporarily evict this project's own `src.*` modules from sys.modules and add production-tools to
    sys.path, so `import src.tools.web_search` resolves to production-tools' package for the duration of
    the `with` block -- then restores everything for this project's own `src.*` imports afterward."""
    saved_src_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for key in saved_src_modules:
        del sys.modules[key]

    saved_path = list(sys.path)
    sys.path.insert(0, str(PRODUCTION_TOOLS_DIR))

    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]
        sys.path[:] = saved_path
        sys.modules.update(saved_src_modules)


def import_production_module(module_name: str):
    """Import and return a whole module from production-tools, e.g. "src.tools.web_search".

    Returns the live module object -- callers can grab classes from it AND
    monkeypatch its internals (e.g. its TAVILY_API_KEY or httpx reference)
    directly on the returned object. That object stays fully usable even
    after this function restores sys.modules back to this project's own
    `src` package; only re-importing it *by name* through the normal import
    system would break, which no caller needs to do.

    Raises ProductionToolsUnavailable if production-tools isn't present or fails to import --
    callers should catch this and fall back to mock-only scenario behavior.
    """
    if not PRODUCTION_TOOLS_DIR.exists():
        raise ProductionToolsUnavailable(f"production-tools not found at {PRODUCTION_TOOLS_DIR}")

    try:
        with _swapped_src_namespace():
            return importlib.import_module(module_name)
    except Exception as e:
        raise ProductionToolsUnavailable(f"Failed to import module {module_name!r} from production-tools: {e}") from e


def import_production_tool(module_name: str, class_name: str) -> Type:
    """Import one class from production-tools by its module path, e.g.:

        WebSearchTool = import_production_tool("src.tools.web_search", "WebSearchTool")
    """
    module = import_production_module(module_name)
    return getattr(module, class_name)
