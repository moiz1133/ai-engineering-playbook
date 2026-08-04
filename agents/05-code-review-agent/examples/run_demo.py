"""Scripted demo: runs the full Critic/Fixer/Tester group chat against all four sample
buggy files in examples/sample_code/, one after another, saving a real review for each
to outputs/. Run from the project root:

    python -m examples.run_demo
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.config import MAX_ROUNDS
from src.main import _run

console = Console()

_SAMPLE_DIR = Path(__file__).parent / "sample_code"
_SAMPLE_FILES = [
    "buggy_fibonacci.py",
    "insecure_auth.py",
    "slow_query.py",
    "missing_tests.py",
]


def main() -> None:
    for filename in _SAMPLE_FILES:
        file_path = _SAMPLE_DIR / filename
        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold cyan]Reviewing: {filename}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")
        _run(str(file_path), None, MAX_ROUNDS)


if __name__ == "__main__":
    main()
