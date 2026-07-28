"""Scripted two-session demo proving episodic and procedural memory persist across separate Assistant instances.

Session 1 tells the assistant three things: a name/job fact, a communication
preference, and a current-learning fact. Session 2 is a brand new Assistant
instance -- exactly what a fresh process would create -- and asks an
unrelated technical question. The reply should come out concise/technical
(procedural), reference the user's RAG learning (episodic), and address them
by name (episodic), without session 2 ever being told any of that directly.

Run with: python examples/multi_session_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel

from src.assistant import Assistant

console = Console()


def _print_memory_state(assistant: Assistant, label: str) -> None:
    console.print(Panel(assistant.memory_summary(), title=label))
    facts = assistant.episodic.list_all()
    if facts:
        console.print("[bold]Episodic facts:[/bold]")
        for f in facts:
            console.print(f"  - {f.content}")
    prefs = assistant.procedural.list_preferences()
    if prefs:
        console.print("[bold]Procedural preferences:[/bold]")
        for p in prefs:
            console.print(f"  - {p.key}: {p.value} (confidence {p.confidence:.2f})")
    console.print()


def main() -> None:
    console.rule("[bold]Session 1[/bold]")
    session1 = Assistant(session_id="demo_session_1")
    # Start from a clean slate so this demo's output is reproducible on a fresh run.
    session1.episodic.clear()
    session1.procedural.clear()

    turns = [
        "My name is Abdul and I work at Afiniti as a Senior Software Engineer.",
        "I prefer concise, technical responses without unnecessary caveats.",
        "I'm currently learning about production RAG systems.",
    ]
    for turn in turns:
        console.print(f"[bold cyan]User:[/bold cyan] {turn}")
        reply = session1.chat(turn)
        console.print(f"[bold green]Assistant:[/bold green] {reply}\n")

    _print_memory_state(session1, "Memory state after Session 1")

    console.rule("[bold]Session 2 (new Assistant instance)[/bold]")
    session2 = Assistant(session_id="demo_session_2")
    # A brand-new instance with no in-process state from session1 -- this is
    # the whole point: everything it knows below must come from disk.
    _print_memory_state(session2, "Memory state at start of Session 2 (before any chat)")

    question = "Can you explain how HNSW works?"
    console.print(f"[bold cyan]User:[/bold cyan] {question}")
    reply = session2.chat(question)
    console.print(f"[bold green]Assistant:[/bold green] {reply}\n")

    _print_memory_state(session2, "Memory state after Session 2")


if __name__ == "__main__":
    main()
