"""CLI entry point: an interactive Rich chat interface wired to the Assistant orchestrator."""

from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.assistant import Assistant

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def _print_header(assistant: Assistant) -> None:
    console.rule("[bold]Assistant Memory System[/bold]")
    console.print(f"Session: [bold]{assistant.session_id}[/bold]")
    console.print(assistant.memory_summary())
    console.print()


def _show_memory(assistant: Assistant) -> None:
    fact_table = Table(title="Episodic Facts", show_lines=True)
    fact_table.add_column("Fact")
    fact_table.add_column("Source", style="dim")
    fact_table.add_column("Accessed", justify="right")
    for fact in assistant.episodic.list_all():
        fact_table.add_row(fact.content, fact.source, str(fact.access_count))
    console.print(fact_table)

    pref_table = Table(title="Procedural Preferences", show_lines=True)
    pref_table.add_column("Key")
    pref_table.add_column("Value")
    pref_table.add_column("Category")
    pref_table.add_column("Confidence", justify="right")
    for pref in assistant.procedural.list_preferences():
        pref_table.add_row(pref.key, pref.value, pref.category, f"{pref.confidence:.2f}")
    console.print(pref_table)


def _forget(assistant: Assistant, query: str) -> None:
    """Remove episodic facts whose content mentions query. Uses semantic search as a candidate filter, then a
    substring check on the results -- semantic search alone can return "closest" facts even when nothing truly
    matches, which would delete unrelated memories."""
    query_term = query.strip().strip('"').lower()
    if not query_term:
        console.print("[red]Usage: /memory forget <query>[/red]")
        return

    candidates = assistant.episodic.search(query_term, top_k=max(assistant.episodic.count(), 1))
    to_forget = [f for f in candidates if query_term in f.content.lower()]

    for fact in to_forget:
        assistant.episodic.forget(fact.fact_id)

    if to_forget:
        console.print(f"[yellow]Removed {len(to_forget)} fact(s) matching {query!r}:[/yellow]")
        for fact in to_forget:
            console.print(f"  - {fact.content}")
    else:
        console.print(f"[dim]No facts matched {query!r}.[/dim]")


def _handle_command(assistant: Assistant, command: str) -> bool:
    """Handle a /slash command. Returns False if the session should end."""
    parts = command.strip().split(maxsplit=2)

    if parts[0] == "/exit":
        return False

    if parts[0] == "/memory" and len(parts) >= 2:
        if parts[1] == "show":
            _show_memory(assistant)
            return True
        if parts[1] == "forget" and len(parts) >= 3:
            _forget(assistant, parts[2])
            return True
        if parts[1] == "clear" and len(parts) >= 3 and parts[2] == "working":
            assistant.working.clear()
            console.print("[yellow]Working memory cleared.[/yellow]")
            return True
        if parts[1] == "clear" and len(parts) >= 3 and parts[2] == "all":
            confirm = console.input(
                "[bold red]This wipes ALL episodic facts and preferences. Type 'yes' to confirm: [/bold red]"
            )
            if confirm.strip().lower() == "yes":
                assistant.working.clear()
                assistant.episodic.clear()
                assistant.procedural.clear()
                console.print("[red]All memory wiped.[/red]")
            else:
                console.print("[dim]Cancelled.[/dim]")
            return True

    console.print(f"[red]Unknown command: {command}[/red]")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal assistant with working, episodic, and procedural memory.")
    parser.add_argument("--session-id", type=str, required=True, help="Session identifier (not a real auth system)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    assistant = Assistant(session_id=args.session_id)
    _print_header(assistant)

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            if not _handle_command(assistant, user_input):
                break
            continue

        facts_before = {f.fact_id for f in assistant.episodic.list_all()}
        response = assistant.chat(user_input)
        console.print(f"[bold green]Assistant:[/bold green] {response}")

        for fact in assistant.episodic.list_all():
            if fact.fact_id not in facts_before:
                console.print(f'[dim][Fact extracted: "{fact.content}"][/dim]')

    console.print("\n[bold]Session ended. Facts saved to episodic memory.[/bold]")


if __name__ == "__main__":
    main()
