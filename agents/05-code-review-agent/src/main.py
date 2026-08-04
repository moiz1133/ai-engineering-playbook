"""CLI entry point.

    python -m src.main --file examples/sample_code/buggy_fibonacci.py
    python -m src.main --code "def add(a, b): return a + b"
    python -m src.main --file examples/sample_code/insecure_auth.py --rounds 4
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import autogen
from rich.console import Console

from src.agents.critic import create_critic
from src.agents.fixer import create_fixer
from src.agents.tester import create_tester
from src.config import HUMAN_INPUT_MODE, LLM_CONFIG, MAX_ROUNDS, OUTPUT_DIR, validate_config
from src.group_chat.manager import create_group_chat
from src.group_chat.termination import is_approved
from src.schemas import ReviewOutput, ReviewVerdict

console = Console()

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CRITIC_SECTION_RE = re.compile(
    r"###\s*(Bugs Found|Security Issues|Performance Issues|Missing Error Handling|Style / Readability)\s*\n"
    r"(.*?)(?=\n###|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_code_block(content: str) -> str:
    """Pulls the first fenced code block out of a message. Returns "" if none is present
    -- callers must not assume every message actually contains one."""
    match = _CODE_BLOCK_RE.search(content or "")
    return match.group(1).strip() if match else ""


def _extract_issue_bullets(content: str) -> list[str]:
    """Pulls every '- ...' bullet out of a Critic message's findings sections, skipping
    the "None found" placeholder AutoGen agents use when a section has nothing to report."""
    issues: list[str] = []
    for _section_name, body in _CRITIC_SECTION_RE.findall(content or ""):
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            text = line.lstrip("-").strip()
            if text and text.lower() not in ("none found", "none"):
                issues.append(text)
    return issues


def extract_output(messages: list[dict[str, Any]], original_code: str) -> ReviewOutput:
    """Parse the group chat's conversation history into a structured ReviewOutput.

    Robust to edge cases: missing code blocks, agents that never got a turn, and an
    incomplete/truncated final message all degrade to empty values rather than raising.
    """
    critic_messages = [m for m in messages if m.get("name") == "Critic"]
    fixer_messages = [m for m in messages if m.get("name") == "Fixer"]
    tester_messages = [m for m in messages if m.get("name") == "Tester"]

    final_code = _extract_code_block(fixer_messages[-1].get("content", "")) if fixer_messages else original_code
    test_suite = _extract_code_block(tester_messages[-1].get("content", "")) if tester_messages else ""

    all_issues: list[str] = []
    for msg in critic_messages:
        all_issues.extend(_extract_issue_bullets(msg.get("content", "")))

    verdict = ReviewVerdict.APPROVED if is_approved(messages) else ReviewVerdict.MAX_ROUNDS_HIT

    return ReviewOutput(
        original_code=original_code,
        final_code=final_code or original_code,
        test_suite=test_suite,
        all_issues_found=all_issues,
        rounds_completed=len(tester_messages),
        verdict=verdict,
        conversation_log=messages,
        output_file="",  # filled in by the caller once the file is actually written
        total_tokens_used=0,  # filled in by the caller from autogen.gather_usage_summary
        total_time_ms=0,  # filled in by the caller
    )


def _load_code(file_path: str | None, code: str | None) -> tuple[str, str]:
    """Returns (code, source_label). Raises FileNotFoundError with a clean message if
    --file points at a path that doesn't exist -- callers turn this into a clean exit,
    not a raw traceback."""
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {file_path}")
        return path.read_text(encoding="utf-8"), path.name
    assert code is not None
    return code, "inline_code"


def _save_review(review: ReviewOutput, source_label: str) -> str:
    """Writes the review to outputs/review_{filename}_{timestamp}.md and returns the path."""
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(source_label).stem
    path = out_dir / f"review_{stem}_{timestamp}.md"

    verdict_line = (
        f"APPROVED after {review.rounds_completed} rounds"
        if review.verdict == ReviewVerdict.APPROVED
        else f"MAX_ROUNDS_HIT after {review.rounds_completed} rounds"
    )

    conversation_section = []
    round_num = 0
    for msg in review.conversation_log:
        name = msg.get("name")
        if name == "Critic":
            round_num += 1
        if name in ("Critic", "Fixer", "Tester"):
            conversation_section.append(f"### Round {round_num} -- {name}\n\n{msg.get('content', '')}\n")

    issues_section = "\n".join(f"- {issue}" for issue in review.all_issues_found) or "- None recorded"

    content = f"""# Code Review: {source_label}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Verdict: {verdict_line}

## Original Code
```python
{review.original_code}
```

## Final Fixed Code
```python
{review.final_code}
```

## Final Test Suite
```python
{review.test_suite}
```

## Issues Found (all rounds)
{issues_section}

## Conversation Log

{chr(10).join(conversation_section)}
"""
    path.write_text(content, encoding="utf-8")
    return str(path)


def _print_summary(review: ReviewOutput, elapsed_seconds: float) -> None:
    console.print("-" * 40)
    if review.verdict == ReviewVerdict.APPROVED:
        console.print(f"[bold green]APPROVED[/bold green] after {review.rounds_completed} rounds ({elapsed_seconds:.1f}s)")
    else:
        console.print(
            f"[bold yellow]MAX_ROUNDS_HIT[/bold yellow] after {review.rounds_completed} rounds ({elapsed_seconds:.1f}s)"
        )
    tests_generated = review.test_suite.count("def test_")
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Rounds: {review.rounds_completed} of {MAX_ROUNDS if MAX_ROUNDS else '?'} max")
    console.print(f"  Issues found: {len(review.all_issues_found)}")
    console.print(f"  Tests generated: {tests_generated}")
    console.print(f"  Tokens used: {review.total_tokens_used:,}")


def _run(file_path: str | None, code: str | None, max_rounds: int) -> None:
    validate_config()

    original_code, source_label = _load_code(file_path, code)

    console.print("[bold]Code Review Multi-Agent System[/bold]")
    console.print(f"Source: {source_label}\n")
    console.print("Starting group chat...")
    console.print("-" * 40)

    critic = create_critic(LLM_CONFIG)
    fixer = create_fixer(LLM_CONFIG)
    tester = create_tester(LLM_CONFIG)
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        human_input_mode=HUMAN_INPUT_MODE,
        max_consecutive_auto_reply=0,
        code_execution_config=False,  # the Tester reasons about tests, never executes them
    )

    groupchat, manager = create_group_chat(critic, fixer, tester, user_proxy, max_rounds)

    start = time.perf_counter()
    user_proxy.initiate_chat(manager, message=original_code)
    elapsed_seconds = time.perf_counter() - start

    usage = autogen.gather_usage_summary([critic, fixer, tester])
    total_tokens = sum(
        data.get("total_tokens", 0)
        for model, data in usage.get("usage_including_cached_inference", {}).items()
        if model != "total_cost"
    )

    review = extract_output(groupchat.messages, original_code)
    review.total_tokens_used = total_tokens
    review.total_time_ms = int(elapsed_seconds * 1000)

    output_file = _save_review(review, source_label)
    review.output_file = output_file

    console.print("-" * 40)
    _print_summary(review, elapsed_seconds)
    console.print(f"\nSaved: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoGen code review multi-agent system on a piece of code.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="Path to a Python file to review.")
    source.add_argument("--code", help="Python code to review, given directly as a string.")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="Maximum Critic/Fixer/Tester cycles.")
    args = parser.parse_args()

    try:
        _run(args.file, args.code, args.rounds)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
