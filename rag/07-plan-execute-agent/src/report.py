"""Synthesizer phase: assembles the final markdown report from all step results via one LLM call, and saves it to disk."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from src.config import OUTPUT_DIR
from src.planner import call_llm
from src.schemas import Plan, ReportData, StepResult

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesizer.txt"
_SYNTH_SYSTEM_PROMPT = "You are a professional research report writer who cites sources rigorously."


def _collect_sources(step_results: List[StepResult]) -> List[str]:
    """Deduplicated URLs across all steps, in first-seen order. Index+1 is the citation number used throughout the report."""
    seen: List[str] = []
    for result in step_results:
        for r in result.search_results:
            if r.url and r.url not in seen:
                seen.append(r.url)
    return seen


def _build_step_results_block(step_results: List[StepResult], sources: List[str]) -> str:
    lines: List[str] = []
    for result in step_results:
        step = result.step
        source_nums = [str(sources.index(r.url) + 1) for r in result.search_results if r.url in sources]
        lines.append(f"### Sub-question {step.step_number}: {step.sub_question}")
        lines.append(f"Summary: {result.summary}")
        lines.append(f"Relevant source numbers: {', '.join(source_nums) if source_nums else 'none'}")
        lines.append("")
    return "\n".join(lines)


def _build_sources_block(sources: List[str]) -> str:
    return "\n".join(f"[{i + 1}] {url}" for i, url in enumerate(sources)) or "(no sources found)"


def _render_sources_section(sources: List[str]) -> str:
    if not sources:
        return "\n## Sources\n\nNo sources were found.\n"
    lines = "\n".join(f"{i + 1}. {url}" for i, url in enumerate(sources))
    return f"\n## Sources\n\n{lines}\n"


def synthesize_report(plan: Plan, step_results: List[StepResult]) -> ReportData:
    """Call the synthesizer LLM to produce the report body, then append the deterministic Sources section.

    Citation numbers are assigned in Python (first-seen URL order) and given
    to the LLM as a fixed reference list, then the Sources section is
    rendered from that same list -- this guarantees every [n] the LLM writes
    matches the right URL, rather than trusting the LLM to enumerate and
    dedupe sources itself. Raises on LLM failure; the caller is responsible
    for falling back to save_fallback() so research work is never lost.
    """
    sources = _collect_sources(step_results)
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        topic=plan.topic,
        step_results_block=_build_step_results_block(step_results, sources),
        sources_block=_build_sources_block(sources),
    )

    markdown_body = call_llm(_SYNTH_SYSTEM_PROMPT, prompt, temperature=0.3).strip()
    full_markdown = markdown_body.rstrip() + "\n" + _render_sources_section(sources)

    return ReportData(
        topic=plan.topic,
        plan=plan,
        step_results=step_results,
        sources=sources,
        markdown_body=markdown_body,
        full_markdown=full_markdown,
    )


def save_report(report: ReportData) -> str:
    """Write the report to outputs/report_YYYYMMDD_HHMMSS.md and return the path."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = output_dir / filename
    path.write_text(report.full_markdown, encoding="utf-8")
    report.output_path = str(path)
    return str(path)


def save_fallback(plan: Plan, step_results: List[StepResult]) -> str:
    """If synthesis fails entirely, dump the raw step results to a fallback file so the research work isn't lost."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = output_dir / filename

    lines = [f"# {plan.topic} (raw research -- synthesis failed)\n"]
    for result in step_results:
        step = result.step
        lines.append(f"## {step.step_number}. {step.sub_question}\n")
        lines.append(f"Summary: {result.summary}\n")
        for r in result.search_results:
            lines.append(f"- [{r.title}]({r.url}): {r.snippet}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
