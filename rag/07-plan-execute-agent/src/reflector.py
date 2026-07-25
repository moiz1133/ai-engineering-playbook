"""Self-reflection phase: critiques a synthesized report and, in a bounded loop, revises it or re-executes
targeted searches to fill gaps -- without ever re-planning the original Plan.

Reuses call_llm() from src.planner (the project's single LLM-call wrapper)
and search_web() from src.tools.web_search, per the existing patterns in
this codebase.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.config import (
    MAX_ADDITIONAL_QUERIES_PER_ITERATION,
    MAX_REFLECTION_ITERATIONS,
    MAX_REFLECTION_TOKEN_BUDGET,
    REFLECTION_CONFIDENCE_THRESHOLD,
    REFLECTION_LOG_DIR,
)
from src.planner import call_llm
from src.schemas import Critique, Plan, ReflectionIteration, ReflectionResult, SearchResult, StepResult
from src.tools.web_search import search_web

logger = logging.getLogger(__name__)

_CRITIC_PROMPT_PATH = Path(__file__).parent / "prompts" / "critic.txt"
_REVISER_PROMPT_PATH = Path(__file__).parent / "prompts" / "reviser.txt"

_CRITIC_SYSTEM_PROMPT = "You are a rigorous, skeptical research reviewer."
_REVISER_SYSTEM_PROMPT = "You are a professional research report writer who cites sources rigorously."


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences an LLM might wrap JSON output in. Mirrors planner._strip_json_fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[len("json") :]
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Rough ~4-chars-per-token estimate, used only as a cost guard since call_llm() doesn't expose exact usage."""
    return max(1, len(text) // 4)


def _collect_sources(step_results: List[StepResult]) -> List[str]:
    """Deduplicated URLs across all steps, in first-seen order. Mirrors report._collect_sources so citation numbers stay consistent with the original report."""
    seen: List[str] = []
    for result in step_results:
        for r in result.search_results:
            if r.url and r.url not in seen:
                seen.append(r.url)
    return seen


def _strip_sources_section(markdown: str) -> str:
    """Remove a trailing '## Sources' section (as rendered by report.py) so revision prompts see body content only."""
    marker = "\n## Sources"
    idx = markdown.rfind(marker)
    return markdown[:idx].rstrip() + "\n" if idx != -1 else markdown.rstrip() + "\n"


def _render_sources_section(sources: List[str]) -> str:
    """Mirrors report._render_sources_section so the final reflected report looks identical in format to a normal one."""
    if not sources:
        return "\n## Sources\n\nNo sources were found.\n"
    lines = "\n".join(f"{i + 1}. {url}" for i, url in enumerate(sources))
    return f"\n## Sources\n\n{lines}\n"


def _render_plan_block(plan: Plan) -> str:
    return "\n".join(f"{s.step_number}. {s.sub_question}" for s in plan.steps)


def _render_critique_block(critique: Critique) -> str:
    lines = [
        f"Overall assessment: {critique.overall_assessment}",
        f"Confidence: {critique.confidence}",
        "Missing information:",
    ]
    lines += [f"- {m}" for m in critique.missing_information] if critique.missing_information else ["- (none noted)"]
    lines.append("Weak sections:")
    lines += [f"- {w}" for w in critique.weak_sections] if critique.weak_sections else ["- (none noted)"]
    return "\n".join(lines)


class Reflector:
    """Runs the bounded self-reflection loop (critique -> decide -> revise/re-execute) on a synthesized report."""

    def __init__(self) -> None:
        self._critic_template = _CRITIC_PROMPT_PATH.read_text(encoding="utf-8")
        self._reviser_template = _REVISER_PROMPT_PATH.read_text(encoding="utf-8")
        self._topic: str = ""
        self._plan: Optional[Plan] = None
        self._sources: List[str] = []  # running, first-seen-order source list across the whole loop
        self._tokens_used_estimate: int = 0

    def _call_llm_tracked(self, system_prompt: str, user_prompt: str, *, temperature: float) -> str:
        response = call_llm(system_prompt, user_prompt, temperature=temperature)
        self._tokens_used_estimate += _estimate_tokens(user_prompt) + _estimate_tokens(response)
        return response

    def _try_parse_critique(self, raw: str) -> Optional[Critique]:
        try:
            data = json.loads(_strip_json_fences(raw))
            return Critique(**data)
        except Exception as e:
            logger.warning("Failed to parse critique JSON: %s", e)
            return None

    def critique(self, report: str, topic: str, plan: Plan) -> Critique:
        """Phase 1: call the critic LLM. Retries once with a stricter prompt if the first response isn't valid JSON."""
        prompt = self._critic_template.format(topic=topic, plan_block=_render_plan_block(plan), report=report)

        raw = self._call_llm_tracked(_CRITIC_SYSTEM_PROMPT, prompt, temperature=0.2)
        parsed = self._try_parse_critique(raw)
        if parsed is not None:
            return parsed

        logger.warning("Critic returned malformed JSON, retrying with a stricter prompt")
        strict_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. Return ONLY a single "
            "valid JSON object, with no markdown fences, no commentary, and no trailing text."
        )
        raw_retry = self._call_llm_tracked(_CRITIC_SYSTEM_PROMPT, strict_prompt, temperature=0.2)
        parsed = self._try_parse_critique(raw_retry)
        if parsed is not None:
            return parsed

        raise ValueError(f"Critic failed to produce valid JSON after one retry. Last response:\n{raw_retry}")

    @staticmethod
    def decide(critique: Critique) -> str:
        """Pure decision logic: Critique -> "accept" | "revise" | "re-execute". No LLM call, easily unit-testable.

        The spec's three rules aren't mutually exclusive as written -- a critique
        can be `is_sufficient=True, confidence>=threshold` (Accept) while also
        listing non-empty `weak_sections` (Revise). The weak-sections signal is
        checked first: a concrete, named problem should never be silently waved
        through by a high confidence score, matching the critic prompt's own
        instruction that an honest "needs improvement" beats a weak "accept".
        """
        if critique.weak_sections and not critique.additional_queries:
            return "revise"
        if critique.is_sufficient and critique.confidence >= REFLECTION_CONFIDENCE_THRESHOLD:
            return "accept"
        if critique.is_sufficient and critique.confidence < REFLECTION_CONFIDENCE_THRESHOLD:
            return "revise"
        if not critique.is_sufficient and critique.additional_queries:
            return "re-execute"
        # Not sufficient, but the critic identified no weak sections and proposed no
        # queries -- nothing concrete to search for, so fall back to revising with
        # what we already have rather than looping with no actionable signal.
        return "revise"

    def _register_new_sources(self, new_results: List[SearchResult]) -> str:
        lines = []
        for r in new_results:
            if r.url and r.url not in self._sources:
                self._sources.append(r.url)
                lines.append(f"[{len(self._sources)}] {r.title} -- {r.url}: {r.snippet}")
        return "\n".join(lines)

    def _revise(self, report: str, critique: Critique, new_results: List[SearchResult]) -> str:
        new_sources_block = self._register_new_sources(new_results)
        prompt = self._reviser_template.format(
            topic=self._topic,
            plan_block=_render_plan_block(self._plan),
            current_report=report,
            critique_block=_render_critique_block(critique),
            new_sources_block=new_sources_block or "(no new sources)",
        )
        return self._call_llm_tracked(_REVISER_SYSTEM_PROMPT, prompt, temperature=0.3).strip()

    def revise_without_search(self, report: str, critique: Critique) -> str:
        """Phase 3 -- Revise path: improve the report using only existing information."""
        return self._revise(report, critique, new_results=[])

    def revise_with_new_search(
        self, report: str, critique: Critique, original_results: List[StepResult]
    ) -> Tuple[str, List[SearchResult]]:
        """Phase 3 -- Re-execute path: run critique.additional_queries through the existing search tool, then revise incorporating the new results."""
        del original_results  # kept in the signature per spec; new queries come from the critique, not the original plan
        new_results: List[SearchResult] = []
        queries = critique.additional_queries[:MAX_ADDITIONAL_QUERIES_PER_ITERATION]
        for query in queries:
            found = search_web(query)
            if not found:
                logger.warning("Additional query %r returned no results; continuing with existing results", query)
            new_results.extend(found)

        revised = self._revise(report, critique, new_results=new_results)
        return revised, new_results

    def run_reflection_loop(
        self,
        report: str,
        topic: str,
        plan: Plan,
        original_results: List[StepResult],
        on_iteration_start: Optional[Callable[[int], None]] = None,
        on_critique_done: Optional[Callable[[int, Critique, str], None]] = None,
        on_search_start: Optional[Callable[[int, List[str]], None]] = None,
        on_revised: Optional[Callable[[int, str], None]] = None,
    ) -> ReflectionResult:
        """Full bounded loop: critique -> decide -> revise/re-execute, up to MAX_REFLECTION_ITERATIONS. Never re-plans."""
        self._topic = topic
        self._plan = plan
        self._sources = _collect_sources(original_results)
        self._tokens_used_estimate = 0

        current_body = _strip_sources_section(report)
        iterations: List[ReflectionIteration] = []
        stop_reason = "max_iterations"

        for i in range(1, MAX_REFLECTION_ITERATIONS + 1):
            if self._tokens_used_estimate >= MAX_REFLECTION_TOKEN_BUDGET:
                logger.warning(
                    "Reflection token budget exceeded (~%d estimated tokens), stopping early",
                    self._tokens_used_estimate,
                )
                stop_reason = "token_budget_exceeded"
                break

            if on_iteration_start:
                on_iteration_start(i)

            try:
                crit = self.critique(current_body, topic, plan)
            except Exception as e:
                logger.warning("Critique failed on iteration %d, stopping reflection: %s", i, e)
                stop_reason = "critic_failed"
                break

            decision = self.decide(crit)
            if on_critique_done:
                on_critique_done(i, crit, decision)

            if decision == "accept":
                iterations.append(
                    ReflectionIteration(
                        iteration=i, critique=crit, decision=decision,
                        improvement_notes="Report accepted as sufficient.",
                    )
                )
                stop_reason = "accepted"
                break

            if decision == "re-execute":
                queries = crit.additional_queries[:MAX_ADDITIONAL_QUERIES_PER_ITERATION]
                if on_search_start:
                    on_search_start(i, queries)
                new_body, new_results = self.revise_with_new_search(current_body, crit, original_results)
                notes = (
                    f"Ran {len(queries)} additional quer{'y' if len(queries) == 1 else 'ies'}, "
                    f"incorporated {len(new_results)} new result(s)."
                )
                iterations.append(
                    ReflectionIteration(
                        iteration=i, critique=crit, decision=decision,
                        additional_queries_run=queries, improvement_notes=notes,
                    )
                )
            else:
                new_body = self.revise_without_search(current_body, crit)
                iterations.append(
                    ReflectionIteration(
                        iteration=i, critique=crit, decision=decision,
                        improvement_notes="Revised using existing information only.",
                    )
                )

            if on_revised:
                on_revised(i, decision)

            if new_body.strip() == current_body.strip():
                logger.info("No change between iterations, stopping reflection (convergence)")
                current_body = new_body
                stop_reason = "no_improvement"
                break

            current_body = new_body

        final_confidence = iterations[-1].critique.confidence if iterations else 0.0
        full_markdown = current_body.rstrip() + "\n" + _render_sources_section(self._sources)

        return ReflectionResult(
            topic=topic,
            final_report_markdown=full_markdown,
            total_iterations=len(iterations),
            iterations=iterations,
            final_confidence=final_confidence,
            stop_reason=stop_reason,
        )


def save_reflection_log(result: ReflectionResult) -> str:
    """Write outputs/reflections/reflection_YYYYMMDD_HHMMSS.json and return the path."""
    log_dir = Path(REFLECTION_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = log_dir / filename

    payload = {
        "topic": result.topic,
        "total_iterations": result.total_iterations,
        "iterations": [i.model_dump() for i in result.iterations],
        "final_confidence": result.final_confidence,
        "stop_reason": result.stop_reason,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
