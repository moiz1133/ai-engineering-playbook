"""Executor phase: runs each fixed plan step's search sequentially and summarizes what was found.

Plan-and-Execute never re-plans mid-execution -- the Plan produced in
Phase 1 is treated as immutable here.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

from src.planner import call_llm
from src.schemas import Plan, PlanStep, SearchResult, StepResult
from src.tools.web_search import search_web

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = "You are a concise research assistant that summarizes search results in 1-2 sentences."


def _summarize_results(sub_question: str, results: List[SearchResult]) -> str:
    """Ask the LLM for a 1-2 sentence summary of what the search snippets say about sub_question."""
    if not results:
        return "No search results were found for this sub-question; data unavailable."

    snippets_block = "\n".join(f"- {r.title}: {r.snippet}" for r in results)
    prompt = (
        f"Sub-question: {sub_question}\n\n"
        f"Search result snippets:\n{snippets_block}\n\n"
        "In 1-2 sentences, summarize what these snippets say in answer to the sub-question. "
        "Be factual and only use information present in the snippets above."
    )
    return call_llm(_SUMMARY_SYSTEM_PROMPT, prompt, temperature=0.2).strip()


def execute_step(step: PlanStep) -> StepResult:
    """Run one plan step: search, then summarize the snippets found."""
    results = search_web(step.search_query)
    if not results:
        logger.warning("Step %d (%r) returned no search results", step.step_number, step.search_query)
    summary = _summarize_results(step.sub_question, results)
    return StepResult(step=step, search_results=results, summary=summary)


def execute_plan(plan: Plan, on_step_start: Optional[Callable[[PlanStep], None]] = None,
                  on_step_done: Optional[Callable[[PlanStep, StepResult, float], None]] = None) -> List[StepResult]:
    """Execute every step in `plan` sequentially, in order. Optional callbacks let the CLI report progress without executor.py depending on Rich."""
    step_results: List[StepResult] = []
    for step in plan.steps:
        if on_step_start:
            on_step_start(step)
        start = time.perf_counter()
        result = execute_step(step)
        elapsed_s = time.perf_counter() - start
        step_results.append(result)
        if on_step_done:
            on_step_done(step, result, elapsed_s)
    return step_results
