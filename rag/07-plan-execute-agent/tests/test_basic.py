"""Minimal sanity checks for schemas, config, and the non-LLM parts of report generation. No LLM or search calls."""

import os

from src.report import _collect_sources, _render_sources_section
from src.schemas import Plan, PlanStep, SearchResult, StepResult


def test_plan_validates_well_formed_dict() -> None:
    data = {
        "topic": "Test topic",
        "steps": [
            {"step_number": 1, "sub_question": "Why?", "search_query": "why query", "rationale": "because"},
        ],
    }
    plan = Plan(**data)
    assert plan.topic == "Test topic"
    assert len(plan.steps) == 1
    assert isinstance(plan.steps[0], PlanStep)


def test_config_loads_without_errors() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-test"
    from src import config

    assert config.GENERATION_MODEL == "gpt-4o-mini"
    assert config.MAX_STEPS_PER_PLAN == 6
    assert config.SEARCH_PROVIDER in {"tavily", "duckduckgo"}


def test_report_sources_and_markdown_are_valid() -> None:
    step = PlanStep(step_number=1, sub_question="What is X?", search_query="X", rationale="testing")
    result = StepResult(
        step=step,
        search_results=[
            SearchResult(title="A", url="http://a.com", snippet="snippet a"),
            SearchResult(title="B", url="http://b.com", snippet="snippet b"),
        ],
        summary="X is a thing.",
    )

    sources = _collect_sources([result])
    assert sources == ["http://a.com", "http://b.com"]

    sources_section = _render_sources_section(sources)
    assert "## Sources" in sources_section
    assert "1. http://a.com" in sources_section
    assert "2. http://b.com" in sources_section
