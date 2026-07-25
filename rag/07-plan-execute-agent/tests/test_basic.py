"""Minimal sanity checks for schemas, config, and the non-LLM parts of report generation and reflection. No real LLM or search calls."""

import json
import os

from src import reflector as reflector_module
from src.reflector import Reflector
from src.report import _collect_sources, _render_sources_section
from src.schemas import Critique, Plan, PlanStep, SearchResult, StepResult


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


def test_critique_validates_well_formed_dict() -> None:
    data = {
        "is_sufficient": False,
        "missing_information": ["no discussion of cost"],
        "weak_sections": ["Intro: too vague"],
        "additional_queries": ["HNSW production cost benchmark"],
        "confidence": 0.4,
        "overall_assessment": "Needs more evidence.",
    }
    critique = Critique(**data)
    assert critique.is_sufficient is False
    assert critique.confidence == 0.4
    assert critique.additional_queries == ["HNSW production cost benchmark"]


def test_decide_accept_when_sufficient_and_confident() -> None:
    critique = Critique(is_sufficient=True, confidence=0.9, overall_assessment="Comprehensive and well-cited.")
    assert Reflector.decide(critique) == "accept"


def test_decide_re_execute_when_insufficient_with_queries() -> None:
    critique = Critique(
        is_sufficient=False,
        confidence=0.4,
        additional_queries=["query one", "query two"],
        overall_assessment="Missing key evidence.",
    )
    assert Reflector.decide(critique) == "re-execute"


def test_reflection_loop_respects_max_iterations(monkeypatch) -> None:
    """With a mocked critic that never accepts and never converges, the loop must stop at MAX_REFLECTION_ITERATIONS."""
    call_count = {"n": 0}

    def fake_call_llm(system_prompt: str, user_prompt: str, *, temperature: float = 0.3) -> str:
        call_count["n"] += 1
        if "reviewer" in system_prompt:
            return json.dumps(
                {
                    "is_sufficient": False,
                    "missing_information": ["still missing something"],
                    "weak_sections": ["Section A: too shallow"],
                    "additional_queries": [],
                    "confidence": 0.5,
                    "overall_assessment": "Needs another pass.",
                }
            )
        # Reviser call: return distinct text each time so the no-improvement
        # convergence check never fires and the loop actually runs to the cap.
        return f"# Topic\n\n## Section A\nrevised content, pass {call_count['n']} [1]\n"

    monkeypatch.setattr(reflector_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(reflector_module, "search_web", lambda query, max_results=4: [])

    step = PlanStep(step_number=1, sub_question="Q?", search_query="q", rationale="r")
    original_results = [
        StepResult(
            step=step,
            search_results=[SearchResult(title="A", url="http://a.com", snippet="s")],
            summary="sum",
        )
    ]
    plan = Plan(topic="Topic", steps=[step])
    draft = "# Topic\n\n## Section A\noriginal content [1]\n\n## Sources\n\n1. http://a.com\n"

    result = Reflector().run_reflection_loop(draft, "Topic", plan, original_results)

    assert result.total_iterations == reflector_module.MAX_REFLECTION_ITERATIONS
    assert result.stop_reason == "max_iterations"
