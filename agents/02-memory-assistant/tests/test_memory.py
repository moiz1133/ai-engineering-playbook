"""Tests for the three memory types and the fact/preference extractors. No real LLM calls except where mocked."""

import json

import pytest

from src.extractors import fact_extractor as fact_extractor_module
from src.extractors.fact_extractor import FactExtractor
from src.memory.episodic import EpisodicMemory
from src.memory.procedural import ProceduralMemory
from src.memory.working import WorkingMemory


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------


def test_working_memory_sliding_window_drops_oldest() -> None:
    wm = WorkingMemory(max_messages=3)
    for i in range(5):
        wm.add("user", f"msg {i}")
    contents = [m.content for m in wm.get_context()]
    assert contents == ["msg 2", "msg 3", "msg 4"]
    assert wm.count() == 3


def test_working_memory_clear() -> None:
    wm = WorkingMemory(max_messages=5)
    wm.add("user", "hello")
    wm.clear()
    assert wm.count() == 0
    assert wm.get_context() == []


def test_working_memory_get_context_format() -> None:
    wm = WorkingMemory(max_messages=5)
    wm.add("user", "hi")
    wm.add("assistant", "hello there")
    ctx = wm.get_context()
    assert [m.role for m in ctx] == ["user", "assistant"]
    assert ctx[0].content == "hi"
    assert ctx[0].timestamp is not None


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------


@pytest.fixture
def procedural(tmp_path):
    mem = ProceduralMemory(db_path=str(tmp_path / "procedural.db"))
    yield mem
    mem.close()


def test_procedural_set_get_roundtrip(procedural: ProceduralMemory) -> None:
    procedural.set_preference("response_style", "concise", "communication", 0.9)
    assert procedural.get_preference("response_style") == "concise"
    assert procedural.count() == 1


def test_procedural_upsert_updates_existing_key(procedural: ProceduralMemory) -> None:
    procedural.set_preference("response_style", "concise", "communication", 0.8)
    procedural.set_preference("response_style", "verbose", "communication", 0.6)
    assert procedural.get_preference("response_style") == "verbose"
    assert procedural.count() == 1  # still one row, not two


def test_procedural_list_preferences_filters_by_category(procedural: ProceduralMemory) -> None:
    procedural.set_preference("response_style", "concise", "communication", 0.9)
    procedural.set_preference("units", "metric", "domain", 0.9)
    comms = procedural.list_preferences(category="communication")
    assert [p.key for p in comms] == ["response_style"]
    all_prefs = procedural.list_preferences()
    assert {p.key for p in all_prefs} == {"response_style", "units"}


def test_procedural_get_missing_key_returns_none(procedural: ProceduralMemory) -> None:
    assert procedural.get_preference("does_not_exist") is None


def test_procedural_update_confidence_clamped(procedural: ProceduralMemory) -> None:
    procedural.set_preference("response_style", "concise", "communication", 0.95)
    procedural.update_confidence("response_style", 0.5)  # would overflow past 1.0
    prefs = procedural.list_preferences()
    assert prefs[0].confidence == 1.0


def test_procedural_forget_preference(procedural: ProceduralMemory) -> None:
    procedural.set_preference("response_style", "concise", "communication", 0.9)
    procedural.forget_preference("response_style")
    assert procedural.get_preference("response_style") is None
    assert procedural.count() == 0


def test_procedural_clear(procedural: ProceduralMemory) -> None:
    procedural.set_preference("a", "1", "x", 0.9)
    procedural.set_preference("b", "2", "y", 0.9)
    procedural.clear()
    assert procedural.count() == 0


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


@pytest.fixture
def episodic(tmp_path):
    return EpisodicMemory(persist_dir=str(tmp_path / "chroma_memory"), collection_name="test_facts")


def test_episodic_add_and_search_returns_fact(episodic: EpisodicMemory) -> None:
    episodic.add_fact("User works at Afiniti as a Senior Software Engineer", source="test", session_id="s1")
    episodic.add_fact("User's daughter Zara turned 4", source="test", session_id="s1")

    results = episodic.search("Where does the user work?", top_k=1)
    assert len(results) == 1
    assert "Afiniti" in results[0].content
    assert results[0].access_count == 1


def test_episodic_search_updates_access_count(episodic: EpisodicMemory) -> None:
    episodic.add_fact("User is learning about production RAG systems", source="test", session_id="s1")
    episodic.search("What is the user learning?")
    second = episodic.search("What is the user learning?")
    assert second[0].access_count == 2


def test_episodic_forget_removes_fact(episodic: EpisodicMemory) -> None:
    fact = episodic.add_fact("User's daughter Zara turned 4", source="test", session_id="s1")
    assert episodic.count() == 1
    episodic.forget(fact.fact_id)
    assert episodic.count() == 0
    assert episodic.list_all() == []


def test_episodic_count_and_list_all(episodic: EpisodicMemory) -> None:
    episodic.add_fact("Fact one", source="test", session_id="s1")
    episodic.add_fact("Fact two", source="test", session_id="s1")
    assert episodic.count() == 2
    assert {f.content for f in episodic.list_all()} == {"Fact one", "Fact two"}


def test_episodic_search_on_empty_store_returns_empty_list(episodic: EpisodicMemory) -> None:
    assert episodic.search("anything") == []


def test_episodic_clear(episodic: EpisodicMemory) -> None:
    episodic.add_fact("Fact one", source="test", session_id="s1")
    episodic.clear()
    assert episodic.count() == 0


# ---------------------------------------------------------------------------
# FactExtractor (LLM always mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor() -> FactExtractor:
    return FactExtractor()


def test_extract_facts_parses_schema_correctly(extractor: FactExtractor, monkeypatch) -> None:
    monkeypatch.setattr(
        fact_extractor_module,
        "call_llm",
        lambda system, prompt: json.dumps(
            {"facts": [{"content": "User has a daughter named Zara", "confidence": 0.95, "category": "personal"}]}
        ),
    )
    facts = extractor.extract_facts("My daughter Zara turned 4", "Happy birthday to Zara!")
    assert len(facts) == 1
    assert facts[0].content == "User has a daughter named Zara"
    assert facts[0].category == "personal"


def test_extract_facts_filters_low_confidence(extractor: FactExtractor, monkeypatch) -> None:
    monkeypatch.setattr(
        fact_extractor_module,
        "call_llm",
        lambda system, prompt: json.dumps(
            {
                "facts": [
                    {"content": "Durable fact", "confidence": 0.9, "category": "personal"},
                    {"content": "Shaky guess", "confidence": 0.3, "category": "other"},
                ]
            }
        ),
    )
    facts = extractor.extract_facts("some message", "some response")
    assert len(facts) == 1
    assert facts[0].content == "Durable fact"


def test_extract_facts_malformed_json_returns_empty_list(extractor: FactExtractor, monkeypatch) -> None:
    monkeypatch.setattr(fact_extractor_module, "call_llm", lambda system, prompt: "not valid json {{{")
    facts = extractor.extract_facts("some message", "some response")
    assert facts == []


def test_extract_preferences_parses_schema_correctly(extractor: FactExtractor, monkeypatch) -> None:
    monkeypatch.setattr(
        fact_extractor_module,
        "call_llm",
        lambda system, prompt: json.dumps(
            {
                "preferences": [
                    {"key": "response_style", "value": "concise", "category": "communication", "confidence": 0.9}
                ]
            }
        ),
    )
    prefs = extractor.extract_preferences("Keep it short please", "Got it, concise from now on.")
    assert len(prefs) == 1
    assert prefs[0].key == "response_style"
    assert prefs[0].value == "concise"


def test_extract_preferences_filters_low_confidence(extractor: FactExtractor, monkeypatch) -> None:
    monkeypatch.setattr(
        fact_extractor_module,
        "call_llm",
        lambda system, prompt: json.dumps(
            {
                "preferences": [
                    {"key": "response_style", "value": "concise", "category": "communication", "confidence": 0.9},
                    {"key": "unit_system", "value": "metric", "category": "domain", "confidence": 0.2},
                ]
            }
        ),
    )
    prefs = extractor.extract_preferences("some message", "some response")
    assert len(prefs) == 1
    assert prefs[0].key == "response_style"
