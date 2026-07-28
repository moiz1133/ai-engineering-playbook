"""Assistant: the orchestrator tying working, episodic, and procedural memory together into one chat loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.config import EPISODIC_TOP_K, FACT_EXTRACTION_ENABLED, WORKING_MEMORY_MAX_MESSAGES
from src.extractors.fact_extractor import FactExtractor, call_llm
from src.memory.episodic import EpisodicMemory
from src.memory.procedural import ProceduralMemory
from src.memory.working import WorkingMemory
from src.schemas import EpisodicFact, Message, Preference

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "assistant.txt"
_SYSTEM_PROMPT = "You are a helpful, personalized assistant."


def _render_preferences(preferences: List[Preference]) -> str:
    if not preferences:
        return "(none learned yet)"
    return "\n".join(f"- {p.key}: {p.value} (category: {p.category})" for p in preferences)


def _render_facts(facts: List[EpisodicFact]) -> str:
    if not facts:
        return "(none relevant to this message)"
    return "\n".join(f"- {f.content}" for f in facts)


def _render_working_memory(messages: List[Message]) -> str:
    prior = messages[:-1]  # the latest user message is rendered separately in the template
    if not prior:
        return "(this is the first message of the session)"
    return "\n".join(f"{m.role}: {m.content}" for m in prior)


class Assistant:
    """Ties working, episodic, and procedural memory into one chat() call, extracting new memories after each turn."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.working = WorkingMemory(max_messages=WORKING_MEMORY_MAX_MESSAGES)
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()
        self.fact_extractor = FactExtractor()
        self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

    def chat(self, user_message: str) -> str:
        """Answer user_message using working+episodic+procedural memory, then extract any new memories from the turn.

        Example:
            reply = assistant.chat("Can you explain how HNSW works?")
        """
        self.working.add("user", user_message)

        relevant_facts = self.episodic.search(user_message, top_k=EPISODIC_TOP_K) if self.episodic.count() > 0 else []
        preferences = self.procedural.list_preferences()

        prompt = self._prompt_template.format(
            procedural_preferences=_render_preferences(preferences),
            episodic_facts=_render_facts(relevant_facts),
            working_memory_messages=_render_working_memory(self.working.get_context()),
            user_message=user_message,
        )

        response_text = call_llm(_SYSTEM_PROMPT, prompt)
        self.working.add("assistant", response_text)

        if FACT_EXTRACTION_ENABLED:
            self._extract_and_store(user_message, response_text)

        return response_text

    def _extract_and_store(self, user_message: str, assistant_response: str) -> None:
        """Extract durable facts/preferences from this turn and store them. Never lets extraction failures break chat()."""
        try:
            facts = self.fact_extractor.extract_facts(user_message, assistant_response)
            # Defense in depth against re-extracting something already known: the
            # extraction prompt is told not to re-derive facts from the assistant
            # merely repeating known info back (e.g. addressing the user by name),
            # but an exact-match dedup here catches it even if the prompt slips.
            existing_contents = {f.content.strip().lower() for f in self.episodic.list_all()}
            for fact in facts:
                normalized = fact.content.strip().lower()
                if normalized in existing_contents:
                    logger.info("Skipping duplicate fact for session %s: %r", self.session_id, fact.content)
                    continue
                self.episodic.add_fact(
                    content=fact.content,
                    source=f"extracted from conversation in session {self.session_id}",
                    session_id=self.session_id,
                )
                existing_contents.add(normalized)
        except Exception:
            logger.exception("Fact extraction failed for session %s", self.session_id)

        try:
            preferences = self.fact_extractor.extract_preferences(user_message, assistant_response)
            for pref in preferences:
                self.procedural.set_preference(pref.key, pref.value, pref.category, pref.confidence)
        except Exception:
            logger.exception("Preference extraction failed for session %s", self.session_id)

    def memory_summary(self) -> str:
        """One-line summary of current memory state, e.g. for the CLI header.

        Example:
            print(assistant.memory_summary())
        """
        return (
            f"Working memory: {self.working.count()} messages | "
            f"Episodic: {self.episodic.count()} facts | "
            f"Procedural: {self.procedural.count()} preferences"
        )
