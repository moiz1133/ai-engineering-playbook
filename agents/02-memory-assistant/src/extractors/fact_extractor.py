"""Decides what from a conversation turn is worth remembering long-term -- the deliberate bridge into
episodic (durable facts) and procedural (structured preferences) memory. Most turns produce nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import EPISODIC_MIN_CONFIDENCE, GENERATION_MODEL, PROCEDURAL_MIN_CONFIDENCE
from src.schemas import ExtractedFact, ExtractedPreference

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_FACT_PROMPT_PATH = _PROMPTS_DIR / "fact_extraction.txt"
_PREFERENCE_PROMPT_PATH = _PROMPTS_DIR / "preference_extraction.txt"

_FACT_SYSTEM_PROMPT = "You extract durable facts about a user from a conversation turn."
_PREFERENCE_SYSTEM_PROMPT = "You extract structured behavioral preferences from a conversation turn."

_client = OpenAI()


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Single wrapper every LLM call in this module goes through -- retries with exponential backoff on rate limits."""
    response = _client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[len("json") :]
    return text.strip()


class FactExtractor:
    """Extracts durable episodic facts and structured procedural preferences from a single conversation turn."""

    def __init__(self) -> None:
        self._fact_template = _FACT_PROMPT_PATH.read_text(encoding="utf-8")
        self._preference_template = _PREFERENCE_PROMPT_PATH.read_text(encoding="utf-8")

    def _parse_facts(self, raw: str) -> List[ExtractedFact]:
        try:
            data = json.loads(_strip_json_fences(raw))
            return [ExtractedFact(**f) for f in data.get("facts", [])]
        except Exception as e:
            logger.warning("Failed to parse extracted facts JSON: %s", e)
            return []

    def extract_facts(self, user_message: str, assistant_response: str) -> List[ExtractedFact]:
        """Identify durable, personally-relevant facts about the user worth storing in episodic memory.

        Only facts meeting EPISODIC_MIN_CONFIDENCE are returned -- callers can
        pass every returned fact straight to EpisodicMemory.add_fact().

        assistant_response is accepted for interface symmetry with
        extract_preferences() but deliberately never shown to the LLM here:
        in practice, gpt-4o-mini reliably "extracted" facts from the
        assistant's own personalized phrasing (it repeating the user's name,
        or connecting an answer to something already known) as if the user
        had newly stated them -- three rounds of explicit prompt instructions
        against this did not fix it. Never seeing that text at all is the
        only fix that actually worked.

        Example:
            facts = extractor.extract_facts("My daughter Zara just turned 4", "Happy birthday to Zara!")
        """
        del assistant_response
        prompt = self._fact_template.format(user_message=user_message)
        raw = call_llm(_FACT_SYSTEM_PROMPT, prompt)
        facts = self._parse_facts(raw)
        kept = [f for f in facts if f.confidence >= EPISODIC_MIN_CONFIDENCE]
        logger.info("fact_extractor: facts extracted=%d kept=%d", len(facts), len(kept))
        return kept

    def _parse_preferences(self, raw: str) -> List[ExtractedPreference]:
        try:
            data = json.loads(_strip_json_fences(raw))
            return [ExtractedPreference(**p) for p in data.get("preferences", [])]
        except Exception as e:
            logger.warning("Failed to parse extracted preferences JSON: %s", e)
            return []

    def extract_preferences(self, user_message: str, assistant_response: str) -> List[ExtractedPreference]:
        """Identify structured behavioral preferences worth storing in procedural memory.

        Only preferences meeting PROCEDURAL_MIN_CONFIDENCE are returned.

        Example:
            prefs = extractor.extract_preferences("Keep answers short please", "Got it, I'll be concise.")
        """
        prompt = self._preference_template.format(user_message=user_message, assistant_response=assistant_response)
        raw = call_llm(_PREFERENCE_SYSTEM_PROMPT, prompt)
        preferences = self._parse_preferences(raw)
        kept = [p for p in preferences if p.confidence >= PROCEDURAL_MIN_CONFIDENCE]
        logger.info("fact_extractor: preferences extracted=%d kept=%d", len(preferences), len(kept))
        return kept
