"""Planner phase: turns a research topic into a fixed Plan of sub-questions via one LLM call (with one retry on malformed JSON).

Also defines call_llm(), the single wrapper every phase (planner, executor,
report) routes its OpenAI calls through, so retries and logging live in
exactly one place.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import GENERATION_MODEL, MAX_STEPS_PER_PLAN
from src.schemas import Plan, PlanStep

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.txt"

_client = OpenAI()


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
def call_llm(system_prompt: str, user_prompt: str, *, temperature: float = 0.3) -> str:
    """Single wrapper for every OpenAI chat completion call in this project. Retries with exponential backoff on rate limits and transient connection errors."""
    logger.info("LLM call: model=%s prompt_chars=%d", GENERATION_MODEL, len(user_prompt))
    response = _client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    logger.info("LLM response received: %d chars", len(content))
    return content


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences an LLM might wrap JSON output in."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[len("json") :]
    return text.strip()


def _try_parse_plan(topic: str, raw: str) -> Optional[Plan]:
    try:
        data = json.loads(_strip_json_fences(raw))
        steps = [PlanStep(**s) for s in data["steps"]]
        return Plan(topic=topic, steps=steps)
    except Exception as e:
        logger.warning("Failed to parse plan JSON: %s", e)
        return None


def generate_plan(topic: str) -> Plan:
    """Call the planner LLM to break `topic` into a fixed Plan of sub-questions.

    Retries once with a stricter prompt if the first response isn't valid
    JSON; raises if the retry also fails to parse.
    """
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(topic=topic, max_steps=MAX_STEPS_PER_PLAN)

    raw = call_llm("You are a meticulous research planning assistant.", prompt)
    plan = _try_parse_plan(topic, raw)
    if plan is not None:
        return plan

    logger.warning("Planner returned malformed JSON, retrying with a stricter prompt")
    strict_prompt = (
        prompt
        + "\n\nIMPORTANT: Your previous response was not valid JSON. Return ONLY a single "
        "valid JSON object, with no markdown fences, no commentary, and no trailing text."
    )
    raw_retry = call_llm("You are a meticulous research planning assistant.", strict_prompt)
    plan = _try_parse_plan(topic, raw_retry)
    if plan is not None:
        return plan

    raise ValueError(f"Planner failed to produce valid JSON after one retry. Last response:\n{raw_retry}")
