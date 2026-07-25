"""Centralized configuration constants for the Plan-and-Execute agent, sourced from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

GENERATION_MODEL: str = "gpt-4o-mini"
MAX_STEPS_PER_PLAN: int = 6
SEARCH_RESULTS_PER_STEP: int = 4
OUTPUT_DIR: str = "./outputs"

# "tavily" or "duckduckgo". Defaults to duckduckgo since it needs no API key --
# override via the SEARCH_PROVIDER env var to switch to Tavily once a key is set.
SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")

# Self-reflection loop (see src/reflector.py)
MAX_REFLECTION_ITERATIONS: int = 3
REFLECTION_CONFIDENCE_THRESHOLD: float = 0.8
MAX_ADDITIONAL_QUERIES_PER_ITERATION: int = 3
REFLECTION_LOG_DIR: str = "./outputs/reflections"
# Approximate token budget for the whole reflection loop (critic + reviser
# calls combined). call_llm() doesn't expose exact usage, so this is enforced
# via a ~4-chars-per-token estimate -- a cost guard, not precise accounting.
MAX_REFLECTION_TOKEN_BUDGET: int = 40000
