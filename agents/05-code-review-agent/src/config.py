"""Central configuration, loaded once from environment variables / .env, plus the
shared prompt-loading helper every agent uses."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key": OPENAI_API_KEY,
    "temperature": 0.1,  # low temp for consistent reviews
}

MAX_ROUNDS = 3
OUTPUT_DIR = "./outputs"
HUMAN_INPUT_MODE = "NEVER"

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt file's contents from src/prompts/ by filename (e.g. 'critic.txt')."""
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def validate_config() -> None:
    """Fail fast if required configuration is missing. Called explicitly at CLI startup,
    not at import time, so this module stays importable in tests without a real key."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
