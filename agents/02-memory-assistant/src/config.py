"""Centralized configuration constants for the assistant and all three memory types."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GENERATION_MODEL: str = "gpt-4o-mini"
EMBEDDING_MODEL: str = "text-embedding-3-small"

WORKING_MEMORY_MAX_MESSAGES: int = 20

EPISODIC_MEMORY_DIR: str = "./data/chroma_memory"
EPISODIC_MEMORY_COLLECTION: str = "user_facts"
EPISODIC_TOP_K: int = 5
EPISODIC_MIN_CONFIDENCE: float = 0.7

PROCEDURAL_DB_PATH: str = "./data/procedural.db"
PROCEDURAL_MIN_CONFIDENCE: float = 0.7

FACT_EXTRACTION_ENABLED: bool = True
