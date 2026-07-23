"""Centralized configuration constants for the project, sourced from environment variables where relevant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL: str = "text-embedding-3-small"
GENERATION_MODEL: str = "gpt-4o-mini"

CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION_NAME: str = "rag_patterns"

DEFAULT_PROMPT_VERSION: str = os.getenv("DEFAULT_PROMPT_VERSION", "v1")

SHADOW_LOG_PATH: str = "./logs/shadow_comparisons.jsonl"

TOP_K: int = 5
