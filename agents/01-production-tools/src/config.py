"""Centralized configuration constants for all six tools, sourced from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Web search (Tavily)
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# Postgres
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "")
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

# File reader
FILE_READER_BASE_DIR: Path = Path(os.getenv("FILE_READER_BASE_DIR", "./data"))

# REST API -- SSRF controls
ALLOWED_DOMAINS: Set[str] = set(os.getenv("ALLOWED_DOMAINS", "").split(",")) if os.getenv("ALLOWED_DOMAINS") else set()
BLOCKED_DOMAINS: Set[str] = {"localhost", "127.0.0.1", "169.254.169.254", "0.0.0.0"}
MAX_RESPONSE_SIZE_MB: int = int(os.getenv("MAX_RESPONSE_SIZE_MB", "10"))

# Code executor (Docker sandbox)
DOCKER_SANDBOX_IMAGE: str = "production-tools-sandbox:latest"
CODE_EXECUTOR_MEMORY_LIMIT: str = "256m"
CODE_EXECUTOR_CPU_LIMIT: float = 0.5

# with_llm.py example
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
