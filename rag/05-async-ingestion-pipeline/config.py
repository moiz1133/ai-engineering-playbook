"""Environment-based configuration for the async ingestion pipeline.

WHAT: all config from environment — no hardcoded values anywhere
WHY: same codebase runs locally (localhost Redis) and in production (managed
     Redis), just by changing environment variables, no code changes
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str
    chroma_persist_dir: str = "./chroma_db"
    max_file_size_mb: int = 10
    chunk_size: int = 400
    chunk_overlap: int = 40
    embedding_model: str = "text-embedding-3-small"
    celery_task_max_retries: int = 3
    celery_retry_backoff: int = 2  # doubles each retry: 2s, 4s, 8s
    dlq_key: str = "ingestion:dlq"
    progress_key_prefix: str = "ingestion:progress:"
    progress_ttl_seconds: int = 86400  # 24h auto-cleanup
    webhook_notify_url: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
