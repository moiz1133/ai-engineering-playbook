"""Custom exception hierarchy for the ingestion pipeline.

WHAT: three error tiers map directly onto exception types
WHY: worker.tasks.ingest_document branches its retry/DLQ behaviour on the
     exception type — non-retryable input errors go straight to the DLQ,
     transient errors get exponential-backoff retries first
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion pipeline errors."""


class FileTooLargeError(IngestionError):
    """Non-retryable — the file itself violates the size limit."""

    def __init__(self, size_mb: float, limit_mb: int):
        self.size_mb = size_mb
        self.limit_mb = limit_mb
        super().__init__(f"File {size_mb:.1f}MB exceeds limit of {limit_mb}MB")


class UnsupportedFileTypeError(IngestionError):
    """Non-retryable — retrying will never make an unsupported extension valid."""

    def __init__(self, ext: str):
        super().__init__(f"Unsupported file type: {ext}. Expected: .pdf or .txt")


class EmbeddingError(IngestionError):
    """Transient — raised when the OpenAI embedding call fails."""


class StorageError(IngestionError):
    """Transient — raised when a ChromaDB write fails."""


class DLQWriteError(IngestionError):
    """Last resort — raised if the DLQ write itself fails."""
