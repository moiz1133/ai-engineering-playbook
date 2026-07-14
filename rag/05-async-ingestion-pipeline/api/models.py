"""Pydantic request/response models for the ingestion API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class IngestResponse(BaseModel):
    doc_id: str
    status: str
    status_url: str


class StatusResponse(BaseModel):
    doc_id: str
    stage: str
    detail: str = ""
    chunk_total: int = 0
    chunks_done: int = 0
    pct_complete: int = 0
    updated_at: str
    collection: Optional[str] = None


class DLQEntry(BaseModel):
    doc_id: str
    filename: str
    error_type: str
    error_message: str
    failed_stage: str
    attempt: int
    task_id: str
    failed_at: str
    file_content_b64: Optional[str] = ""
    collection_name: Optional[str] = "default"


class DLQListResponse(BaseModel):
    entries: List[DLQEntry]
    count: int


class DLQRetryResponse(BaseModel):
    retried: Optional[DLQEntry]
    remaining_in_dlq: int


class DLQClearResponse(BaseModel):
    cleared: int


class HealthResponse(BaseModel):
    status: str
    redis: str
    dlq_depth: int
