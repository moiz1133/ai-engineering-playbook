"""FastAPI app: webhook ingestion endpoint plus status/DLQ/health endpoints.

WHAT: POST /ingest returns 202 immediately and hands off to a Celery worker
WHY: ingestion takes 10-60s; HTTP clients commonly time out around 30s, so
     the request/response cycle must never block on the actual work
"""

from __future__ import annotations

import base64
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from api.deps import get_dlq, get_progress_tracker, get_redis_client
from api.models import (
    DLQClearResponse,
    DLQListResponse,
    DLQRetryResponse,
    HealthResponse,
    IngestResponse,
    StatusResponse,
)
from config import settings
from exceptions import FileTooLargeError, UnsupportedFileTypeError
from storage.dlq import DeadLetterQueue
from storage.progress import ProgressTracker
from worker.tasks import ingest_document

app = FastAPI(title="Async RAG Ingestion API")

ALLOWED_EXTENSIONS = (".pdf", ".txt")


@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
    tracker: ProgressTracker = Depends(get_progress_tracker),
) -> IngestResponse:
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(status_code=413,
                             detail=str(FileTooLargeError(size_mb, settings.max_file_size_mb)))
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=str(UnsupportedFileTypeError(ext)))

    doc_id = str(uuid4())
    tracker.set_stage(doc_id, "queued", f"Queued: {filename}")

    # WHAT: Celery task args must be JSON-serialisable -- base64-encode the file
    file_content_b64 = base64.b64encode(file_bytes).decode("ascii")
    ingest_document.apply_async(
        kwargs={
            "doc_id": doc_id,
            "filename": filename,
            "file_content_b64": file_content_b64,
            "collection_name": collection_name,
        },
        task_id=doc_id,
    )
    # WHAT: 202 Accepted -- do NOT wait for processing
    # WHY: ingestion takes 10-60s; HTTP clients time out at ~30s
    # WHAT: doc_id == Celery task_id -- a single ID tracks both
    return IngestResponse(doc_id=doc_id, status="queued", status_url=f"/status/{doc_id}")


@app.get("/status/{doc_id}", response_model=StatusResponse)
async def get_status(
    doc_id: str,
    tracker: ProgressTracker = Depends(get_progress_tracker),
) -> StatusResponse:
    progress = tracker.get(doc_id)
    if progress is None:
        raise HTTPException(status_code=404, detail=f"No progress found for doc_id={doc_id}")
    return StatusResponse(**progress)


@app.get("/dlq", response_model=DLQListResponse)
async def list_dlq(dlq: DeadLetterQueue = Depends(get_dlq)) -> DLQListResponse:
    return DLQListResponse(entries=dlq.list_all(), count=dlq.count())


@app.post("/dlq/retry", response_model=DLQRetryResponse)
async def retry_dlq(dlq: DeadLetterQueue = Depends(get_dlq)) -> DLQRetryResponse:
    entry = dlq.pop_and_retry(ingest_document)
    return DLQRetryResponse(retried=entry, remaining_in_dlq=dlq.count())


@app.delete("/dlq", response_model=DLQClearResponse)
async def clear_dlq(dlq: DeadLetterQueue = Depends(get_dlq)) -> DLQClearResponse:
    return DLQClearResponse(cleared=dlq.clear())


@app.get("/health", response_model=HealthResponse)
async def health(dlq: DeadLetterQueue = Depends(get_dlq)) -> HealthResponse:
    redis_client = get_redis_client()
    try:
        redis_client.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "unreachable"
    return HealthResponse(status="ok", redis=redis_status, dlq_depth=dlq.count())
