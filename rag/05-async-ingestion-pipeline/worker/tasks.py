"""The ingestion task: extract -> chunk -> embed -> store, with progress
tracking at every stage and three-tier error handling that routes permanent
failures to the Dead Letter Queue.

ASYNC INGESTION -- INTERVIEW EXPLANATION

WHY ASYNC?
  Sync: POST /ingest -> wait 30-60s -> response. HTTP times out.
  Async: POST /ingest -> 202 immediately -> worker processes in background
         -> GET /status/{id} to poll progress. Scales to N concurrent uploads.

WHY REDIS AS QUEUE?
  Redis lists are a natural queue (LPUSH/BRPOP). Celery wraps this.
  Redis also stores progress keys and the DLQ -- single dependency, three roles.

WHY CELERY?
  Manages worker processes, retries, routing to named queues, result backend.
  task_acks_late=True is the critical reliability flag -- without it, a worker
  crash mid-task loses the task silently. With it, the task requeues.

WHY DEAD LETTER QUEUE?
  Failed tasks must not disappear silently.
  DLQ stores: doc_id, filename, error_type, error_message, stage, attempt.
  Without DLQ: fail -> Celery marks failed -> error is gone.
  With DLQ:    fail -> DLQ entry -> visible at GET /dlq -> retry at POST /dlq/retry.

THREE ERROR TIERS:
  Non-retryable (bad file type, empty file) -> straight to DLQ
  Retryable (EmbeddingError, StorageError)  -> backoff retry -> DLQ if exhausted
  Unexpected (any other Exception)          -> DLQ always for human inspection

BRIDGING TO PRIOR EXPERIENCE:
  Airflow DAGs      = Celery tasks (both async task graphs with retry)
  Grafana           = Flower monitor (both show queue depth, task latency)
  Incident pipeline = ingestion pipeline (both process data async in background)
  The pattern is the same infrastructure shape -- only the tool names changed.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Dict, List

import redis
from openai import OpenAI

from config import settings
from exceptions import (
    EmbeddingError,
    FileTooLargeError,
    IngestionError,
    StorageError,
    UnsupportedFileTypeError,
)
from storage.dlq import DeadLetterQueue
from storage.progress import ProgressTracker
from worker.celery_app import celery_app
from worker.pipeline import extract_pdf_text, recursive_chunk, store_chunks


@celery_app.task(
    bind=True,
    name="worker.tasks.ingest_document",
    max_retries=settings.celery_task_max_retries,
    default_retry_delay=settings.celery_retry_backoff,
)
def ingest_document(self, doc_id: str, filename: str,
                     file_content_b64: str, collection_name: str) -> dict:
    # WHAT: instantiate clients inside the task -- not at module level
    # WHY: Celery workers fork processes; module-level sockets don't survive forks
    redis_client = redis.from_url(settings.redis_url)
    tracker = ProgressTracker(redis_client)
    dlq = DeadLetterQueue(redis_client)

    try:
        # -- STAGE 1: EXTRACT ------------------------------------------------
        tracker.set_stage(doc_id, "extracting", f"Reading {filename}")
        file_bytes = base64.b64decode(file_content_b64)

        if filename.endswith(".pdf"):
            text = extract_pdf_text(file_bytes)
        elif filename.endswith(".txt"):
            text = file_bytes.decode("utf-8")
        else:
            raise UnsupportedFileTypeError(filename.split(".")[-1])

        if not text.strip():
            raise IngestionError("Document contains no extractable text")

        # -- STAGE 2: CHUNK ---------------------------------------------------
        tracker.set_stage(doc_id, "chunking", "Splitting into chunks")
        chunks = recursive_chunk(text, settings.chunk_size, settings.chunk_overlap)
        tracker.set_stage(doc_id, "chunking", f"Created {len(chunks)} chunks",
                           chunk_total=len(chunks))

        # -- STAGE 3: EMBED (with per-batch progress) -------------------------
        tracker.set_stage(doc_id, "embedding", "Starting embedding",
                           chunk_total=len(chunks), chunks_done=0)
        openai_client = OpenAI(api_key=settings.openai_api_key)
        embedded_chunks = embed_with_progress(chunks, openai_client, doc_id, tracker)

        # -- STAGE 4: STORE -----------------------------------------------------
        tracker.set_stage(doc_id, "storing", f"Writing to collection '{collection_name}'",
                           chunk_total=len(chunks), chunks_done=len(chunks))
        store_chunks(embedded_chunks, collection_name, doc_id)

        # -- COMPLETE -------------------------------------------------------------
        tracker.set_complete(doc_id, len(chunks), collection_name)
        notify_webhook(doc_id, collection_name, len(chunks))
        return {"doc_id": doc_id, "status": "complete", "chunk_count": len(chunks)}

    except (UnsupportedFileTypeError, FileTooLargeError) as e:
        # NON-RETRYABLE: bad input -- retrying will never fix this
        # NOTE: FileTooLargeError is validated and rejected at the API layer
        # (413, before the task is ever dispatched) -- caught here too as
        # defense in depth for any future direct-task-call path that skips it
        tracker.set_failed(doc_id, str(e), "validation")
        dlq.push(doc_id, filename, type(e).__name__, str(e), "validation",
                 self.request.retries, self.request.id,
                 file_content_b64=file_content_b64, collection_name=collection_name)
        raise  # raise without self.retry = no retry

    except (EmbeddingError, StorageError) as e:
        # RETRYABLE: transient API/DB failure -- retry with exponential backoff
        stage = "embedding" if isinstance(e, EmbeddingError) else "storing"
        tracker.set_stage(doc_id, stage,
                           f"Error (attempt {self.request.retries + 1}/{self.max_retries}): {e}")
        if self.request.retries >= self.max_retries:
            tracker.set_failed(doc_id, str(e), stage)
            dlq.push(doc_id, filename, type(e).__name__, str(e), stage,
                     self.request.retries, self.request.id,
                     file_content_b64=file_content_b64, collection_name=collection_name)
            raise
        countdown = settings.celery_retry_backoff * (2 ** self.request.retries)
        raise self.retry(exc=e, countdown=countdown)
        # WHAT: countdown doubles: 2s -> 4s -> 8s (exponential backoff)
        # WHY: gives transient API errors time to resolve

    except Exception as e:
        # UNEXPECTED (includes plain IngestionError, e.g. empty-text docs):
        # always route to DLQ for human inspection
        tracker.set_failed(doc_id, str(e), "unknown")
        dlq.push(doc_id, filename, type(e).__name__, str(e), "unknown",
                 self.request.retries, self.request.id,
                 file_content_b64=file_content_b64, collection_name=collection_name)
        raise

    # THREE ERROR TIERS SUMMARY:
    # 1. Non-retryable (bad file type)  -> straight to DLQ, no retry
    # 2. Retryable (API timeout)        -> exponential backoff -> DLQ if exhausted
    # 3. Unexpected (incl. empty docs)  -> DLQ always, for human inspection


def embed_with_progress(chunks: List[Dict], openai_client: OpenAI, doc_id: str,
                         tracker: ProgressTracker) -> List[Dict]:
    result = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            response = openai_client.embeddings.create(
                model=settings.embedding_model,
                input=[c["text"] for c in batch],
            )
        except Exception as e:
            raise EmbeddingError(f"Batch {i // batch_size + 1} failed: {e}")
        for chunk, emb in zip(batch, response.data):
            chunk["embedding"] = emb.embedding
            result.append(chunk)
        tracker.set_stage(doc_id, "embedding",
                           f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks",
                           chunk_total=len(chunks),
                           chunks_done=min(i + batch_size, len(chunks)))
    return result


def notify_webhook(doc_id: str, collection: str, chunk_count: int) -> None:
    if not settings.webhook_notify_url:
        return
    try:
        import httpx
        httpx.post(settings.webhook_notify_url, json={
            "event": "ingestion_complete", "doc_id": doc_id,
            "collection": collection, "chunk_count": chunk_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, timeout=5.0)
    except Exception as e:
        print(f"[WEBHOOK] Notify failed (non-fatal): {e}")
        # WHAT: webhook failure is non-fatal -- the doc is already ingested
        # WHY: don't fail the whole task because an outbound notification timed out
