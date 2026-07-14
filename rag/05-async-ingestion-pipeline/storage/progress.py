"""Redis-backed progress tracker for in-flight ingestion tasks.

WHAT: one JSON blob per doc_id, keyed under a TTL so it self-cleans
WHY: lets GET /status/{doc_id} answer instantly without touching Celery's
     result backend or the task object itself — progress is a first-class,
     independently-queryable piece of state
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import redis

from config import settings


class ProgressTracker:
    STAGES = ["queued", "extracting", "chunking", "embedding", "storing", "complete", "failed"]

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def _key(self, doc_id: str) -> str:
        return f"{settings.progress_key_prefix}{doc_id}"

    def set_stage(self, doc_id: str, stage: str, detail: str = "",
                  chunk_total: int = 0, chunks_done: int = 0) -> None:
        payload = {
            "doc_id": doc_id,
            "stage": stage,
            "detail": detail,
            "chunk_total": chunk_total,
            "chunks_done": chunks_done,
            "pct_complete": int(chunks_done / chunk_total * 100) if chunk_total else 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.setex(self._key(doc_id), settings.progress_ttl_seconds, json.dumps(payload))
        # WHAT: setex = set + expire atomically
        # WHY: prevents orphaned keys if the process dies before a separate
        #      EXPIRE call would otherwise have run

    def get(self, doc_id: str) -> Optional[dict]:
        raw = self.redis.get(self._key(doc_id))
        return json.loads(raw) if raw else None

    def set_failed(self, doc_id: str, error: str, stage: str) -> None:
        self.set_stage(doc_id, "failed", detail=f"Failed at {stage}: {error}")

    def set_complete(self, doc_id: str, chunk_count: int, collection: str) -> None:
        payload = {
            "doc_id": doc_id,
            "stage": "complete",
            "detail": f"Ingested {chunk_count} chunks into '{collection}'",
            "chunk_total": chunk_count,
            "chunks_done": chunk_count,
            "pct_complete": 100,
            "collection": collection,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.setex(self._key(doc_id), settings.progress_ttl_seconds, json.dumps(payload))
