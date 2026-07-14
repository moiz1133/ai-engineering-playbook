"""Dead Letter Queue for ingestion tasks that failed permanently.

WHAT: a Redis list used as a FIFO/LIFO-addressable queue of failure records
WHY: failed tasks must not disappear silently — without a DLQ, a failure is
     just a log line; with one, it's visible at GET /dlq and retryable at
     POST /dlq/retry
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable, List, Optional

import redis

from config import settings


class DeadLetterQueue:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key = settings.dlq_key

    def push(self, doc_id: str, filename: str, error_type: str,
             error_message: str, failed_stage: str,
             attempt: int, task_id: str,
             file_content_b64: str = "", collection_name: str = "default") -> None:
        """Record a permanent failure.

        WHAT: file_content_b64 and collection_name are stored alongside the
              failure metadata, beyond just doc_id/filename/error info
        WHY: pop_and_retry() re-dispatches the ORIGINAL file as a new Celery
             task — without the file content stored here, a "retry" would
             have nothing to actually re-process
        """
        entry = {
            "doc_id": doc_id,
            "filename": filename,
            "error_type": error_type,
            "error_message": error_message,
            "failed_stage": failed_stage,
            "attempt": attempt,
            "task_id": task_id,
            "file_content_b64": file_content_b64,
            "collection_name": collection_name,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.lpush(self.key, json.dumps(entry))
        # WHAT: LPUSH prepends — most recent failure at index 0
        # WHY: Redis list = natural DLQ; a single LRANGE retrieves all failures

    def list_all(self) -> List[dict]:
        return [json.loads(e) for e in self.redis.lrange(self.key, 0, -1)]

    def pop_and_retry(self, task_fn: Callable) -> Optional[dict]:
        """Pop the oldest entry (FIFO) and re-queue it as a new Celery task."""
        raw = self.redis.rpop(self.key)  # rpop = oldest entry
        if not raw:
            return None
        entry = json.loads(raw)
        task_fn.apply_async(
            kwargs={
                "doc_id": entry["doc_id"],
                "filename": entry["filename"],
                "file_content_b64": entry.get("file_content_b64", ""),
                "collection_name": entry.get("collection_name", "default"),
            },
            task_id=f"retry-{entry['doc_id'][:8]}",
        )
        return entry
        # WHAT: RPOP = right pop = oldest item (FIFO retry order)
        # WHY: retry the oldest failures first — they've been waiting longest

    def count(self) -> int:
        return self.redis.llen(self.key)

    def clear(self) -> int:
        n = self.redis.llen(self.key)
        self.redis.delete(self.key)
        return n
