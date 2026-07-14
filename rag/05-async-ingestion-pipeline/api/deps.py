"""Shared FastAPI dependencies: a process-wide Redis client, and the
progress tracker / DLQ built on top of it.

WHAT: one cached Redis client per API process, reused across requests
WHY: unlike Celery tasks (which fork and can't share a client across worker
     processes), the FastAPI process is long-lived, so one client amortises
     connection setup across every request instead of reconnecting each time
"""

from __future__ import annotations

from functools import lru_cache

import redis

from config import settings
from storage.dlq import DeadLetterQueue
from storage.progress import ProgressTracker


@lru_cache
def get_redis_client() -> redis.Redis:
    return redis.from_url(settings.redis_url)


def get_progress_tracker() -> ProgressTracker:
    return ProgressTracker(get_redis_client())


def get_dlq() -> DeadLetterQueue:
    return DeadLetterQueue(get_redis_client())
