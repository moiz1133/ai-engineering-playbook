"""Celery application instance and configuration.

WHAT: task_acks_late=True — a task is acknowledged only AFTER it completes
WHY: the default (ack on receipt) loses the task silently if the worker
     process crashes mid-processing; task_acks_late requeues it instead
WHAT: worker_prefetch_multiplier=1 — fetch one task at a time per worker
WHY: embedding calls are slow and heavy — prefetching more would let one
     worker hoard several tasks while sitting idle on an API call, starving
     other workers
"""

from __future__ import annotations

from celery import Celery

from config import settings

celery_app = Celery("ingestion", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_routes={"worker.tasks.ingest_document": {"queue": "ingestion"}},
)
