"""End-to-end demo of the ingestion pipeline without HTTP: calls
worker.tasks.ingest_document directly (bypassing Celery's broker /
apply_async) and polls storage.progress.ProgressTracker the same way a real
client would poll GET /status/{doc_id}.

SANDBOX-ONLY SUBSTITUTION: this environment has no Docker and no root access
to install a real Redis server, so redis.from_url is monkeypatched below to a
shared in-memory fakeredis instance, and Celery is set to task_always_eager
so dlq.pop_and_retry()'s call to apply_async() runs synchronously instead of
publishing to a real broker. Every other module (config.py, worker/tasks.py,
api/*) is completely unaware of this and is written exactly as it would run
against real Redis via docker-compose. Nothing here is a stand-in for the
actual ingestion logic -- only for the Redis/broker transport.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo-root .env (gitignored) holds OPENAI_API_KEY - load it into the process
# environment before config.Settings() is constructed by any import below.
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", ".env"))

import fakeredis
import redis as redis_module

_fake_server = fakeredis.FakeServer()


def _fake_from_url(url: str, *args, **kwargs) -> fakeredis.FakeStrictRedis:
    return fakeredis.FakeStrictRedis(server=_fake_server)


redis_module.from_url = _fake_from_url  # sandbox-only, see module docstring

from storage.dlq import DeadLetterQueue
from storage.progress import ProgressTracker
from worker.celery_app import celery_app
from worker.tasks import ingest_document

# Sandbox-only: makes apply_async() (used by DeadLetterQueue.pop_and_retry)
# run the task synchronously in-process instead of publishing to a real
# Celery broker, which this sandbox doesn't have either.
celery_app.conf.task_always_eager = True

DEMO_DOCS = [
    ("ml_basics.txt",
     "Machine learning is a field of artificial intelligence in which "
     "algorithms improve their performance on a task through exposure to "
     "data rather than explicit rules. Supervised learning trains on "
     "labeled examples, while unsupervised learning finds structure in "
     "data with no labels at all.",
     "default"),
    ("climate.txt",
     "Climate science studies the mechanisms behind Earth's changing "
     "climate, including greenhouse gas radiative forcing, ice-albedo "
     "feedback loops, and shifts in ocean circulation driven by human "
     "carbon emissions.",
     "default"),
    ("bad_file.xyz", "Should fail -- bad extension", "default"),
    ("empty_file.txt", "", "default"),
]


def _poll_until_done(tracker: ProgressTracker, doc_id: str, timeout_s: float = 10.0) -> dict:
    """Poll the same way a real client would poll GET /status/{doc_id}.

    NOTE: because ingest_document is called synchronously below (not via a
    background worker), the task has always already reached a terminal stage
    by the time this loop runs its first check -- in a real deployment, with
    a worker processing this asynchronously, this loop would actually wait
    across several iterations.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        progress = tracker.get(doc_id)
        if progress and progress["stage"] in ("complete", "failed"):
            return progress
        time.sleep(0.5)
    return tracker.get(doc_id) or {}


def main() -> None:
    redis_client = redis_module.from_url("redis://fake")
    tracker = ProgressTracker(redis_client)
    dlq = DeadLetterQueue(redis_client)

    demo_log = []

    for filename, content, collection_name in DEMO_DOCS:
        doc_id = str(uuid4())
        print(f"\n=== Ingesting {filename} (doc_id={doc_id}) ===")
        file_content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")

        tracker.set_stage(doc_id, "queued", f"Queued: {filename}")

        task_result = None
        task_error = None
        try:
            task_result = ingest_document(doc_id, filename, file_content_b64, collection_name)
            print(f"  task returned: {task_result}")
        except Exception as e:
            task_error = f"{type(e).__name__}: {e}"
            print(f"  task raised: {task_error}")

        final_status = _poll_until_done(tracker, doc_id)
        print(f"  final status: stage={final_status.get('stage')} | "
              f"detail={final_status.get('detail')}")

        demo_log.append({
            "doc_id": doc_id,
            "filename": filename,
            "task_result": task_result,
            "task_error": task_error,
            "final_status": final_status,
        })

    print("\n=== Dead Letter Queue ===")
    entries = dlq.list_all()
    for entry in entries:
        print(f"  {entry['filename']}: {entry['error_type']} - {entry['error_message']}")
    dlq_depth_before_retry = dlq.count()
    print(f"DLQ depth: {dlq_depth_before_retry}")

    print("\n=== Retrying oldest DLQ entry ===")
    retried_entry = dlq.pop_and_retry(ingest_document)
    print(f"  retried: {retried_entry['filename'] if retried_entry else None}")
    dlq_depth_after_retry = dlq.count()
    print(f"  DLQ depth immediately after pop: "
          f"{dlq_depth_before_retry - 1 if retried_entry else dlq_depth_before_retry}")
    print(f"  DLQ depth after eager re-run (re-failed entries land back in the DLQ): "
          f"{dlq_depth_after_retry}")

    results = {
        "demo_log": demo_log,
        "dlq_before_retry": entries,
        "dlq_depth_before_retry": dlq_depth_before_retry,
        "retried_entry": retried_entry,
        "dlq_after_retry": dlq.list_all(),
        "dlq_depth_after_retry": dlq_depth_after_retry,
    }
    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults.json written to {results_path}")


if __name__ == "__main__":
    main()
