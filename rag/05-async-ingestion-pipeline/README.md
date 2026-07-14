# Async RAG Ingestion Pipeline

A production-style asynchronous document ingestion pipeline: documents are
uploaded via a webhook endpoint, queued in Redis, processed by Celery
workers (extract -> chunk -> embed -> store), and tracked with a progress
API. Permanently failed tasks route to a Dead Letter Queue instead of
disappearing. Pure Python + FastAPI + Celery + Redis + ChromaDB + OpenAI. No
LangChain.

## Why async

```
Sync:   POST /ingest ---- wait 10-60s ----> response
                     (HTTP client times out around 30s)

Async:  POST /ingest -> 202 immediately
                          |
                          v
                    Celery worker processes in the background
                          |
                          v
        GET /status/{id} -> poll for progress until "complete" or "failed"
```

Ingestion (extracting text, chunking, embedding every chunk, writing to
ChromaDB) routinely takes longer than an HTTP client is willing to wait. The
webhook hands the file to a queue and returns immediately; a worker process
does the actual work out-of-band.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              FastAPI (api/)                │
                    │                                            │
  client ──POST──── │  /ingest  → validate size/ext, 202 + doc_id │
  client ──GET────── │  /status/{doc_id}  → read progress          │
  client ──GET────── │  /dlq  → list failures                      │
  client ──POST───── │  /dlq/retry → re-queue oldest failure       │
  client ──DELETE─── │  /dlq  → clear                              │
  client ──GET────── │  /health → Redis ping + DLQ depth            │
                    └──────────────┬─────────────────────────────┘
                                   │ apply_async(kwargs=..., task_id=doc_id)
                                   v
                    ┌──────────────────────────────┐        ┌───────────────┐
                    │     Redis (broker + store)     │<─────>│ Flower (5555) │
                    │  - Celery task queue            │      │ queue monitor │
                    │  - ingestion:progress:{doc_id}  │      └───────────────┘
                    │  - ingestion:dlq (list)         │
                    └──────────────┬───────────────┘
                                   │ worker picks up task
                                   v
                    ┌───────────────────────────────────────────┐
                    │        Celery worker (worker/tasks.py)       │
                    │                                              │
                    │  extract (pypdf/.txt) ─┐                      │
                    │  chunk (recursive)     ├─ progress.set_stage  │
                    │  embed (OpenAI, batch) │  at every stage       │
                    │  store (ChromaDB)     ─┘                      │
                    │                                              │
                    │  on failure: 3 error tiers → DLQ if permanent │
                    │  on success: notify_webhook() (best-effort)   │
                    └───────────────┬─────────────────────────────┘
                                    v
                         ┌────────────────────┐
                         │  ChromaDB (persist)  │
                         │  collection per call  │
                         └────────────────────┘
```

## The three error tiers

`worker/tasks.py`'s `ingest_document` branches retry/DLQ behaviour by
exception type, not by a single generic `except`:

| Tier | Exceptions | Behaviour |
|---|---|---|
| Non-retryable | `UnsupportedFileTypeError`, `FileTooLargeError` | Straight to DLQ, no retry — retrying a bad file type can never succeed |
| Retryable | `EmbeddingError`, `StorageError` | Exponential backoff (2s, 4s, 8s) up to `celery_task_max_retries`, then DLQ |
| Unexpected | anything else (e.g. plain `IngestionError` for an empty document) | Always routed to the DLQ for human inspection |

Without a DLQ, a permanent failure is just a Celery log line that vanishes.
With one, it's a durable Redis entry visible at `GET /dlq` and retryable at
`POST /dlq/retry` — and the DLQ entry stores the base64 file content and
target collection alongside the error metadata, so a retry actually
re-processes the original file instead of just re-describing what failed.

## Reused from prior projects

`worker/pipeline.py`'s `recursive_chunk` is the same recursive
character-splitting algorithm (same `chunk_size=400`/`chunk_overlap=40`
defaults, same `char_start` tracking) as
`rag/03-retrieval-evaluation/retrieval_eval/corpus_builder.py` — reused
rather than re-derived. PDF extraction and ChromaDB storage are new here
since this project writes to a caller-supplied `collection_name` rather than
one fixed eval corpus.

## Real demo run (no HTTP, no Docker needed)

`run_demo.py` calls `worker.tasks.ingest_document` directly — bypassing
Celery's `apply_async`/broker — and polls `ProgressTracker` the same way a
real client would poll `GET /status/{doc_id}`. This is a genuine run against
the real OpenAI API (see `results.json` for full output):

```
=== Ingesting ml_basics.txt ===
  task returned: {'status': 'complete', 'chunk_count': 1}
  final status: stage=complete | detail=Ingested 1 chunks into 'default'

=== Ingesting climate.txt ===
  task returned: {'status': 'complete', 'chunk_count': 1}
  final status: stage=complete | detail=Ingested 1 chunks into 'default'

=== Ingesting bad_file.xyz ===
  task raised: UnsupportedFileTypeError: Unsupported file type: xyz. Expected: .pdf or .txt
  final status: stage=failed | detail=Failed at validation: ...

=== Ingesting empty_file.txt ===
  task raised: IngestionError: Document contains no extractable text
  final status: stage=failed | detail=Failed at unknown: ...

=== Dead Letter Queue ===
  empty_file.txt: IngestionError - Document contains no extractable text
  bad_file.xyz: UnsupportedFileTypeError - Unsupported file type: xyz. Expected: .pdf or .txt
DLQ depth: 2

=== Retrying oldest DLQ entry ===
  retried: bad_file.xyz
  DLQ depth immediately after pop: 1
  DLQ depth after eager re-run (re-failed entries land back in the DLQ): 2
```

Both successful documents were actually embedded (`text-embedding-3-small`)
and persisted into a real ChromaDB collection at `./chroma_db` — confirmed by
querying it directly after the run:

```
count: 2
<doc_id>_0  {'doc_id': '...', 'char_start': 0, 'chunk_id': 0} -> Machine learning is a field of artificial intelligence in wh...
<doc_id>_0  {'doc_id': '...', 'char_start': 0, 'chunk_id': 0} -> Climate science studies the mechanisms behind Earth's changi...
```

`bad_file.xyz` retried and failed again for the same reason (its extension
is still invalid) — exactly the expected behaviour for a non-retryable
failure, and proof that `pop_and_retry` genuinely re-dispatches the original
file rather than just replaying a description of the failure.

A FastAPI smoke test (`TestClient`, not part of the committed code) confirmed
the HTTP layer end-to-end on top of the same pipeline: `202` from `/ingest`,
`200` with `stage=complete` from `/status/{doc_id}`, `415` for a bad
extension, `413` for an oversized file, and `404` for an unknown `doc_id`.

### Sandbox note on this run

This sandbox has no Docker and no root access to install a system Redis, so
`run_demo.py` monkeypatches `redis.from_url` to a shared in-memory
`fakeredis` instance and sets Celery's `task_always_eager=True` so
`pop_and_retry`'s `apply_async` call runs synchronously instead of publishing
to a real broker. **Every other module — `config.py`, `worker/tasks.py`,
`api/*` — is completely unaware of this and is written exactly as it would
run against a real Redis via `docker-compose up -d`.** The substitution is
confined entirely to `run_demo.py`.

## Endpoints

| Method | Path | Description | Success | Failure modes |
|---|---|---|---|---|
| POST | `/ingest` | Upload a `.pdf`/`.txt` file (`multipart/form-data`: `file`, optional `collection_name`) | `202 {doc_id, status, status_url}` | `413` file too large, `415` unsupported extension |
| GET | `/status/{doc_id}` | Current ingestion stage/progress | `200 {stage, pct_complete, detail, chunk_total, chunks_done, updated_at}` | `404` unknown doc_id |
| GET | `/dlq` | List all Dead Letter Queue entries | `200 {entries, count}` | — |
| POST | `/dlq/retry` | Re-queue the oldest DLQ entry as a new task | `200 {retried, remaining_in_dlq}` | `retried: null` if DLQ empty |
| DELETE | `/dlq` | Clear the entire DLQ | `200 {cleared}` | — |
| GET | `/health` | Redis connectivity + DLQ depth | `200 {status, redis, dlq_depth}` | — |

## How to run

```
pip install -r requirements.txt
docker-compose up -d          # Redis on :6379, Flower on :5555

export OPENAI_API_KEY=...     # or put it in a local .env (see config.py)

# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — worker
celery -A worker.celery_app worker --loglevel=info -Q ingestion

# Terminal 3 — try it
curl -F "file=@mydoc.pdf" -F "collection_name=demo" http://localhost:8000/ingest
curl http://localhost:8000/status/<doc_id>
curl http://localhost:8000/dlq

# Or, without any of the above running (no HTTP, no Docker):
python run_demo.py
```

Flower (Celery's queue monitor) is at http://localhost:5555 once
`docker-compose up -d` is running and a worker is connected.

## Files

```
05-async-ingestion-pipeline/
├── requirements.txt
├── docker-compose.yml       Redis + Flower
├── config.py                env-based Settings (Redis URL, OpenAI key, chunk size, ...)
├── exceptions.py             IngestionError hierarchy -> the three error tiers
├── run_demo.py                end-to-end demo, no HTTP/Docker required
├── results.json               real output from the last demo run
├── api/
│   ├── main.py                FastAPI app: webhook + status + DLQ + health
│   ├── models.py               Pydantic request/response models
│   └── deps.py                 shared Redis client / tracker / DLQ dependencies
├── worker/
│   ├── celery_app.py           Celery instance + crash-safe config
│   ├── tasks.py                ingest_document task + three-tier DLQ routing
│   └── pipeline.py              extract / chunk / store (chunker reused from retrieval_eval)
└── storage/
    ├── progress.py              Redis-backed progress tracker (TTL'd JSON per doc_id)
    └── dlq.py                    Dead Letter Queue (Redis list, push/list/retry/clear)
```
