"""FastAPI app: one /query endpoint wiring prompt versioning, baseline+shadow retrieval, and streaming generation."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from src.prompts.registry import PromptRegistry
from src.retrieval.baseline import RetrievedChunk, retrieve_baseline
from src.retrieval.shadow import log_shadow_comparison, retrieve_shadow
from src.streaming.generator import stream_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Production Patterns")

prompt_registry = PromptRegistry()
_sync_client = OpenAI()
_async_client = AsyncOpenAI()


class QueryRequest(BaseModel):
    question: str
    prompt_version: Optional[str] = None


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Join retrieved chunks into one context block, each tagged with its chunk_id for citation."""
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def _run_shadow_and_log(question: str, baseline_chunks: List[RetrievedChunk], baseline_latency_ms: float) -> None:
    """Background task: run shadow retrieval and log the comparison. Runs after the response is sent; never seen by the user."""
    start = time.perf_counter()
    try:
        shadow_chunks = retrieve_shadow(_sync_client, question)
    except Exception:
        logger.exception("Shadow retrieval failed for query: %s", question)
        return
    shadow_latency_ms = (time.perf_counter() - start) * 1000
    log_shadow_comparison(question, baseline_chunks, shadow_chunks, baseline_latency_ms, shadow_latency_ms)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "prompt_versions_loaded": prompt_registry.list_versions()}


@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    version = request.prompt_version or prompt_registry.default_version
    try:
        template = prompt_registry.get(version)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start = time.perf_counter()
    try:
        baseline_chunks = retrieve_baseline(_sync_client, request.question)
    except Exception as e:
        logger.exception("Baseline retrieval failed")
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}")
    baseline_latency_ms = (time.perf_counter() - start) * 1000

    prompt = template.format(context=format_context(baseline_chunks), question=request.question)

    background_tasks.add_task(_run_shadow_and_log, request.question, baseline_chunks, baseline_latency_ms)

    return StreamingResponse(stream_answer(_async_client, prompt), media_type="text/plain")
