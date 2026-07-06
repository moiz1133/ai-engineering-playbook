"""Cohere rerank retriever: retrieve wide with cosine, rerank narrow with a cross-encoder.

WHAT: retrieve wide with cheap cosine, rerank narrow with cross-encoder
WHY: cosine embeds query and chunk independently — misses query-chunk relationships.
     Cohere's cross-encoder sees both together, catches "similar vocab, different meaning"
WHAT: initial_k=20 means reranker sees 20 candidates; final_k=5 is what gets scored
WHY: retrieving more candidates increases the chance the correct chunk is in the pool
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import chromadb
from cohere.errors import TooManyRequestsError
from openai import OpenAI

from corpus_builder import EMBED_MODEL

RERANK_MODEL = "rerank-english-v3.0"
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 65.0  # trial Cohere keys use a rolling 60s window
MIN_CALL_INTERVAL_SECONDS = 6.5  # paces us under the trial key's 10 calls/minute cap

_last_call_time = 0.0
_wait_seconds_since_last_read = 0.0


def _throttle() -> None:
    """Sleep just enough to keep rerank calls under 10/minute (trial key limit)."""
    global _last_call_time, _wait_seconds_since_last_read
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < MIN_CALL_INTERVAL_SECONDS:
        wait = MIN_CALL_INTERVAL_SECONDS - elapsed
        time.sleep(wait)
        _wait_seconds_since_last_read += wait
    _last_call_time = time.monotonic()


def get_and_reset_wait_seconds() -> float:
    """Return time spent sleeping for rate-limit pacing/backoff since the last
    call, then reset the counter. Lets callers exclude deliberate throttling
    from measured retrieval latency."""
    global _wait_seconds_since_last_read
    wait = _wait_seconds_since_last_read
    _wait_seconds_since_last_read = 0.0
    return wait


def retrieve_with_rerank(query: str, collection: chromadb.Collection,
                          openai_client: OpenAI, cohere_client,
                          initial_k: int = 20, final_k: int = 5) -> Optional[List[str]]:
    """Retrieve initial_k cosine candidates, then rerank them with Cohere."""
    if cohere_client is None or not os.environ.get("COHERE_API_KEY"):
        print("COHERE_API_KEY not set — skipping rerank experiment")
        return None

    embedding = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[query]
    ).data[0].embedding

    candidates = collection.query(
        query_embeddings=[embedding],
        n_results=initial_k,
        include=["documents", "metadatas"],
    )
    candidate_ids = candidates["ids"][0]
    candidate_texts = candidates["documents"][0]

    if not candidate_ids:
        return []

    global _wait_seconds_since_last_read
    for attempt in range(1, MAX_RETRIES + 1):
        _throttle()
        try:
            response = cohere_client.rerank(
                model=RERANK_MODEL,
                query=query,
                documents=candidate_texts,
                top_n=min(final_k, len(candidate_texts)),
            )
            break
        except TooManyRequestsError:
            if attempt == MAX_RETRIES:
                raise
            print(f"Cohere rate limit hit — waiting {RETRY_WAIT_SECONDS:.0f}s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_WAIT_SECONDS)
            _wait_seconds_since_last_read += RETRY_WAIT_SECONDS

    reranked_ids = [candidate_ids[result.index] for result in response.results]
    return reranked_ids
