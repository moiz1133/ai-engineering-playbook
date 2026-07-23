"""Shadow retrieval: a deliberately different strategy run in parallel with baseline, for offline comparison only.

The user never sees shadow results -- they only exist to answer "would a
different retrieval strategy have picked different chunks?" via the log at
logs/shadow_comparisons.jsonl.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from openai import OpenAI

from src.config import SHADOW_LOG_PATH, TOP_K
from src.retrieval.baseline import RetrievedChunk, embed_query, get_collection, to_chunks


def retrieve_shadow(client: OpenAI, query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """Fetch the top-(2*top_k) nearest chunks, then keep the top_k longest by character count.

    Length-based re-scoring is an arbitrary choice -- the point of shadow mode
    is comparing *any* alternate strategy against baseline, not this specific one.
    """
    query_embedding = embed_query(client, query)
    results = get_collection().query(query_embeddings=[query_embedding], n_results=top_k * 2)
    candidates = to_chunks(results)
    candidates.sort(key=lambda c: len(c["text"]), reverse=True)
    return candidates[:top_k]


def log_shadow_comparison(
    query: str,
    baseline_chunks: List[RetrievedChunk],
    shadow_chunks: List[RetrievedChunk],
    baseline_latency_ms: float,
    shadow_latency_ms: float,
) -> None:
    """Append one JSON line to the shadow comparison log."""
    baseline_ids = [c["chunk_id"] for c in baseline_chunks]
    shadow_ids = [c["chunk_id"] for c in shadow_chunks]

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "baseline_chunk_ids": baseline_ids,
        "shadow_chunk_ids": shadow_ids,
        "baseline_latency_ms": baseline_latency_ms,
        "shadow_latency_ms": shadow_latency_ms,
        "overlap": len(set(baseline_ids) & set(shadow_ids)),
    }

    log_path = Path(SHADOW_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
