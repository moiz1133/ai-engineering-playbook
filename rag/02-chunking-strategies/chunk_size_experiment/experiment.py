"""Chunk-size experiment harness.

Holds chunking method (recursive_chunk) and overlap percentage fixed, varies
only chunk_size, and measures retrieval quality and cost for each setting
against the same 15 TEST_QUERIES.
"""

import time
from typing import Dict, List

import numpy as np
from openai import OpenAI

from chunker import recursive_chunk
from embedder import embed_chunks
from queries import TEST_QUERIES
from retriever import retrieve


def _contains_all_keywords(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in keywords)


def _keyword_hit(top_chunks: List[dict], keywords: List[str]) -> bool:
    """True if ANY single retrieved chunk contains every ground-truth keyword."""
    return any(_contains_all_keywords(c["text"], keywords) for c in top_chunks)


def _context_completeness(top_chunks: List[dict], keywords: List[str]) -> str:
    """"concentrated" if one chunk has every keyword, "split" if the keywords
    are only present when combined across more than one retrieved chunk,
    "missed" if some keyword doesn't appear in any of the top chunks at all
    (a clean retrieval miss, not a fragmentation problem).
    """
    if _keyword_hit(top_chunks, keywords):
        return "concentrated"

    found = set()
    for c in top_chunks:
        text_lower = c["text"].lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                found.add(kw.lower())

    if found == {kw.lower() for kw in keywords}:
        return "split"
    return "missed"


def run_size_experiment(corpus_paths: List[str], client: OpenAI,
                         chunk_sizes: List[int] = [150, 300, 600]) -> Dict[int, dict]:
    texts = []
    for path in corpus_paths:
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    full_text = "\n\n".join(texts)

    results: Dict[int, dict] = {}

    for chunk_size in chunk_sizes:
        t0 = time.perf_counter()
        chunks = recursive_chunk(full_text, chunk_size)
        chunks = embed_chunks(chunks, client)
        build_time_s = time.perf_counter() - t0

        embedding_tokens = sum(len(c["text"]) for c in chunks)

        hits: List[bool] = []
        top1_similarities: List[float] = []
        query_latencies_ms: List[float] = []
        hits_by_type: Dict[str, List[bool]] = {"factual": [], "multi_concept": [], "rephrased": []}
        split_flags: List[bool] = []  # multi_concept queries only

        for q in TEST_QUERIES:
            t_q0 = time.perf_counter()
            top_chunks = retrieve(q["query"], chunks, client, top_k=3)
            query_latencies_ms.append((time.perf_counter() - t_q0) * 1000)

            hit = _keyword_hit(top_chunks, q["ground_truth_keywords"])
            hits.append(hit)
            hits_by_type[q["type"]].append(hit)
            top1_similarities.append(top_chunks[0]["score"] if top_chunks else 0.0)

            if q["type"] == "multi_concept":
                completeness = _context_completeness(top_chunks, q["ground_truth_keywords"])
                split_flags.append(completeness == "split")

        results[chunk_size] = {
            "chunk_count": len(chunks),
            "build_time_s": build_time_s,
            "embedding_tokens": embedding_tokens,
            "hit_rate": sum(hits) / len(TEST_QUERIES),
            "hit_rate_by_type": {
                t: (sum(v) / len(v) if v else 0.0) for t, v in hits_by_type.items()
            },
            "avg_top1_similarity": float(np.mean(top1_similarities)),
            "split_context_rate": (sum(split_flags) / len(split_flags)) if split_flags else 0.0,
            "avg_query_latency_ms": float(np.mean(query_latencies_ms)),
        }

    return results
