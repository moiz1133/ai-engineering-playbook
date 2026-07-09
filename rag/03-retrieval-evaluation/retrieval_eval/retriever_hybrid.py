# pip install rank-bm25
"""Hybrid retriever: BM25 keyword search fused with vector search via RRF.

WHAT: BM25 keyword retrieval + cosine vector retrieval, fused with Reciprocal
      Rank Fusion (RRF) into a single ranked list
WHY: pure vector search misses exact-term queries (semantic drift); pure BM25
     misses semantic similarity (misses paraphrases). Fusing both ranked lists
     surfaces chunks that are either a strong keyword match, a strong semantic
     match, or both — without needing to normalise two incompatible score scales
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from corpus_builder import EMBED_MODEL

# ------------------------------------------------------------------------------------
# HYBRID SEARCH — INTERVIEW EXPLANATION
#
# Problem with pure vector search:
#   Embeds query and chunk INDEPENDENTLY then compares.
#   A chunk about "transformers" may rank higher than one mentioning "BERT" literally
#   if the query is "BERT architecture" — semantic drift from exact-term queries.
#
# Problem with pure BM25:
#   Counts keyword overlap — misses semantic similarity entirely.
#   Query "how does attention work" → BM25 needs the word "attention" to be in the chunk.
#   A chunk explaining "query-key-value mechanism" scores zero despite being the answer.
#
# Why hybrid fixes both:
#   BM25 ranked list + vector ranked list → RRF → single fused list.
#   A chunk that is BOTH a keyword match AND a semantic match scores highest.
#   A chunk that only appears in one list still surfaces — just ranked lower.
#
# Why RRF over linear interpolation:
#   Linear: final_score = α * vector_score + (1-α) * bm25_score
#   Problem: vector scores are cosine distances (0–2), BM25 scores are TF-IDF weights
#            (unbounded). Normalising them requires knowing the max score in the corpus,
#            which changes as documents are added.
#   RRF:    uses only rank POSITION (integers) — no normalisation needed.
#           Adding new documents doesn't invalidate existing fusion scores.
#           This is why RRF is the production default.
#
# RRF formula: score(chunk) = Σ 1/(k + rank_in_list_i)  for each list i
# k=60 standard — dampens rank-1 advantage so neither list dominates.
#
# Latency:
#   BM25 runs in-memory on tokenised text — typically <5ms.
#   Vector search is the bottleneck (embedding API + ChromaDB).
#   Hybrid total latency ≈ vector latency + ~5ms — negligible overhead.
# ------------------------------------------------------------------------------------


# WHAT: BM25Okapi is the standard keyword retrieval algorithm used in Elasticsearch
# WHY: it scores chunks by term frequency + inverse document frequency — exact keyword
#      matches score very high regardless of semantic meaning
# WHAT: we build BM25 over the same chunks already stored in ChromaDB — no new storage
# WHY: single source of truth for chunks; BM25 index is rebuilt in-memory at startup
#      (fast for small corpora; for production you'd persist it with pickle)
def build_bm25_index(collection: chromadb.Collection) -> Tuple[BM25Okapi, List[str]]:
    """Fetch every chunk from ChromaDB and build an in-memory BM25 index over it."""
    result = collection.get(include=["documents", "metadatas"])
    chunk_ids: List[str] = result["ids"]
    documents: List[str] = result["documents"]

    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25_index = BM25Okapi(tokenized_docs)
    return bm25_index, chunk_ids


# WHAT: BM25 returns a score per chunk — higher = more keyword overlap with query
# WHY: top_k=20 (wide pool) because RRF will fuse this with the vector top-20
def retrieve_bm25(query: str, bm25_index: BM25Okapi, chunk_ids: List[str],
                   top_k: int = 20) -> List[Tuple[str, float]]:
    """Score every chunk against the query with BM25 and return the top_k by score."""
    query_tokens = query.lower().split()
    scores = bm25_index.get_scores(query_tokens)

    ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


# WHAT: retrieve wide vector candidates for RRF — same embedding as baseline
# WHY: RRF uses RANK POSITION only, not raw scores — so cosine distance and BM25
#      scores don't need to be on the same scale. This is RRF's key advantage
#      over linear interpolation (which requires normalising two different score scales)
def retrieve_vector_for_fusion(query: str, collection: chromadb.Collection,
                                openai_client: OpenAI, top_k: int = 20) -> List[Tuple[str, float]]:
    """Standard cosine retrieval, widened to top_k=20 candidates for RRF fusion."""
    embedding = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[query]
    ).data[0].embedding

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    distances = result["distances"][0]
    return list(zip(ids, distances))


# WHAT: RRF computes a fusion score for each chunk across all ranked lists
# HOW:  for each list, score(chunk) += 1 / (k + rank)
#        rank 1  → adds 1/61  ≈ 0.0164
#        rank 5  → adds 1/65  ≈ 0.0154
#        rank 20 → adds 1/80  ≈ 0.0125
# WHY:  chunks that appear HIGH in MULTIPLE lists get high scores
#        chunks that appear in only one list get lower scores
# WHAT: k=60 is the standard RRF constant (from the original 2009 paper)
# WHY:  k dampens the advantage of rank 1 vs rank 2 — prevents one list from
#        dominating when its rank 1 chunk is much less relevant than the other list's
# KEY ADVANTAGE OVER LINEAR INTERPOLATION:
#        RRF needs no score normalisation — BM25 scores and cosine distances are
#        on completely different scales, but RRF only uses rank position (integers)
def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    """Fuse multiple ranked chunk-id lists into one, scored by 1/(k + rank)."""
    scores: Dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)


# WHAT: hybrid = BM25 ranked list + vector ranked list → RRF → fused ranked list
# WHY: a chunk ranking #1 in BM25 (exact match) and #4 in vector (semantic match)
#      will outscore a chunk that only appears in one list — best of both worlds
# EXAMPLE: query "BERT architecture" —
#   BM25: chunk mentioning "BERT" literally at rank 1
#   vector: chunk about "transformer models" at rank 1 (semantically similar)
#   RRF: both chunks in top-3, BERT chunk likely #1 due to dual-list presence
def retrieve_hybrid(query: str, collection: chromadb.Collection,
                     bm25_index: BM25Okapi, chunk_ids: List[str],
                     openai_client: OpenAI, top_k: int = 5,
                     rrf_k: int = 60) -> List[str]:
    """Fuse BM25 and vector top-20 candidates via RRF and return the top_k chunk_ids."""
    bm25_results = retrieve_bm25(query, bm25_index, chunk_ids, top_k=20)
    bm25_ranked = [chunk_id for chunk_id, _ in bm25_results]

    vector_results = retrieve_vector_for_fusion(query, collection, openai_client, top_k=20)
    vector_ranked = [chunk_id for chunk_id, _ in vector_results]

    fused_ranked = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=rrf_k)
    return fused_ranked[:top_k]


def explain_hybrid_result(query: str, chunk_id: str, bm25_ranked: List[str],
                           vector_ranked: List[str], rrf_score: float) -> Dict:
    """Break down why a chunk scored the way it did — its rank in each source list."""
    bm25_rank = bm25_ranked.index(chunk_id) + 1 if chunk_id in bm25_ranked else None
    vector_rank = vector_ranked.index(chunk_id) + 1 if chunk_id in vector_ranked else None
    return {
        "chunk_id": chunk_id,
        "bm25_rank": bm25_rank,
        "vector_rank": vector_rank,
        "rrf_score": rrf_score,
        "in_bm25_top5": bm25_rank is not None and bm25_rank <= 5,
        "in_vector_top5": vector_rank is not None and vector_rank <= 5,
    }


def _first_relevant_rank(ranked: List[str], relevant_ids: List[str]) -> Optional[int]:
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant_set:
            return rank
    return None


# WHAT: disagreement analysis shows WHEN hybrid beats baseline and when it doesn't
# WHY: the most common pattern — hybrid wins on exact-term queries where BM25 rank 1
#      pulls a correctly-matched chunk that vector alone ranked at 4 or 5
def analyse_disagreements(eval_set: List[dict], baseline_results: List[List[str]],
                           hybrid_results: List[List[str]], bm25_all: List[List[str]],
                           vector_all: List[List[str]]) -> List[dict]:
    """Find queries where baseline and hybrid disagree on the rank-1 chunk."""
    disagreements: List[dict] = []
    for item, baseline_ranked, hybrid_ranked, bm25_ranked, vector_ranked in zip(
        eval_set, baseline_results, hybrid_results, bm25_all, vector_all
    ):
        baseline_rank1 = baseline_ranked[0] if baseline_ranked else None
        hybrid_rank1 = hybrid_ranked[0] if hybrid_ranked else None
        if baseline_rank1 == hybrid_rank1:
            continue
        relevant_ids = item.get("relevant_chunk_ids", [])
        disagreements.append({
            "query": item["question"],
            "baseline_rank1": baseline_rank1,
            "hybrid_rank1": hybrid_rank1,
            "bm25_rank_of_correct": _first_relevant_rank(bm25_ranked, relevant_ids),
            "vector_rank_of_correct": _first_relevant_rank(vector_ranked, relevant_ids),
        })
    return disagreements
