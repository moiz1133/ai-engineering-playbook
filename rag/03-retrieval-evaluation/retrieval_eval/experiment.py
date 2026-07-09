"""Run all retrieval methods over the shared eval set and collect metrics.

WHAT: single shared eval set used by all four retrieval methods
WHY: fair comparison requires identical corpus and embeddings — only retrieval
     logic changes between baseline / rerank / MMR / hybrid (BM25+RRF)
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import chromadb
from openai import OpenAI

import retriever_rerank
from metrics import hit_at_k, metrics_report, mrr
from retriever_baseline import retrieve_baseline
from retriever_hybrid import (
    analyse_disagreements,
    build_bm25_index,
    retrieve_bm25,
    retrieve_hybrid,
    retrieve_vector_for_fusion,
)
from retriever_mmr import retrieve_mmr
from retriever_rerank import retrieve_with_rerank

LAMBDA_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RRF_K_SWEEP = [10, 30, 60, 100]


def _run_method(label: str, questions: List[str], relevant_ids_list: List[List[str]],
                 query_types: List[str], retrieve_fn: Callable[[str], Optional[List[str]]]) -> Dict:
    results_list: List[List[str]] = []
    latencies: List[float] = []
    for question in questions:
        start = time.perf_counter()
        ids = retrieve_fn(question)
        elapsed = time.perf_counter() - start
        # Exclude deliberate rate-limit pacing/backoff sleeps (Cohere trial key)
        # from measured latency — that's a client-side throttle, not API latency.
        elapsed -= retriever_rerank.get_and_reset_wait_seconds()
        latencies.append(max(elapsed, 0.0))
        results_list.append(ids if ids else [])

    report = metrics_report(label, results_list, relevant_ids_list, query_types)
    report["avg_latency_ms"] = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0
    return report


def run_all_experiments(collection: chromadb.Collection, openai_client: OpenAI,
                         cohere_client, eval_set: List[dict]) -> Dict:
    """Run baseline, Cohere rerank, MMR (multiple lambdas), and hybrid BM25+RRF
    (plus a BM25-only baseline and an RRF k sweep) over eval_set."""
    answerable = [item for item in eval_set if item.get("relevant_chunk_ids")]
    questions = [item["question"] for item in answerable]
    relevant_ids_list = [item["relevant_chunk_ids"] for item in answerable]
    query_types = [item["query_type"] for item in answerable]

    results: Dict = {}

    results["baseline"] = _run_method(
        "Baseline cosine", questions, relevant_ids_list, query_types,
        lambda q: retrieve_baseline(q, collection, openai_client, top_k=5),
    )

    cohere_available = cohere_client is not None and bool(os.environ.get("COHERE_API_KEY"))
    if cohere_available:
        results["rerank"] = _run_method(
            "Cohere rerank", questions, relevant_ids_list, query_types,
            lambda q: retrieve_with_rerank(q, collection, openai_client, cohere_client,
                                            initial_k=20, final_k=5),
        )
    else:
        print("COHERE_API_KEY not set — skipping rerank experiment")
        results["rerank"] = None

    results["mmr"] = _run_method(
        "MMR (λ=0.5)", questions, relevant_ids_list, query_types,
        lambda q: retrieve_mmr(q, collection, openai_client, top_k=5, lambda_param=0.5),
    )
    results["mmr_high_div"] = _run_method(
        "MMR (λ=0.3, div)", questions, relevant_ids_list, query_types,
        lambda q: retrieve_mmr(q, collection, openai_client, top_k=5, lambda_param=0.3),
    )
    results["mmr_low_div"] = _run_method(
        "MMR (λ=0.7, rel)", questions, relevant_ids_list, query_types,
        lambda q: retrieve_mmr(q, collection, openai_client, top_k=5, lambda_param=0.7),
    )

    sweep: Dict[float, Dict] = {}
    for lam in LAMBDA_SWEEP:
        results_list = [
            retrieve_mmr(q, collection, openai_client, top_k=5, lambda_param=lam)
            for q in questions
        ]
        sweep[lam] = {
            "mrr": mrr(results_list, relevant_ids_list),
            "hit_at_3": hit_at_k(results_list, relevant_ids_list, k=3),
        }
    results["mmr_lambda_sweep"] = sweep

    # Build BM25 index once — shared across all hybrid and BM25-only calls
    bm25_index, bm25_chunk_ids = build_bm25_index(collection)

    results["hybrid_rrf"] = _run_method(
        "Hybrid (BM25+RRF)", questions, relevant_ids_list, query_types,
        lambda q: retrieve_hybrid(q, collection, bm25_index, bm25_chunk_ids, openai_client, top_k=5),
    )

    # BM25-only as a standalone baseline for comparison
    results["bm25_only"] = _run_method(
        "BM25 only", questions, relevant_ids_list, query_types,
        lambda q: [cid for cid, _ in retrieve_bm25(q, bm25_index, bm25_chunk_ids, top_k=5)],
    )

    # WHAT: k controls how much RRF dampens rank-1 advantage
    # k=10: rank 1 = 1/11 = 0.091, rank 2 = 1/12 = 0.083 → large gap (rank 1 dominates)
    # k=60: rank 1 = 1/61 = 0.016, rank 2 = 1/62 = 0.016 → small gap (smoother fusion)
    # WHY sweep: optimal k depends on corpus — 60 is the standard default but check empirically
    rrf_k_sweep: Dict[int, Dict] = {}
    for rrf_k in RRF_K_SWEEP:
        sweep_results = [
            retrieve_hybrid(q, collection, bm25_index, bm25_chunk_ids, openai_client,
                             top_k=5, rrf_k=rrf_k)
            for q in questions
        ]
        rrf_k_sweep[rrf_k] = {
            "mrr": mrr(sweep_results, relevant_ids_list),
            "hit_at_3": hit_at_k(sweep_results, relevant_ids_list, k=3),
        }
    results["rrf_k_sweep"] = rrf_k_sweep

    # Disagreement analysis needs the raw BM25-only and vector-only top-20 pools
    # that fed into the hybrid fusion for each query (not just the fused top-5).
    bm25_all = [
        [cid for cid, _ in retrieve_bm25(q, bm25_index, bm25_chunk_ids, top_k=20)]
        for q in questions
    ]
    vector_all = [
        [cid for cid, _ in retrieve_vector_for_fusion(q, collection, openai_client, top_k=20)]
        for q in questions
    ]
    results["disagreements"] = analyse_disagreements(
        answerable,
        results["baseline"]["raw_results"],
        results["hybrid_rrf"]["raw_results"],
        bm25_all,
        vector_all,
    )

    # Per query-type count of how often hybrid's reciprocal rank beats baseline's —
    # feeds the "When hybrid beats baseline" README section with real counts.
    baseline_raw = results["baseline"]["raw_results"]
    hybrid_raw = results["hybrid_rrf"]["raw_results"]
    hybrid_vs_baseline_by_type: Dict[str, Dict] = {}
    for qtype in ("factual", "multi_concept", "rephrased"):
        idxs = [i for i, t in enumerate(query_types) if t == qtype]
        improved = sum(
            1 for i in idxs
            if mrr([hybrid_raw[i]], [relevant_ids_list[i]]) >
            mrr([baseline_raw[i]], [relevant_ids_list[i]])
        )
        hybrid_vs_baseline_by_type[qtype] = {"hybrid_improved": improved, "total": len(idxs)}
    results["hybrid_vs_baseline_by_type"] = hybrid_vs_baseline_by_type

    results["_meta"] = {
        "total_eval_items": len(eval_set),
        "answerable_eval_items": len(answerable),
        "cohere_available": cohere_available,
    }

    return results


# ------------------------------------------------------------------------------------
# RETRIEVAL EVAL — INTERVIEW EXPLANATION
#
# MRR (Mean Reciprocal Rank):
#   Measures WHERE the first correct answer appears in the ranked list.
#   RR = 1/rank; MRR = mean across all queries.
#   Penalises burying the correct answer at rank 3 vs rank 1.
#   MRR=1.0 = correct answer always at rank 1.
#   MRR=0.33 = correct answer always at rank 3.
#
# Hit@k:
#   Binary — did the correct answer appear anywhere in top k?
#   MRR and Hit@k are complementary: MRR rewards rank, Hit@k rewards presence.
#
# Cohere reranking:
#   Retrieve wide (top-20 cosine) -> rerank narrow (cross-encoder sees query+chunk together).
#   Cross-encoder fixes false positives from cosine: "similar vocab, different meaning".
#   Cost: ~600ms extra latency per query. Worth it when MRR delta is meaningful.
#
# MMR (Maximal Marginal Relevance):
#   score = λ * sim(chunk, query) - (1-λ) * max_sim(chunk, already_selected)
#   Selects chunks that are relevant AND different from each other.
#   Useful for multi-concept queries — avoids 3 paraphrases of the same fact.
#   λ=1.0 -> pure cosine. λ=0.0 -> pure diversity. λ=0.5 -> balanced.
#   Does NOT improve MRR vs baseline in most single-doc retrieval scenarios —
#   its value is diversity of information, not rank position of correct answer.
#
# Hybrid search (BM25 + vector RRF):
#   BM25 ranked list + vector ranked list -> Reciprocal Rank Fusion -> single fused list.
#   Fixes exact-term queries that pure cosine misses (semantic drift) while still
#   catching paraphrases that pure BM25 misses (no shared vocabulary).
#   RRF uses only rank position, not raw scores, so BM25 and cosine — which live on
#   totally different scales — never need score normalisation. See retriever_hybrid.py.
#
# When to use each:
#   Cosine baseline -> fast, good enough for focused single-concept queries
#   Cohere rerank   -> when precision matters and 600ms latency is acceptable
#   MMR             -> when downstream LLM needs diverse context (multi-step reasoning)
#   Hybrid BM25+RRF -> when queries mix exact-term lookups and semantic questions,
#                      at near-zero extra latency over vector search alone
# ------------------------------------------------------------------------------------
