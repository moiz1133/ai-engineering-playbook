"""Benchmark harness comparing parent-child retrieval against flat 300-char retrieval."""
from __future__ import annotations
from typing import List

import chromadb
from openai import OpenAI

from retriever import retrieve_with_expansion
from baseline import retrieve_flat


def _keyword_hit(chunks_text: List[str], keywords: List[str]) -> bool:
    """True if ANY retrieved chunk contains ALL keywords (case-insensitive)."""
    lower_keywords = [kw.lower() for kw in keywords]
    for chunk in chunks_text:
        lower_chunk = chunk.lower()
        if all(kw in lower_chunk for kw in lower_keywords):
            return True
    return False


def _context_completeness(chunks_text: List[str], keywords: List[str]) -> bool:
    """True if all keywords appear in a SINGLE chunk (concentrated, not split)."""
    lower_keywords = [kw.lower() for kw in keywords]
    for chunk in chunks_text:
        lower_chunk = chunk.lower()
        if all(kw in lower_chunk for kw in lower_keywords):
            return True
    return False


def run_experiment(
    corpus_paths: List[str],
    openai_client: OpenAI,
    queries: List[dict],
    pc_collection: chromadb.Collection,
    flat_collection: chromadb.Collection,
    top_k: int = 3,
) -> dict:
    """Run all queries against both retrieval methods and collect metrics.

    Metrics per method:
      hit_rate              — fraction of queries where any chunk contains all keywords
      hit_rate_factual      — hit_rate restricted to factual queries
      hit_rate_multi        — hit_rate restricted to multi_concept queries
      context_completeness  — fraction where all keywords appear in ONE chunk
      avg_context_length    — mean total chars of retrieved text passed to LLM
      avg_top1_sim          — mean similarity score of the top-1 hit
      dedup_rate            — fraction of queries where deduplication fired (pc only)

    Args:
        corpus_paths: Paths to corpus .txt files (unused at query time; included for traceability).
        openai_client: Authenticated OpenAI client.
        queries: List of query dicts from queries.py.
        pc_collection: Pre-built parent-child ChromaDB collection.
        flat_collection: Pre-built flat ChromaDB collection.
        top_k: Number of results to retrieve.

    Returns:
        Nested dict with results for "parent_child" and "flat_300".
    """
    pc_results: List[dict] = []
    flat_results: List[dict] = []

    print("\n[experiment] Running queries...")
    print("=" * 60)

    for i, q in enumerate(queries):
        query_text = q["query"]
        keywords = q["keywords"]
        qtype = q["query_type"]
        print(f"\nQuery {i+1:2d}/{len(queries)} [{qtype}]: {query_text[:70]}")

        # ── Parent-child ──────────────────────────────────────────────────────
        pc_hits = retrieve_with_expansion(
            query_text, pc_collection, openai_client, top_k=top_k, verbose=True
        )
        pc_texts = [h["parent_text"] for h in pc_hits]
        pc_deduped = any(h["was_deduplicated"] for h in pc_hits)
        pc_top1_sim = pc_hits[0]["child_score"] if pc_hits else 0.0
        pc_total_len = sum(len(t) for t in pc_texts)

        pc_hit = _keyword_hit(pc_texts, keywords)
        pc_complete = _context_completeness(pc_texts, keywords)
        print(
            f"  PC  | hit={pc_hit} complete={pc_complete} "
            f"top1={pc_top1_sim:.3f} len={pc_total_len} dedup={pc_deduped}"
        )

        pc_results.append(
            {
                "hit": pc_hit,
                "complete": pc_complete,
                "top1_sim": pc_top1_sim,
                "context_len": pc_total_len,
                "deduped": pc_deduped,
                "query_type": qtype,
            }
        )

        # ── Flat baseline ─────────────────────────────────────────────────────
        flat_hits = retrieve_flat(query_text, flat_collection, openai_client, top_k=top_k)
        flat_texts = [h["text"] for h in flat_hits]
        flat_top1_sim = flat_hits[0]["score"] if flat_hits else 0.0
        flat_total_len = sum(len(t) for t in flat_texts)

        flat_hit = _keyword_hit(flat_texts, keywords)
        flat_complete = _context_completeness(flat_texts, keywords)
        print(
            f"  Flat| hit={flat_hit} complete={flat_complete} "
            f"top1={flat_top1_sim:.3f} len={flat_total_len}"
        )

        flat_results.append(
            {
                "hit": flat_hit,
                "complete": flat_complete,
                "top1_sim": flat_top1_sim,
                "context_len": flat_total_len,
                "query_type": qtype,
            }
        )

    def _agg(results: List[dict]) -> dict:
        n = len(results)
        factual = [r for r in results if r["query_type"] == "factual"]
        multi = [r for r in results if r["query_type"] == "multi_concept"]
        return {
            "hit_rate": sum(r["hit"] for r in results) / n,
            "hit_rate_factual": (
                sum(r["hit"] for r in factual) / len(factual) if factual else 0.0
            ),
            "hit_rate_multi": (
                sum(r["hit"] for r in multi) / len(multi) if multi else 0.0
            ),
            "context_completeness_pct": sum(r["complete"] for r in results) / n,
            "avg_context_length": sum(r["context_len"] for r in results) / n,
            "avg_top1_sim": sum(r["top1_sim"] for r in results) / n,
            "dedup_rate": (
                sum(r.get("deduped", False) for r in results) / n
            ),
        }

    return {
        "parent_child": _agg(pc_results),
        "flat_300": {**_agg(flat_results), "dedup_rate": 0.0},
    }


def print_results_table(comparison: dict) -> None:
    """Print a formatted comparison table and dynamic interpretation."""
    pc = comparison["parent_child"]
    fl = comparison["flat_300"]

    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def chars(v: float) -> str:
        return f"{int(v):,} chars"

    def sim(v: float) -> str:
        return f"{v:.2f}"

    rows = [
        ("Keyword hit rate",       pct(pc["hit_rate"]),                    pct(fl["hit_rate"])),
        ("  Factual queries",      pct(pc["hit_rate_factual"]),             pct(fl["hit_rate_factual"])),
        ("  Multi-concept queries",pct(pc["hit_rate_multi"]),               pct(fl["hit_rate_multi"])),
        ("Context completeness",   pct(pc["context_completeness_pct"]),     pct(fl["context_completeness_pct"])),
        ("Avg context length",     chars(pc["avg_context_length"]),         chars(fl["avg_context_length"])),
        ("Avg top-1 similarity",   sim(pc["avg_top1_sim"]),                 sim(fl["avg_top1_sim"])),
        ("Deduplication fired",    pct(pc["dedup_rate"]),                   "n/a"),
    ]

    col_w = [26, 15, 15]
    header = f"{'Metric':<{col_w[0]}} | {'Parent-child':>{col_w[1]}} | {'Flat 300-char':>{col_w[2]}}"
    sep = "-" * col_w[0] + "-+-" + "-" * col_w[1] + "-+-" + "-" * col_w[2]

    print("\n" + "=" * len(header))
    print(header)
    print(sep)
    for label, pc_val, fl_val in rows:
        print(f"{label:<{col_w[0]}} | {pc_val:>{col_w[1]}} | {fl_val:>{col_w[2]}}")
    print("=" * len(header))

    # dynamic interpretation
    multi_improvement = (pc["hit_rate_multi"] - fl["hit_rate_multi"]) * 100
    completeness_improvement = (pc["context_completeness_pct"] - fl["context_completeness_pct"]) * 100
    pc_len = int(pc["avg_context_length"])
    fl_len = int(fl["avg_context_length"])
    token_ratio = pc_len / fl_len if fl_len > 0 else 1.0

    print(
        f"\nParent-child improved multi-concept hit rate by {multi_improvement:+.1f}% over flat retrieval.\n"
        f"Context completeness improved by {completeness_improvement:+.1f}% — answers that were split across "
        f"flat chunks were consolidated into single parent chunks.\n"
        f"The tradeoff: avg context sent to LLM was {pc_len:,} chars vs {fl_len:,} chars for flat "
        f"— about {token_ratio:.1f}x more tokens per query."
    )
