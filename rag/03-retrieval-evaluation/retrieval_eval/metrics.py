"""MRR and Hit@k retrieval metrics.

WHAT: MRR (Mean Reciprocal Rank) and Hit@k are the two metrics used to score
      every retrieval method in this project.
WHY: they are complementary — MRR rewards rank position, Hit@k rewards mere
     presence in the top-k results.
"""

from __future__ import annotations

from typing import Dict, List


def _reciprocal_rank(result_ids: List[str], relevant_ids: List[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(result_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mrr(results_list: List[List[str]], relevant_ids_list: List[List[str]]) -> float:
    """Mean Reciprocal Rank across all queries.

    EXAMPLE: correct chunk at rank 1 -> RR=1.0; rank 2 -> RR=0.5;
             rank 3 -> RR=0.33; not found -> RR=0
    """
    if not results_list:
        return 0.0
    scores = [
        _reciprocal_rank(results, relevant)
        for results, relevant in zip(results_list, relevant_ids_list)
    ]
    return sum(scores) / len(scores)


def hit_at_k(results_list: List[List[str]], relevant_ids_list: List[List[str]],
             k: int = 3) -> float:
    """Fraction of queries where a relevant chunk appears anywhere in the top-k results."""
    if not results_list:
        return 0.0
    hits = []
    for results, relevant in zip(results_list, relevant_ids_list):
        relevant_set = set(relevant)
        hits.append(1.0 if any(cid in relevant_set for cid in results[:k]) else 0.0)
    return sum(hits) / len(hits)


def metrics_report(label: str, results_list: List[List[str]],
                    relevant_ids_list: List[List[str]],
                    query_types: List[str] = None) -> Dict:
    """Compute MRR, Hit@1/3/5, a per-query-type MRR/Hit@3 breakdown, and stash
    raw_results (the ranked chunk_ids per query) for downstream analysis."""
    report = {
        "method": label,
        "mrr": mrr(results_list, relevant_ids_list),
        "hit_at_1": hit_at_k(results_list, relevant_ids_list, k=1),
        "hit_at_3": hit_at_k(results_list, relevant_ids_list, k=3),
        "hit_at_5": hit_at_k(results_list, relevant_ids_list, k=5),
        "by_type": {},
        # Raw per-query ranked chunk_ids — kept so downstream analysis (e.g.
        # disagreement analysis between two methods) doesn't need to re-run retrieval.
        "raw_results": results_list,
    }

    if query_types is not None:
        for qtype in ("factual", "multi_concept", "rephrased"):
            idxs = [i for i, t in enumerate(query_types) if t == qtype]
            if not idxs:
                report["by_type"][qtype] = {"mrr": 0.0, "hit_at_3": 0.0}
                continue
            sub_results = [results_list[i] for i in idxs]
            sub_relevant = [relevant_ids_list[i] for i in idxs]
            report["by_type"][qtype] = {
                "mrr": mrr(sub_results, sub_relevant),
                "hit_at_3": hit_at_k(sub_results, sub_relevant, k=3),
            }

    return report
