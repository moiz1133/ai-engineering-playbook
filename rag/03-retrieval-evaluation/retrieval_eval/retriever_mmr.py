"""Maximal Marginal Relevance (MMR) retriever.

WHAT: MMR selects chunks that are relevant to the query BUT dissimilar to
      already-selected chunks
WHY: pure cosine top-k often returns several near-duplicate chunks (paraphrases
     of the same fact); MMR trades some relevance for diversity so each new
     chunk adds new information
WHAT: lambda_param controls the tradeoff
      lambda=1.0 -> pure relevance (same as cosine)
      lambda=0.0 -> pure diversity (ignores query)
      lambda=0.5 -> balanced (default)
"""

from __future__ import annotations

from typing import List

import chromadb
import numpy as np
from openai import OpenAI

from corpus_builder import EMBED_MODEL


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_mmr(query: str, collection: chromadb.Collection,
                  openai_client: OpenAI, top_k: int = 5,
                  lambda_param: float = 0.5,
                  candidate_k: int = 20) -> List[str]:
    """Retrieve candidate_k cosine candidates, then greedily select top_k via MMR."""
    query_embedding = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[query]
    ).data[0].embedding
    query_vec = np.array(query_embedding)

    candidates = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "embeddings"],
    )
    candidate_ids = candidates["ids"][0]
    if not candidate_ids:
        return []
    candidate_vecs = [np.array(e) for e in candidates["embeddings"][0]]

    query_sims = [_cosine_sim(query_vec, vec) for vec in candidate_vecs]

    remaining = list(range(len(candidate_ids)))
    selected: List[int] = []

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -float("inf")
        for i in remaining:
            sim_to_query = query_sims[i]
            if selected:
                sim_to_selected = max(
                    _cosine_sim(candidate_vecs[i], candidate_vecs[j]) for j in selected
                )
            else:
                sim_to_selected = 0.0
            mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_ids[i] for i in selected]
