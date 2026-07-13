"""Embedding-based semantic cache for LLM query/response pairs.

WHAT: in-memory list of cached queries with their embeddings and responses
WHY: no external cache store needed — corpus is small enough for an in-memory
     cosine scan
NOTE: for production, swap the entries list for Redis with a vector extension
      (e.g. RedisVL)

THREAD SAFETY: this class is NOT thread-safe — self.entries is mutated by both
lookup() (read) and store() (append/evict) without any locking. Concurrent
access from multiple threads can race on eviction or read a partially-updated
list. For threaded use, guard lookup()/store() calls with a threading.Lock();
for asyncio, use asyncio.Lock().
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI


class SemanticCache:
    def __init__(self, openai_client: OpenAI, similarity_threshold: float = 0.95,
                 max_entries: int = 1000, embedding_model: str = "text-embedding-3-small"):
        self.client = openai_client
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.embedding_model = embedding_model
        self.entries: List[Dict] = []
        self.stats = {"hits": 0, "misses": 0, "total_saved_usd": 0.0}

    def _embed(self, text: str) -> np.ndarray:
        """Embed text and L2-normalise it to a unit vector.

        WHAT: normalise so cosine similarity reduces to a plain dot product
        WHY: avoids the division in the cosine formula when both vectors are
             already unit length
        """
        response = self.client.embeddings.create(model=self.embedding_model, input=[text])
        vec = np.array(response.data[0].embedding)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # valid because both are unit vectors

    def lookup(self, query: str) -> Optional[Dict]:
        """Return the cached response for the most similar cached query if its
        similarity clears the threshold; otherwise None.

        WHAT: similarity >= 0.95 means the new query is semantically
              near-identical to a cached one
        WHY: threshold=0.95 is conservative — 0.90 risks returning wrong
             cached responses for similar-sounding but different questions
        NOTE: log best_similarity even on a miss — helps calibrate the
              threshold over time
        """
        query_vec = self._embed(query)

        best_sim = -1.0
        best_entry = None
        for entry in self.entries:
            sim = self._cosine_similarity(query_vec, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.threshold:
            best_entry["hit_count"] += 1
            self.stats["hits"] += 1
            print(f"[SEMANTIC CACHE] HIT | similarity={best_sim:.3f} | "
                  f"query_preview={query[:60]!r}")
            return {
                "response": best_entry["response"],
                "similarity": best_sim,
                "cache_hit": True,
                "matched_query": best_entry["query"],
            }

        self.stats["misses"] += 1
        print(f"[SEMANTIC CACHE] MISS | best_similarity={max(best_sim, 0.0):.3f} | "
              f"query_preview={query[:60]!r}")
        return None

    def store(self, query: str, query_embedding: np.ndarray, response: str,
              cost_usd: float = 0.0) -> None:
        """Cache a query/response pair, evicting the oldest entry if full.

        WHAT: max_entries cap prevents unbounded memory growth
        WHY: LRU eviction keeps the most recently used queries — old ones are
             less likely to be hit
        NOTE: pass in the embedding computed during the lookup attempt — do
              NOT re-embed here, lookup() already paid for that API call
        """
        if len(self.entries) >= self.max_entries:
            oldest_idx = min(range(len(self.entries)),
                              key=lambda i: self.entries[i]["timestamp"])
            self.entries.pop(oldest_idx)

        self.entries.append({
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hit_count": 0,
        })
        self.stats["total_saved_usd"] += cost_usd

    def summary(self) -> Dict:
        hits = self.stats["hits"]
        misses = self.stats["misses"]
        total = hits + misses
        return {
            **self.stats,
            "total_entries": len(self.entries),
            "threshold": self.threshold,
            "hit_rate": hits / total if total > 0 else 0.0,
        }
