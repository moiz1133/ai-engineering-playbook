from __future__ import annotations
from typing import List

import chromadb
from openai import OpenAI

from store import embed_query


# WHAT: deduplication prevents sending the same parent twice when two children hit
# WHY: without dedup, LLM receives duplicate context that wastes tokens


def retrieve_with_expansion(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 3,
    verbose: bool = False,
) -> List[dict]:
    """Search child chunks and expand each hit to its parent.

    Steps:
      1. Embed the query.
      2. Retrieve top_k child chunks by cosine similarity.
      3. For each child hit, fetch parent_text from metadata.
      4. Deduplicate: if two children share the same parent_id, return that
         parent only once (keeping the child with the highest similarity score).

    Args:
        query: Natural-language query string.
        collection: ChromaDB collection of child embeddings.
        openai_client: Authenticated OpenAI client.
        top_k: Number of child chunks to retrieve before deduplication.
        verbose: If True, print deduplication events.

    Returns:
        List of result dicts, one per unique parent:
          {parent_text, child_text, child_score, parent_id, was_deduplicated}
    """
    query_embedding = embed_query(query, openai_client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents: List[str] = results["documents"][0]
    metadatas: List[dict] = results["metadatas"][0]
    # ChromaDB cosine distance = 1 - similarity; convert to similarity score
    distances: List[float] = results["distances"][0]
    scores = [1.0 - d for d in distances]

    seen_parents: dict[int, dict] = {}
    dedup_fired = False

    for child_text, meta, score in zip(documents, metadatas, scores):
        parent_id: int = meta["parent_id"]
        parent_text: str = meta["parent_text"]

        if parent_id in seen_parents:
            # keep the hit with the higher similarity score
            if score > seen_parents[parent_id]["child_score"]:
                seen_parents[parent_id]["child_score"] = score
                seen_parents[parent_id]["child_text"] = child_text
            seen_parents[parent_id]["was_deduplicated"] = True
            dedup_fired = True
        else:
            seen_parents[parent_id] = {
                "parent_text": parent_text,
                "child_text": child_text,
                "child_score": score,
                "parent_id": parent_id,
                "was_deduplicated": False,
            }

    if dedup_fired and verbose:
        print(f"  [dedup] Query triggered deduplication: '{query[:60]}...'")

    return list(seen_parents.values())
