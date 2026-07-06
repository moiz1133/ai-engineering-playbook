"""Baseline retriever: pure cosine similarity, no post-processing.

WHAT: pure cosine similarity, no post-processing
WHY: this is the control — every other method is measured as delta from this
"""

from __future__ import annotations

from typing import List

import chromadb
from openai import OpenAI

from corpus_builder import EMBED_MODEL


def retrieve_baseline(query: str, collection: chromadb.Collection,
                       openai_client: OpenAI, top_k: int = 5) -> List[str]:
    """Embed the query and return the top_k nearest chunk_ids by cosine distance."""
    embedding = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[query]
    ).data[0].embedding

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return result["ids"][0]
