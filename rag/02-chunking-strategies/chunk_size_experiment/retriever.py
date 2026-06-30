"""In-memory cosine-similarity retrieval over embedded chunks.

No vector DB: the corpus here is small enough (a few hundred chunks at most)
that a single NumPy matrix-vector product is faster than standing up an
index would be worth.
"""

from typing import List

import numpy as np
from openai import OpenAI

from embedder import EMBEDDING_MODEL


def retrieve(query: str, chunks: List[dict], client: OpenAI, top_k: int = 3) -> List[dict]:
    """Embed `query`, score it against every chunk's embedding by cosine
    similarity, and return the top_k chunks (each with a "score" key added),
    sorted by score descending.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)

    corpus = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    query_norm = query_vec / np.clip(np.linalg.norm(query_vec), 1e-12, None)
    corpus_norm = corpus / np.clip(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12, None)
    scores = corpus_norm @ query_norm

    top_indices = np.argsort(-scores)[:top_k]
    results: List[dict] = []
    for idx in top_indices:
        result = dict(chunks[idx])
        result["score"] = float(scores[idx])
        results.append(result)
    return results
