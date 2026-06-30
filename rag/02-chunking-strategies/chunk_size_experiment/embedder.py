"""Batch embedding for chunk dicts produced by chunker.py."""

from typing import List

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-large"


def embed_chunks(chunks: List[dict], client: OpenAI, batch_size: int = 20) -> List[dict]:
    """Embed every chunk's text with `EMBEDDING_MODEL`, batched to stay well
    under per-request rate limits, and attach the result as chunk["embedding"].

    Mutates and returns the same list of dicts.
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[c["text"] for c in batch],
        )
        for chunk, item in zip(batch, response.data):
            chunk["embedding"] = item.embedding
    return chunks
