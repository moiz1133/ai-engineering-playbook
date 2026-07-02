from __future__ import annotations
from typing import List

import chromadb
from openai import OpenAI


# WHAT: only child chunks go into ChromaDB — parents are NOT indexed
# WHY: we search the precise child space, then expand to the richer parent

COLLECTION_NAME = "child_chunks"
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100


def _embed_texts(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Embed a list of texts in batches to respect API limits."""
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def build_child_index(
    child_chunks: List[dict],
    openai_client: OpenAI,
) -> chromadb.Collection:
    """Build a ChromaDB collection of child chunk embeddings.

    Only child text is indexed. Parent text is stored in metadata so the
    retriever can expand to the parent without a second lookup.

    Args:
        child_chunks: Output of build_parent_child_chunks().
        openai_client: Authenticated OpenAI client.

    Returns:
        A ChromaDB Collection ready for similarity queries.
    """
    chroma_client = chromadb.Client()

    # drop and recreate so the function is idempotent
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in child_chunks]
    embeddings = _embed_texts(texts, openai_client)

    ids = [str(c["child_id"]) for c in child_chunks]
    metadatas = [
        {
            "parent_id": c["parent_id"],
            "parent_text": c["parent_text"],
            "char_start": c["char_start"],
        }
        for c in child_chunks
    ]

    # add in batches
    for i in range(0, len(ids), BATCH_SIZE):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=texts[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    print(f"[store] Indexed {len(ids)} child chunks into '{COLLECTION_NAME}'.")
    return collection


def embed_query(query: str, openai_client: OpenAI) -> List[float]:
    """Embed a single query string."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding
