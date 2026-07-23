"""Baseline retrieval: plain top-k vector search against the persistent ChromaDB collection."""

from __future__ import annotations

from typing import List, TypedDict

import chromadb
from openai import OpenAI

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TOP_K


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    source_file: str
    distance: float


_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_collection = _chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def get_collection() -> chromadb.Collection:
    """Return the shared persistent ChromaDB collection used by all retrieval strategies."""
    return _collection


def embed_query(client: OpenAI, query: str) -> List[float]:
    """Embed a single query string with the configured embedding model."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def to_chunks(results: dict) -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=results["metadatas"][0][i]["chunk_id"],
            text=results["documents"][0][i],
            source_file=results["metadatas"][0][i]["source_file"],
            distance=results["distances"][0][i],
        )
        for i in range(len(results["ids"][0]))
    ]


def retrieve_baseline(client: OpenAI, query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """Return the top_k nearest chunks to the query via plain vector search. This is what the user's answer is built from."""
    query_embedding = embed_query(client, query)
    results = _collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return to_chunks(results)
