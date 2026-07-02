"""Flat 300-char chunking baseline — same embedding model, same top-k as parent-child."""
from __future__ import annotations
from typing import List

import chromadb
from openai import OpenAI

from store import EMBED_MODEL, BATCH_SIZE, embed_query

FLAT_COLLECTION_NAME = "flat_chunks_300"
FLAT_CHUNK_SIZE = 300
FLAT_OVERLAP = 30


def _recursive_split_flat(text: str, size: int, overlap: int) -> List[dict]:
    """Split text into flat chunks of ~size chars with overlap."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[dict] = []

    def _split(s: str, base_offset: int, seps: List[str]) -> None:
        if len(s) <= size:
            if s.strip():
                chunks.append({"text": s, "char_start": base_offset})
            return
        sep = seps[0] if seps else ""
        remaining = seps[1:] if seps else []

        if sep and sep in s:
            parts = s.split(sep)
            current = ""
            current_offset = base_offset
            for i, part in enumerate(parts):
                candidate = current + (sep if current else "") + part
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current.strip():
                        if len(current) > size:
                            _split(current, current_offset, remaining)
                        else:
                            chunks.append({"text": current, "char_start": current_offset})
                    next_offset = base_offset + s.index(part, max(0, len(current) - len(part)))
                    current_offset = next_offset
                    current = part
            if current.strip():
                chunks.append({"text": current, "char_start": current_offset})
        else:
            start = 0
            while start < len(s):
                end = min(start + size, len(s))
                chunk = s[start:end]
                if chunk.strip():
                    chunks.append({"text": chunk, "char_start": base_offset + start})
                if end == len(s):
                    break
                start = end - overlap

    _split(text, 0, separators)
    return chunks


def build_flat_index(texts: List[str], openai_client: OpenAI) -> chromadb.Collection:
    """Build a ChromaDB collection of flat 300-char chunk embeddings."""
    all_chunks: List[dict] = []
    for text in texts:
        all_chunks.extend(_recursive_split_flat(text, FLAT_CHUNK_SIZE, FLAT_OVERLAP))

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(FLAT_COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=FLAT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunk_texts = [c["text"] for c in all_chunks]
    embeddings: List[List[float]] = []
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i : i + BATCH_SIZE]
        response = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([item.embedding for item in response.data])

    ids = [str(i) for i in range(len(all_chunks))]
    metadatas = [{"char_start": c["char_start"]} for c in all_chunks]

    for i in range(0, len(ids), BATCH_SIZE):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=chunk_texts[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    print(f"[baseline] Indexed {len(ids)} flat chunks (300-char) into '{FLAT_COLLECTION_NAME}'.")
    return collection


def retrieve_flat(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 3,
) -> List[dict]:
    """Standard flat retrieval — no parent expansion, no deduplication."""
    query_embedding = embed_query(query, openai_client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    documents: List[str] = results["documents"][0]
    distances: List[float] = results["distances"][0]
    scores = [1.0 - d for d in distances]

    return [
        {
            "text": doc,
            "score": score,
            "was_deduplicated": False,
        }
        for doc, score in zip(documents, scores)
    ]
