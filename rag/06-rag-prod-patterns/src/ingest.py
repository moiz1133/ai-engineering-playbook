"""One-shot script: chunk data/corpus/*.md, embed each chunk, and store in a persistent ChromaDB collection.

Run with: python -m src.ingest
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import chromadb
import tiktoken
from openai import OpenAI

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

_encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    """Split text into overlapping chunks of roughly chunk_size tokens each."""
    tokens = _encoding.encode(text)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(_encoding.decode(window))
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts with the configured embedding model."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main() -> None:
    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    files = sorted(CORPUS_DIR.glob("*.md"))
    logger.info("Found %d corpus file(s) in %s", len(files), CORPUS_DIR)

    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            logger.warning("Skipping empty file: %s", path.name)
            continue

        # Deterministic IDs (source_file + chunk_index) plus upsert (not add)
        # is what makes re-running this script idempotent: a re-ingest
        # overwrites the same chunk rows instead of appending duplicates.
        ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
        embeddings = embed_texts(openai_client, chunks)
        metadatas = [
            {"source_file": path.name, "chunk_index": i, "chunk_id": ids[i]} for i in range(len(chunks))
        ]

        collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        logger.info("Ingested %s -> %d chunk(s)", path.name, len(chunks))

    logger.info("Done. Collection %r now has %d chunk(s).", CHROMA_COLLECTION_NAME, collection.count())


if __name__ == "__main__":
    main()
