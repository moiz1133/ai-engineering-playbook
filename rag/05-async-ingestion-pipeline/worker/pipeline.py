"""Chunk -> embed -> store pipeline steps.

WHAT: PDF/text extraction and the recursive character splitter are reused
      (unchanged algorithm) from rag/03-retrieval-evaluation/retrieval_eval/
      corpus_builder.py's _recursive_split/_merge_splits/chunk_document —
      same chunk_size=400/chunk_overlap=40 defaults, same char_start tracking
WHY: no need to re-derive a chunking algorithm that's already been built and
     used elsewhere in this repo; embedding and storage are new here because
     this project writes to a dynamic, per-request collection_name rather
     than one fixed eval corpus
"""

from __future__ import annotations

import io
from typing import Dict, List

import chromadb
from pypdf import PdfReader

from config import settings


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from all pages of a PDF given as raw bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# ----------------------------------------------------------------------------
# Recursive character splitter, reused from corpus_builder.py (same algorithm,
# same defaults). See that file for the original WHAT/WHY on the separator
# cascade and merge/overlap logic.
# ----------------------------------------------------------------------------

def _merge_splits(splits: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for s in splits:
        if len(current) + len(s) <= chunk_size:
            current += s
            continue
        if current:
            chunks.append(current)
        overlap_text = current[-chunk_overlap:] if chunk_overlap and current else ""
        current = overlap_text + s
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap:]
    if current:
        chunks.append(current)
    return chunks


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int,
                      separators: List[str]) -> List[str]:
    sep = separators[-1]
    for s in separators:
        if s == "" or s in text:
            sep = s
            break

    splits = text.split(sep) if sep else list(text)

    final_chunks: List[str] = []
    good_splits: List[str] = []
    for i, s in enumerate(splits):
        piece = s + sep if sep and i < len(splits) - 1 else s
        if len(piece) < chunk_size:
            good_splits.append(piece)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits, chunk_size, chunk_overlap))
                good_splits = []
            remaining = separators[separators.index(sep) + 1:]
            if remaining:
                final_chunks.extend(_recursive_split(piece, chunk_size, chunk_overlap, remaining))
            else:
                final_chunks.append(piece)
    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, chunk_size, chunk_overlap))
    return [c for c in final_chunks if c.strip()]


def recursive_chunk(text: str, chunk_size: int = 400, chunk_overlap: int = 40) -> List[Dict]:
    """Split text into overlapping chunks, tracking each chunk's char_start."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _recursive_split(text, chunk_size, chunk_overlap, separators)

    chunks = []
    search_pos = 0
    for chunk_text in raw_chunks:
        probe = chunk_text[:30].strip()
        idx = text.find(probe, max(0, search_pos - chunk_overlap)) if probe else search_pos
        if idx == -1:
            idx = search_pos
        chunks.append({"text": chunk_text, "char_start": idx})
        search_pos = idx + len(chunk_text)
    return chunks


def store_chunks(embedded_chunks: List[Dict], collection_name: str, doc_id: str) -> None:
    """Write embedded chunks into a (dynamically named) persistent ChromaDB collection.

    WHAT: chunk ids are namespaced by doc_id (f"{doc_id}_{i}")
    WHY: two different documents' chunk 0 must never collide in the same
         collection, since collection_name is caller-supplied and multiple
         documents commonly share one collection
    """
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(collection_name)

    ids = [f"{doc_id}_{i}" for i in range(len(embedded_chunks))]
    documents = [c["text"] for c in embedded_chunks]
    metadatas = [{"doc_id": doc_id, "chunk_id": i, "char_start": c["char_start"]}
                 for i, c in enumerate(embedded_chunks)]
    embeddings = [c["embedding"] for c in embedded_chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
