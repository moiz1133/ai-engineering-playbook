# PARENT-CHILD CHUNKING
# 1. INDEX small (child) chunks for precise cosine matching
# 2. RETURN large (parent) chunks to the LLM for full context
# 3. DEDUPLICATION: two children → same parent → send parent once, not twice
# 4. WHY IT WORKS: small chunks fix embedding dilution; parent expansion fixes fragmentation
#    — it solves both failure modes of naive fixed-size chunking simultaneously
# 5. TRADEOFF: more tokens per query (parent > child) → higher LLM cost per call
# 6. WHEN TO USE: corpus has multi-sentence answers where context around a fact matters

from __future__ import annotations
from typing import List, Tuple


def _recursive_split(text: str, size: int, overlap: int) -> List[Tuple[str, int]]:
    """Split text into chunks of ~size chars with overlap, respecting sentence/word boundaries.

    Returns list of (chunk_text, char_start_in_original).
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[Tuple[str, int]] = []

    def _split(s: str, base_offset: int, seps: List[str]) -> None:
        if len(s) <= size:
            if s.strip():
                chunks.append((s, base_offset))
            return

        sep = seps[0] if seps else ""
        remaining_seps = seps[1:] if seps else []

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
                        _split(current, current_offset, remaining_seps) if len(current) > size else chunks.append((current, current_offset))
                    # compute next offset considering the sep we split on
                    # track where the current part starts in the original text
                    current_offset = base_offset + s.index(part, len(current) + len(sep) if i > 0 else 0)
                    current = part
            if current.strip():
                chunks.append((current, current_offset))
        else:
            # no separator found: hard-split at size boundary with overlap
            start = 0
            while start < len(s):
                end = min(start + size, len(s))
                chunk = s[start:end]
                if chunk.strip():
                    chunks.append((chunk, base_offset + start))
                if end == len(s):
                    break
                start = end - overlap

    _split(text, 0, separators)
    return chunks


def _split_into_parent_chunks(text: str, parent_size: int) -> List[Tuple[str, int]]:
    return _recursive_split(text, parent_size, overlap=0)


def _split_into_child_chunks(
    parent_text: str, parent_start: int, child_size: int, child_overlap: int
) -> List[Tuple[str, int]]:
    raw = _recursive_split(parent_text, child_size, child_overlap)
    # adjust offsets to be relative to the full document
    return [(chunk, parent_start + offset) for chunk, offset in raw]


def build_parent_child_chunks(
    text: str,
    parent_size: int = 600,
    child_size: int = 150,
    child_overlap: int = 15,
) -> List[dict]:
    """Split text into parent chunks then child chunks.

    # WHAT: child chunks = small, precise embeddings for matching
    # WHY: smaller chunks embed a single idea → higher cosine similarity on focused queries
    # WHAT: parent_text stored on child so retrieval can expand without a second lookup
    # WHY: expanding to parent gives LLM the full surrounding context, not a 150-char fragment

    Returns a flat list of child chunk dicts. Each dict:
      {child_id, text, char_start, parent_id, parent_text, chunk_size_setting}

    Parents are stored on each child for easy expansion. Call get_parents() if you
    need the parent-level view.
    """
    parent_chunks_raw = _split_into_parent_chunks(text, parent_size)

    child_chunks: List[dict] = []
    parents: List[dict] = []
    child_id_counter = 0

    for parent_id, (parent_text, parent_start) in enumerate(parent_chunks_raw):
        child_ids: List[int] = []
        raw_children = _split_into_child_chunks(parent_text, parent_start, child_size, child_overlap)

        for child_text, child_start in raw_children:
            child_chunks.append(
                {
                    "child_id": child_id_counter,
                    "text": child_text,
                    "char_start": child_start,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "chunk_size_setting": "child",
                }
            )
            child_ids.append(child_id_counter)
            child_id_counter += 1

        parents.append(
            {
                "parent_id": parent_id,
                "text": parent_text,
                "char_start": parent_start,
                "child_ids": child_ids,
            }
        )

    return child_chunks


def get_parents_from_children(child_chunks: List[dict]) -> List[dict]:
    """Reconstruct the parent-level view from a flat child list."""
    seen: dict[int, dict] = {}
    for c in child_chunks:
        pid = c["parent_id"]
        if pid not in seen:
            seen[pid] = {
                "parent_id": pid,
                "text": c["parent_text"],
                "child_ids": [],
            }
        seen[pid]["child_ids"].append(c["child_id"])
    return list(seen.values())
