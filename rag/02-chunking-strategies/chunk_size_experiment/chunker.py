"""
Recursive character chunker for the chunk-size experiment.

Method (recursive separator splitting) and overlap percentage are held fixed
here; chunk_size is the only thing the experiment varies, and that variation
happens entirely at the call site (experiment.py), not in this module.
"""

from typing import List

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on_separator(text: str, separator: str) -> List[str]:
    """Split on `separator`, re-attaching it to every piece but the last so
    concatenating the pieces reproduces `text` exactly. That exactness is
    what lets char_start tracking use a running counter instead of having to
    re-search the source text later.
    """
    parts = text.split(separator)
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _split_into_pieces(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively break `text` into pieces that each fit within chunk_size,
    preferring the coarsest separator (paragraph) that actually occurs in the
    text and only falling back to finer ones for whichever piece still
    doesn't fit. A hard character cut is the implicit last resort once every
    separator in the list has been tried and a piece still doesn't fit.
    """
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, remaining = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)
    if len(pieces) == 1:
        # Separator doesn't occur in this text at all - try the next, finer one.
        return _split_into_pieces(text, chunk_size, remaining)

    result: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > chunk_size:
            result.extend(_split_into_pieces(piece, chunk_size, remaining))
        else:
            result.append(piece)
    return result


def recursive_chunk(text: str, chunk_size: int, overlap_pct: float = 0.10) -> List[dict]:
    """Recursive-separator chunking with overlap held at a fixed percentage
    of chunk_size.

    # WHAT: overlap scales with chunk_size — 10% of 150 ≈ 15 chars, 10% of 600 ≈ 60 chars
    # WHY: keeping overlap PERCENTAGE constant (not absolute) is what makes 150 vs 600 a fair comparison
    """
    overlap = int(chunk_size * overlap_pct)
    pieces = _split_into_pieces(text, chunk_size, SEPARATORS)

    # Greedily merge atomic pieces into groups <= chunk_size, tracking each
    # group's (start, end) offset with a running character cursor. Pieces are
    # always verbatim, in-order substrings of `text` (separators are
    # re-attached, never dropped), so the cursor is exact without any string
    # searching.
    spans: List[tuple] = []
    pos = 0
    group_start = 0
    group_len = 0
    for piece in pieces:
        piece_start = pos
        pos += len(piece)
        if group_len and group_len + len(piece) > chunk_size:
            spans.append((group_start, group_start + group_len))
            group_start = piece_start
            group_len = 0
        group_len += len(piece)
    if group_len:
        spans.append((group_start, group_start + group_len))

    chunks: List[dict] = []
    for i, (start, end) in enumerate(spans):
        char_start = max(0, start - overlap) if i > 0 else start
        chunks.append({
            "chunk_id": i,
            "text": text[char_start:end],
            "char_start": char_start,
            "chunk_size_setting": chunk_size,
        })
    return chunks
