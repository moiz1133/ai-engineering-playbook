"""
Chunking Strategies Benchmark
==============================

Implements four chunking strategies (fixed-size, recursive, semantic,
document-aware), runs all of them against the same PDF corpus, and reports
build time, chunk shape, sentence-boundary cleanliness, retrieval recall@3,
and indexing token cost in a single comparison table.

Run:
    python benchmark.py
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Repo-root .env (gitignored) holds OPENAI_API_KEY - load it here so running
# this script from within chunking-benchmark/ still picks it up.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))

EMBEDDING_MODEL = "text-embedding-3-large"
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------
def load_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF, one page at a time.

    Trailing whitespace is stripped from every line so that a line containing
    only spacing (how a real paragraph gap usually survives PDF extraction)
    collapses to a truly empty line - that's what lets the "\\n\\n" separator
    in SEPARATORS actually match paragraph breaks instead of every separator
    after it silently no-oping.
    """
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    raw = "\n\n".join(pages)
    lines = [line.rstrip() for line in raw.split("\n")]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# 1. Fixed-size chunking
# --------------------------------------------------------------------------
def chunk_text_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """Slide a fixed-width window over the raw character stream.

    Content-blind by design: this is the baseline every other strategy is
    measured against, precisely because it doesn't try to respect sentences,
    paragraphs, or sections.
    """
    stride = chunk_size - overlap
    n = len(text)
    chunks: List[dict] = []
    start = 0
    chunk_id = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append({
            "chunk_id": chunk_id,
            "text": text[start:end],
            "char_start": start,
            "strategy": "fixed",
        })
        if end == n:
            break
        chunk_id += 1
        start += stride
    return chunks


# --------------------------------------------------------------------------
# 2. Recursive chunking
# --------------------------------------------------------------------------
def _split_on_separator(text: str, separator: str) -> List[str]:
    """Split on `separator`, re-attaching it to every piece but the last so
    concatenating the result reproduces `text` exactly - that exactness is
    what lets char_start tracking later use a running counter instead of
    re-searching the source text.
    """
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _split_into_pieces(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively break `text` into pieces that each fit within chunk_size,
    preferring the coarsest separator (paragraph) that actually occurs in the
    text and only falling back to finer ones - line, sentence, word, then a
    hard character cut - for whichever piece still doesn't fit.
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


def _merge_pieces_with_spans(pieces: List[str], chunk_size: int) -> List[Tuple[int, int]]:
    """Greedily merge consecutive atomic pieces into groups <= chunk_size,
    returning each group's (start, end) offset into the original text.

    Pieces from `_split_into_pieces` are verbatim, in-order substrings of the
    source text (separators are always re-attached, never dropped), so a
    running character cursor is enough to recover exact offsets - no string
    searching needed.
    """
    spans: List[Tuple[int, int]] = []
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
    return spans


def chunk_text_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """Split on natural boundaries first (paragraph > line > sentence > word),
    falling back to a hard character cut only when nothing else fits, then
    apply the same trailing-overlap behavior as the fixed-size strategy.
    """
    pieces = _split_into_pieces(text, chunk_size, SEPARATORS)
    spans = _merge_pieces_with_spans(pieces, chunk_size)

    chunks: List[dict] = []
    for i, (start, end) in enumerate(spans):
        char_start = max(0, start - overlap) if i > 0 else start
        chunks.append({
            "chunk_id": i,
            "text": text[char_start:end],
            "char_start": char_start,
            "strategy": "recursive",
        })
    return chunks


# --------------------------------------------------------------------------
# Shared embedding / similarity helpers
# --------------------------------------------------------------------------
def _embed_texts(client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL,
                  batch_size: int = 20) -> np.ndarray:
    """Embed `texts` in batches of `batch_size` to stay well under per-request
    rate limits, returning a single (len(texts), dim) float32 array.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return np.array(vectors, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _cosine_sim_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between every query row and every corpus
    row, via normalized dot products - no sklearn.
    """
    q_norm = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    c_norm = corpus / np.clip(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12, None)
    return q_norm @ c_norm.T


# --------------------------------------------------------------------------
# 3. Semantic chunking
# --------------------------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    """Split into sentences on ". ", re-adding the period that the split
    consumed (the last sentence keeps whatever punctuation it already had).
    """
    raw = text.replace("\n", " ").split(". ")
    sentences: List[str] = []
    for i, piece in enumerate(raw):
        piece = piece.strip()
        if not piece:
            continue
        if i < len(raw) - 1:
            piece += "."
        sentences.append(piece)
    return sentences


def chunk_text_semantic(text: str, openai_client: OpenAI, threshold: float = 0.85) -> List[dict]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [{"chunk_id": 0, "text": sentences[0], "strategy": "semantic", "sentence_count": 1}]

    embeddings = _embed_texts(openai_client, sentences)

    # WHAT: boundary = topic shift detected by embedding similarity drop
    # WHY: chunks align with topic changes, not arbitrary character counts
    boundary_after = set()
    for i in range(len(sentences) - 1):
        if _cosine_sim(embeddings[i], embeddings[i + 1]) < threshold:
            boundary_after.add(i)

    chunks: List[dict] = []
    current: List[str] = []
    chunk_id = 0
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        if i in boundary_after or i == len(sentences) - 1:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current),
                "strategy": "semantic",
                "sentence_count": len(current),
            })
            chunk_id += 1
            current = []
    return chunks


# --------------------------------------------------------------------------
# 4. Document-aware chunking
# --------------------------------------------------------------------------
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _MARKDOWN_HEADING_RE.match(stripped):
        return True
    # ALL-CAPS heuristic: short, has at least one letter, no lowercase letters.
    return len(stripped) <= 80 and any(c.isalpha() for c in stripped) and stripped == stripped.upper()


def _heading_text(line: str) -> str:
    stripped = line.strip()
    return _MARKDOWN_HEADING_RE.sub("", stripped).strip() or stripped


def _split_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """Break `text` into (header, body) pairs at each detected structural
    marker. Content before the first heading gets header=None.
    """
    sections: List[Tuple[Optional[str], str]] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []

    for line in text.split("\n"):
        if _is_heading(line):
            sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = _heading_text(line)
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_header, "\n".join(current_lines).strip()))
    return sections


def _merge_text_pieces(pieces: List[str], max_size: int) -> List[str]:
    """Greedily merge atomic pieces into <= max_size chunks, no overlap -
    a section heading already gives every sub-chunk its context, so unlike
    fixed/recursive chunking there's no need to carry trailing text forward.
    """
    merged: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_size:
            merged.append(current)
            current = piece
        else:
            current += piece
    if current:
        merged.append(current)
    return merged


def chunk_text_document(text: str, max_chunk_size: int = 1000) -> List[dict]:
    sections = _split_into_sections(text)
    chunks: List[dict] = []
    chunk_id = 0

    for header, body in sections:
        if not body:
            continue  # edge case: empty section (e.g. back-to-back headings)

        if len(body) <= max_chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "text": body,
                "strategy": "document",
                "section_header": header,
            })
            chunk_id += 1
            continue

        pieces = _split_into_pieces(body, max_chunk_size, SEPARATORS)
        for sub_text in _merge_text_pieces(pieces, max_chunk_size):
            if not sub_text:
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "text": sub_text,
                "strategy": "document",
                "section_header": header,
            })
            chunk_id += 1

    return chunks


# --------------------------------------------------------------------------
# Benchmark metrics
# --------------------------------------------------------------------------
def _avg_chunk_size(chunks: List[dict]) -> float:
    return float(np.mean([len(c["text"]) for c in chunks])) if chunks else 0.0


def _boundary_penalty_pct(chunks: List[dict]) -> float:
    """% of chunks whose text starts with a lowercase letter - a cheap proxy
    for "this chunk was cut off mid-sentence" without doing real NLP.
    """
    if not chunks:
        return 0.0
    mid_sentence = sum(1 for c in chunks if c["text"] and c["text"][0].islower())
    return mid_sentence / len(chunks) * 100


def _index_tokens(chunks: List[dict]) -> int:
    return sum(len(c["text"].split()) for c in chunks)


def _locate_span(source: str, chunk_text: str, search_from: int) -> Tuple[int, int]:
    """Best-effort (start, end) offset of `chunk_text` inside `source`.

    Fixed and recursive chunks already carry an exact char_start; semantic and
    document-aware chunks don't (their text is reassembled from sentences or
    merged pieces), so this locates them by searching for their first ~40
    characters - just precise enough to measure span overlap for recall@3,
    not meant to be an exact offset.
    """
    needle = chunk_text.strip()[:40]
    if not needle:
        return (search_from, search_from)
    idx = source.find(needle, search_from)
    if idx == -1:
        idx = source.find(needle)
    if idx == -1:
        return (search_from, search_from + len(chunk_text))
    return (idx, idx + len(chunk_text))


def _chunk_spans(source: str, chunks: List[dict]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for c in chunks:
        if "char_start" in c:
            start = c["char_start"]
        else:
            start, _ = _locate_span(source, c["text"], cursor)
        end = start + len(c["text"])
        spans.append((start, end))
        cursor = max(cursor, start)
    return spans


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int], min_overlap_chars: int = 20) -> bool:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return (end - start) >= min_overlap_chars


def _recall_at_3(source: str, fixed_chunks: List[dict], fixed_embeddings: np.ndarray,
                  strategy_chunks: List[dict], strategy_embeddings: np.ndarray,
                  query_embeddings: np.ndarray) -> float:
    """Recall@3 against fixed-size as ground truth.

    Different strategies produce different chunk boundaries over the same
    source text, so chunk *indices* aren't comparable across strategies - two
    strategies can both retrieve "the right passage" using completely
    different chunk objects. Instead, for each query, take fixed-size's top-3
    chunks as the ground-truth source-text spans, take this strategy's top-3
    chunks, and score the fraction of the strategy's picks whose span
    actually overlaps one of those ground-truth spans.
    """
    k = 3
    if len(strategy_chunks) == 0 or len(fixed_chunks) == 0:
        return 0.0

    fixed_spans = _chunk_spans(source, fixed_chunks)
    strategy_spans = _chunk_spans(source, strategy_chunks)

    fixed_sims = _cosine_sim_matrix(query_embeddings, fixed_embeddings)
    strategy_sims = _cosine_sim_matrix(query_embeddings, strategy_embeddings)

    k_fixed = min(k, len(fixed_chunks))
    k_strategy = min(k, len(strategy_chunks))

    scores = []
    for qi in range(query_embeddings.shape[0]):
        ground_truth = [fixed_spans[i] for i in np.argsort(-fixed_sims[qi])[:k_fixed]]
        retrieved = np.argsort(-strategy_sims[qi])[:k_strategy]
        hits = sum(1 for si in retrieved if any(_spans_overlap(strategy_spans[si], gt) for gt in ground_truth))
        scores.append(hits / k)  # divide by k (not k_strategy) so a short chunk list is penalized, not rewarded
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# Benchmark harness
# --------------------------------------------------------------------------
def run_benchmark(pdf_path: str, openai_client: OpenAI, test_queries: List[str]) -> Dict[str, Dict[str, float]]:
    text = load_pdf(pdf_path)

    builders = {
        "fixed": lambda: chunk_text_fixed(text),
        "recursive": lambda: chunk_text_recursive(text),
        "semantic": lambda: chunk_text_semantic(text, openai_client),
        "document": lambda: chunk_text_document(text),
    }

    chunks_by_strategy: Dict[str, List[dict]] = {}
    build_time_s: Dict[str, float] = {}
    for name, build in builders.items():
        t0 = time.perf_counter()
        chunks_by_strategy[name] = build()
        build_time_s[name] = time.perf_counter() - t0

    # Embed every strategy's chunks and the test queries once up front so the
    # recall@3 comparison below can reuse them instead of re-embedding per pair.
    embeddings_by_strategy = {
        name: _embed_texts(openai_client, [c["text"] for c in chunks])
        for name, chunks in chunks_by_strategy.items()
    }
    query_embeddings = _embed_texts(openai_client, test_queries)

    fixed_chunks = chunks_by_strategy["fixed"]
    fixed_embeddings = embeddings_by_strategy["fixed"]

    results: Dict[str, Dict[str, float]] = {}
    for name, chunks in chunks_by_strategy.items():
        recall = _recall_at_3(
            text, fixed_chunks, fixed_embeddings,
            chunks, embeddings_by_strategy[name], query_embeddings,
        )
        results[name] = {
            "build_time_s": build_time_s[name],
            "chunk_count": len(chunks),
            "avg_chunk_size": _avg_chunk_size(chunks),
            "boundary_penalty_pct": _boundary_penalty_pct(chunks),
            "recall_at_3": recall,
            "index_tokens": _index_tokens(chunks),
        }
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
_STRATEGY_LABELS = [
    ("fixed", "Fixed-size"),
    ("recursive", "Recursive"),
    ("semantic", "Semantic"),
    ("document", "Doc-aware"),
]
_COLUMN_WIDTHS = [12, 6, 8, 8, 9, 8, 7]


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    headers = ["Strategy", "Chunks", "Avg size", "Build(s)", "Boundary%", "Recall@3", "Tokens"]
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, _COLUMN_WIDTHS))
    print(header_row)
    print("-|-".join("-" * w for w in _COLUMN_WIDTHS))

    for key, label in _STRATEGY_LABELS:
        r = results[key]
        cells = [
            label,
            str(r["chunk_count"]),
            str(round(r["avg_chunk_size"])),
            f'{r["build_time_s"]:.2f}s',
            f'{r["boundary_penalty_pct"]:.0f}%',
            f'{r["recall_at_3"]:.2f}',
            f'{int(r["index_tokens"]):,}',
        ]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, _COLUMN_WIDTHS)))


def recommendation(results: Dict[str, Dict[str, float]]) -> str:
    """Pick the best recall/cost tradeoff among the non-baseline strategies.

    Fixed-size is excluded: it's the ground truth recall@3 is measured
    against, so its recall is trivially 1.00 and isn't a meaningful "win".
    """
    candidates = {name: r for name, r in results.items() if name != "fixed"}
    best_name = max(candidates, key=lambda n: (candidates[n]["recall_at_3"], -candidates[n]["index_tokens"]))
    best = candidates[best_name]
    label = dict(_STRATEGY_LABELS)[best_name]

    return (
        f"For this corpus, {label} gives the best recall/cost tradeoff because it reaches "
        f"recall@3 of {best['recall_at_3']:.2f} using {int(best['index_tokens']):,} index tokens "
        f"and a {best['build_time_s']:.2f}s build, without the per-sentence embedding overhead "
        f"semantic chunking pays at index time."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "rag_test_document.pdf")

    openai_client = OpenAI()  # reads OPENAI_API_KEY from the environment

    test_queries = [
        "How does HNSW improve vector search performance?",
        "What embedding model should I use for semantic search?",
        "What is the difference between fixed-size and semantic chunking?",
        "How do you measure retrieval quality with recall and precision?",
        "What are the cost considerations when deploying a RAG system in production?",
    ]

    print(f"Loading corpus: {pdf_path}")
    results = run_benchmark(pdf_path, openai_client, test_queries)

    print()
    print_results_table(results)
    print()
    print(recommendation(results))


if __name__ == "__main__":
    main()


# CHUNKING STRATEGIES — INTERVIEW SUMMARY
# Fixed-size: O(1) build, content-blind, ~15-20% boundary cuts
# Recursive: tries natural separators first — best general default
# Semantic: topic-aligned chunks, ~8x slower to build, highest precision
# Document-aware: structure-first, requires consistent document formatting
# Rule of thumb: start recursive, upgrade to semantic if precision < 80%
