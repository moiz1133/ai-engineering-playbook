"""End-to-end runner — regenerates corpus index and runs the full benchmark."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chunker import build_parent_child_chunks
from store import build_child_index
from baseline import build_flat_index
from experiment import run_experiment, print_results_table
from queries import QUERIES

# WHAT: self-contained corpus — no external files needed
CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_FILES = [
    CORPUS_DIR / "ml_fundamentals.txt",
    CORPUS_DIR / "climate_science.txt",
    CORPUS_DIR / "internet_history.txt",
]

PARENT_SIZE = 600
CHILD_SIZE = 150
CHILD_OVERLAP = 15
TOP_K = 3


def main() -> None:
    # Repo-root .env (gitignored) holds OPENAI_API_KEY.
    load_dotenv(Path(__file__).parent / ".." / ".." / ".." / ".env")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Set it in the repo-root .env or as an environment variable.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=5)

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    print("[runner] Loading corpus...")
    corpus_texts: list[str] = []
    for path in CORPUS_FILES:
        if not path.exists():
            print(f"ERROR: corpus file not found: {path}", file=sys.stderr)
            sys.exit(1)
        corpus_texts.append(path.read_text(encoding="utf-8"))
    print(f"[runner] Loaded {len(corpus_texts)} documents.")

    # ── 2. Build parent-child chunks across all documents ────────────────────
    print("[runner] Building parent-child chunks...")
    all_child_chunks: list[dict] = []
    # offset child_id and parent_id per document to keep them globally unique
    child_id_offset = 0
    parent_id_offset = 0
    for text in corpus_texts:
        chunks = build_parent_child_chunks(
            text,
            parent_size=PARENT_SIZE,
            child_size=CHILD_SIZE,
            child_overlap=CHILD_OVERLAP,
        )
        for c in chunks:
            c["child_id"] += child_id_offset
            c["parent_id"] += parent_id_offset
        if chunks:
            child_id_offset = max(c["child_id"] for c in chunks) + 1
            parent_id_offset = max(c["parent_id"] for c in chunks) + 1
        all_child_chunks.extend(chunks)

    print(
        f"[runner] Total child chunks: {len(all_child_chunks)}, "
        f"unique parents: {len({c['parent_id'] for c in all_child_chunks})}"
    )

    # ── 3. Build vector stores ────────────────────────────────────────────────
    print("[runner] Embedding child chunks (parent-child index)...")
    pc_collection = build_child_index(all_child_chunks, client)

    print("[runner] Embedding flat 300-char chunks (baseline index)...")
    flat_collection = build_flat_index(corpus_texts, client)

    # ── 4. Run experiment ─────────────────────────────────────────────────────
    comparison = run_experiment(
        corpus_paths=[str(p) for p in CORPUS_FILES],
        openai_client=client,
        queries=QUERIES,
        pc_collection=pc_collection,
        flat_collection=flat_collection,
        top_k=TOP_K,
    )

    # ── 5. Print results ──────────────────────────────────────────────────────
    print_results_table(comparison)


if __name__ == "__main__":
    main()
