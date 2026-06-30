# Chunking Strategies Benchmark

A benchmark comparing four ways to split a document into retrievable chunks for RAG:
fixed-size, recursive, semantic, and document-aware. All four run against the same
PDF corpus so their chunk shape, build cost, and retrieval quality can be compared
head-to-head.

Code: [`chunking-benchmark/benchmark.py`](chunking-benchmark/benchmark.py)

## Files

| File | Purpose |
|---|---|
| `chunking-benchmark/benchmark.py` | All four chunkers, the benchmark harness, and the results table |
| `chunking-benchmark/generate_test_pdf.py` | One-off script that builds `rag_test_document.pdf` |
| `chunking-benchmark/rag_test_document.pdf` | The test corpus: six sections covering RAG, vector DBs, embeddings, chunking, retrieval evaluation, and deployment cost |
| `chunking-benchmark/requirements.txt` | `numpy`, `openai`, `pypdf`, `fpdf2`, `python-dotenv` |

## How the corpus is built

`generate_test_pdf.py` writes six topically distinct sections, alternating between
markdown-style headings (`#`, `##`, `###`) and ALL-CAPS plain-text headings, so that
document-aware chunking has to detect both heading styles, and semantic chunking has
real topic shifts to detect — a single rambling topic would make every strategy
produce near-identical chunks and the benchmark wouldn't show any contrast.

One detail worth knowing: PDF text extraction normally loses the blank line between
paragraphs — `pdf.ln()` (vertical spacing) draws no glyphs, so there's nothing for
`pypdf` to extract there. To keep paragraph breaks intact, the generator writes a
line containing a single space character between paragraphs instead of just spacing.
`load_pdf()` then `rstrip()`s every line, turning that whitespace-only line into a
genuinely empty one, which is what lets `"\n\n"` work as a real separator instead of
silently never matching.

## `load_pdf(pdf_path) -> str`

Reads every page with `pypdf.PdfReader`, joins pages with `"\n\n"`, and strips
trailing whitespace from each line (see above for why). Returns one big string —
all four chunkers operate on plain text, not PDF objects.

## The four strategies

### 1. Fixed-size — `chunk_text_fixed(text, chunk_size=500, overlap=50)`

The baseline. Slides a fixed-width window over the raw character stream, stepping by
`chunk_size - overlap` each time, with no awareness of words, sentences, or
paragraphs. Each chunk records its exact `char_start` in the source text. This is
deliberately the simplest and fastest strategy — and the one every other strategy's
`recall@3` is measured against (see below).

### 2. Recursive — `chunk_text_recursive(text, chunk_size=500, overlap=50)`

Tries to respect natural text boundaries before falling back to a hard cut. It works
in two passes:

1. **`_split_into_pieces`** recursively breaks the text into pieces that each fit
   under `chunk_size`, trying separators in this order: `"\n\n"` (paragraph) →
   `"\n"` (line) → `". "` (sentence) → `" "` (word) → `""` (character). It tries the
   coarsest separator first; if that separator doesn't even occur in the text, it
   moves to the next, finer one. Any piece still too big after a split recurses into
   the *remaining* separators only — it never retries a coarser separator on a
   smaller piece.
2. **`_merge_pieces_with_spans`** greedily glues those small pieces back together up
   to `chunk_size`, tracking each merged group's `(start, end)` offset with a running
   character counter rather than re-searching the text — `_split_on_separator`
   always re-attaches the separator it split on, so concatenating pieces in order
   reproduces the original text exactly, and offsets follow for free.

`chunk_text_recursive` then applies the same trailing-overlap idea as fixed-size: for
every chunk after the first, it pulls `char_start` back by up to `overlap`
characters so consecutive chunks share some context.

### 3. Semantic — `chunk_text_semantic(text, openai_client, threshold=0.85)`

Boundaries are placed where the *meaning* shifts, not where a character count runs
out:

1. `_split_sentences` splits on `". "` and re-attaches the period that the split
   consumed.
2. Every sentence gets embedded with `text-embedding-3-large` (`_embed_texts`,
   batched 20 sentences per API call to stay under rate limits).
3. Walking consecutive sentence pairs, a chunk boundary is inserted after sentence
   `i` whenever `cosine_similarity(sentence[i], sentence[i+1]) < threshold` — a big
   similarity drop signals a topic change.
4. Sentences between boundaries are joined with `" "` into one chunk, which also
   records `sentence_count`.

This is the only strategy with no fixed size limit — chunk size is whatever the
topic naturally spans — and the only one that pays a per-sentence embedding cost at
*build* time rather than just at index/query time.

### 4. Document-aware — `chunk_text_document(text, max_chunk_size=1000)`

Splits at the document's own structure first, falling back to recursive splitting
only inside oversized sections:

1. `_split_into_sections` walks the text line by line. `_is_heading` flags a line as
   a structural marker if it's a markdown heading (`^#{1,6}\s+`) or an ALL-CAPS line
   (short, has a letter, no lowercase). Everything between two headings becomes one
   `(header, body)` section; content before the first heading gets `header=None`.
2. A section that already fits under `max_chunk_size` becomes a single chunk.
3. A section that's too big gets run through the same `_split_into_pieces` used by
   recursive chunking, then `_merge_text_pieces` glues the pieces back up to
   `max_chunk_size` — with no overlap, since every sub-chunk already inherits its
   `section_header`, so it doesn't need borrowed trailing context the way fixed and
   recursive chunks do.
4. Empty sections (e.g. two headings back-to-back) are skipped — handled by the
   `if not body: continue` guard.

## Benchmark metrics — `run_benchmark(pdf_path, openai_client, test_queries)`

For each strategy, `run_benchmark`:

1. Times the chunking call itself with `time.perf_counter()` → **build_time_s**.
2. Counts chunks → **chunk_count**, and averages `len(chunk["text"])` → **avg_chunk_size**.
3. Computes **boundary_penalty_pct**: the % of chunks whose text starts with a
   lowercase letter — a cheap heuristic for "this chunk was almost certainly cut off
   mid-sentence," since a clean sentence/paragraph start is capitalized.
4. Computes **index_tokens**: `sum(len(chunk["text"].split()))` across all chunks — a
   whitespace-word-count proxy for embedding API cost, not a real tokenizer count.
5. Computes **recall_at_3** (see below).

### Why recall@3 needs span overlap, not chunk-index overlap

Each strategy chunks the *same* source text differently, so chunk index `5` in
fixed-size and chunk index `5` in semantic are unrelated pieces of text — comparing
indices directly would be meaningless. `_recall_at_3` instead compares **source-text
spans**:

- Fixed-size's top-3 chunks (by cosine similarity to the query) become the
  ground-truth spans, using their exact `char_start`.
- For semantic and document-aware chunks, which don't carry `char_start`,
  `_locate_span` finds where each chunk's text actually sits in the source by
  searching for its first ~40 characters — good enough for an overlap check, not
  meant to be exact.
- For each query, a strategy's top-3 retrieved chunk counts as a hit if its span
  overlaps any ground-truth span by at least 20 characters (`_spans_overlap`).
  `recall@3` is hits ÷ 3, averaged across all test queries.

Fixed-size compared against itself trivially scores `1.00` — that's expected, it's
the baseline the metric is built around, not evidence fixed-size "won."

## Running it

```bash
cd chunking-benchmark
pip install -r requirements.txt
python benchmark.py
```

`benchmark.py` loads `OPENAI_API_KEY` from a `.env` file at the repo root via
`python-dotenv` (gitignored — add your own key there). `main()` runs five test
queries spanning the document's six topics, prints the results table, and prints a
plain-English recommendation that picks the best non-baseline strategy by
`(recall@3, -index_tokens)` — fixed-size is excluded from that pick since its
recall@3 is trivially 1.00 by construction.

## Benchmark results

Measured on `rag_test_document.pdf` (the 6-section, ~6,200-character corpus described
above) with `text-embedding-3-large`, 5 test queries spanning all six topics:

```
Strategy     | Chunks | Avg size | Build(s) | Boundary% | Recall@3 | Tokens
-------------|--------|----------|----------|-----------|----------|-------
Fixed-size   | 14     | 492      | 0.00s    | 71%       | 1.00     | 1,070
Recursive    | 17     | 414      | 0.00s    | 76%       | 1.00     | 1,099
Semantic     | 34     | 182      | 9.83s    | 0%        | 0.87     | 969
Doc-aware    | 10     | 598      | 0.00s    | 0%        | 0.80     | 939
```

> For this corpus, Recursive gives the best recall/cost tradeoff because it reaches
> recall@3 of 1.00 using 1,099 index tokens and a 0.00s build, without the
> per-sentence embedding overhead semantic chunking pays at index time.

### Reading these numbers

- **Boundary% is high (71-76%) for fixed and recursive here**, well above the
  ~15-20% you'd expect on a longer corpus. This document's paragraphs run
  500-900 characters — close to `chunk_size=500` itself — so a 500-character window
  frequently lands inside a paragraph rather than at its edge. Boundary% is a
  property of *how a document's natural unit size compares to chunk_size*, not a
  fixed cost of the strategy; recursive only avoids it when chunk_size comfortably
  exceeds the surrounding paragraph/sentence length.
- **Semantic and document-aware both score 0% boundary penalty by construction** —
  semantic chunks always start at a sentence boundary (joined whole sentences) and
  document-aware chunks always start at a section boundary (or a recursive split
  point that itself respects sentence/paragraph edges), so the "starts lowercase"
  heuristic never fires for either.
- **Recursive ties fixed-size at recall@3 = 1.00** mainly because this corpus is
  small enough (six sections, ~6,200 characters) that both strategies' chunks
  densely cover the same handful of topics — the ground-truth comparison is against
  fixed-size itself, so any strategy whose chunks span similar text inevitably scores
  well too.
- **Semantic is ~9.8s to build versus ~0.00s for the others** — the one-embedding-
  call-per-sentence cost from the design doc shows up directly; on a larger corpus
  this gap only grows, since fixed/recursive/document-aware build cost is pure string
  manipulation while semantic's build cost is bounded by the OpenAI API.
- **Doc-aware uses the fewest tokens (939) for the fewest chunks (10)** — by chunking
  along section boundaries it produces the largest, most self-contained chunks here,
  trading some recall (0.80) for the lowest indexing cost.

Re-run `python benchmark.py` after editing `rag_test_document.pdf` or the test
queries — these numbers are corpus-specific, not universal claims about any one
strategy.

## Rule of thumb

- **Fixed-size**: O(1) build, content-blind, routinely cuts mid-sentence.
- **Recursive**: tries natural separators first — best general-purpose default.
- **Semantic**: topic-aligned chunks, pays a per-sentence embedding cost at build
  time, highest topical precision.
- **Document-aware**: structure-first, but only as good as the document's own
  heading consistency.
- Start with recursive; reach for semantic only once measured precision on your own
  corpus says you need it.
