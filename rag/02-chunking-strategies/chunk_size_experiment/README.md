# Chunk Size Experiment

A standalone, controlled experiment isolating the effect of **chunk size alone** on RAG
retrieval quality and cost. Chunking method (recursive separator splitting) and overlap
percentage (10% of `chunk_size`) are held fixed; only `chunk_size` varies, across three
values: **150, 300, and 600 characters**. This is a separate project from
[`../chunking-benchmark/`](../chunking-benchmark/) — no shared code, no shared corpus.

Code: [`run_experiment.py`](run_experiment.py)

## Why hold everything else constant

Comparing chunk sizes is only meaningful if chunk size is the *only* thing that changed.
If overlap were held at a fixed character count instead of a fixed percentage, a 150-char
chunk with a 50-char overlap would have a third of its content duplicated from the
previous chunk, while a 600-char chunk with the same 50-char overlap would barely overlap
at all — that's not a fair comparison, it's confounding two variables at once. Scaling
overlap as 10% of `chunk_size` (15 / 30 / 60 chars respectively) keeps the *relative*
context-carryover identical across all three sizes, so any difference in the results can
be attributed to chunk size itself.

## Project structure

```
chunk_size_experiment/
├── corpus/                    # 3 generated .txt documents (~2,000-2,800 words each)
├── corpus_generator.py        # generate_test_corpus() — writes the synthetic corpus
├── chunker.py                 # recursive_chunk() — the one fixed chunking method
├── queries.py                 # TEST_QUERIES — 15 fixed test queries
├── embedder.py                # embed_chunks() — batched OpenAI embedding calls
├── retriever.py                # retrieve() — NumPy cosine-similarity top-k search
├── experiment.py              # run_size_experiment() — the controlled-variable harness
├── run_experiment.py          # entry point: table, interpretation, results.json
├── results.json               # raw output of the last run (see below)
└── requirements.txt           # numpy, openai, python-dotenv
```

## The corpus

`corpus_generator.generate_test_corpus()` writes three synthetic, topically distinct
documents — `machine_learning_fundamentals.txt`, `climate_science_basics.txt`, and
`history_of_the_internet.txt` — each roughly 2,000-2,800 words of real, factual prose
covering a different general-knowledge subject.

```
# WHAT: synthetic corpus so the experiment is fully self-contained and reproducible
# WHY: using real PDFs would make this project dependent on external files
```

`run_experiment.py` regenerates these files on every run (overwriting them with the same
deterministic content), then loads and concatenates all three into one text blob before
chunking — so the experiment always starts from a known, version-controllable corpus
rather than depending on documents that might not exist on another machine.

## The chunker — `recursive_chunk(text, chunk_size, overlap_pct=0.10)`

Tries separators `["\n\n", "\n", ". ", " "]` in order — paragraph, then line, then
sentence, then word — using whichever separator actually occurs in a piece of text and
keeps the resulting pieces under `chunk_size`; a hard character cut is the implicit last
resort once every separator has been tried and a piece still doesn't fit. Adjacent pieces
are then greedily merged back into chunks up to `chunk_size`, and every chunk after the
first pulls its `char_start` back by `overlap` characters so consecutive chunks share
some context — identical logic to the recursive chunker in the sibling
`chunking-benchmark` project, just reimplemented from scratch here per the "no dependency
on prior code" requirement.

```
# WHAT: overlap scales with chunk_size — 10% of 150 ≈ 15 chars, 10% of 600 ≈ 60 chars
# WHY: keeping overlap PERCENTAGE constant (not absolute) is what makes 150 vs 600 a fair comparison
```

## The 15 test queries — `queries.py`

Written *after* reading the generated corpus, so every `ground_truth_keywords` list is
verified to actually appear in the source text (see the keyword-presence check run during
development — every one of the 15 keyword sets was confirmed present in the concatenated
corpus before being committed to `queries.py`):

- **5 factual** — single-fact lookups needing minimal context (e.g. "When was the IPCC
  established?" → keywords `["IPCC", "1988"]`).
- **5 multi-concept** — need a fuller passage that ties two or more ideas together (e.g.
  "How do thermal expansion and melting ice both contribute to sea level rise?" → keywords
  `["thermal expansion", "melting", "sea level"]`).
- **5 rephrased** — deliberately worded differently from the source text to test semantic
  match rather than keyword match (e.g. "Why does Earth's atmosphere trap heat and keep
  the planet's surface warm?" → keywords `["greenhouse effect", "infrared"]`, neither of
  which appears in the question itself).

## Metrics — `run_size_experiment()`

For each `chunk_size`, the harness chunks the full corpus, embeds every chunk with
`text-embedding-3-large`, and runs all 15 queries against the resulting index:

- **chunk_count** / **embedding_tokens** (`sum(len(text))` across chunks — a character-
  count proxy for embedding API cost) / **build_time_s** (chunking + embedding, timed
  together with `time.perf_counter()`).
- **hit_rate**: fraction of the 15 queries where *any* of the top-3 retrieved chunks
  contains *every* one of that query's `ground_truth_keywords` (case-insensitive substring
  match) — an automated relevance proxy that needs no human judge and is fully
  reproducible.
- **hit_rate_by_type**: the same hit check, broken down across factual / multi-concept /
  rephrased.
- **avg_top1_similarity**: mean cosine similarity of the top-1 result across all 15
  queries — a proxy for retrieval confidence.
- **split_context_rate**: for multi-concept queries only, whether the keywords were found
  together in a single retrieved chunk (`"concentrated"`) or only when combined across
  more than one of the top-3 chunks (`"split"`) — see below for why this needs a third,
  unlabeled outcome.
- **avg_query_latency_ms**: mean wall-clock time of the `retrieve()` call across all 15
  queries.

### Why "split" needs a third, implicit outcome

A multi-concept query's keywords can end up in exactly one of three states relative to the
top-3 retrieved chunks: all together in one chunk (**concentrated**), spread across two or
more of the retrieved chunks but still fully present somewhere in the top-3
(**split** — literal fragmentation, the retriever found the right region but the
chunk boundaries cut it apart), or simply absent from the top-3 entirely (a plain
retrieval **miss**, not a fragmentation problem). `_context_completeness()` in
`experiment.py` checks for "concentrated" first, then checks whether the *union* of
keywords found across all three chunks covers the full keyword set before labeling
something "split" — a query that flat-out missed isn't counted as fragmented just because
it also failed to hit.

## Running it

```bash
pip install -r requirements.txt
python run_experiment.py
```

Reads `OPENAI_API_KEY` from a `.env` file at the repo root (gitignored) via
`python-dotenv`, same as the sibling `chunking-benchmark` project.

## Results

Measured end-to-end against real `text-embedding-3-large` embeddings, 15 fixed queries,
3 corpus documents (~6,900 words / ~43,200 characters total):

```
Chunk size | Chunks | Hit rate | Factual | Multi-concept  | Rephrased | Split ctx | Avg sim | Tokens
-----------|--------|----------|---------|----------------|-----------|-----------|---------|---------
150 chars  | 313    | 53.3%    | 100%    | 60%            | 0%        | 40%       | 0.59    | 47,913
300 chars  | 185    | 73.3%    | 100%    | 40%            | 80%       | 60%       | 0.62    | 48,753
600 chars  | 90     | 93.3%    | 100%    | 80%            | 100%      | 20%       | 0.59    | 48,573
```

Build time and per-query latency (from `results.json`, not shown in the printed table):

| Chunk size | Build time (chunk + embed) | Avg query latency |
|---|---|---|
| 150 chars | 15.31s | 468ms |
| 300 chars | 9.20s | 491ms |
| 600 chars | 4.70s | 489ms |

Full raw output, including every field per chunk size, is in
[`results.json`](results.json).

### Interpretation (generated by `build_interpretation()`, reproduced verbatim)

> 150 chars: hit rate 53.3% (factual 100%, multi-concept 60%, rephrased 0%), split-context
> rate 40%, avg top-1 similarity 0.59, 47,913 embedding tokens across 313 chunks. 300
> chars: hit rate 73.3% (factual 100%, multi-concept 40%, rephrased 80%), split-context
> rate 60%, avg top-1 similarity 0.62, 48,753 embedding tokens across 185 chunks. 600
> chars: hit rate 93.3% (factual 100%, multi-concept 80%, rephrased 100%), split-context
> rate 20%, avg top-1 similarity 0.59, 48,573 embedding tokens across 90 chunks.
> Split-context rate was highest at 60% with 300-char chunks and lowest at 20% with
> 600-char chunks, a non-monotonic pattern across the chunk sizes tested in this run.
> Average top-1 similarity peaked at 0.62 with 300-char chunks and was lowest at 0.59 with
> 600-char chunks - consistent with larger chunks covering more than one topic and
> diluting the embedding match for any single query even when the answer is present. 600
> chars gave the best overall hit rate (93.3%) in this experiment.

### Reading these numbers honestly

This run's actual pattern is *not* the textbook "small chunks fragment, big chunks
dilute, middle wins" story — on this corpus, **600 chars won outright on hit rate**
(93.3%, vs. 73.3% at 300 and 53.3% at 150), and the relationship wasn't monotonic on
every metric:

- **Rephrased-query hit rate climbs steeply with chunk size** — 0% at 150 chars, 80% at
  300, 100% at 600. The five rephrased queries deliberately avoid the source text's exact
  wording, so they depend on the embedding capturing enough surrounding context to make
  the semantic link; a 150-char chunk (≈ 1-2 sentences) often just doesn't contain enough
  context to do that, regardless of how good the embedding model is.
- **Factual hit rate is 100% at every size** — single-fact lookups with literal,
  short ground-truth keywords are easy to retrieve correctly no matter how the corpus is
  chunked, since the chunk only needs to contain one short, distinctive sentence.
  This metric alone wouldn't have shown any difference between chunk sizes.
- **Split-context rate is non-monotonic** (40% → 60% → 20%): it actually peaked at 300
  chars, not at the smallest size. With 5 multi-concept queries, each percentage point
  swing is one query flipping outcome, so this metric is the noisiest one in the table —
  it would need more queries (or repeated runs) to separate a real effect from sampling
  noise at this sample size.
- **Avg top-1 similarity also peaked at 300, not at the smallest size** (0.59 → 0.62 →
  0.59) — the "smaller chunks = higher similarity" intuition didn't hold here; 150-char
  chunks scored about the same as 600-char chunks despite being four times smaller.
- **Embedding token cost is nearly flat across all three sizes** (~47,900-48,750) because
  it's driven by total corpus character count and overlap duplication, not chunk count —
  more chunks at smaller sizes doesn't mean meaningfully more total characters embedded,
  since overlap percentage (not overlap chars) was held constant.
- **Build time scales clearly with chunk count**: 15.3s at 150 chars (313 chunks) down to
  4.7s at 600 chars (90 chunks) — more chunks means more embedding API calls in total,
  even at the same `batch_size=20`.

The takeaway isn't "always use 600-char chunks" — it's that **the effect of chunk size is
corpus- and query-dependent enough that you have to measure it on your own data**, which
is exactly the point of running this experiment instead of assuming a textbook default.
Re-run `python run_experiment.py` after editing the corpus or `TEST_QUERIES` and these
numbers, and the interpretation paragraph above, will update automatically.

## How to explain this in an interview

```
# CHUNK SIZE EXPERIMENT — HOW TO EXPLAIN IN AN INTERVIEW
# 1. Method and overlap percentage held constant — chunk_size is the ONLY independent variable
# 2. Keyword hit rate is an automated proxy for relevance — no LLM judge needed, fully reproducible
# 3. Split-context rate measures the boundary problem directly: does a multi-part answer
#    get fragmented across chunks the retriever then can't reassemble
# 4. Smaller chunks → higher precision per chunk but more fragmentation risk for complex answers
# 5. Larger chunks → less fragmentation but diluted embeddings (multiple topics per chunk)
#    pull the cosine similarity score down even when the chunk DOES contain the answer
# 6. The "best" size is empirical, not theoretical — that's why this experiment exists
```
