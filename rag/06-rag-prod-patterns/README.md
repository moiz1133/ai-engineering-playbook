# RAG Production Patterns

## What This Is

This project is a minimal, focused demonstration of three production RAG patterns: streaming responses, prompt versioning, and shadow-mode retrieval. It intentionally ignores everything else a "real" RAG product might need — the goal is to show each pattern clearly, in isolation, on top of a small but genuine FastAPI + ChromaDB + OpenAI stack. Every result quoted in this README came from actually running the code against the real OpenAI API and a real local ChromaDB collection, not from hand-written examples.

## The Three Patterns

**Streaming Response.** The `/query` endpoint streams the generated answer token-by-token instead of waiting for the full completion and returning it in one block. In production this matters for perceived latency: a user watching text appear within a few hundred milliseconds feels the system responding immediately, even if the *total* generation time is the same five seconds it would have taken to return the answer all at once. It also lets a client abandon a request early (e.g., the user navigates away) without paying for tokens that were never read. This project implements it with FastAPI's `StreamingResponse` wrapping an async generator (`src/streaming/generator.py`) around the OpenAI SDK's native `stream=True` chat completion.

**Prompt Versioning.** Prompts live as plain `.txt` files under `src/prompts/versions/`, loaded at startup by `PromptRegistry` and selectable per-request via a `prompt_version` field. In production, prompts change constantly — a tweak to reduce hallucination, a new instruction to cite sources, a rewording that improves tone — and treating them as hardcoded strings buried in application code makes it impossible to A/B test changes, roll back a regression, or know which prompt produced a given historical answer. Versioning prompts as named, independently loadable files turns "which prompt generated this?" into a one-line lookup instead of a git-archaeology exercise.

**Shadow Mode.** Every `/query` request runs two retrieval strategies: `baseline` (top-5 plain vector search), which the user's answer is actually built from, and `shadow` (top-10 candidates re-scored by chunk length, then top-5 kept), which runs in the background and is logged to `logs/shadow_comparisons.jsonl` — the user never sees it. This is how production teams safely evaluate a new retrieval strategy: run it for real, on real traffic, side-by-side with the strategy currently serving users, and compare the two offline before ever routing a single real answer through the new path. It turns "will this new retrieval approach actually help?" from a guess into a question you can answer with real comparison data, with zero risk to what users see.

## Setup

```bash
git clone <this-repo-url>
cd rag-production-patterns
pip install -r requirements.txt
cp .env.example .env   # then fill in your OPENAI_API_KEY
python -m src.ingest
```

## Running

Start the server:

```bash
uvicorn src.main:app --reload
```

Query with the default prompt version (v1):

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does HNSW achieve fast approximate nearest neighbor search?"}'
```

Real output from this exact command:

```
HNSW achieves fast approximate nearest neighbor search by building a multi-layer graph where
each node represents a vector and edges connect "close" vectors. The search process starts at
a fixed entry point in the topmost layer and uses greedy descent to move towards the query
vector at each layer... Overall, HNSW can achieve approximately O(log n) search time and
typically returns 95-99% of the true nearest neighbors...
```

Query with the v2 prompt (citations + confidence level):

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the tradeoff between chunk overlap and storage cost?", "prompt_version": "v2"}'
```

Real output from this exact command:

```
The tradeoff between chunk overlap and storage cost involves duplicated storage and embedding
calls due to overlapping text in consecutive chunks. For example, a 50-token overlap on
300-token chunks means that approximately 17% of all embedded content is redundant across the
corpus... [chunking_overlap_tradeoffs_0]

However, while overlap incurs these costs, it serves to mitigate the boundary-splitting
problem... [chunking_overlap_tradeoffs_1]

What is missing from the context is specific quantitative data or examples demonstrating how
these costs scale with increased corpus size...

Confidence: medium
```

Health check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","prompt_versions_loaded":["v1","v2"]}
```

## Reading the Shadow Log

Each line in `logs/shadow_comparisons.jsonl` is one request's baseline-vs-shadow comparison: the query, both strategies' retrieved `chunk_id` lists, both latencies, and an `overlap` count (how many chunk IDs both strategies agreed on). To analyze retrieval strategy differences offline, load the file with one JSON object per line (e.g., `pandas.read_json(path, lines=True)` or a simple loop with `json.loads`) and look at the distribution of `overlap` — a low average overlap across many real queries means the two strategies are making meaningfully different retrieval decisions, which is the signal you'd investigate before ever considering promoting a shadow strategy to baseline. Two real lines from this project's own test run:

```json
{"timestamp": "2026-07-23T13:51:17.881651+00:00", "query": "How does HNSW achieve fast approximate nearest neighbor search?", "baseline_chunk_ids": ["hnsw_overview_1", "hnsw_overview_0", "hnsw_search_algorithm_0", "vector_db_indexing_algorithms_1", "hnsw_search_algorithm_1"], "shadow_chunk_ids": ["vector_db_scaling_0", "vector_db_indexing_algorithms_0", "hnsw_parameters_0", "hnsw_overview_0", "hnsw_construction_0"], "baseline_latency_ms": 5309.97, "shadow_latency_ms": 349.88, "overlap": 1}
{"timestamp": "2026-07-23T13:51:27.246161+00:00", "query": "What is the tradeoff between chunk overlap and storage cost?", "baseline_chunk_ids": ["chunking_overlap_tradeoffs_0", "chunking_overlap_tradeoffs_1", "chunking_fixed_size_1", "chunking_strategies_overview_0", "chunking_strategies_overview_1"], "shadow_chunk_ids": ["chunking_fixed_size_0", "chunking_strategies_overview_0", "embeddings_dimensionality_0", "chunking_overlap_tradeoffs_0", "vector_db_scaling_1"], "baseline_latency_ms": 347.48, "shadow_latency_ms": 342.88, "overlap": 2}
```

The first query shows only 1-of-5 overlap and a much higher baseline latency — likely a cold-start embedding/HTTP round trip rather than a real strategy difference, which is exactly the kind of noise a shadow log needs enough volume to average out before drawing conclusions.

## What This Deliberately Does NOT Include

- **Authentication** — no API keys, no user accounts, no rate limiting. Adding auth is a well-understood, separate concern that would only add noise to a demo about retrieval and streaming patterns.
- **Caching** — no semantic or exact-match response cache. Caching interacts with streaming and shadow mode in ways worth their own dedicated project (see this repo's `04-semantic-caching`), not bolted onto this one.
- **Monitoring / metrics / tracing** — no Prometheus, no Langfuse, no dashboards. Observability is valuable but orthogonal to demonstrating the three patterns themselves.
- **Reranking** — baseline and shadow both do plain vector search (with a length-based re-score for shadow only as a way to produce a *different* strategy, not a `production reranking model`). Reranking is its own pattern, covered in `03-retrieval-evaluation`.
- **Multiple embedding models** — one embedding model (`text-embedding-3-small`), one generation model (`gpt-4o-mini`), fixed in `config.py`. Model selection and routing is its own concern, covered in `04-semantic-caching`'s model tier router.
- **A frontend** — this is an API meant to be tested with `curl`. A UI would add real engineering surface area without teaching anything new about the three patterns above.

Every omission above is a deliberate scope boundary, not an oversight — the value of this project is in showing each pattern clearly, and every one of these additions would dilute that focus.
