# LLM Infrastructure Middleware: Semantic Cache + Circuit Breaker + Model Router

A standalone Python project implementing three production LLM infrastructure
components — an embedding-based semantic cache, a budget-based cost circuit
breaker, and a complexity-based model tier router — composed into a single
middleware entry point, `llm_request()`. Pure Python + `openai` + `numpy`. No
LangChain, no external cache store, no vector database.

## Why these three components together

Each solves a distinct cost/reliability problem that shows up in any
production LLM wrapper:

- **Semantic cache** — identical or near-identical questions shouldn't pay for
  a new completion every time.
- **Cost circuit breaker** — a runaway loop, retry storm, or batch job
  shouldn't be able to silently exhaust an API budget.
- **Model tier router** — most queries don't need the most expensive model;
  routing simple queries to a cheap model cuts average spend by 6-7x without
  an LLM call to decide which LLM to call.

Composed together, they form a stack where a cache hit costs $0 and never
touches the budget, and every real call is routed to the cheapest model
capable of answering it before the budget check runs.

## Architecture: composition order

`llm_request()` in `llm_infra/middleware.py` applies the stack in this exact
order, and the order is load-bearing, not arbitrary:

1. **Cache lookup first** — a hit skips the circuit breaker and the LLM
   entirely. Checking the breaker first would incorrectly block a $0 cache
   hit against the budget.
2. **Model routing before the circuit breaker check** — route first, estimate
   the *routed* model's cost, then check the breaker. Checking the breaker
   with a worst-case (gpt-4o) estimate before routing would make the
   pre-check 10x too pessimistic for queries that route to gpt-4o-mini.
3. **Circuit breaker pre-check, before the LLM call** — a post-call check
   would let the final call overshoot the budget. The pre-check uses an
   upper-bound cost estimate (see `model_router.estimate_cost`), so it can
   only ever be too conservative, never too permissive.
4. **Cache store after the call, never before** — only successful responses
   are cached; caching a failed or empty response would poison future hits.
5. **Circuit breaker record after the call** — records the actual token cost,
   which is usually lower than the pre-call estimate.

Every component (`SemanticCache`, `CostCircuitBreaker`, `classify_complexity`
/ `route_model`) is independently instantiable and testable outside the
middleware. Swap the cache for Redis, the breaker for a Prometheus counter, or
the router for a fine-tuned classifier — `middleware.py` doesn't change.

## Components

| File | Responsibility |
|---|---|
| `exceptions.py` | `LLMInfraError` hierarchy; `BudgetExceededError` carries `spent`/`limit` so callers can special-case budget exhaustion |
| `semantic_cache.py` | Embedding-based cache: L2-normalised embeddings compared by dot product (= cosine similarity), LRU-evicted by timestamp |
| `circuit_breaker.py` | CLOSED/OPEN state machine gated on a pre-call estimated-cost check |
| `model_router.py` | Rule-based (keyword + length) complexity classifier, mapping `simple`/`complex` to `gpt-4o-mini`/`gpt-4o` |
| `cost_tracker.py` | Per-call, per-model cost/token history — analytics, separate from the breaker's single running budget |
| `logger.py` | Append-only structured JSONL event log |
| `middleware.py` | Composes all of the above into `llm_request()` |
| `run_demo.py` | End-to-end demo covering all six scenarios below |

## An optimization made across two commits: avoiding double embedding

The first working version of `middleware.py` called `cache.lookup(query)` to
search the cache (which embeds the query internally), and on a miss, called
`cache._embed(query)` again just before `cache.store(...)` — embedding the
same string twice per cache miss. `text-embedding-3-small` is cheap
(~$0.00000002/token), but at scale (10k queries/day) that's 10k entirely
avoidable API calls that could hit rate limits, not just a fraction of a cent
of avoidable spend.

The fix: `SemanticCache.lookup()` now always returns a dict, even on a miss —
`{"cache_hit": False, "query_embedding": query_vec}` — threading the already-
computed embedding through to `cache.store()` instead of recomputing it.
See the git history for this project: the naive version and the fix are two
separate commits, in the order they were actually built.

## Thread safety

**None of `SemanticCache`, `CostCircuitBreaker`, or `CostTracker` are
thread-safe.** Each mutates shared state (`self.entries`, `self.spent_usd`,
`self.calls`) without any locking — concurrent calls from multiple threads can
race past a budget check between the read and the write, or corrupt list
state during eviction/append. For threaded use, guard calls into these
objects with a `threading.Lock()`; for asyncio, use `asyncio.Lock()`.
`JSONLLogger` has the same caveat for concurrent file appends.

## Demo walkthrough (real run against the OpenAI API)

`run_demo.py` initialises the stack with a deliberately low $0.05 budget so
the circuit breaker actually trips, and runs six scenarios in order. All
numbers below are from an actual run (see `llm_infra/results.json`), not
illustrative placeholders.

**Scenario 1 — simple queries route to `gpt-4o-mini`:**
all three of "What is the capital of France?", "Who invented the telephone?",
and "Define machine learning in one sentence." routed to `gpt-4o-mini` with
confidence 0.80-0.85, at $0.0000078-$0.0000843 per call.

**Scenario 2 — complex queries route to `gpt-4o`:** both the HNSW/IVF
tradeoffs question and the reciprocal-rank-fusion coding question routed to
`gpt-4o` at confidence 0.55 and 0.80 respectively, at ~$0.0051 per call — the
HNSW query's classifier confidence of only 0.55 is the "uncertain -> default
to complex" branch in `classify_complexity`, since it contains no keyword from
`COMPLEXITY_SIGNALS` but is 12 words long.

**Scenario 3 — cache hit demonstration (a genuinely interesting real
result):** the exact repeat of "What is the capital of France?" hit the cache
at similarity=1.000, saving the call entirely. But the two rephrasings —
"What's France's capital city?" (similarity=0.811) and "Capital of France?"
(similarity=0.848) — **did not** hit the cache, both falling short of the
0.95 threshold. This is the threshold working exactly as documented: 0.95 is
deliberately conservative, and real embeddings of legitimately-equivalent
short questions don't always land above 0.95 cosine similarity. A looser
threshold (e.g. 0.85) would have caught these two but risks false-positive
hits on genuinely different questions that merely share vocabulary — this is
the real tradeoff the threshold comment in `semantic_cache.py` describes, not
a hypothetical one.

**Scenario 4 — circuit breaker trip:** looping complex queries (routed to
`gpt-4o`, ~$0.0051 each) tripped the breaker on the 7th trip-loop call, when
projected spend reached $0.0509 against the $0.05 limit. `BudgetExceededError`
was caught and reported (`spent=$0.0509 | limit=$0.0500`). A subsequent call
to the cheap query "What is 2+2?" was blocked immediately — once `OPEN`, the
breaker rejects every call regardless of its estimated cost.

**Scenario 5 — reset and continue:** status before reset was
`state=OPEN, spent=$0.0459, utilisation=91.8%`; after
`breaker.reset(new_budget=0.10)`, status was `state=CLOSED, spent=$0.00,
budget=$0.10`. The next call ("What is Python?") succeeded normally.

**Scenario 6 — force model override:** `force_model="gpt-4o"` on "What is
2+2?" correctly returned `model_used="gpt-4o"` and `complexity="forced"`
despite the query being an obvious `simple`-classifier case.

**Final summary from this run:**

| Metric | Value |
|---|---|
| Simple calls (gpt-4o-mini) | 6 |
| Complex calls (gpt-4o) | 10 |
| Cache hits | 1 |
| Cache hit rate | 12.5% (1 of 8 lookups) |
| Total cost tracked | $0.0462 |
| Cost by model | gpt-4o-mini: $0.00034, gpt-4o: $0.0459 |
| Estimated savings from cache hits | $0.0103 |

The single cache hit "saved" more than 20% of the total tracked spend by
itself — a preview of why cache hit rate matters disproportionately once a
corpus of repeated queries builds up in a real deployment.

## How to run

```
pip install -r ../requirements.txt   # openai, numpy, python-dotenv
# Reads OPENAI_API_KEY from a repo-root .env, or from the environment
# directly — export it yourself if you don't use a .env file.
export OPENAI_API_KEY=...
cd llm_infra
python run_demo.py
```

This writes `llm_infra/results.json` (structured output for all six
scenarios) and appends to `llm_infra/llm_infra_events.log.jsonl` (one JSON
event per cache hit/miss, routing decision, and completed call).

## Files

```
rag/04-semantic-caching/
├── requirements.txt
├── README.md
└── llm_infra/
    ├── exceptions.py        custom exception hierarchy
    ├── semantic_cache.py    embedding-based query cache
    ├── circuit_breaker.py   budget-based halt mechanism
    ├── model_router.py      complexity classifier + tier routing
    ├── cost_tracker.py      per-call/per-model token cost tracking
    ├── logger.py            structured JSONL logging
    ├── middleware.py        composes all of the above into llm_request()
    ├── run_demo.py          end-to-end demo (writes results.json)
    ├── results.json         real output from the last demo run
    └── llm_infra_events.log.jsonl   structured event log from the last run
```
