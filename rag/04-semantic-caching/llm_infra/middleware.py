"""Compose the semantic cache, cost circuit breaker, and model router into a
single llm_request() entry point.

LLM MIDDLEWARE STACK — INTERVIEW EXPLANATION

The stack applies in this exact order for a reason:

1. CACHE FIRST — a hit skips circuit breaker AND LLM entirely
   If you checked the circuit breaker first, you'd block a cache hit that costs $0.

2. MODEL ROUTING BEFORE CIRCUIT BREAKER CHECK
   Route first, estimate the routed model's cost, THEN check the breaker.
   If you checked the breaker with a gpt-4o cost but then routed to mini,
   your pre-check would be 10x too pessimistic.

3. CIRCUIT BREAKER PRE-CHECK (before the LLM call)
   Post-call checking lets the last call overshoot the budget.
   Pre-call checking with an estimated cost prevents overshoot.
   Estimated cost is an upper bound — actual is usually lower.

4. CACHE STORE AFTER THE CALL — not before
   Only store successful responses. Caching failures poisons future hits.

5. CIRCUIT BREAKER RECORD AFTER CACHE STORE
   Order doesn't matter here — both happen post-call.
   Record actual tokens, not estimated.

Composability: each component (cache, breaker, router) is independently
testable. Swap the cache for Redis, the breaker for a Prometheus counter, or
the router for a fine-tuned classifier — middleware.py doesn't change.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from openai import OpenAI

from circuit_breaker import CostCircuitBreaker
from cost_tracker import CostTracker
from logger import JSONLLogger
from model_router import PRICE_TABLE, estimate_cost, route_model
from semantic_cache import SemanticCache


def llm_request(
    query: str,
    openai_client: OpenAI,
    cache: SemanticCache,
    breaker: CostCircuitBreaker,
    system_prompt: str = "You are a helpful assistant.",
    force_model: Optional[str] = None,
    max_tokens: int = 500,
    skip_cache: bool = False,
    cost_tracker: Optional[CostTracker] = None,
    logger: Optional[JSONLLogger] = None,
) -> dict:
    """Single entry point for all LLM calls. Applies the full middleware stack:
      1. Semantic cache lookup
      2. Cost circuit breaker pre-check
      3. Model tier routing
      4. LLM call
      5. Cache store
      6. Circuit breaker record
    Returns a result dict with full metadata.
    """
    call_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    # -- STEP 1: CACHE LOOKUP ------------------------------------------------
    query_embedding = None
    if not skip_cache:
        cached = cache.lookup(query)
        if cached["cache_hit"]:
            latency_ms = int((time.perf_counter() - start) * 1000)
            if logger:
                logger.log("cache_hit", call_id=call_id, similarity=cached["similarity"],
                            matched_query=cached["matched_query"], latency_ms=latency_ms)
            return {
                "call_id": call_id,
                "response": cached["response"],
                "source": "cache",
                "similarity": cached["similarity"],
                "matched_query": cached["matched_query"],
                "model_used": None,
                "cost_usd": 0.0,
                "latency_ms": latency_ms,
            }
        # WHAT: thread the embedding through lookup -> store to avoid a second
        #       API call — see semantic_cache.SemanticCache.lookup for the why
        query_embedding = cached["query_embedding"]
    # WHAT: cache lookup FIRST — if hit, never touch circuit breaker or LLM
    # WHY: cache hits cost $0 and bypass the budget entirely

    # -- STEP 2: MODEL ROUTING ------------------------------------------------
    model, complexity, confidence = route_model(query, force_model)
    estimated = estimate_cost(query, model, max_tokens)

    # -- STEP 3: CIRCUIT BREAKER PRE-CHECK ------------------------------------
    # Raises BudgetExceededError if budget is exhausted — let it propagate
    try:
        breaker.check(estimated)
    except Exception as e:
        if logger:
            logger.log("circuit_breaker_blocked", call_id=call_id, model=model,
                        estimated_cost=estimated, error=str(e))
        raise
    # WHAT: check() raises before the LLM call if budget would be exceeded
    # WHY: post-call check would allow the last call to overshoot the budget

    # -- STEP 4: LLM CALL ------------------------------------------------------
    # WHAT: no try/except here — if this call raises, it propagates directly
    # WHY: breaker.record() and cache.store() are both later in this function,
    #      so a failed call never reaches them; cost is only ever recorded and
    #      responses only ever cached for calls that actually succeeded
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    response_text = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    actual_cost = (
        prompt_tokens * PRICE_TABLE[model]["prompt"] +
        completion_tokens * PRICE_TABLE[model]["completion"]
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    # -- STEP 5: RECORD COST ---------------------------------------------------
    breaker.record(actual_cost)
    if cost_tracker:
        cost_tracker.record(model, prompt_tokens, completion_tokens, actual_cost, call_id)

    # -- STEP 6: CACHE STORE ----------------------------------------------------
    if not skip_cache:
        # Reuse the embedding computed during STEP 1's lookup — do NOT re-embed
        cache.store(query, query_embedding, response_text, actual_cost)
    # WHAT: store AFTER the call — cache only successful LLM responses
    # WHY: caching a failed or empty response would poison future cache hits

    if logger:
        logger.log("llm_call_complete", call_id=call_id, model=model, complexity=complexity,
                    cost_usd=actual_cost, latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    return {
        "call_id": call_id,
        "response": response_text,
        "source": "llm",
        "model_used": model,
        "complexity": complexity,
        "routing_confidence": confidence,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": actual_cost,
        "latency_ms": latency_ms,
        "breaker_status": breaker.status(),
    }
