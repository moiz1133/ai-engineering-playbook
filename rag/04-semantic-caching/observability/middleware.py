"""Thin observability wrapper around the existing llm_request() middleware.

observed_llm_request() calls llm_request() EXACTLY ONCE and instruments
Prometheus metrics + a Langfuse trace purely from its return value (or from
catching BudgetExceededError) -- it never re-checks the cache, re-runs model
routing, or re-checks the circuit breaker itself.

WHY NOT duplicate those checks here: llm_request()'s own cache.lookup() call
is the ONLY thing that determines hit vs miss, and its own cache.store() call
is the ONLY thing that ever populates the cache. Calling cache.lookup() a
second time here, purely to gather metrics before delegating, would either
(a) trigger a second, redundant embedding API call, or -- if paired with
skip_cache=True to avoid that -- (b) ALSO skip llm_request's internal
cache.store() call, since the same `if not skip_cache:` guard covers both the
lookup and the store. That would silently break caching for every observed
request: repeat queries would never hit, because nothing would ever get
stored. Reading everything off the single call's return value avoids both
failure modes.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

_LLM_INFRA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_infra"
)
if _LLM_INFRA_DIR not in sys.path:
    sys.path.insert(0, _LLM_INFRA_DIR)
# WHAT: llm_infra/'s modules (middleware.py, circuit_breaker.py, ...) import
#       each other with bare `from circuit_breaker import ...`-style imports,
#       which only resolve if llm_infra/ itself is on sys.path
# WHY: keeps that requirement self-contained inside this package instead of
#      asking every entry point (run_observed_demo.py, metrics_server.py) to
#      remember to fix up sys.path before importing us

from openai import OpenAI  # noqa: E402

from circuit_breaker import CostCircuitBreaker  # noqa: E402
from exceptions import BudgetExceededError  # noqa: E402
from middleware import llm_request  # noqa: E402 -- existing code, unmodified
from semantic_cache import SemanticCache  # noqa: E402

from observability.langfuse_tracer import LangfuseTracer
from observability.prometheus_metrics import (
    budget_remaining_usd,
    budget_trips_total,
    cache_entries_count,
    cache_hit_rate,
    cache_hits_total,
    cache_misses_total,
    cache_similarity_score,
    cost_per_request_usd,
    llm_requests_total,
    request_latency_seconds,
    session_spend_usd,
)

_tracer = LangfuseTracer()


def observed_llm_request(
    query: str,
    openai_client: OpenAI,
    cache: SemanticCache,
    breaker: CostCircuitBreaker,
    system_prompt: str = "You are a helpful assistant.",
    force_model: Optional[str] = None,
    max_tokens: int = 500,
    skip_cache: bool = False,
    session_id: Optional[str] = None,
) -> dict:
    """Drop-in replacement for llm_request() with Prometheus + Langfuse layered
    on top. Same signature (plus an optional session_id for Langfuse), same
    return value -- existing code can call this instead of llm_request()."""
    start = time.perf_counter()
    trace = _tracer.trace_request(query, session_id)

    try:
        result = llm_request(
            query, openai_client, cache, breaker,
            system_prompt=system_prompt, force_model=force_model,
            max_tokens=max_tokens, skip_cache=skip_cache,
        )
    except BudgetExceededError as e:
        budget_trips_total.inc()
        _tracer.span_circuit_breaker(trace, e.spent, e.limit, tripped=True)
        _tracer.finalise(trace, "BUDGET_EXCEEDED", 0.0, "breaker",
                          int((time.perf_counter() - start) * 1000))
        raise

    latency_s = time.perf_counter() - start

    if result["source"] == "cache":
        # -- PROMETHEUS --------------------------------------------------
        llm_requests_total.labels(source="cache", model="none", complexity="none").inc()
        cache_hits_total.inc()
        request_latency_seconds.labels(source="cache", model="none").observe(latency_s)
        cache_similarity_score.observe(result["similarity"])
        cache_entries_count.set(len(cache.entries))
        _update_hit_rate(cache)

        # -- LANGFUSE ------------------------------------------------------
        _tracer.span_cache_lookup(trace, query, hit=True,
                                   similarity=result["similarity"],
                                   matched_query=result["matched_query"])
        _tracer.finalise(trace, result["response"], 0.0, "cache", result["latency_ms"])
        return result

    # result["source"] == "llm"
    cache_misses_total.inc()
    _tracer.span_cache_lookup(trace, query, hit=False)

    model = result["model_used"]
    complexity = result["complexity"]
    _tracer.span_model_routing(trace, query, model, complexity, result["routing_confidence"])

    breaker_status = result["breaker_status"]
    _tracer.span_circuit_breaker(trace, breaker_status["spent_usd"],
                                  breaker_status["budget_usd"], tripped=False)

    # -- PROMETHEUS ----------------------------------------------------------
    llm_requests_total.labels(source="llm", model=model, complexity=complexity).inc()
    request_latency_seconds.labels(source="llm", model=model).observe(latency_s)
    cost_per_request_usd.labels(model=model, complexity=complexity).observe(result["cost_usd"])
    session_spend_usd.set(breaker_status["spent_usd"])
    budget_remaining_usd.set(breaker_status["remaining_usd"])
    cache_entries_count.set(len(cache.entries))
    _update_hit_rate(cache)

    # -- LANGFUSE --------------------------------------------------------------
    _tracer.generation(
        trace, model, prompt=query, response=result["response"],
        prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"],
        cost_usd=result["cost_usd"], latency_ms=result["latency_ms"],
    )
    _tracer.finalise(trace, result["response"], result["cost_usd"], "llm", result["latency_ms"])

    return result


def _update_hit_rate(cache: SemanticCache) -> None:
    hits = cache.stats["hits"]
    total = hits + cache.stats["misses"]
    cache_hit_rate.set(hits / total if total > 0 else 0.0)


# ------------------------------------------------------------------------------------
# OBSERVABILITY LAYER -- INTERVIEW EXPLANATION
#
# PROMETHEUS:
#   Pull-based metrics system. Your app exposes /metrics; Prometheus scrapes it.
#   Three metric types used here:
#     Counter   -- only goes up (requests_total, cache_hits_total)
#     Histogram -- bucketed distribution; gives p50/p95/p99 via histogram_quantile()
#     Gauge     -- current value (session_spend, cache_hit_rate, budget_remaining)
#   Histograms are the key insight: you can't compute p95 from a Counter.
#   Bucket boundaries must be chosen to match your expected value range.
#
# LANGFUSE:
#   Trace/root span = one user request end-to-end (input query -> final output)
#   Span            = one stage within the trace (cache lookup, routing, circuit breaker)
#   Generation      = special observation type for LLM calls (understands token usage + cost)
#   The waterfall view in the Langfuse UI shows exactly where latency comes from.
#   flush() forces immediate send -- important in short-lived scripts; a long-running
#   server relies on Langfuse's own background batching instead.
#
# DESIGN DECISION -- wrapper, not modification:
#   observed_llm_request() calls the existing llm_request() unchanged, exactly once.
#   WHY: observability code should be additive, never modify business logic, and
#   never duplicate a decision (cache hit/miss, routing, budget) that llm_request()
#   already makes authoritatively.
#   If Langfuse is down or unconfigured, disable it with one env var -- zero code change.
#   If the Prometheus scraper fails, the app keeps running regardless.
#   This is the "observability as middleware" pattern used in production.
#
# GRAFANA QUERY TO MEMORISE FOR INTERVIEW:
#   histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m]))
#   Reads: "the p95 latency computed from the rate of bucket increments over 5 minutes"
# ------------------------------------------------------------------------------------
