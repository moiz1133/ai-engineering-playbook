"""End-to-end demo exercising every llm_infra component through llm_request():
simple/complex routing, semantic cache hits, a circuit breaker trip and
reset, and a forced model override. Writes results.json with every scenario's
output next to this script.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from openai import OpenAI

from circuit_breaker import CostCircuitBreaker
from cost_tracker import CostTracker
from exceptions import BudgetExceededError
from logger import JSONLLogger
from middleware import llm_request
from semantic_cache import SemanticCache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Repo-root .env (gitignored) holds OPENAI_API_KEY - load it here so running
# `python run_demo.py` works without exporting the var manually.
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", "..", ".env"))

MAX_TRIP_ATTEMPTS = 30  # safety cap so the demo can't loop forever if pricing assumptions drift


def _print_result(label: str, result: dict, fields: list = None) -> None:
    print(f"\n--- {label} ---")
    if fields:
        for f in fields:
            print(f"  {f}: {result.get(f)}")
    else:
        print(json.dumps(result, indent=2, default=str))


def main() -> None:
    openai_client = OpenAI()
    cache = SemanticCache(openai_client, similarity_threshold=0.95)
    breaker = CostCircuitBreaker(budget_usd=0.05, warn_threshold=0.80)
    cost_tracker = CostTracker()
    logger = JSONLLogger(os.path.join(SCRIPT_DIR, "llm_infra_events.log.jsonl"))

    all_results: dict = {}

    def call(query: str, **kwargs) -> dict:
        return llm_request(query, openai_client, cache, breaker,
                            cost_tracker=cost_tracker, logger=logger, **kwargs)

    # ------------------------------------------------------------------
    # SCENARIO 1 — Simple queries -> gpt-4o-mini
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 1 -- Simple queries (expect gpt-4o-mini)")
    print("=" * 70)
    scenario_1 = []
    for q in [
        "What is the capital of France?",
        "Who invented the telephone?",
        "Define machine learning in one sentence.",
    ]:
        result = call(q)
        _print_result(q, result, ["model_used", "cost_usd", "complexity", "routing_confidence"])
        scenario_1.append(result)
    all_results["scenario_1_simple"] = scenario_1

    # ------------------------------------------------------------------
    # SCENARIO 2 — Complex queries -> gpt-4o
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 2 -- Complex queries (expect gpt-4o)")
    print("=" * 70)
    scenario_2 = []
    for q in [
        "Explain the tradeoffs between HNSW and IVF indexing for a production RAG system.",
        "Write a Python function that implements reciprocal rank fusion for two ranked lists.",
    ]:
        result = call(q)
        _print_result(q, result, ["model_used", "cost_usd", "complexity", "routing_confidence"])
        scenario_2.append(result)
    all_results["scenario_2_complex"] = scenario_2

    # ------------------------------------------------------------------
    # SCENARIO 3 — Cache hit demonstration
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 3 -- Cache hit demonstration")
    print("=" * 70)
    scenario_3 = []
    for q in [
        "What is the capital of France?",  # identical to Q1
        "What's France's capital city?",  # rephrased — should still hit cache
        "Capital of France?",  # even shorter — should hit cache
    ]:
        result = call(q)
        _print_result(q, result, ["source", "similarity"])
        scenario_3.append(result)
    all_results["scenario_3_cache_hits"] = scenario_3

    # ------------------------------------------------------------------
    # SCENARIO 4 — Circuit breaker trip
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 4 -- Circuit breaker trip")
    print("=" * 70)
    trip_queries = itertools.cycle([
        "Explain why distributed consensus algorithms like Raft are hard to "
        "implement correctly, in detail.",
        "Compare and contrast SQL and NoSQL databases for a high-write "
        "analytics workload, comprehensively.",
        "Analyse the pros and cons of microservices versus a monolithic "
        "architecture step by step.",
        "Design a migration plan from a monolith to microservices and "
        "reason through the tradeoffs.",
        "Debate the tradeoffs of strong versus eventual consistency in "
        "distributed systems in detail.",
    ])

    calls_before_trip = []
    tripped = None
    # skip_cache=True: these queries exist purely to drain the budget, not to
    # demonstrate caching, and skipping the cache also skips its embed calls.
    for attempt, q in zip(range(MAX_TRIP_ATTEMPTS), trip_queries):
        try:
            result = call(q, skip_cache=True)
            _print_result(f"trip call {attempt + 1}", result, ["model_used", "cost_usd"])
            calls_before_trip.append(result)
        except BudgetExceededError as e:
            print(f"[CIRCUIT BREAKER TRIPPED] spent=${e.spent:.4f} | limit=${e.limit:.4f}")
            tripped = {"spent": e.spent, "limit": e.limit}
            break

    blocked_after_trip = False
    if tripped:
        try:
            call("What is 2+2?", skip_cache=True)
        except BudgetExceededError:
            print("All calls blocked after budget trip -- even simple ones")
            blocked_after_trip = True

    all_results["scenario_4_circuit_breaker"] = {
        "calls_before_trip": calls_before_trip,
        "tripped_at": tripped,
        "blocked_after_trip": blocked_after_trip,
    }

    # ------------------------------------------------------------------
    # SCENARIO 5 — Reset and continue
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 5 -- Reset and continue")
    print("=" * 70)
    status_before = breaker.status()
    print("Breaker status BEFORE reset:", status_before)
    breaker.reset(new_budget=0.10)
    status_after = breaker.status()
    print("Breaker status AFTER reset:", status_after)

    result_after_reset = call("What is Python?", skip_cache=True)
    _print_result("What is Python?", result_after_reset, ["model_used", "cost_usd", "source"])

    all_results["scenario_5_reset"] = {
        "status_before_reset": status_before,
        "status_after_reset": status_after,
        "result_after_reset": result_after_reset,
    }

    # ------------------------------------------------------------------
    # SCENARIO 6 — Force model override
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO 6 -- Force model override")
    print("=" * 70)
    result_forced = call("What is 2+2?", force_model="gpt-4o", skip_cache=True)
    _print_result("What is 2+2? (forced gpt-4o)", result_forced,
                  ["model_used", "complexity", "routing_confidence"])
    all_results["scenario_6_force_model"] = result_forced

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    cache_summary = cache.summary()
    breaker_status = breaker.status()
    cost_summary = cost_tracker.summary()

    print("Cache summary:", cache_summary)
    print("Breaker status:", breaker_status)
    print("Cost tracker summary:", cost_summary)

    simple_calls = cost_summary["calls_by_model"].get("gpt-4o-mini", 0)
    complex_calls = cost_summary["calls_by_model"].get("gpt-4o", 0)
    cache_hits = cache_summary["hits"]
    print(f"Total calls: {simple_calls} simple (gpt-4o-mini), "
          f"{complex_calls} complex (gpt-4o), {cache_hits} cache hits")
    print(f"Estimated savings from cache hits: ${cache_summary['total_saved_usd']:.4f}")

    all_results["final_summary"] = {
        "cache_summary": cache_summary,
        "breaker_status": breaker_status,
        "cost_tracker_summary": cost_summary,
        "simple_calls": simple_calls,
        "complex_calls": complex_calls,
        "cache_hits": cache_hits,
    }

    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nresults.json written to {results_path}")


if __name__ == "__main__":
    main()
