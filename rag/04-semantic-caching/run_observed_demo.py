"""End-to-end demo using observed_llm_request().

Shows Prometheus metrics incrementing, and -- if LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are set -- Langfuse traces appearing at
cloud.langfuse.com. Without those keys, tracing silently no-ops and
everything else still runs.
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "llm_infra"))
# WHAT: both this project's root (for `observability`) and llm_infra/ (for its
#       flat sibling imports like `from circuit_breaker import ...`) explicitly
# WHY: don't rely on observability.middleware's own sys.path fixup running
#      first as a side effect of import order -- that's fragile against an
#      import-sorter reordering these lines

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# Repo-root .env (gitignored) holds OPENAI_API_KEY - load it before any
# client below is constructed.
load_dotenv(os.path.join(SCRIPT_DIR, "..", "..", ".env"))

from openai import OpenAI
from prometheus_client import generate_latest

from observability.middleware import observed_llm_request
from observability.prometheus_metrics import REGISTRY
from circuit_breaker import CostCircuitBreaker  # noqa: E402
from semantic_cache import SemanticCache  # noqa: E402

openai_client = OpenAI()
cache = SemanticCache(openai_client, similarity_threshold=0.95)
breaker = CostCircuitBreaker(budget_usd=0.10)

QUERIES = [
    ("What is the capital of France?", "simple"),
    ("Who invented the telephone?", "simple"),
    ("What is the capital of France?", "cache hit -- same as Q1"),
    ("France's capital city?", "cache hit -- rephrased"),
    ("Explain HNSW vs IVF indexing for a production RAG system at scale.", "complex"),
    ("Compare transformer and LSTM architectures for sequence modelling.", "complex"),
    ("What is the capital of France?", "cache hit -- third time"),
]

print("=" * 60)
print("Running observed demo")
print("=" * 60)

for query, note in QUERIES:
    print(f"\n[{note.upper()}]")
    print(f"Query: {query}")
    try:
        result = observed_llm_request(
            query, openai_client, cache, breaker,
            session_id="demo-session-001",
        )
        print(f"Source:  {result['source']}")
        print(f"Model:   {result.get('model_used') or 'n/a (cache)'}")
        print(f"Cost:    ${result['cost_usd']:.6f}")
        print(f"Latency: {result['latency_ms']}ms")
        if result["source"] == "cache":
            print(f"Similarity: {result.get('similarity', 0):.4f}")
    except Exception as e:
        print(f"Error: {e}")

# -- PRINT PROMETHEUS METRICS SNAPSHOT ---------------------------------------
print("\n" + "=" * 60)
print("PROMETHEUS METRICS SNAPSHOT")
print("=" * 60)
metrics_text = generate_latest(REGISTRY).decode()
for line in metrics_text.split("\n"):
    if line and not line.startswith("#"):
        print(line)

# -- PRINT CACHE SUMMARY -----------------------------------------------------
print("\n" + "=" * 60)
print("CACHE SUMMARY")
print("=" * 60)
summary = cache.summary()
for k, v in summary.items():
    print(f"  {k}: {v}")

# -- GRAFANA DASHBOARD HINTS --------------------------------------------------
print("\n" + "=" * 60)
print("GRAFANA QUERIES TO USE")
print("=" * 60)
print("""
  p50 latency:  histogram_quantile(0.50, rate(llm_request_latency_seconds_bucket[5m]))
  p95 latency:  histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m]))
  p99 latency:  histogram_quantile(0.99, rate(llm_request_latency_seconds_bucket[5m]))
  cache hit rate:  cache_hit_rate
  cost p95:     histogram_quantile(0.95, rate(llm_cost_per_request_usd_bucket[5m]))
  total spend:  session_spend_usd
  budget left:  budget_remaining_usd
  requests/min: rate(llm_requests_total[1m]) * 60
""")
