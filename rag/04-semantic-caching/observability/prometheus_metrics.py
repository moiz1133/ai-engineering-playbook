"""Prometheus metric definitions for the semantic-caching LLM middleware.

WHAT: a custom CollectorRegistry (not the global default) holding every
      metric this project exposes
WHY: a custom registry means this module can be imported multiple times
     (e.g. in tests) without "Duplicated timeseries" errors that the global
     default registry raises on re-registration
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

# ── COUNTERS ──────────────────────────────────────────────────────────────
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests by source and model",
    labelnames=["source", "model", "complexity"],  # source: llm | cache
    registry=REGISTRY,
)
# WHAT: labels let you filter in Grafana -- e.g. show only cache hits
# WHY: source="cache" requests cost $0 and should be separated in dashboards

cache_hits_total = Counter(
    "cache_hits_total",
    "Total semantic cache hits",
    registry=REGISTRY,
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total semantic cache misses",
    registry=REGISTRY,
)

budget_trips_total = Counter(
    "budget_trips_total",
    "Total times the cost circuit breaker tripped",
    registry=REGISTRY,
)

# ── HISTOGRAMS ────────────────────────────────────────────────────────────
request_latency_seconds = Histogram(
    "llm_request_latency_seconds",
    "End-to-end request latency in seconds",
    labelnames=["source", "model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)
# WHAT: buckets define the boundaries for p50/p95/p99 calculation
# WHY: Prometheus computes percentiles from bucket counts -- bucket choice matters.
#      0.05-0.25 captures cache hits; 1-10s captures LLM calls; 30s for slow models
# NOTE: Prometheus histogram_quantile(0.95, ...) gives p95 latency in Grafana

cost_per_request_usd = Histogram(
    "llm_cost_per_request_usd",
    "Cost in USD per LLM request (excludes cache hits)",
    labelnames=["model", "complexity"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.50],
    registry=REGISTRY,
)
# WHAT: cost histogram -- p95 cost per request visible in Grafana
# WHY: average cost hides outliers; a histogram shows the full distribution --
#      a p99 cost spike means one query type is extremely expensive

cache_similarity_score = Histogram(
    "cache_similarity_score",
    "Cosine similarity score on cache hits",
    buckets=[0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
    registry=REGISTRY,
)
# WHAT: distribution of similarity scores on cache hits
# WHY: if most hits cluster near 0.95 (the threshold), the threshold may be too low --
#      scores near 1.0 mean near-identical queries; near 0.95 means borderline matches

embedding_latency_seconds = Histogram(
    "embedding_latency_seconds",
    "Latency of OpenAI embedding calls",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    registry=REGISTRY,
)

# ── GAUGES ────────────────────────────────────────────────────────────────
session_spend_usd = Gauge(
    "session_spend_usd",
    "Cumulative session spend in USD",
    registry=REGISTRY,
)

budget_remaining_usd = Gauge(
    "budget_remaining_usd",
    "Remaining budget in USD before the circuit breaker trips",
    registry=REGISTRY,
)

cache_entries_count = Gauge(
    "cache_entries_count",
    "Number of entries currently in the semantic cache",
    registry=REGISTRY,
)

cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Rolling cache hit rate (hits / total requests)",
    registry=REGISTRY,
)
# WHAT: Gauge = current value, not cumulative -- right for rates and balances
# WHY: hit_rate and budget_remaining change over time (including downward);
#      a Counter can only go up, so it would be the wrong type for either
