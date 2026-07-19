# pip install langfuse
# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from environment
# Free tier at cloud.langfuse.com -- no self-hosting needed for a demo
"""Langfuse tracing for llm_request() calls.

NOTE ON SDK VERSION: `pip install langfuse` today installs the v4
(OpenTelemetry-based) SDK. Its API is NOT the `client.trace(...)` /
`trace.span(...)` / `trace.generation(...)` chained-object API from older
Langfuse releases -- that API (and `langfuse.model.CreateTrace` etc.) no
longer exists. v4 instead uses `client.start_observation(...)` to create a
span/generation, `.start_observation(...)` on that span again to create a
child, and `.update()` / `.end()` to close it out, with
`propagate_attributes()` as a context manager for trace-level session_id/tags.
This module is written against the actual installed v4 API (verified via
`inspect.signature` against langfuse==4.14.0), not the older API shape.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from typing import Any, Dict, Optional

from langfuse import Langfuse, propagate_attributes


class LangfuseTracer:
    def __init__(self) -> None:
        self.enabled = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
        self.client: Optional[Langfuse] = None
        if self.enabled:
            self.client = Langfuse(
                public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
                host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
        else:
            print("[LANGFUSE] Keys not set -- tracing disabled, calls pass through")
        # WHAT: enabled flag means the app works without Langfuse keys
        # WHY: tracing is observability infrastructure -- it must never block the main path

    def trace_request(self, query: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Start a root span for one llm_request() call.

        Returns a small context dict (root span + the propagate_attributes
        context manager, held open across the whole request) or None if
        tracing is disabled -- every other method on this class treats a
        None/falsy trace as a no-op.
        """
        if not self.enabled:
            return None
        stack = ExitStack()
        stack.enter_context(propagate_attributes(
            session_id=session_id, tags=["production", "semantic-cache-project"],
        ))
        root_span = self.client.start_observation(
            name="llm_request", as_type="span", input={"query": query},
        )
        return {"root": root_span, "stack": stack}
        # WHAT: one root span = one user-facing request end-to-end
        # WHY: every child observation (cache lookup, routing, circuit breaker,
        #      generation) nests under this span -- the Langfuse UI shows the
        #      full waterfall from input to output

    def span_cache_lookup(self, trace: Optional[Dict[str, Any]], query: str, hit: bool,
                           similarity: Optional[float] = None,
                           matched_query: Optional[str] = None) -> None:
        if not self.enabled or not trace:
            return
        trace["root"].start_observation(
            name="semantic_cache_lookup", as_type="span",
            input={"query": query},
            output={"hit": hit, "similarity": similarity, "matched_query": matched_query},
            metadata={"threshold": 0.95},
            level="DEFAULT" if hit else "WARNING",
        ).end()
        # WHAT: span captures one stage of the pipeline with input/output
        # WHY: seeing "cache_lookup -> hit=True, similarity=0.97" in the trace tells
        #      you exactly why a request was cheap and fast

    def span_model_routing(self, trace: Optional[Dict[str, Any]], query: str,
                            model: str, complexity: str, confidence: float) -> None:
        if not self.enabled or not trace:
            return
        trace["root"].start_observation(
            name="model_routing", as_type="span",
            input={"query": query},
            output={"model": model, "complexity": complexity, "confidence": confidence},
        ).end()

    def span_circuit_breaker(self, trace: Optional[Dict[str, Any]],
                              spent: float, budget: float, tripped: bool) -> None:
        if not self.enabled or not trace:
            return
        trace["root"].start_observation(
            name="circuit_breaker_check", as_type="span",
            input={"spent": spent, "budget": budget},
            output={"tripped": tripped, "remaining": budget - spent},
            level="ERROR" if tripped else "DEFAULT",
        ).end()

    def generation(self, trace: Optional[Dict[str, Any]], model: str, prompt: str,
                    response: str, prompt_tokens: int, completion_tokens: int,
                    cost_usd: float, latency_ms: int) -> None:
        if not self.enabled or not trace:
            return
        trace["root"].start_observation(
            name="llm_generation", as_type="generation", model=model,
            input=prompt, output=response,
            usage_details={
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            cost_details={"total": cost_usd},
            metadata={"latency_ms": latency_ms},
        ).end()
        # WHAT: as_type="generation" is Langfuse's dedicated observation type for LLM calls
        # WHY: it understands token usage and cost and renders them in the UI --
        #      shows up separately from plain spans in the trace waterfall

    def finalise(self, trace: Optional[Dict[str, Any]], output: str, cost_usd: float,
                 source: str, latency_ms: int) -> None:
        if not self.enabled or not trace:
            return
        trace["root"].update(
            output=output,
            metadata={"total_cost_usd": cost_usd, "source": source, "latency_ms": latency_ms},
        )
        trace["root"].end()
        trace["stack"].close()  # exits the propagate_attributes() context opened in trace_request
        self.client.flush()
        # WHAT: update() adds the final output to the root span before closing it
        # WHY: root span input=query, output=answer -- visible in Langfuse's trace list view
        # WHAT: flush() forces immediate send instead of Langfuse's default background batching
        # WHY: a short-lived demo/script can exit before a batched flush would fire on its own
