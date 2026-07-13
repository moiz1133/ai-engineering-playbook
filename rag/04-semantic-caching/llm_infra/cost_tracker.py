"""Token cost tracking, factored out of the LLM call site so any wrapper
(middleware, batch jobs, notebooks) can reuse the same accounting without
duplicating the token-to-dollar math.

WHAT: records every LLM call's token usage and dollar cost
WHY: the circuit breaker only needs a single running total against a budget;
     this tracker keeps the full per-call, per-model history for analytics
     (cost by model, average cost per call, etc.) — a separate concern that
     shouldn't be bolted onto the breaker's budget-enforcement logic

THREAD SAFETY: this class is NOT thread-safe — self.calls is mutated by
record() without any locking. For threaded use, guard calls to record() with
a threading.Lock(); for asyncio, use asyncio.Lock().
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional


class CostTracker:
    def __init__(self) -> None:
        self.calls: List[Dict] = []
        self.total_cost_usd: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               cost_usd: float, call_id: Optional[str] = None) -> Dict:
        """Record one LLM call's token usage and cost."""
        entry = {
            "call_id": call_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.calls.append(entry)
        self.total_cost_usd += cost_usd
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return entry

    def cost_by_model(self) -> Dict[str, float]:
        """Total cost broken down by model name."""
        breakdown: Dict[str, float] = {}
        for call in self.calls:
            breakdown[call["model"]] = breakdown.get(call["model"], 0.0) + call["cost_usd"]
        return breakdown

    def calls_by_model(self) -> Dict[str, int]:
        """Call count broken down by model name."""
        counts: Dict[str, int] = {}
        for call in self.calls:
            counts[call["model"]] = counts.get(call["model"], 0) + 1
        return counts

    def summary(self) -> Dict:
        """Aggregate totals: cost, tokens, per-model breakdown, avg cost/call."""
        n = len(self.calls)
        return {
            "total_calls": n,
            "total_cost_usd": self.total_cost_usd,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "avg_cost_per_call_usd": self.total_cost_usd / n if n else 0.0,
            "cost_by_model": self.cost_by_model(),
            "calls_by_model": self.calls_by_model(),
        }
