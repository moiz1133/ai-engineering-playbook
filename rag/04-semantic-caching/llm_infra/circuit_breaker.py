"""Budget-based circuit breaker for LLM API spend.

WHAT: circuit breaker pattern from distributed systems, applied to LLM budget
WHY: without this, a runaway loop or batch job exhausts your API budget silently
STATE: CLOSED = calls allowed (normal operating state)
       OPEN   = budget exceeded, all calls halt immediately

THREAD SAFETY: this class is NOT thread-safe — self.spent_usd and self.state
are mutated by check() and record() without any locking. Concurrent calls from
multiple threads can race past the budget check between the read and the
write. For threaded use, guard check()/record() with a threading.Lock(); for
asyncio, use asyncio.Lock().
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from exceptions import BudgetExceededError


class CostCircuitBreaker:
    def __init__(self, budget_usd: float, warn_threshold: float = 0.80):
        self.budget_usd = budget_usd
        self.warn_threshold = warn_threshold  # warn at 80% by default
        self.spent_usd: float = 0.0
        self.call_count: int = 0
        self.state: str = "CLOSED"
        self.opened_at: Optional[str] = None

    def check(self, estimated_cost: float = 0.0) -> None:
        """Call BEFORE making an LLM API call. Raises BudgetExceededError if open.

        WHAT: check() is called before the LLM call — not after
        WHY: post-call checking would let the final call overshoot the budget;
             pre-call checking with estimated_cost prevents the overshoot
        """
        if self.state == "OPEN":
            raise BudgetExceededError(self.spent_usd, self.budget_usd)

        projected = self.spent_usd + estimated_cost
        if projected >= self.budget_usd:
            self.state = "OPEN"
            self.opened_at = datetime.now(timezone.utc).isoformat()
            raise BudgetExceededError(projected, self.budget_usd)

        if projected >= self.budget_usd * self.warn_threshold:
            pct = (projected / self.budget_usd) * 100
            print(f"[CIRCUIT BREAKER WARNING] {pct:.1f}% of budget used "
                  f"(${projected:.4f} / ${self.budget_usd:.4f})")

    def record(self, actual_cost: float) -> None:
        """Call AFTER a successful LLM call to record actual spend.

        WHAT: actual_cost may differ from estimated_cost — always record the
              real number
        WHY: estimated_cost is an upper bound; actual tokens used may be fewer
        """
        self.spent_usd += actual_cost
        self.call_count += 1
        remaining = self.budget_usd - self.spent_usd
        print(f"[CIRCUIT BREAKER] spent=${self.spent_usd:.4f} | "
              f"remaining=${remaining:.4f} | calls={self.call_count}")

    def reset(self, new_budget: Optional[float] = None) -> None:
        """Manually reset for a new session or after budget is recharged."""
        self.spent_usd = 0.0
        self.call_count = 0
        self.state = "CLOSED"
        self.opened_at = None
        if new_budget:
            self.budget_usd = new_budget

    def status(self) -> Dict:
        return {
            "state": self.state,
            "budget_usd": self.budget_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.budget_usd - self.spent_usd,
            "utilisation_pct": (self.spent_usd / self.budget_usd) * 100,
            "call_count": self.call_count,
            "opened_at": self.opened_at,
        }
