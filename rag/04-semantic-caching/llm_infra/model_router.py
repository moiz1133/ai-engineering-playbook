"""Complexity classifier and model tier router.

WHAT: routes each query to a cheap or expensive model based on a rule-based
      complexity classifier
WHY: a lightweight classifier avoids spending an LLM call to decide which LLM
     to call (that would be ironic and costly)
"""

from __future__ import annotations

from typing import Optional, Tuple

PRICE_TABLE = {
    "gpt-4o-mini": {"prompt": 0.00000015, "completion": 0.0000006},
    "gpt-4o":      {"prompt": 0.0000025,  "completion": 0.000010},
}

COMPLEXITY_SIGNALS = {
    "simple": [
        "what is", "define", "who is", "when was", "how many",
        "yes or no", "true or false", "list the", "name the",
        "what does", "spell", "translate",
    ],
    "complex": [
        "explain why", "analyse", "compare", "evaluate", "design",
        "write a", "generate", "summarise", "reason", "debate",
        "pros and cons", "how would you", "what would happen if",
        "step by step", "in detail", "comprehensively",
    ],
}


def classify_complexity(query: str) -> Tuple[str, float]:
    """Return (complexity_label, confidence) where label is 'simple' or 'complex'.

    WHAT: rule-based classifier using keyword signals and query length
    WHY default to complex when uncertain: sending a complex query to mini
        risks poor quality; sending a simple query to gpt-4o just wastes
        ~10x the cost — the asymmetric downside favours caution
    """
    query_lower = query.lower()
    simple_hits = sum(1 for s in COMPLEXITY_SIGNALS["simple"] if s in query_lower)
    complex_hits = sum(1 for s in COMPLEXITY_SIGNALS["complex"] if s in query_lower)
    word_count = len(query.split())

    score = 0  # positive = complex, negative = simple
    score += complex_hits * 2
    score -= simple_hits * 2
    score += max(0, (word_count - 15) // 5)  # long queries lean complex
    score -= max(0, (10 - word_count) // 3)  # short queries lean simple

    if score >= 2:
        label, confidence = "complex", min(0.95, 0.70 + score * 0.05)
    elif score <= -1:
        label, confidence = "simple", min(0.95, 0.70 + abs(score) * 0.05)
    else:
        label, confidence = "complex", 0.55  # uncertain -> default to complex (safe)

    return label, confidence


def route_model(query: str, force_model: Optional[str] = None) -> Tuple[str, str, float]:
    """Return (model_name, complexity_label, confidence).

    WHAT: maps complexity label to model tier
    WHY: gpt-4o-mini is ~10x cheaper than gpt-4o per token; routing 60-70% of
         queries to mini cuts average cost by 6-7x
    WHAT: force_model bypasses routing — useful for testing or guaranteed-
          quality paths
    """
    if force_model:
        return force_model, "forced", 1.0

    label, confidence = classify_complexity(query)
    model = "gpt-4o-mini" if label == "simple" else "gpt-4o"

    print(f"[MODEL ROUTER] query_type={label} | confidence={confidence:.2f} "
          f"| model={model} | query_preview={query[:60]}")

    return model, label, confidence


def estimate_cost(query: str, model: str, max_completion_tokens: int = 500) -> float:
    """Rough pre-call cost estimate for circuit breaker check().

    WHAT: rough upper-bound estimate BEFORE the call (actual tokens are only
          known after)
    WHY: circuit breaker needs an estimate to do its pre-call check — better
         to over-estimate slightly than to let a call overshoot the budget
    """
    estimated_prompt_tokens = len(query.split()) * 1.3  # rough token estimate
    prompt_cost = estimated_prompt_tokens * PRICE_TABLE[model]["prompt"]
    completion_cost = max_completion_tokens * PRICE_TABLE[model]["completion"]
    return prompt_cost + completion_cost
