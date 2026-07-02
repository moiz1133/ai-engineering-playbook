"""15 test queries with ground-truth keywords for the benchmark.

Each query dict:
  query         — the natural-language question
  keywords      — all must appear (case-insensitive) in at least one retrieved chunk to count as a hit
  query_type    — "factual" (single concept) or "multi_concept" (spans multiple ideas)
  topic         — which corpus document this targets
"""
from __future__ import annotations
from typing import List

QUERIES: List[dict] = [
    # ── ML Fundamentals ──────────────────────────────────────────────────────
    {
        "query": "What is supervised learning and how does classification differ from regression?",
        "keywords": ["supervised", "classification", "regression"],
        "query_type": "multi_concept",
        "topic": "ml_fundamentals",
    },
    {
        "query": "How does backpropagation work in neural networks?",
        "keywords": ["backpropagation", "gradient", "weights"],
        "query_type": "factual",
        "topic": "ml_fundamentals",
    },
    {
        "query": "What is overfitting and what regularization techniques prevent it?",
        "keywords": ["overfitting", "regularization", "dropout"],
        "query_type": "multi_concept",
        "topic": "ml_fundamentals",
    },
    {
        "query": "Explain the difference between L1 and L2 regularization.",
        "keywords": ["L1", "L2", "weights"],
        "query_type": "multi_concept",
        "topic": "ml_fundamentals",
    },
    {
        "query": "How do convolutional neural networks process image data?",
        "keywords": ["convolutional", "filters", "images"],
        "query_type": "factual",
        "topic": "ml_fundamentals",
    },
    {
        "query": "What is transfer learning and why is it useful for small datasets?",
        "keywords": ["transfer learning", "pre-trained", "fine-tuned"],
        "query_type": "factual",
        "topic": "ml_fundamentals",
    },
    # ── Climate Science ───────────────────────────────────────────────────────
    {
        "query": "What causes the greenhouse effect and which gases are responsible?",
        "keywords": ["greenhouse", "carbon dioxide", "methane"],
        "query_type": "multi_concept",
        "topic": "climate_science",
    },
    {
        "query": "How does ocean acidification affect marine ecosystems?",
        "keywords": ["acidification", "pH", "corals"],
        "query_type": "factual",
        "topic": "climate_science",
    },
    {
        "query": "What are climate tipping points and give an example.",
        "keywords": ["tipping points", "irreversible", "permafrost"],
        "query_type": "factual",
        "topic": "climate_science",
    },
    {
        "query": "How do climate feedbacks like ice-albedo and water vapor amplify warming?",
        "keywords": ["ice-albedo", "water vapor", "feedback"],
        "query_type": "multi_concept",
        "topic": "climate_science",
    },
    {
        "query": "What contributes to sea level rise and how much could it increase by 2100?",
        "keywords": ["sea level", "thermal expansion", "ice sheets"],
        "query_type": "multi_concept",
        "topic": "climate_science",
    },
    # ── Internet History ──────────────────────────────────────────────────────
    {
        "query": "What is packet switching and how did it improve on circuit switching?",
        "keywords": ["packet switching", "circuit switching", "distributed"],
        "query_type": "multi_concept",
        "topic": "internet_history",
    },
    {
        "query": "How did TCP/IP unify different networks into the internet?",
        "keywords": ["TCP", "IP", "protocol"],
        "query_type": "factual",
        "topic": "internet_history",
    },
    {
        "query": "Who invented the World Wide Web and what technologies does it use?",
        "keywords": ["Berners-Lee", "HTML", "HTTP"],
        "query_type": "multi_concept",
        "topic": "internet_history",
    },
    {
        "query": "What is the Domain Name System and why was it created?",
        "keywords": ["DNS", "domain", "IP addresses"],
        "query_type": "factual",
        "topic": "internet_history",
    },
]
