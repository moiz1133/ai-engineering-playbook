# Choosing Between Embedding Models

Picking an embedding model is one of the highest-leverage decisions in a RAG system, because switching later means re-embedding and re-indexing the entire corpus. A few axes matter most in practice.

General-purpose commercial APIs, like OpenAI's `text-embedding-3-small`/`large` or Cohere's `embed-v3`, are trained on broad web-scale data and perform well across most domains without any fine-tuning. They're the easiest starting point: no GPU infrastructure to manage, consistent latency, and strong out-of-the-box performance on benchmarks like MTEB (Massive Text Embedding Benchmark). Their downside is per-token cost and a dependency on an external API being available and fast enough for your latency budget.

Open-weight models — such as the BGE family, E5, or Nomic Embed — can be self-hosted, giving full control over latency, cost at scale, and data residency (nothing leaves your infrastructure). Many of these models now rival or beat commercial APIs on MTEB's retrieval subtasks. The tradeoff is operational: you now own GPU provisioning, batching, and model updates yourself.

Domain-specific or fine-tuned embeddings matter most when the corpus uses vocabulary that general models weren't trained to distinguish well — legal contract language, medical terminology, or a codebase's internal jargon are common examples. Fine-tuning a smaller open-weight model on domain-specific pairs (queries and their known-relevant passages) frequently outperforms even the largest general-purpose commercial model on that specific domain, at a fraction of the inference cost.

In practice, most teams should start with a strong general-purpose model, measure retrieval quality with a real evaluation set, and only invest in self-hosting or fine-tuning once they've confirmed the general-purpose model is the actual bottleneck rather than chunking, retrieval strategy, or prompt design.
