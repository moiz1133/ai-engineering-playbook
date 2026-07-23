# Building a Golden Dataset for RAG Evaluation

A golden dataset is a curated set of (query, expected relevant chunk IDs, and often a reference answer) triples used as ground truth for evaluating retrieval and generation quality. Without one, teams are reduced to eyeballing a handful of example queries, which doesn't scale and doesn't catch regressions reliably.

Building a good golden dataset starts with query sourcing. The strongest source is real user queries, sampled from production logs (with any necessary privacy review) — these reflect actual phrasing, ambiguity, and topic distribution far better than queries a developer invents while staring at the corpus. When real query logs don't exist yet (e.g., pre-launch), synthetic query generation using an LLM prompted with corpus chunks is a reasonable substitute, though it tends to produce queries that are unnaturally well-aligned with the exact wording of the source chunk — a bias worth correcting for by manually rephrasing a portion of generated queries.

Labeling relevance is the more labor-intensive part. For each query, a human (or a carefully prompted and spot-checked LLM) reviews candidate chunks and marks which ones actually contain information relevant to answering the query — not just topically related. A common failure is labeling anything "about the same topic" as relevant, which inflates recall scores without reflecting real usefulness.

Golden datasets should be small enough to maintain (25-100 queries is a common, tractable starting size for many projects) but large enough to cover the query types the system will realistically see: factual lookups, multi-hop questions requiring several chunks, and queries the corpus genuinely can't answer (to test that the system says so rather than hallucinating).

Crucially, a golden dataset needs periodic refreshing — as the corpus and real user query patterns evolve, a stale golden set can silently stop reflecting production reality, giving false confidence in metrics that no longer matter.
