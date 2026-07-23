# Core Metrics for Evaluating RAG Retrieval

Evaluating a RAG system's retrieval quality requires metrics that separate "did we find the right chunks" from "did the model write a good answer" — conflating the two makes debugging nearly impossible when something goes wrong.

Recall@k measures, out of all truly relevant chunks for a query, what fraction appear somewhere in the top k retrieved results. It answers "did we get the right information into the context window at all," and is the most fundamental retrieval health check — if recall@k is low, no amount of prompt engineering can fix the resulting answers, since the model simply never sees the needed information.

Precision@k measures, out of the k chunks retrieved, what fraction are actually relevant. High recall with low precision means the system is retrieving the right chunks but burying them among irrelevant ones, which increases both cost (more tokens sent to the LLM) and the risk of the model getting distracted by noise — a well-documented failure mode sometimes called "lost in the middle."

Mean Reciprocal Rank (MRR) measures, on average, the reciprocal of the rank position of the first relevant result (1/rank). MRR rewards systems that put a relevant chunk at position 1 much more than one at position 5, making it well suited for evaluating systems where only one chunk needs to be right (e.g., a single-fact lookup) rather than systems needing broad coverage.

NDCG (Normalized Discounted Cumulative Gain) extends this idea to graded relevance — instead of binary relevant/not-relevant labels, chunks can be scored on a scale (e.g., 0-3), and NDCG rewards placing the most relevant results earliest while accounting for the fact that a "somewhat relevant" chunk at rank 2 still has value.

No single metric tells the whole story — production evaluation suites typically track several of these simultaneously, segmented by query type, since a system can look strong in aggregate while failing badly on a specific important query category.
