# Why Chunking Strategy Matters

Chunking — splitting source documents into smaller pieces before embedding and indexing — is one of the most consequential and most underrated decisions in a RAG pipeline. Get it wrong and no amount of better embeddings, retrieval algorithms, or prompting can fully compensate, because the fundamental unit of information the system retrieves is broken from the start.

Chunks that are too large dilute relevance: a 2000-token chunk covering five different subtopics might get retrieved because one sentence matches the query, but the embedding for the whole chunk represents an average of all five subtopics, making it a poor match for a specific, narrow question — and it wastes context window space with mostly-irrelevant text once retrieved.

Chunks that are too small lose context: a 20-token chunk might contain a sentence like "this reduces latency by roughly 40%" with no indication of what "this" refers to, since the referent lived in a different chunk. The embedding for such a chunk is often uninformative, and even if retrieved correctly, the LLM generating an answer has no way to recover the missing context.

The right chunk size is workload-dependent: FAQ-style content with self-contained question/answer pairs can use small chunks effectively, while narrative or highly cross-referential documents (contracts, technical specifications) usually need larger chunks or explicit context-preservation techniques (like prepending a summary or heading to each chunk) to remain useful in isolation.

Beyond size, the boundary-finding method — how a document decides where one chunk ends and the next begins — matters just as much as the target size, which is why fixed-size, semantic, and overlap-based strategies (covered separately) each represent a different tradeoff between simplicity and context preservation.
