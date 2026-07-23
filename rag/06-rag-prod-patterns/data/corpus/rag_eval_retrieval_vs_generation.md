# Separating Retrieval Evaluation from Generation Evaluation

A common mistake in RAG evaluation is judging the whole pipeline end-to-end — "is the final answer correct?" — without first checking whether retrieval succeeded. This conflates two very different failure modes and makes root-causing painfully slow.

Retrieval evaluation asks a narrower, more mechanical question: given a query, did the system fetch chunks that actually contain the information needed to answer it? This can be measured with a labeled dataset of (query, relevant chunk IDs) pairs, using metrics like recall@k and MRR, entirely independent of any LLM call. Because it doesn't require generating or grading text, retrieval evaluation is cheap, fast, and fully deterministic — the same query against the same index always produces the same retrieved set, making it ideal for regression testing every time chunking, embeddings, or the retrieval algorithm changes.

Generation evaluation asks whether, given good context, the LLM produced a good answer — faithful to the provided context, relevant to the question, and appropriately hedged when the context is insufficient. This is harder to measure automatically and often relies on LLM-as-judge scoring or human review, since correctness of natural language is inherently fuzzier than "was chunk X in the top 5."

The practical value of separating these two evaluations is diagnostic speed: if end-to-end answer quality drops, checking retrieval metrics first tells you immediately whether to look at the retrieval pipeline (embeddings, chunking, index parameters) or the generation pipeline (prompt, model choice, context formatting) — without this split, every regression turns into an open-ended investigation across the entire stack.

A mature RAG evaluation suite runs both continuously: retrieval metrics on every change to ingestion or search code, and periodic generation quality checks (often smaller-sample, since LLM-as-judge calls cost money and time) to catch drift in the answer-writing half of the system.
