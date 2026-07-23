# Fixed-Size Chunking

Fixed-size chunking splits text into chunks of a constant length — typically measured in tokens rather than characters, since token count is what actually determines embedding model input limits and cost. A simple implementation encodes the full document with a tokenizer (e.g., `tiktoken` for OpenAI models), then slices the resulting token list into fixed-length windows, decoding each window back into text before embedding.

Its biggest advantage is simplicity and predictability: every chunk costs roughly the same to embed, fits comfortably within any model's context window, and the implementation requires no document-structure awareness — it works identically on Markdown, plain text, or extracted PDF text, which makes it a reasonable default when ingesting heterogeneous document types with no consistent internal structure to exploit.

Its biggest weakness is that fixed-size boundaries have no relationship to the document's actual structure. A chunk boundary can land in the middle of a sentence, splitting a single idea across two chunks that each get embedded (and potentially retrieved) independently, neither containing the complete thought. This is especially damaging for content with a strict logical flow, like step-by-step instructions or legal clauses, where a mid-thought split can make both halves individually useless.

Choosing the fixed size itself is an empirical exercise, not a formula: common starting points are 200-500 tokens, but the right value depends on how self-contained typical passages in the corpus are, and should be validated with real retrieval evaluation (recall@k against a labeled query set) rather than picked from a rule of thumb alone.

Fixed-size chunking is almost always paired with overlap between consecutive chunks specifically to mitigate the mid-thought-split problem, trading some storage and embedding cost for a lower chance that any single idea is split with zero surrounding context in either chunk.
