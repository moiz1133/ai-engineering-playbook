# HNSW Query-Time Search

Searching an HNSW index for the k nearest neighbors of a query vector happens in two phases. In the first phase, the search starts at a fixed entry point in the topmost layer and performs greedy descent: at each layer, it repeatedly moves to the neighbor closest to the query vector until no neighbor is closer than the current node, then drops to the next layer down using that node as the new entry point. This phase is cheap because upper layers contain very few nodes.

The second phase happens at layer 0 (or sometimes starting a layer or two above it) and is more thorough. Instead of pure greedy descent, the algorithm maintains a candidate list of size `ef` (the query-time analog of `efConstruction`) and explores outward from the current best candidates, keeping the `ef` closest vectors seen so far. Once no unexplored candidate could possibly improve the current best-`ef` set, the search stops and the top `k` results are returned.

The `ef` parameter is the primary query-time recall/speed knob: setting `ef` equal to `k` gives the fastest, lowest-recall search, while raising `ef` well above `k` (e.g., `ef=100` for `k=10`) explores more of the graph and typically pushes recall above 95%. Because `ef` is set per query rather than baked into the index, the same HNSW index can serve both a fast, lower-recall "autocomplete" use case and a slower, higher-recall "final ranking" use case without rebuilding anything.

Unlike exact search, HNSW provides no hard recall guarantee — it is an approximate algorithm by design, and pathological data distributions (or a poorly tuned `ef`) can occasionally cause it to miss a true nearest neighbor. This is the fundamental tradeoff every vector database user is implicitly accepting.
