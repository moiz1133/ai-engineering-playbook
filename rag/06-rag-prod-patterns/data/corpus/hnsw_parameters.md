# Tuning HNSW: M, efConstruction, and ef

Three parameters control almost every practical tradeoff in an HNSW deployment, and getting them wrong is the most common cause of disappointing retrieval quality in production RAG systems.

`M` controls how many bidirectional edges each node keeps per layer. Small values (8-12) produce a sparser, faster-to-build, lower-memory graph, but connectivity suffers, which shows up as lower recall — especially on datasets with many "clusters" of similar vectors, since sparse graphs can fail to bridge between clusters. Larger values (32-64) improve recall and robustness but increase both memory usage and construction time roughly linearly. Most production deployments land between 16 and 32.

`efConstruction` controls the breadth of the search performed while inserting each new vector during index build. It only affects build time and final graph quality — never query latency directly, since it isn't used after the index is built. Values from 100-200 are common defaults; going higher (400+) yields diminishing returns for most datasets but can meaningfully help on datasets with tricky, non-uniform vector distributions.

`ef` (sometimes called `efSearch` to distinguish it from `efConstruction`) is set per query and controls the same breadth-of-search tradeoff at retrieval time. Unlike the other two parameters, it requires zero rebuilding to change, making it the cheapest lever to tune in production — many teams start with a conservative `ef` for low-latency paths and raise it for offline evaluation or high-stakes queries.

A practical tuning approach: fix `M` and `efConstruction` at reasonable defaults, build the index once, then sweep `ef` against a labeled evaluation set (see RAG evaluation) to find the minimum `ef` that clears your recall bar — this avoids paying for search quality you don't need.
