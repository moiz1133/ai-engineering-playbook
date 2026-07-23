# Indexing Algorithms Behind Vector Databases

While HNSW dominates modern vector database defaults, it isn't the only indexing algorithm in use, and understanding the alternatives clarifies why HNSW won out for most workloads.

IVF (Inverted File Index) partitions the vector space into clusters (using k-means or similar), assigning each vector to its nearest cluster centroid at index time. A query first identifies the closest few cluster centroids to itself, then only searches within those clusters rather than the whole dataset — an approach reminiscent of an inverted index in traditional search, hence the name. IVF is faster to build than HNSW and uses less memory, but recall tends to degrade more sharply as data grows unless the number of clusters and the number of clusters searched per query (`nprobe`) are both tuned carefully.

Product Quantization (PQ) is a compression technique rather than a full indexing algorithm on its own — it splits each vector into sub-vectors and replaces each sub-vector with the index of its nearest centroid from a small codebook, dramatically shrinking storage (often 8-16x) at the cost of some accuracy from the quantization error. PQ is frequently combined with IVF (as "IVF-PQ") to get both fast candidate-cluster search and compact storage, a combination popular in systems handling billions of vectors where full-precision HNSW's memory footprint becomes prohibitive.

Flat (brute-force) indexing simply compares the query against every stored vector with no index structure at all. It guarantees exact results with no recall loss, and remains a reasonable choice for small collections (tens of thousands of vectors or fewer) where the O(n) cost per query is still fast enough — and where a small, deterministic system is easier to reason about than a tuned approximate index.

The practical takeaway: HNSW is the right default for most mid-sized RAG corpora because it offers the best recall-per-byte-of-memory tradeoff at that scale, but very large-scale or extremely memory-constrained deployments increasingly reach for compressed and clustered variants instead.
