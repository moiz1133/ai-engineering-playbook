# HNSW: An Overview

Hierarchical Navigable Small World (HNSW) is the most widely deployed algorithm for approximate nearest neighbor (ANN) search in high-dimensional vector spaces. It powers the default indexes in ChromaDB, Qdrant, Weaviate, and Milvus, and is available as an index type in pgvector and Elasticsearch's dense vector fields.

The core idea is to build a multi-layer graph where each node is a vector and edges connect vectors that are "close" under some distance metric (typically cosine similarity or Euclidean distance). The graph is organized into layers, with the top layer containing a sparse subset of nodes connected by long-range edges, and the bottom layer containing all nodes connected by short-range edges. A search starts at an entry point in the top layer, greedily moves toward the query vector, and drops down a layer once no closer neighbor can be found — repeating until it reaches the bottom layer and returns the best candidates found.

This layered structure is why HNSW is called "navigable small world": each layer behaves like a small-world network, where most vertices are reachable from any other in a small number of hops, similar to the "six degrees of separation" phenomenon in social graphs.

Compared to exact nearest-neighbor search — which requires comparing a query against every vector in the dataset, an O(n) operation — HNSW achieves approximately O(log n) search time by trading a small amount of recall for a massive speedup. In practice, HNSW indexes routinely return 95-99% of the true nearest neighbors while searching orders of magnitude fewer candidates, which is why virtually every production vector database defaults to it rather than brute-force search.

The tradeoff is memory: HNSW stores the full graph structure in memory (each node needs its neighbor lists at every layer it belongs to), so index size scales with both vector count and dimensionality.
