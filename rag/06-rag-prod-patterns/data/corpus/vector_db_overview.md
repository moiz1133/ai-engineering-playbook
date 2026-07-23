# What a Vector Database Actually Does

A vector database is a system purpose-built to store high-dimensional vectors and answer nearest-neighbor queries against them efficiently at scale — given a query vector, quickly return the k stored vectors most similar to it, typically alongside associated metadata and the original text or document reference.

Under the hood, a vector database is usually a combination of three things: an approximate nearest-neighbor index (most commonly HNSW, though IVF and other structures exist), a metadata store that lets queries filter results by structured fields (e.g., only search chunks where `source_file = "policy.md"`), and a persistence/durability layer so the index survives restarts and can be backed up. The "database" framing matters because production use needs more than just a search algorithm — it needs inserts, updates, deletes, filtering, and reliability guarantees around all of them.

Purpose-built vector databases (Qdrant, Weaviate, Milvus, Pinecone) are optimized specifically for this workload and often support advanced features like hybrid search (combining keyword and vector search) and horizontal sharding for very large collections. General-purpose databases with vector extensions (pgvector for PostgreSQL, or vector search in Elasticsearch/OpenSearch) let teams add vector search to an existing database they already operate, trading some specialized performance for operational simplicity — one less system to run, back up, and monitor.

Lightweight, embedded vector stores like ChromaDB (used in this project) or FAISS run in-process or as a lightweight local server, persisting to disk without requiring a separate database service to deploy and manage. These are well suited to prototypes, small-to-medium corpora, and single-machine deployments, while the larger purpose-built systems become more attractive once a corpus needs to be sharded across machines or served with strict multi-tenant isolation.

Choosing among these options is primarily a question of scale and operational constraints, not fundamentally different retrieval quality — the underlying ANN algorithms are broadly similar across systems.
