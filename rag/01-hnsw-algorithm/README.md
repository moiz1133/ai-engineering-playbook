# HNSW vs. Brute-Force Vector Search Benchmark

An educational benchmark comparing exact brute-force nearest-neighbor search against
HNSW (Hierarchical Navigable Small World) approximate search, to show concretely why
HNSW matters once a vector index grows past a few thousand items.

Code: [`vector-search-benchmark/benchmark.py`](vector-search-benchmark/benchmark.py)

## How HNSW Works

### Graph construction

HNSW builds a multi-layer graph over the indexed vectors instead of a flat list:

- Each vector becomes a node. Nodes are inserted one at a time; each insertion is
  randomly assigned a "top layer," and the node exists in every layer from there down
  to layer 0. Higher layers are sparse (few nodes, long-range links), layer 0 contains
  every node (dense, short-range links) — like a skip list, but for similarity instead
  of sorted order.
- `M`: the max number of bidirectional links each node keeps per layer. Higher M means
  a denser, more accurate graph, but more RAM and a slower build.
- `ef_construction`: while inserting a new node, the index runs a greedy graph search
  to find the best M neighbors to link it to. Higher `ef_construction` searches more
  thoroughly before picking those links, producing a higher-quality graph at the cost
  of slower builds.

### Search

- A query starts at a fixed entry point in the topmost layer and greedily walks toward
  whichever neighbor is closest to the query, until no neighbor improves on the current
  best — then it drops down one layer and repeats, using the previous layer's best node
  as the new starting point.
- At layer 0, instead of tracking just the single best node, the search keeps a
  candidate list of size `ef_search` and expands it by visiting each candidate's
  neighbors. A larger `ef_search` means more of the graph gets explored before
  settling on the final top-k, which raises the odds the *true* nearest neighbors are
  among the candidates (higher recall) at the cost of visiting more nodes per query
  (higher latency).
- Routing through long-range links at the top layers first, then narrowing down,
  is why HNSW query time scales roughly O(log N) instead of brute force's O(N): each
  layer prunes the search space geometrically rather than scanning every vector once.

## Recall vs. Latency Trade-off

`ef_search` is a query-time-only knob — changing it doesn't require rebuilding the
index — which is what makes it the natural dial for trading speed against accuracy.
Swept `ef_search` ∈ [10, 50, 100, 200, 500] on a 100,000-vector index
(`M=16`, `ef_construction=200`):

| ef_search | Mean Query (ms) | Recall@10 |
|---|---|---|
| 10  | 0.106 | 0.771 |
| 50  | 0.213 | 0.986 |
| 100 | 0.257 | 0.998 |
| 200 | 0.312 | 0.999 |
| 500 | 0.608 | 0.999 |

![Recall vs Latency](benchmark_output/recall_latency_curve.png)

Recall climbs steeply from `ef_search` 10→100 (77%→99.8%) then flattens — past
`ef_search` ≈ 200 there's almost nothing left to gain, only added latency. This is the
classic ANN-benchmarks "knee" curve: the useful operating range is the steep part, not
the tail.
