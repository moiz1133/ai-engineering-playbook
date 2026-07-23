# How HNSW Graphs Are Built

Building an HNSW index means inserting vectors one at a time into a growing multi-layer graph. When a new vector arrives, the algorithm first decides which layer it will "top out" at. This is chosen randomly using an exponentially decaying probability distribution, controlled by a parameter often called `mL` — most vectors are only inserted into layer 0 (the bottom, densest layer), while progressively fewer vectors reach layer 1, layer 2, and so on. This mirrors skip-list construction, which HNSW's inventors (Malkov and Yashunin) cite as a direct inspiration.

Once a vector's top layer is chosen, the algorithm inserts it into every layer from that top layer down to layer 0. At each layer, it searches for the `efConstruction` nearest already-inserted neighbors using the current graph, then connects the new vector to the best `M` of them (M is typically 12-48). If a neighbor already has `M` connections, the algorithm may prune its weakest edge to make room, using a heuristic that tries to preserve graph connectivity rather than just keeping the raw-closest neighbors.

Two parameters dominate the construction-time tradeoffs: `M`, the maximum number of connections per node per layer, and `efConstruction`, the size of the candidate list explored during insertion. Higher values of both produce a higher-quality graph (better recall at query time) at the cost of slower index builds and more memory. A common starting point is `M=16` and `efConstruction=200`, tuned upward for recall-sensitive workloads and downward when index build time dominates.

Because insertion order affects which edges get formed, HNSW graphs built from the same vectors in a different order are not identical — though their search quality is generally very similar in practice.
