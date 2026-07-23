# Comparing Embeddings: Cosine, Dot Product, and Euclidean Distance

Once text is embedded into vectors, "similarity" needs a precise mathematical definition, and the three most common choices are cosine similarity, dot product, and Euclidean (L2) distance.

Cosine similarity measures the angle between two vectors, ignoring their magnitude entirely — it's computed as the dot product of the two vectors divided by the product of their norms. Its output ranges from -1 (opposite direction) to 1 (identical direction), with 0 meaning orthogonal (unrelated). Cosine similarity is the default choice for most text embedding use cases because it's insensitive to vector length, which matters since some embedding models produce vectors whose magnitude correlates with text length or frequency rather than meaning.

Dot product (also called inner product) is simply the sum of elementwise products of two vectors, without normalizing by their magnitudes. If vectors are already L2-normalized (scaled to unit length), dot product and cosine similarity produce identical rankings — this is why many systems normalize vectors once at embedding time and then use the cheaper dot product at query time, avoiding the extra division on every comparison. OpenAI's embedding models are commonly used this way: normalize once, then use dot product for all subsequent comparisons.

Euclidean distance measures the straight-line distance between two vectors' endpoints in space. Unlike cosine similarity, it is sensitive to vector magnitude, which can be desirable or undesirable depending on whether magnitude carries meaningful signal for a given embedding model. Smaller Euclidean distance means more similar; this is the opposite direction from cosine similarity's "higher is more similar," a detail that trips up many first-time implementations of top-k retrieval logic.

Most vector databases let you choose the metric per collection at creation time, and it must match whichever metric the embedding model was trained/evaluated against — an embedding model optimized for cosine similarity is not guaranteed to preserve ranking quality under raw Euclidean distance.
