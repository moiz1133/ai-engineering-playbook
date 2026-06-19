"""
HNSW vs. Brute-Force Vector Search Benchmark
=============================================

Educational benchmark that demonstrates *why* HNSW (Hierarchical Navigable
Small World graphs) matters once a vector index grows past a few thousand
items.

We deliberately do NOT use ChromaDB's query path for this comparison -
ChromaDB is installed (see requirements.txt) because it's a common way
people first encounter HNSW, but it wraps hnswlib/its own index internally
and hides the tunable parameters (M, ef_construction, ef_search) we need to
expose for the lesson. Going straight to `hnswlib` gives us full control and
a clean head-to-head against a from-scratch numpy brute-force search.

Run:
    python benchmark.py
    python benchmark.py --sizes 1000,10000,100000 --include-500k
    python benchmark.py --ef-search 50 --m 16 --ef-construction 200
"""

import argparse
import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
from tabulate import tabulate

# --------------------------------------------------------------------------
# Global baseline RAM reading.
#
# A bare Python interpreter + numpy + hnswlib + matplotlib already occupies
# tens of MB of resident memory before we've stored a single vector. If we
# report raw process RSS, every chart would be dominated by that fixed
# overhead instead of the thing we actually care about: how much *extra*
# memory the index itself costs. So we snapshot RSS once, right at the top
# of main(), and every later measurement subtracts this baseline.
# --------------------------------------------------------------------------
BASELINE_RSS = None
MB = 1024 * 1024


def _rss_bytes() -> int:
    """Current resident set size of this process, in bytes."""
    return psutil.Process(os.getpid()).memory_info().rss


def _extra_mb() -> float:
    """RSS right now, minus the baseline captured at script start."""
    if BASELINE_RSS is None:
        raise RuntimeError("BASELINE_RSS not set - call main() first")
    return (_rss_bytes() - BASELINE_RSS) / MB


@contextmanager
def measure():
    """Context manager that captures wall-clock time and RAM for a block.

    We call gc.collect() before AND after the timed region. Before, so a
    pending garbage collection from earlier work doesn't randomly fire
    *during* the block we're timing and inflate its cost. After, so memory
    that Python could reclaim doesn't get counted as "belonging" to this
    block just because the collector hasn't run yet.

    We use perf_counter() (a monotonic, high-resolution clock meant for
    benchmarking) rather than time.time() (a wall-clock clock that can jump
    backwards on NTP sync and has coarser resolution on some platforms).
    """
    gc.collect()
    start_time = time.perf_counter()
    stats = {}
    try:
        yield stats
    finally:
        elapsed = time.perf_counter() - start_time
        gc.collect()
        stats["time_s"] = elapsed
        stats["ram_mb"] = _extra_mb()


@dataclass
class PhaseStats:
    time_s: float
    ram_mb: float


@dataclass
class QueryStats:
    times_s: np.ndarray  # one entry per query, in seconds
    ram_mb: float

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_s) * 1000)

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_s) * 1000)

    def percentile_ms(self, p: float) -> float:
        return float(np.percentile(self.times_s, p) * 1000)


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------
def generate_clustered_dataset(n: int, dim: int, seed: int, num_queries: int,
                                num_clusters: int = None, cluster_std: float = 1.0,
                                center_scale: float = 10.0):
    """Generate indexed vectors AND query vectors from the same mixture of
    Gaussian clusters, instead of pure uniform/Gaussian noise.

    Why clusters and not pure random noise: in ~384 dimensions, independent
    random vectors are nearly equidistant from each other (a "concentration
    of measure" effect) - there is no local neighborhood structure for HNSW's
    graph to exploit. Measured directly: pure random noise at 10k vectors
    drove recall@10 down to ~35% at the default ef_search, and even cranking
    ef_search to 500 only recovered ~94% recall while making HNSW *slower*
    than brute force. That's the opposite of the lesson this benchmark is
    supposed to teach. Real embeddings (sentence/image vectors) cluster
    semantically, which is exactly the structure HNSW's graph is built to
    exploit, so synthetic data needs at least a crude version of that
    structure to behave like the real thing.

    Centers are spread far apart (center_scale, default 10 per-dimension std)
    relative to the noise added around each center (cluster_std, default 1
    per-dimension std) so clusters are well separated but not literally
    identical points. Queries are drawn from the SAME clusters as the
    indexed vectors (different specific noise draws) - this mirrors a real
    search where the query is similar-but-not-identical to indexed items,
    rather than coming from a totally unrelated distribution.

    float32 throughout to match real embedding models (and to keep memory
    for the 500k case manageable - float64 would double every RAM number in
    this benchmark for no pedagogical benefit).
    """
    rng = np.random.RandomState(seed)
    if num_clusters is None:
        # Roughly 50 points per cluster, capped so we always keep meaningful
        # cluster separation even at 500k vectors.
        num_clusters = max(2, min(256, n // 50))

    centers = rng.randn(num_clusters, dim).astype(np.float32) * center_scale
    total = n + num_queries
    assignments = rng.randint(0, num_clusters, size=total)
    noise = rng.randn(total, dim).astype(np.float32) * cluster_std
    all_vectors = (centers[assignments] + noise).astype(np.float32)

    vectors, queries = all_vectors[:n], all_vectors[n:]
    return vectors, queries


# --------------------------------------------------------------------------
# Brute-force search (the O(N) baseline)
# --------------------------------------------------------------------------
def run_brute_force(vectors: np.ndarray, queries: np.ndarray, k: int):
    """Exact nearest-neighbor search by scanning every vector, every query.

    Metric: cosine similarity. We normalize once up front so that "search"
    degenerates into a single matrix-vector dot product per query - this is
    the fastest a brute-force numpy implementation can reasonably be, which
    is the point: even the *best-case* brute force still loses to HNSW at
    scale because it's fundamentally O(N) per query.
    """
    with measure() as build:
        # "Build" for brute force is just normalizing the vectors so cosine
        # similarity becomes a dot product. This is the only up-front cost
        # brute force ever pays - no graph, no index structure - which is
        # exactly why its build time is tiny but its query time is not.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms
    build_stats = PhaseStats(build["time_s"], build["ram_mb"])

    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    normalized_queries = queries / query_norms

    # Warm up: the first few matrix multiplications pay one-time costs
    # (BLAS thread pool spin-up, CPU cache warming, page faults touching
    # memory for the first time). Without a warm-up, query #1's latency is
    # not representative of steady-state performance and would skew p50/p99.
    for q in normalized_queries[:5]:
        sims = normalized_vectors @ q
        np.argpartition(-sims, k - 1)[:k]

    times = np.empty(len(normalized_queries))
    results = np.empty((len(normalized_queries), k), dtype=np.int64)
    with measure() as query_phase:
        for i, q in enumerate(normalized_queries):
            t0 = time.perf_counter()
            sims = normalized_vectors @ q  # cosine similarity (unit vectors)
            # argpartition finds the top-k in O(N) without fully sorting all
            # N similarities - then we sort just those k for a ranked result.
            top_k_unranked = np.argpartition(-sims, k - 1)[:k]
            top_k = top_k_unranked[np.argsort(-sims[top_k_unranked])]
            times[i] = time.perf_counter() - t0
            results[i] = top_k
    query_stats = QueryStats(times, query_phase["ram_mb"])

    return {"build": build_stats, "query": query_stats, "results": results}


# --------------------------------------------------------------------------
# HNSW search (the approximate, sub-linear index)
# --------------------------------------------------------------------------
def build_hnsw_index(vectors: np.ndarray, M: int, ef_construction: int, space: str = "cosine"):
    """Build an hnswlib index. Kept separate from querying so the parameter
    sweep in the stretch experiment can build ONCE and vary ef_search many
    times, instead of paying build cost repeatedly for a query-time-only
    parameter.
    """
    dim = vectors.shape[1]
    index = hnswlib.Index(space=space, dim=dim)
    # M: max number of bidirectional links per node. Higher M = more
    # accurate graph but more RAM and slower build.
    # ef_construction: how exhaustively we search for neighbors while
    # inserting each node. Higher = better graph quality, slower build.
    index.init_index(max_elements=vectors.shape[0], M=M, ef_construction=ef_construction)

    with measure() as build:
        index.add_items(vectors, np.arange(vectors.shape[0]))
    build_stats = PhaseStats(build["time_s"], build["ram_mb"])
    return index, build_stats


def query_hnsw_index(index: "hnswlib.Index", queries: np.ndarray, k: int, ef_search: int):
    """Query a pre-built HNSW index.

    ef_search: how many candidate nodes the graph search keeps "active" at
    once. It's the query-time knob that trades latency for recall - unlike
    M/ef_construction, it costs nothing to change and doesn't require
    rebuilding, which is exactly why the stretch experiment sweeps it alone.
    """
    index.set_ef(ef_search)

    # Warm up for the same reason as brute force: first few hnswlib calls
    # pay one-time costs that don't reflect steady-state query latency.
    for q in queries[:5]:
        index.knn_query(q.reshape(1, -1), k=k)

    times = np.empty(len(queries))
    results = np.empty((len(queries), k), dtype=np.int64)
    with measure() as query_phase:
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            labels, _distances = index.knn_query(q.reshape(1, -1), k=k)
            times[i] = time.perf_counter() - t0
            results[i] = labels[0]
    query_stats = QueryStats(times, query_phase["ram_mb"])
    return results, query_stats


def run_hnsw(vectors: np.ndarray, queries: np.ndarray, k: int, M: int, ef_construction: int, ef_search: int):
    index, build_stats = build_hnsw_index(vectors, M, ef_construction)
    results, query_stats = query_hnsw_index(index, queries, k, ef_search)
    return {"build": build_stats, "query": query_stats, "results": results, "index": index}


# --------------------------------------------------------------------------
# Accuracy: how close is "approximate" to "exact"?
# --------------------------------------------------------------------------
def compute_recall(approx_results: np.ndarray, ground_truth: np.ndarray) -> float:
    """Recall@k: for each query, what fraction of the TRUE top-k (from exact
    brute-force search) did the approximate method (HNSW) actually find?
    Averaged across all queries. This is the number that quantifies the
    "accuracy you give up" half of the speed/accuracy trade-off.
    """
    recalls = []
    for approx_row, true_row in zip(approx_results, ground_truth):
        hits = len(set(approx_row.tolist()) & set(true_row.tolist()))
        recalls.append(hits / len(true_row))
    return float(np.mean(recalls))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class SizeResult:
    size: int
    bf_build: PhaseStats
    bf_query: QueryStats
    hnsw_build: PhaseStats
    hnsw_query: QueryStats
    recall: float


def run_one_size(size: int, dim: int, num_queries: int, k: int, seed: int, M: int, ef_construction: int, ef_search: int,
                  num_clusters: int = None, cluster_std: float = 1.0) -> SizeResult:
    print(f"\n=== Dataset size: {size:,} vectors (dim={dim}) ===", flush=True)

    print(f"  [{size:,}] generating clustered synthetic data (seed={seed})...", flush=True)
    # Queries come from the SAME clusters as the indexed vectors (see
    # generate_clustered_dataset's docstring for why pure random noise
    # gives misleadingly bad HNSW recall in high dimensions).
    vectors, queries = generate_clustered_dataset(size, dim, seed, num_queries, num_clusters, cluster_std)

    print(f"  [{size:,}] running brute-force (ground truth)...", flush=True)
    bf = run_brute_force(vectors, queries, k)
    print(
        f"  [{size:,}] brute-force done: build={bf['build'].time_s:.4f}s, "
        f"mean query={bf['query'].mean_ms:.3f}ms",
        flush=True,
    )

    print(f"  [{size:,}] running HNSW (M={M}, ef_construction={ef_construction}, ef_search={ef_search})...", flush=True)
    hnsw = run_hnsw(vectors, queries, k, M, ef_construction, ef_search)
    print(
        f"  [{size:,}] HNSW done: build={hnsw['build'].time_s:.4f}s, "
        f"mean query={hnsw['query'].mean_ms:.3f}ms",
        flush=True,
    )

    recall = compute_recall(hnsw["results"], bf["results"])
    print(f"  [{size:,}] recall@{k} = {recall:.4f}", flush=True)

    # Free the large arrays/index before the next (larger) size so RAM
    # measurements at the next size aren't inflated by this size's leftovers,
    # and so we don't run out of memory holding several 500k-scale datasets
    # in memory at once.
    del vectors, queries, bf["results"], hnsw["results"], hnsw["index"]
    gc.collect()

    return SizeResult(
        size=size,
        bf_build=bf["build"],
        bf_query=bf["query"],
        hnsw_build=hnsw["build"],
        hnsw_query=hnsw["query"],
        recall=recall,
    )


def print_summary_table(results: list):
    rows = []
    for r in results:
        rows.append([f"{r.size:,}", "Brute-force", f"{r.bf_build.time_s:.4f}s", f"{r.bf_build.ram_mb:.1f} MB",
                     f"{r.bf_query.mean_ms:.3f}", f"{r.bf_query.percentile_ms(95):.3f}", "1.0000 (exact)"])
        rows.append([f"{r.size:,}", "HNSW", f"{r.hnsw_build.time_s:.4f}s", f"{r.hnsw_build.ram_mb:.1f} MB",
                     f"{r.hnsw_query.mean_ms:.3f}", f"{r.hnsw_query.percentile_ms(95):.3f}", f"{r.recall:.4f}"])
    headers = ["Dataset Size", "Method", "Build Time", "Index RAM", "Mean Query (ms)", "p95 (ms)", "Recall@10"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))


def print_latency_detail_table(results: list):
    rows = []
    for r in results:
        for label, qs in (("Brute-force", r.bf_query), ("HNSW", r.hnsw_query)):
            rows.append([
                f"{r.size:,}", label,
                f"{qs.mean_ms:.3f}", f"{qs.percentile_ms(50):.3f}",
                f"{qs.percentile_ms(95):.3f}", f"{qs.percentile_ms(99):.3f}",
                f"{qs.std_ms:.3f}",
            ])
    headers = ["Dataset Size", "Method", "Mean (ms)", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Std Dev (ms)"]
    print("\nFull query latency distribution (100 queries/size):")
    print(tabulate(rows, headers=headers, tablefmt="github"))


def plot_latency_vs_size(results: list, out_dir: str):
    sizes = [r.size for r in results]
    bf_means = [r.bf_query.mean_ms for r in results]
    hnsw_means = [r.hnsw_query.mean_ms for r in results]

    plt.figure(figsize=(7, 5))
    # log-log axes: O(N) brute force becomes a straight line with slope 1;
    # O(log N) HNSW becomes nearly flat. The growing visual gap between the
    # two lines IS the argument for HNSW - no statistics needed to see it.
    plt.loglog(sizes, bf_means, marker="o", label="Brute-force (O(N))")
    plt.loglog(sizes, hnsw_means, marker="o", label="HNSW (~O(log N))")
    plt.xlabel("Dataset size (vectors, log scale)")
    plt.ylabel("Mean query latency (ms, log scale)")
    plt.title("Query Latency vs. Dataset Size")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    path = os.path.join(out_dir, "query_latency_vs_size.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_ram_vs_size(results: list, out_dir: str):
    sizes = [r.size for r in results]
    bf_ram = [r.bf_build.ram_mb for r in results]
    hnsw_ram = [r.hnsw_build.ram_mb for r in results]

    x = np.arange(len(sizes))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, bf_ram, width, label="Brute-force (raw vectors)")
    plt.bar(x + width / 2, hnsw_ram, width, label="HNSW (graph + vectors)")
    plt.xticks(x, [f"{s:,}" for s in sizes])
    plt.xlabel("Dataset size (vectors)")
    plt.ylabel("Additional RAM vs. baseline (MB)")
    plt.title("Index RAM Usage vs. Dataset Size")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    path = os.path.join(out_dir, "ram_usage_vs_size.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def print_final_summary(results: list):
    largest = max(results, key=lambda r: r.size)
    speedup = largest.bf_query.mean_ms / largest.hnsw_query.mean_ms
    ram_overhead_mb = largest.hnsw_build.ram_mb - largest.bf_build.ram_mb
    ram_overhead_pct = (ram_overhead_mb / largest.bf_build.ram_mb * 100) if largest.bf_build.ram_mb > 0 else float("nan")
    recall_loss_pct = (1 - largest.recall) * 100

    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY (at largest tested size: {largest.size:,} vectors)")
    print("=" * 70)
    print(f"  Speedup (brute-force time / HNSW time): {speedup:.1f}x faster with HNSW")
    print(f"  RAM overhead of HNSW vs. brute-force:    +{ram_overhead_mb:.1f} MB "
          f"({ram_overhead_pct:.1f}% more than brute-force's raw vector storage)")
    print(f"  Recall@10 trade-off:                     {largest.recall:.4f} "
          f"(gave up {recall_loss_pct:.2f}% accuracy for {speedup:.1f}x speed)")
    print("=" * 70)


# --------------------------------------------------------------------------
# Stretch experiment: ef_search sweep -> the classic recall/latency curve
# --------------------------------------------------------------------------
def run_ef_search_sweep(size: int, dim: int, num_queries: int, k: int, seed: int, M: int, ef_construction: int, ef_search_values: list, out_dir: str,
                         num_clusters: int = None, cluster_std: float = 1.0):
    print(f"\n=== Stretch experiment: ef_search sweep at {size:,} vectors ===", flush=True)
    print("  This teaches that HNSW is a TUNABLE trade-off, not a single fixed algorithm:", flush=True)
    print("  raising ef_search trades query latency for higher recall, with diminishing returns.", flush=True)

    vectors, queries = generate_clustered_dataset(size, dim, seed, num_queries, num_clusters, cluster_std)

    print(f"  [sweep] computing exact ground truth via brute-force...", flush=True)
    bf = run_brute_force(vectors, queries, k)
    ground_truth = bf["results"]

    print(f"  [sweep] building HNSW index ONCE (M={M}, ef_construction={ef_construction})...", flush=True)
    index, build_stats = build_hnsw_index(vectors, M, ef_construction)
    print(f"  [sweep] build done in {build_stats.time_s:.4f}s", flush=True)

    sweep_rows = []
    for ef in ef_search_values:
        print(f"  [sweep] ef_search={ef} ...", flush=True)
        results, query_stats = query_hnsw_index(index, queries, k, ef)
        recall = compute_recall(results, ground_truth)
        sweep_rows.append((ef, query_stats.mean_ms, recall))
        print(f"    -> mean query={query_stats.mean_ms:.3f}ms, recall@{k}={recall:.4f}", flush=True)

    headers = ["ef_search", "Mean Query (ms)", "Recall@10"]
    table_rows = [[ef, f"{ms:.3f}", f"{r:.4f}"] for ef, ms, r in sweep_rows]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="github"))

    # The classic ANN-benchmarks plot: recall on the y-axis, latency on the
    # x-axis. The "best" parameter choice lives on the upper-left frontier
    # of this curve - this single plot is the entire pitch for why ANN
    # libraries expose ef_search as a runtime knob instead of hard-coding it.
    plt.figure(figsize=(7, 5))
    xs = [ms for _, ms, _ in sweep_rows]
    ys = [r for _, _, r in sweep_rows]
    plt.plot(xs, ys, marker="o")
    for ef, ms, r in sweep_rows:
        plt.annotate(f"ef={ef}", (ms, r), textcoords="offset points", xytext=(6, -4))
    plt.xlabel("Mean query latency (ms)")
    plt.ylabel(f"Recall@{k}")
    plt.title(f"Recall vs. Latency Trade-off (HNSW, {size:,} vectors)")
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "recall_latency_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

    del vectors, queries, bf["results"], ground_truth, index
    gc.collect()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="HNSW vs. brute-force vector search benchmark")
    p.add_argument("--sizes", type=str, default="1000,10000,100000",
                    help="Comma-separated dataset sizes to test (default: 1000,10000,100000)")
    p.add_argument("--include-500k", action="store_true",
                    help="Also test 500,000 vectors. Needs ~3-4 GB free RAM for the brute-force copy + HNSW graph; off by default.")
    p.add_argument("--dim", type=int, default=384, help="Vector dimensionality (default: 384, matches all-MiniLM-L6-v2)")
    p.add_argument("--num-queries", type=int, default=100, help="Number of query vectors per dataset size (default: 100)")
    p.add_argument("--k", type=int, default=10, help="Top-k for nearest neighbor search and recall@k (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument("--m", type=int, default=16, help="HNSW M parameter: max links per node (default: 16)")
    p.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction (default: 200)")
    p.add_argument("--ef-search", type=int, default=50, help="HNSW ef_search (default: 50)")
    p.add_argument("--num-clusters", type=int, default=None,
                    help="Number of synthetic clusters to draw vectors from (default: n//50, capped at 256). "
                         "Pure random noise has no cluster structure in high dimensions and makes HNSW recall artificially terrible.")
    p.add_argument("--cluster-std", type=float, default=1.0,
                    help="Per-dimension noise std within each cluster (default: 1.0). Cluster centers are spread "
                         "with std=10 per dimension, so this controls how tight/loose each cluster is relative to that spread.")
    p.add_argument("--skip-stretch", action="store_true", help="Skip the ef_search sweep stretch experiment")
    p.add_argument("--out-dir", type=str, default="benchmark_output", help="Directory for plots (default: ./benchmark_output)")
    return p.parse_args()


def main():
    global BASELINE_RSS

    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if args.include_500k and 500_000 not in sizes:
        sizes.append(500_000)
    sizes = sorted(set(sizes))

    os.makedirs(args.out_dir, exist_ok=True)

    print("HNSW vs. Brute-Force Vector Search Benchmark")
    print(f"Seed: {args.seed}  |  Dim: {args.dim}  |  k: {args.k}  |  Queries/size: {args.num_queries}")
    print(f"HNSW params: M={args.m}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    print(f"Dataset sizes: {sizes}")

    # Snapshot baseline RSS *after* importing all heavy libraries (numpy,
    # hnswlib, matplotlib are already loaded by this point) but *before* we
    # allocate a single benchmark vector. Everything measured from here on
    # is "extra" memory attributable to the benchmark itself.
    gc.collect()
    BASELINE_RSS = _rss_bytes()
    print(f"Baseline RSS (interpreter + libraries, excluded from all RAM numbers below): {BASELINE_RSS / MB:.1f} MB")

    results = []
    for size in sizes:
        r = run_one_size(size, args.dim, args.num_queries, args.k, args.seed, args.m, args.ef_construction, args.ef_search,
                          args.num_clusters, args.cluster_std)
        results.append(r)
        # Print progressively after EACH size completes, so a slow 500k run
        # doesn't leave the user staring at a blank terminal wondering if
        # the script hung.
        print_summary_table(results)

    print_latency_detail_table(results)
    plot_latency_vs_size(results, args.out_dir)
    plot_ram_vs_size(results, args.out_dir)
    print_final_summary(results)

    if not args.skip_stretch and 100_000 in sizes:
        run_ef_search_sweep(
            size=100_000, dim=args.dim, num_queries=args.num_queries, k=args.k, seed=args.seed,
            M=args.m, ef_construction=args.ef_construction,
            ef_search_values=[10, 50, 100, 200, 500], out_dir=args.out_dir,
            num_clusters=args.num_clusters, cluster_std=args.cluster_std,
        )
    elif not args.skip_stretch:
        print("\n(Skipping ef_search sweep stretch experiment - 100,000 is not in --sizes)")


if __name__ == "__main__":
    main()
