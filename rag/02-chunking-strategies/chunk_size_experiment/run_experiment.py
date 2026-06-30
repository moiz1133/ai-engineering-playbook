"""
Chunk Size Experiment — entry point.

Holds chunking method (recursive separator splitting) and overlap percentage
(10% of chunk_size) constant, and varies ONLY chunk_size across 150, 300, and
600 characters, to isolate the effect of chunk size alone on RAG retrieval
quality and cost.

Run:
    python run_experiment.py
"""

import json
import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

from corpus_generator import generate_test_corpus
from experiment import run_size_experiment

_COLUMNS = [
    ("Chunk size", 10),
    ("Chunks", 6),
    ("Hit rate", 8),
    ("Factual", 7),
    ("Multi-concept", 14),
    ("Rephrased", 9),
    ("Split ctx", 9),
    ("Avg sim", 7),
    ("Tokens", 8),
]


def print_results_table(results: Dict[int, dict]) -> None:
    header = " | ".join(h.ljust(w) for h, w in _COLUMNS)
    print(header)
    print("-|-".join("-" * w for _, w in _COLUMNS))

    for size in sorted(results):
        r = results[size]
        by_type = r["hit_rate_by_type"]
        cells = [
            f"{size} chars",
            str(r["chunk_count"]),
            f'{r["hit_rate"] * 100:.1f}%',
            f'{by_type["factual"] * 100:.0f}%',
            f'{by_type["multi_concept"] * 100:.0f}%',
            f'{by_type["rephrased"] * 100:.0f}%',
            f'{r["split_context_rate"] * 100:.0f}%',
            f'{r["avg_top1_similarity"]:.2f}',
            f'{r["embedding_tokens"]:,}',
        ]
        print(" | ".join(c.ljust(w) for c, (_, w) in zip(cells, _COLUMNS)))


def build_interpretation(results: Dict[int, dict]) -> str:
    """Plain-English summary built entirely from `results` — every number in
    the output is read out of the results dict, not hardcoded.
    """
    sizes = sorted(results)
    smallest, largest = sizes[0], sizes[-1]

    per_size_summary = []
    for size in sizes:
        r = results[size]
        per_size_summary.append(
            f"{size} chars: hit rate {r['hit_rate'] * 100:.1f}% "
            f"(factual {r['hit_rate_by_type']['factual'] * 100:.0f}%, "
            f"multi-concept {r['hit_rate_by_type']['multi_concept'] * 100:.0f}%, "
            f"rephrased {r['hit_rate_by_type']['rephrased'] * 100:.0f}%), "
            f"split-context rate {r['split_context_rate'] * 100:.0f}%, "
            f"avg top-1 similarity {r['avg_top1_similarity']:.2f}, "
            f"{r['embedding_tokens']:,} embedding tokens across {r['chunk_count']} chunks."
        )

    # Compare the actual max/min across every size tested, not just the smallest
    # vs. largest endpoints — chunk-size effects aren't guaranteed to be
    # monotonic, and an endpoints-only comparison can miss (or misreport) a
    # peak or trough that happens to land on the middle size.
    split_values = {s: results[s]["split_context_rate"] for s in sizes}
    max_split_size = max(split_values, key=split_values.get)
    min_split_size = min(split_values, key=split_values.get)
    if max_split_size == min_split_size:
        fragmentation_sentence = (
            f"Split-context rate was identical ({split_values[max_split_size] * 100:.0f}%) across every "
            f"chunk size tested."
        )
    else:
        fragmentation_sentence = (
            f"Split-context rate was highest at {split_values[max_split_size] * 100:.0f}% with "
            f"{max_split_size}-char chunks and lowest at {split_values[min_split_size] * 100:.0f}% with "
            f"{min_split_size}-char chunks"
        )
        if max_split_size == smallest:
            fragmentation_sentence += (
                ", consistent with smaller chunks more often splitting a multi-concept answer's keywords "
                "across separate chunks the retriever can't reassemble into one result."
            )
        elif max_split_size == largest:
            fragmentation_sentence += ", with larger chunks fragmenting multi-concept answers more often in this run."
        else:
            fragmentation_sentence += ", a non-monotonic pattern across the chunk sizes tested in this run."

    sim_values = {s: results[s]["avg_top1_similarity"] for s in sizes}
    max_sim_size = max(sim_values, key=sim_values.get)
    min_sim_size = min(sim_values, key=sim_values.get)
    if max_sim_size == min_sim_size:
        similarity_sentence = (
            f"Average top-1 similarity was identical ({sim_values[max_sim_size]:.2f}) across every chunk "
            f"size tested."
        )
    else:
        similarity_sentence = (
            f"Average top-1 similarity peaked at {sim_values[max_sim_size]:.2f} with {max_sim_size}-char "
            f"chunks and was lowest at {sim_values[min_sim_size]:.2f} with {min_sim_size}-char chunks"
        )
        if min_sim_size == largest:
            similarity_sentence += (
                " - consistent with larger chunks covering more than one topic and diluting the embedding "
                "match for any single query even when the answer is present."
            )
        elif min_sim_size == smallest:
            similarity_sentence += (
                " - in this run, smaller chunks scored lower similarity on average than larger ones."
            )
        else:
            similarity_sentence += ", a non-monotonic pattern across the chunk sizes tested in this run."

    best_size = max(results, key=lambda s: results[s]["hit_rate"])
    best_sentence = (
        f"{best_size} chars gave the best overall hit rate "
        f"({results[best_size]['hit_rate'] * 100:.1f}%) in this experiment."
    )

    return " ".join(per_size_summary) + " " + fragmentation_sentence + " " + similarity_sentence + " " + best_sentence


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Repo-root .env (gitignored) holds OPENAI_API_KEY.
    load_dotenv(os.path.join(base_dir, "..", "..", "..", ".env"))

    corpus_dir = os.path.join(base_dir, "corpus")
    corpus_paths = generate_test_corpus(corpus_dir)

    client = OpenAI()

    print("Running chunk-size experiment: 150 / 300 / 600 chars, overlap fixed at 10% of chunk_size...")
    results = run_size_experiment(corpus_paths, client)

    print()
    print_results_table(results)
    print()
    print(build_interpretation(results))

    results_path = os.path.join(base_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {results_path}")


if __name__ == "__main__":
    main()


# CHUNK SIZE EXPERIMENT — HOW TO EXPLAIN IN AN INTERVIEW
# 1. Method and overlap percentage held constant — chunk_size is the ONLY independent variable
# 2. Keyword hit rate is an automated proxy for relevance — no LLM judge needed, fully reproducible
# 3. Split-context rate measures the boundary problem directly: does a multi-part answer
#    get fragmented across chunks the retriever then can't reassemble
# 4. Smaller chunks → higher precision per chunk but more fragmentation risk for complex answers
# 5. Larger chunks → less fragmentation but diluted embeddings (multiple topics per chunk)
#    pull the cosine similarity score down even when the chunk DOES contain the answer
# 6. The "best" size is empirical, not theoretical — that's why this experiment exists
