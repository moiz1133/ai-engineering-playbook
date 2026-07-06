"""Print the results table and auto-generate README.md from actual experiment numbers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

METHOD_ORDER: List[Tuple[str, str]] = [
    ("baseline", "Baseline cosine"),
    ("rerank", "Cohere rerank"),
    ("mmr", "MMR (λ=0.5)"),
    ("mmr_high_div", "MMR (λ=0.3, div)"),
    ("mmr_low_div", "MMR (λ=0.7, rel)"),
]


def _available_methods(results: Dict) -> List[Tuple[str, str, Dict]]:
    out = []
    for key, label in METHOD_ORDER:
        report = results.get(key)
        if report is not None:
            out.append((key, label, report))
    return out


def _fmt_row(cells: List[str], widths: List[int]) -> str:
    return " | ".join(cell.ljust(w) for cell, w in zip(cells, widths))


def _build_main_table_lines(results: Dict) -> List[str]:
    headers = ["Method", "MRR", "Hit@1", "Hit@3", "Hit@5", "Avg latency"]
    rows = []
    for _, label, report in _available_methods(results):
        rows.append([
            label,
            f"{report['mrr']:.2f}",
            f"{report['hit_at_1']:.2f}",
            f"{report['hit_at_3']:.2f}",
            f"{report['hit_at_5']:.2f}",
            f"{report['avg_latency_ms']:.0f}ms",
        ])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    lines = [_fmt_row(headers, widths), "-|-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(_fmt_row(row, widths))
    return lines


def _build_by_type_table_lines(results: Dict) -> List[str]:
    headers = ["Method", "Factual MRR", "Multi-concept MRR", "Rephrased MRR"]
    rows = []
    for _, label, report in _available_methods(results):
        by_type = report.get("by_type", {})
        rows.append([
            label,
            f"{by_type.get('factual', {}).get('mrr', 0.0):.2f}",
            f"{by_type.get('multi_concept', {}).get('mrr', 0.0):.2f}",
            f"{by_type.get('rephrased', {}).get('mrr', 0.0):.2f}",
        ])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    lines = [_fmt_row(headers, widths), "-|-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(_fmt_row(row, widths))
    return lines


def _build_lambda_sweep_table_lines(results: Dict) -> List[str]:
    headers = ["Lambda", "MRR", "Hit@3"]
    sweep = results.get("mmr_lambda_sweep", {})
    rows = []
    for lam in sorted(sweep.keys()):
        stats = sweep[lam]
        rows.append([f"{lam:.1f}", f"{stats['mrr']:.2f}", f"{stats['hit_at_3']:.2f}"])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    lines = [_fmt_row(headers, widths), "-|-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(_fmt_row(row, widths))
    return lines


def print_results_table(results: Dict) -> None:
    print("\n=== Method comparison ===")
    for line in _build_main_table_lines(results):
        print(line)

    print("\n=== Per query-type breakdown ===")
    for line in _build_by_type_table_lines(results):
        print(line)

    print("\n=== MMR lambda sweep ===")
    for line in _build_lambda_sweep_table_lines(results):
        print(line)
    print()


def _best_lambda(results: Dict) -> Tuple[float, Dict]:
    sweep = results.get("mmr_lambda_sweep", {})
    best_lam = max(sweep, key=lambda lam: sweep[lam]["mrr"])
    return best_lam, sweep[best_lam]


def _key_findings(results: Dict) -> List[str]:
    findings = []
    baseline = results.get("baseline")
    rerank = results.get("rerank")
    mmr_report = results.get("mmr")

    if baseline and rerank:
        baseline_mrr = baseline["mrr"]
        rerank_mrr = rerank["mrr"]
        delta = (rerank_mrr - baseline_mrr) / baseline_mrr if baseline_mrr else 0.0
        findings.append(
            f"Cohere reranking improved MRR from {baseline_mrr:.2f} to {rerank_mrr:.2f} "
            f"({delta:+.0%} relative change)."
        )
        baseline_rephrase = baseline["by_type"].get("rephrased", {}).get("mrr", 0.0)
        rerank_rephrase = rerank["by_type"].get("rephrased", {}).get("mrr", 0.0)
        findings.append(
            f"Rephrased queries showed a gap between baseline and rerank: "
            f"{baseline_rephrase:.2f} vs {rerank_rephrase:.2f} MRR."
        )
        findings.append(
            f"Cohere rerank added ~{rerank['avg_latency_ms'] - baseline['avg_latency_ms']:.0f}ms "
            f"of latency per query versus baseline cosine ({rerank['avg_latency_ms']:.0f}ms vs "
            f"{baseline['avg_latency_ms']:.0f}ms)."
        )
    elif baseline:
        findings.append(
            "Cohere rerank experiment was skipped (COHERE_API_KEY not set) — "
            "only cosine baseline and MMR variants were evaluated."
        )

    if baseline and mmr_report:
        findings.append(
            f"MMR with λ=0.5 scored Hit@3={mmr_report['hit_at_3']:.2f} versus baseline "
            f"Hit@3={baseline['hit_at_3']:.2f}, trading some relevance for diversity among "
            f"the retrieved chunks."
        )

    best_lam, best_stats = _best_lambda(results)
    findings.append(
        f"Optimal MMR lambda on this corpus: {best_lam} (MRR={best_stats['mrr']:.2f}, "
        f"Hit@3={best_stats['hit_at_3']:.2f})."
    )

    return findings


def generate_readme(results: Dict, output_path: str = "README.md") -> None:
    """Write README.md using only numbers pulled dynamically from results."""
    main_table = "\n".join(_build_main_table_lines(results))
    by_type_table = "\n".join(_build_by_type_table_lines(results))
    sweep_table = "\n".join(_build_lambda_sweep_table_lines(results))
    findings = _key_findings(results)
    findings_md = "\n".join(f"- {f}" for f in findings)

    meta = results.get("_meta", {})
    cohere_note = (
        "Cohere rerank ran successfully." if meta.get("cohere_available")
        else "Cohere rerank was skipped because `COHERE_API_KEY` was not set."
    )

    readme = f"""# Retrieval Evaluation Suite

## Overview

This project measures how three retrieval strategies — plain cosine similarity,
Cohere cross-encoder reranking, and Maximal Marginal Relevance (MMR) — perform
against a shared, automatically-generated corpus and a shared 25-item Q&A eval
set. All three methods query the same ChromaDB collection built from the same
embeddings, so any difference in scores is attributable purely to retrieval
logic rather than corpus or embedding differences. Results are scored with
Mean Reciprocal Rank (MRR) and Hit@k, using automated keyword-based ground
truth annotation instead of human judges. {cohere_note}

## Eval set

- 25 Q&A pairs across 5 documents ({meta.get('answerable_eval_items', '?')} of {meta.get('total_eval_items', '?')} had at least one matching ground-truth chunk)
- 10 factual / 10 multi-concept / 5 rephrased
- Ground truth: keyword-based automated annotation

## Metrics explained

MRR (Mean Reciprocal Rank) measures where the first correct chunk appears in
the ranked results — a hit at rank 1 scores 1.0, at rank 3 scores 0.33, and a
miss scores 0. Hit@k is a simpler yes/no signal: did any correct chunk show up
anywhere in the top k results? MRR and Hit@k are complementary — MRR rewards
precise ranking, while Hit@k only cares whether the answer was retrieved at all.

## Results

```
{main_table}
```

## Key findings

{findings_md}

## Lambda sweep (MMR)

```
{sweep_table}
```

## Per query-type breakdown

```
{by_type_table}
```

## How to run

```
pip install openai chromadb cohere numpy python-dotenv
# Reads OPENAI_API_KEY / COHERE_API_KEY from a repo-root .env, or from the
# environment directly — export them yourself if you don't use a .env file.
export OPENAI_API_KEY=...
export COHERE_API_KEY=...   # optional — rerank experiment skipped if not set
python run_all.py
```

## Files

- `corpus/` — auto-generated documents used as the retrieval corpus
- `eval_set.py` — 25 annotated Q&A pairs plus automated ground-truth annotation
- `corpus_builder.py` — generates the corpus, chunks it, embeds it, and builds the ChromaDB index
- `retriever_baseline.py` — vanilla cosine top-k retriever (the control)
- `retriever_rerank.py` — cosine retrieval followed by Cohere cross-encoder reranking
- `retriever_mmr.py` — Maximal Marginal Relevance retriever for diversity-aware selection
- `metrics.py` — MRR and Hit@k implementations
- `experiment.py` — runs all methods over the eval set and collects results
- `report.py` — prints result tables and writes this README
- `run_all.py` — single entry point that runs the full pipeline
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readme)
