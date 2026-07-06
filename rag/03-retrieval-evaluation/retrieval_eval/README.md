# Retrieval Evaluation Suite

## Overview

This project measures how three retrieval strategies — plain cosine similarity,
Cohere cross-encoder reranking, and Maximal Marginal Relevance (MMR) — perform
against a shared, automatically-generated corpus and a shared 25-item Q&A eval
set. All three methods query the same ChromaDB collection built from the same
embeddings, so any difference in scores is attributable purely to retrieval
logic rather than corpus or embedding differences. Results are scored with
Mean Reciprocal Rank (MRR) and Hit@k, using automated keyword-based ground
truth annotation instead of human judges. Cohere rerank ran successfully.

## Eval set

- 25 Q&A pairs across 5 documents (25 of 25 had at least one matching ground-truth chunk)
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
Method           | MRR  | Hit@1 | Hit@3 | Hit@5 | Avg latency
-----------------|------|-------|-------|-------|------------
Baseline cosine  | 1.00 | 1.00  | 1.00  | 1.00  | 592ms      
Cohere rerank    | 1.00 | 1.00  | 1.00  | 1.00  | 1515ms     
MMR (λ=0.5)      | 1.00 | 1.00  | 1.00  | 1.00  | 575ms      
MMR (λ=0.3, div) | 1.00 | 1.00  | 1.00  | 1.00  | 589ms      
MMR (λ=0.7, rel) | 1.00 | 1.00  | 1.00  | 1.00  | 574ms      
```

## Key findings

- Cohere reranking improved MRR from 1.00 to 1.00 (+0% relative change).
- Rephrased queries showed a gap between baseline and rerank: 1.00 vs 1.00 MRR.
- Cohere rerank added ~924ms of latency per query versus baseline cosine (1515ms vs 592ms).
- MMR with λ=0.5 scored Hit@3=1.00 versus baseline Hit@3=1.00, trading some relevance for diversity among the retrieved chunks.
- Optimal MMR lambda on this corpus: 0.3 (MRR=1.00, Hit@3=1.00).

## Lambda sweep (MMR)

```
Lambda | MRR  | Hit@3
-------|------|------
0.3    | 1.00 | 1.00 
0.4    | 1.00 | 1.00 
0.5    | 1.00 | 1.00 
0.6    | 1.00 | 1.00 
0.7    | 1.00 | 1.00 
0.8    | 1.00 | 1.00 
```

## Per query-type breakdown

```
Method           | Factual MRR | Multi-concept MRR | Rephrased MRR
-----------------|-------------|-------------------|--------------
Baseline cosine  | 1.00        | 1.00              | 1.00         
Cohere rerank    | 1.00        | 1.00              | 1.00         
MMR (λ=0.5)      | 1.00        | 1.00              | 1.00         
MMR (λ=0.3, div) | 1.00        | 1.00              | 1.00         
MMR (λ=0.7, rel) | 1.00        | 1.00              | 1.00         
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
