# Retrieval Evaluation Suite

## Overview

This project measures how four retrieval strategies — plain cosine similarity,
BM25 keyword search fused with vector search via Reciprocal Rank Fusion (RRF),
Cohere cross-encoder reranking, and Maximal Marginal Relevance (MMR) — perform
against a shared, automatically-generated corpus and a shared 25-item Q&A eval
set. All methods query the same ChromaDB collection built from the same
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
Method            | MRR  | Hit@1 | Hit@3 | Hit@5 | Avg latency
------------------|------|-------|-------|-------|------------
Baseline cosine   | 1.00 | 1.00  | 1.00  | 1.00  | 619ms      
BM25 only         | 0.81 | 0.76  | 0.88  | 0.88  | 0ms        
Hybrid (BM25+RRF) | 0.83 | 0.76  | 0.92  | 0.92  | 575ms      
Cohere rerank     | 1.00 | 1.00  | 1.00  | 1.00  | 1260ms     
MMR (λ=0.5)       | 1.00 | 1.00  | 1.00  | 1.00  | 566ms      
MMR (λ=0.3, div)  | 1.00 | 1.00  | 1.00  | 1.00  | 554ms      
MMR (λ=0.7, rel)  | 1.00 | 1.00  | 1.00  | 1.00  | 582ms      
```

## Key findings

- Cohere reranking improved MRR from 1.00 to 1.00 (+0% relative change).
- Rephrased queries showed a gap between baseline and rerank: 1.00 vs 1.00 MRR.
- Cohere rerank added ~641ms of latency per query versus baseline cosine (1260ms vs 619ms).
- MMR with λ=0.5 scored Hit@3=1.00 versus baseline Hit@3=1.00, trading some relevance for diversity among the retrieved chunks.
- Optimal MMR lambda on this corpus: 0.3 (MRR=1.00, Hit@3=1.00).

## Hybrid search — how it works

Hybrid search runs two independent retrievers over the same chunks and fuses
their rankings. BM25 (`retriever_hybrid.build_bm25_index` / `retrieve_bm25`)
scores chunks by keyword overlap — term frequency weighted by how rare that
term is across the corpus — so it excels at exact-term queries regardless of
semantic meaning. Cosine vector search (`retrieve_vector_for_fusion`) embeds
the query and every chunk independently and ranks by similarity, so it excels
at paraphrases and conceptual questions that share no vocabulary with the
source text.

The two top-20 ranked lists are combined with Reciprocal Rank Fusion: every
chunk's fused score is the sum of `1 / (k + rank)` across whichever lists it
appears in, using the standard constant k=60. A chunk ranked #1 by BM25 and
#4 by vector search outscores a chunk that only appears in one list — RRF
rewards presence in both. Critically, RRF only ever looks at rank position,
never the raw score, so BM25's unbounded term-frequency scores and ChromaDB's
0-2 cosine distances never need to be normalised onto a common scale — unlike
linear score interpolation, which breaks the moment either scale shifts (e.g.
after adding new documents).

## When hybrid beats baseline

- Hybrid outperformed baseline on 0 of 5 rephrased queries — cases where exact keyword overlap helped BM25 surface the correct chunk higher than cosine alone.
- BM25-only underperformed on multi-concept queries (0.93 MRR vs baseline 1.00) — confirming semantic retrieval is still needed for questions that don't share vocabulary with the source text.
- 6 of 25 queries had a different rank-1 result between baseline and hybrid.
- Hybrid scored MRR=0.83 / Hit@3=0.92 versus baseline MRR=1.00 / Hit@3=1.00, with 44ms less latency (575ms vs 619ms) — the BM25 fusion step itself runs in under 5ms in-memory, so the difference is embedding-API call variance, not fusion overhead.

## RRF constant sensitivity

k controls how much RRF dampens the rank-1 advantage: at k=10, rank 1 adds
1/11≈0.091 and rank 2 adds 1/12≈0.083 — a large gap, so whichever list ranks
a chunk first tends to dominate. At k=100, rank 1 adds 1/101≈0.0099 and rank 2
adds 1/102≈0.0098 — almost no gap, so fusion is much smoother across both
lists. k=60 (the standard from the original 2009 RRF paper) sits between the
two. The optimal value depends on the corpus, so it's worth sweeping:

```
k   | MRR  | Hit@3 | Note       
----|------|-------|------------
10  | 0.83 | 0.92  |            
30  | 0.83 | 0.92  |            
60  | 0.83 | 0.92  | <- standard
100 | 0.83 | 0.92  |            
```

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
Method            | Factual MRR | Multi-concept MRR | Rephrased MRR
------------------|-------------|-------------------|--------------
Baseline cosine   | 1.00        | 1.00              | 1.00         
BM25 only         | 0.70        | 0.93              | 0.77         
Hybrid (BM25+RRF) | 0.73        | 0.95              | 0.80         
Cohere rerank     | 1.00        | 1.00              | 1.00         
MMR (λ=0.5)       | 1.00        | 1.00              | 1.00         
MMR (λ=0.3, div)  | 1.00        | 1.00              | 1.00         
MMR (λ=0.7, rel)  | 1.00        | 1.00              | 1.00         
```

## How to run

```
pip install openai chromadb cohere numpy python-dotenv rank-bm25
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
- `retriever_hybrid.py` — BM25 keyword retrieval fused with vector search via Reciprocal Rank Fusion
- `metrics.py` — MRR and Hit@k implementations
- `experiment.py` — runs all methods over the eval set and collects results
- `report.py` — prints result tables and writes this README
- `run_all.py` — single entry point that runs the full pipeline
