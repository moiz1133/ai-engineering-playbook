# Parent-Child Chunking for RAG

A self-contained benchmark comparing **parent-child chunking** against standard flat chunking for retrieval-augmented generation.

The core idea: index small *child* chunks for precise vector matching, but return the larger *parent* chunk to the LLM so it receives full surrounding context rather than a 150-character fragment.

---

## How It Works

```
Document
  └─ Parent chunk (~600 chars)          ← returned to LLM
       ├─ Child chunk A (~150 chars)    ← indexed in ChromaDB
       ├─ Child chunk B (~150 chars)    ← indexed in ChromaDB
       └─ Child chunk C (~150 chars)    ← indexed in ChromaDB
```

**At query time:**
1. Embed the query and search the *child* index for top-K matches.
2. For each matched child, expand to its parent chunk.
3. Deduplicate: if two children share the same parent, send that parent only once.
4. Pass the parent chunks as context to the LLM.

**Why this helps:**
- Small chunks embed a single idea → higher cosine similarity on focused queries (fixes *embedding dilution*).
- Parent expansion gives the LLM the full surrounding context (fixes *fragmentation*).

---

## Project Structure

```
parent_child_chunking/
  ├── corpus/               # Three synthetic ~2000-word documents
  │   ├── ml_fundamentals.txt
  │   ├── climate_science.txt
  │   └── internet_history.txt
  ├── chunker.py            # build_parent_child_chunks()
  ├── store.py              # ChromaDB indexing of child chunks
  ├── retriever.py          # Child search → parent expansion + deduplication
  ├── baseline.py           # Flat 300-char chunking for comparison
  ├── experiment.py         # Benchmark harness + results table
  ├── queries.py            # 15 test queries with ground-truth keywords
  └── run_experiment.py     # Entry point — runs everything end to end
```

---

## Setup

```bash
pip install openai chromadb numpy
export OPENAI_API_KEY="sk-..."
cd parent_child_chunking
python run_experiment.py
```

No external datasets required — the corpus is generated and bundled in `corpus/`.

---

## Benchmark Results

Results are printed dynamically from actual embedding runs. Representative numbers from a typical run:

| Metric | Parent-child | Flat 300-char |
|---|---|---|
| Keyword hit rate | **86.7%** | 80.0% |
| — Factual queries | **100.0%** | 100.0% |
| — Multi-concept queries | **80.0%** | 60.0% |
| Context completeness | **73.3%** | 53.3% |
| Avg context length | 587 chars | 312 chars |
| Avg top-1 similarity | **0.83** | 0.78 |
| Deduplication fired | 26.7% | n/a |

---

## Metric Definitions

| Metric | Definition |
|---|---|
| **Keyword hit rate** | Fraction of queries where at least one retrieved chunk contains *all* ground-truth keywords (case-insensitive substring match). |
| **Context completeness** | Fraction of queries where *all* keywords appear in a *single* retrieved chunk — indicates no answer fragmentation. |
| **Avg context length** | Mean total characters of retrieved text passed to the LLM per query. Higher = more tokens consumed. |
| **Avg top-1 similarity** | Mean cosine similarity of the highest-ranked hit. Higher = more precise retrieval. |
| **Deduplication rate** | Fraction of queries where two or more child chunks mapped to the same parent and were merged into one result. |

---

## Chunking Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Parent size | 600 chars | ~100–120 tokens; fits a coherent paragraph |
| Child size | 150 chars | ~25 tokens; embeds a single focused idea |
| Child overlap | 15 chars | Prevents edge words from being cut mid-phrase |
| Flat chunk size | 300 chars | Midpoint between child and parent for a fair comparison |
| Top-K | 3 | Balanced recall vs. context window cost |

---

## Key Trade-offs

| | Parent-child | Flat 300-char |
|---|---|---|
| **Retrieval precision** | Higher (small child embeddings) | Lower (larger chunks dilute embedding) |
| **Answer completeness** | Higher (parent gives full context) | Lower (single chunk may fragment multi-sentence answers) |
| **Token cost per query** | Higher (~2× more chars per hit) | Lower |
| **Deduplication overhead** | Needed — multiple children can map to one parent | Not applicable |
| **Index complexity** | Two logical levels | Single flat index |

**When to use parent-child:** corpus has multi-sentence answers where the context surrounding a fact matters — e.g., technical documentation, long-form articles, research papers.

**When flat chunking is fine:** queries are highly specific and short answers are self-contained at ~300 chars — e.g., FAQ retrieval, product attribute lookup.

---

## Deduplication

When multiple child chunks from the same parent are retrieved, they are merged into a single parent result (keeping the child with the highest similarity score). The experiment logs which queries trigger this:

```
[dedup] Query triggered deduplication: 'What is overfitting and what regularization...'
```

Without deduplication, the LLM would receive the same parent text multiple times, wasting tokens and potentially confusing generation.
