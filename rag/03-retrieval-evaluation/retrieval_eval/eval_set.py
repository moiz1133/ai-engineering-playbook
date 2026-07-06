"""25 annotated Q&A pairs used to evaluate all three retrieval methods.

WHAT: relevant_chunk_ids = the ground truth for MRR and Hit@k calculation
WHY: automated annotation — no human judges needed; deterministic and reproducible
"""

from __future__ import annotations

from typing import Dict, List

import chromadb

EVAL_SET: List[dict] = [
    # ---- machine_learning.txt (5) ----
    {
        "id": 1,
        "question": "What kind of training data does supervised learning use?",
        "answer_keywords": ["labeled", "supervised learning"],
        "relevant_doc": "machine_learning.txt",
        "query_type": "factual",
    },
    {
        "id": 2,
        "question": "What algorithm is commonly used for clustering in unsupervised learning?",
        "answer_keywords": ["k-means", "clustering"],
        "relevant_doc": "machine_learning.txt",
        "query_type": "factual",
    },
    {
        "id": 3,
        "question": "How do precision, recall, and F1 score relate to each other?",
        "answer_keywords": ["precision", "recall", "F1"],
        "relevant_doc": "machine_learning.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 4,
        "question": "What causes overfitting and how does dropout help address it?",
        "answer_keywords": ["overfitting", "dropout"],
        "relevant_doc": "machine_learning.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 5,
        "question": "How does a neural network figure out how much to adjust each connection weight after making an error?",
        "answer_keywords": ["chain rule", "gradient descent"],
        "relevant_doc": "machine_learning.txt",
        "query_type": "rephrased",
    },
    # ---- climate_science.txt (5) ----
    {
        "id": 6,
        "question": "What causes the greenhouse effect?",
        "answer_keywords": ["greenhouse effect", "carbon dioxide"],
        "relevant_doc": "climate_science.txt",
        "query_type": "factual",
    },
    {
        "id": 7,
        "question": "What is the ice-albedo feedback loop?",
        "answer_keywords": ["ice-albedo", "feedback loop"],
        "relevant_doc": "climate_science.txt",
        "query_type": "factual",
    },
    {
        "id": 8,
        "question": "How do permafrost thaw and methane release interact to create a tipping point?",
        "answer_keywords": ["permafrost", "methane", "tipping point"],
        "relevant_doc": "climate_science.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 9,
        "question": "What role does the AMOC play in ocean circulation and climate tipping points?",
        "answer_keywords": ["AMOC", "ocean currents"],
        "relevant_doc": "climate_science.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 10,
        "question": "What happens to Earth's climate system when certain thresholds are crossed permanently?",
        "answer_keywords": ["tipping point", "irreversible"],
        "relevant_doc": "climate_science.txt",
        "query_type": "rephrased",
    },
    # ---- internet_history.txt (5) ----
    {
        "id": 11,
        "question": "What was ARPANET and what networking technique did it pioneer?",
        "answer_keywords": ["ARPANET", "packet switching"],
        "relevant_doc": "internet_history.txt",
        "query_type": "factual",
    },
    {
        "id": 12,
        "question": "Who invented the World Wide Web?",
        "answer_keywords": ["Tim Berners-Lee", "World Wide Web"],
        "relevant_doc": "internet_history.txt",
        "query_type": "factual",
    },
    {
        "id": 13,
        "question": "How did the TCP/IP protocol suite enable different networks to interconnect?",
        "answer_keywords": ["TCP/IP", "interconnect"],
        "relevant_doc": "internet_history.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 14,
        "question": "How did broadband technologies like DSL and cable change home internet access?",
        "answer_keywords": ["DSL", "cable internet"],
        "relevant_doc": "internet_history.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 15,
        "question": "How did the rise of smartphones change how people access the internet?",
        "answer_keywords": ["smartphones", "mobile web"],
        "relevant_doc": "internet_history.txt",
        "query_type": "rephrased",
    },
    # ---- llm_systems.txt (5) ----
    {
        "id": 16,
        "question": "What mechanism allows transformers to weigh the importance of different tokens?",
        "answer_keywords": ["self-attention", "transformer"],
        "relevant_doc": "llm_systems.txt",
        "query_type": "factual",
    },
    {
        "id": 17,
        "question": "What does RLHF stand for and what is it used for?",
        "answer_keywords": ["RLHF", "human feedback"],
        "relevant_doc": "llm_systems.txt",
        "query_type": "factual",
    },
    {
        "id": 18,
        "question": "How does retrieval-augmented generation reduce hallucination?",
        "answer_keywords": ["Retrieval-augmented generation", "hallucination"],
        "relevant_doc": "llm_systems.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 19,
        "question": "How do LLM agents use tool calling to complete multi-step tasks?",
        "answer_keywords": ["agents", "external tools"],
        "relevant_doc": "llm_systems.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 20,
        "question": "Why do language models sometimes confidently state incorrect facts?",
        "answer_keywords": ["hallucination", "confidently"],
        "relevant_doc": "llm_systems.txt",
        "query_type": "rephrased",
    },
    # ---- database_systems.txt (5) ----
    {
        "id": 21,
        "question": "What does ACID stand for in database transactions?",
        "answer_keywords": ["ACID", "atomicity"],
        "relevant_doc": "database_systems.txt",
        "query_type": "factual",
    },
    {
        "id": 22,
        "question": "What data structure is commonly used for indexing in relational databases?",
        "answer_keywords": ["B-tree", "index"],
        "relevant_doc": "database_systems.txt",
        "query_type": "factual",
    },
    {
        "id": 23,
        "question": "How do document, key-value, and graph databases differ within the NoSQL family?",
        "answer_keywords": ["document databases", "key-value stores"],
        "relevant_doc": "database_systems.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 24,
        "question": "How do vector databases use approximate nearest neighbor search like HNSW to find similar embeddings?",
        "answer_keywords": ["HNSW", "approximate nearest neighbor"],
        "relevant_doc": "database_systems.txt",
        "query_type": "multi_concept",
    },
    {
        "id": 25,
        "question": "What guarantees ensure that a bank transfer either fully completes or doesn't happen at all?",
        "answer_keywords": ["Atomicity", "bank accounts"],
        "relevant_doc": "database_systems.txt",
        "query_type": "rephrased",
    },
]


def ground_truth_chunk_ids(eval_set: List[dict], collection: chromadb.Collection) -> List[dict]:
    """Scan the index and attach relevant_chunk_ids to each eval item.

    A chunk is relevant if it contains ALL of the item's answer_keywords
    (case-insensitive substring match). Items with no matching chunk are
    left with an empty list and are excluded from metric computation.
    """
    all_data = collection.get(include=["documents", "metadatas"])
    chunk_ids: List[str] = all_data["ids"]
    chunk_texts: List[str] = all_data["documents"]

    annotated: List[dict] = []
    unanswerable = 0

    for item in eval_set:
        keywords_lower = [kw.lower() for kw in item["answer_keywords"]]
        matches = [
            chunk_id
            for chunk_id, text in zip(chunk_ids, chunk_texts)
            if all(kw in text.lower() for kw in keywords_lower)
        ]
        new_item = dict(item)
        new_item["relevant_chunk_ids"] = matches
        if not matches:
            unanswerable += 1
            print(f"WARNING: eval item {item['id']} ({item['question']!r}) has no "
                  f"matching chunk for keywords {item['answer_keywords']} — excluded from metrics")
        annotated.append(new_item)

    print(f"Ground truth annotation complete: {len(annotated) - unanswerable}/{len(annotated)} "
          f"answerable, {unanswerable} unanswerable")
    return annotated
