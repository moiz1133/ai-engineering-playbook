"""Single entry point: build the corpus/index, annotate ground truth, run all
retrieval experiments, print results, and write README.md + results.json.
"""

from __future__ import annotations

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cohere
from dotenv import load_dotenv
from openai import OpenAI

from corpus_builder import build_index, generate_corpus
from eval_set import EVAL_SET, ground_truth_chunk_ids
from experiment import run_all_experiments
from report import generate_readme, print_results_table

# Repo-root .env (gitignored) holds OPENAI_API_KEY / COHERE_API_KEY - load it
# here so running `python run_all.py` works without exporting vars manually.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))

if __name__ == "__main__":
    openai_client = OpenAI()

    cohere_api_key = os.environ.get("COHERE_API_KEY")
    cohere_client = cohere.Client(cohere_api_key) if cohere_api_key else None

    corpus_paths = generate_corpus("./corpus")
    collection = build_index(corpus_paths, openai_client)

    eval_set = ground_truth_chunk_ids(EVAL_SET, collection)
    answerable = sum(1 for e in eval_set if e["relevant_chunk_ids"])
    print(f"Annotated {len(eval_set)} items — {answerable} answerable")

    results = run_all_experiments(collection, openai_client, cohere_client, eval_set)

    print_results_table(results)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    generate_readme(results, "README.md")
    print("README.md written with actual experiment results")
