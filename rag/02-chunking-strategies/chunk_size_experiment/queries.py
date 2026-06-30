"""
15 test queries for the chunk-size experiment, written against the actual
content of corpus_generator.py's three documents (machine learning
fundamentals, climate science basics, history of the internet).

Each query carries a "type" — factual / multi_concept / rephrased — and a
ground_truth_keywords list: words that MUST appear (case-insensitive
substring match) in a correctly-retrieved chunk. This is what lets retrieval
quality be scored automatically, without a human judge or an LLM grader.
"""

from typing import List, TypedDict


class TestQuery(TypedDict):
    query: str
    ground_truth_keywords: List[str]
    type: str


TEST_QUERIES: List[TestQuery] = [
    # -- 5 factual lookup queries: single fact, minimal context needed -----
    {
        "query": "In what year did ARPANET make its first successful connection, and between which two institutions?",
        "ground_truth_keywords": ["1969", "UCLA"],
        "type": "factual",
    },
    {
        "query": "When was the IPCC established?",
        "ground_truth_keywords": ["IPCC", "1988"],
        "type": "factual",
    },
    {
        "query": "Who invented the World Wide Web, and at what institution?",
        "ground_truth_keywords": ["Tim Berners-Lee", "CERN"],
        "type": "factual",
    },
    {
        "query": "Who developed TCP/IP?",
        "ground_truth_keywords": ["Vint Cerf", "Bob Kahn"],
        "type": "factual",
    },
    {
        "query": "What algorithm did Google use to rank search results?",
        "ground_truth_keywords": ["PageRank", "Google"],
        "type": "factual",
    },

    # -- 5 multi-concept queries: need a fuller passage to answer ----------
    {
        "query": "Explain the bias-variance tradeoff and how regularization or cross-validation help manage it.",
        "ground_truth_keywords": ["bias-variance", "regularization", "cross-validation"],
        "type": "multi_concept",
    },
    {
        "query": "How do thermal expansion and melting ice both contribute to sea level rise?",
        "ground_truth_keywords": ["thermal expansion", "melting", "sea level"],
        "type": "multi_concept",
    },
    {
        "query": "What is the difference between mitigation and adaptation as climate change responses?",
        "ground_truth_keywords": ["mitigation", "adaptation"],
        "type": "multi_concept",
    },
    {
        "query": "How did the dot-com boom turn into the dot-com bust?",
        "ground_truth_keywords": ["dot-com boom", "dot-com bust", "2000"],
        "type": "multi_concept",
    },
    {
        "query": "What role do precision, recall, and the F1 score play in evaluating a classifier?",
        "ground_truth_keywords": ["precision", "recall", "F1"],
        "type": "multi_concept",
    },

    # -- 5 rephrased queries: different wording than the source text -------
    {
        "query": "Why might combining many simple prediction models work better than relying on just one?",
        "ground_truth_keywords": ["random forest", "ensemble"],
        "type": "rephrased",
    },
    {
        "query": "Why does Earth's atmosphere trap heat and keep the planet's surface warm?",
        "ground_truth_keywords": ["greenhouse effect", "infrared"],
        "type": "rephrased",
    },
    {
        "query": "How did the spread of smartphones change the way people connect to the internet?",
        "ground_truth_keywords": ["iPhone", "2007", "mobile"],
        "type": "rephrased",
    },
    {
        "query": "What economic tools can governments use to make polluting more expensive?",
        "ground_truth_keywords": ["carbon tax", "cap-and-trade"],
        "type": "rephrased",
    },
    {
        "query": "How can a model reuse knowledge learned from one task to perform well on a different, smaller task?",
        "ground_truth_keywords": ["transfer learning", "fine-tune"],
        "type": "rephrased",
    },
]
