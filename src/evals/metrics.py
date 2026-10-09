"""Retrieval and Faithfulness Evaluation Metrics.

Computes Hit Rate@k, Recall@k, Mean Reciprocal Rank (MRR@k), and
Faithfulness / Hallucination verification scores against golden benchmarks.
"""

import re
from src.vectorstore.base import SearchResult


def is_chunk_relevant(chunk_content: str, chunk_doc_id: str, expected_doc_id: str | None, key_phrases: list[str]) -> bool:
    """Determine if a retrieved chunk matches ground truth expectations."""
    if expected_doc_id is None:
        # Negative test case (out of domain)
        return False

    # Check document ID match if provided
    if expected_doc_id and chunk_doc_id and expected_doc_id.lower() in chunk_doc_id.lower():
        # If document matches, check phrase overlap
        if not key_phrases:
            return True

    # Check key phrase matches in text content
    content_lower = chunk_content.lower()
    matches = sum(1 for phrase in key_phrases if phrase.lower() in content_lower)
    return matches >= max(1, len(key_phrases) // 2)


def calculate_hit_rate(retrieved: list[SearchResult], expected_doc_id: str | None, key_phrases: list[str], k: int = 5) -> float:
    """Hit Rate@k: Returns 1.0 if at least one relevant passage appears in top-k, else 0.0."""
    top_hits = retrieved[:k]
    for hit in top_hits:
        if is_chunk_relevant(hit.chunk.content, hit.chunk.document_id, expected_doc_id, key_phrases):
            return 1.0
    return 0.0


def calculate_mrr(retrieved: list[SearchResult], expected_doc_id: str | None, key_phrases: list[str], k: int = 5) -> float:
    """Mean Reciprocal Rank (MRR@k): 1 / rank of first relevant hit within top-k."""
    top_hits = retrieved[:k]
    for rank, hit in enumerate(top_hits, start=1):
        if is_chunk_relevant(hit.chunk.content, hit.chunk.document_id, expected_doc_id, key_phrases):
            return 1.0 / rank
    return 0.0


def calculate_recall(retrieved: list[SearchResult], key_phrases: list[str], k: int = 5) -> float:
    """Recall@k: Fraction of key expected regulatory phrases retrieved in top-k."""
    if not key_phrases:
        return 1.0

    aggregated_content = " ".join([r.chunk.content.lower() for r in retrieved[:k]])
    found = sum(1 for phrase in key_phrases if phrase.lower() in aggregated_content)
    return round(found / len(key_phrases), 4)


def calculate_answer_faithfulness(answer_text: str, retrieved_chunks: list[SearchResult]) -> float:
    """Lexical faithfulness: measures fraction of answer sentences grounded in retrieved text."""
    sentences = [s.strip() for s in re.split(r"[.!?]", answer_text) if len(s.strip().split()) >= 4]
    if not sentences:
        return 1.0

    context_text = " ".join([r.chunk.content.lower() for r in retrieved_chunks])
    context_words = set(re.findall(r"\w+", context_text))

    grounded_count = 0
    for sentence in sentences:
        s_words = set(re.findall(r"\w+", sentence.lower()))
        content_words = [w for w in s_words if len(w) > 3]
        if not content_words:
            continue
        overlap = sum(1 for w in content_words if w in context_words)
        if (overlap / len(content_words)) >= 0.50:
            grounded_count += 1

    return round(grounded_count / max(1, len(sentences)), 4)
