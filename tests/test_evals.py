"""Unit tests for Stage 7: Evaluation Metrics."""

from src.core.models import Chunk, ChunkingStrategy, DocumentType
from src.evals.metrics import (
    calculate_answer_faithfulness,
    calculate_hit_rate,
    calculate_mrr,
    calculate_recall,
)
from src.vectorstore.base import SearchResult


def test_eval_metrics_hit_rate_and_mrr():
    chunk = Chunk(
        chunk_id="chunk_01",
        document_id="doc_rbi_dlg",
        document_title="DLG Guidelines",
        issuer="RBI",
        circular_number="RBI/2023-24/41",
        date="2023-06-08",
        doc_type=DocumentType.CIRCULAR,
        page_number=1,
        section_heading="Scope",
        clause_number="1.0",
        content="Regulated Entities shall ensure total DLG cover does not exceed 5% of loan portfolio.",
        content_hash="h1",
        chunk_index=0,
        token_count=15,
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
    )

    results = [SearchResult(chunk=chunk, score=0.9, source="hybrid", rank=1)]
    key_phrases = ["does not exceed 5%", "loan portfolio"]

    # Rank 1 hit
    hit_1 = calculate_hit_rate(results, "doc_rbi_dlg", key_phrases, k=1)
    assert hit_1 == 1.0

    mrr = calculate_mrr(results, "doc_rbi_dlg", key_phrases, k=5)
    assert mrr == 1.0

    recall = calculate_recall(results, key_phrases, k=5)
    assert recall == 1.0


def test_eval_metrics_miss():
    chunk = Chunk(
        chunk_id="chunk_02",
        document_id="doc_sop",
        document_title="SOP",
        issuer="Bank",
        circular_number=None,
        date="2024-01-01",
        doc_type=DocumentType.SOP,
        page_number=1,
        section_heading="Cutoff",
        clause_number="1.0",
        content="RTGS cutoff is 16:30 hours.",
        content_hash="h2",
        chunk_index=0,
        token_count=6,
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
    )
    results = [SearchResult(chunk=chunk, score=0.9, source="dense", rank=1)]
    key_phrases = ["DLG cover", "5% cap"]

    hit = calculate_hit_rate(results, "doc_rbi_dlg", key_phrases, k=1)
    assert hit == 0.0

    mrr = calculate_mrr(results, "doc_rbi_dlg", key_phrases, k=5)
    assert mrr == 0.0


def test_answer_faithfulness():
    chunk = Chunk(
        chunk_id="chunk_01",
        document_id="doc_01",
        document_title="Test",
        issuer="RBI",
        circular_number=None,
        date=None,
        doc_type=DocumentType.CIRCULAR,
        page_number=1,
        section_heading="Heading",
        clause_number=None,
        content="Default loss guarantee is capped strictly at five percent of the overall credit portfolio.",
        content_hash="h1",
        chunk_index=0,
        token_count=15,
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
    )
    results = [SearchResult(chunk=chunk, score=0.9, source="hybrid", rank=1)]

    faithful_answer = "Default loss guarantee is capped strictly at five percent of credit portfolio."
    score = calculate_answer_faithfulness(faithful_answer, results)
    assert score >= 0.8
