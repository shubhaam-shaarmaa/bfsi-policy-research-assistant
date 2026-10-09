"""Unit tests for Stage 5: Grounded Generation, Citations, and Hallucination Verification."""

import pytest
from src.core.models import Chunk, ChunkingStrategy, DocumentType
from src.generation.context_builder import ContextBuilder
from src.generation.generator import GroundedGenerator
from src.generation.mock_provider import MockLLMProvider
from src.generation.models import Citation
from src.generation.verifier import CitationVerifier
from src.vectorstore.base import SearchResult


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    chunk1 = Chunk(
        chunk_id="chunk_rbi_dlg#001",
        document_id="doc_rbi_dlg",
        document_title="Default Loss Guarantee in Digital Lending",
        issuer="Reserve Bank of India (RBI)",
        circular_number="RBI/2023-24/41",
        date="2023-06-08",
        doc_type=DocumentType.CIRCULAR,
        page_number=2,
        section_heading="Structure of DLG",
        clause_number="3.1",
        content="Regulated Entities shall ensure total DLG cover does not exceed 5% of loan portfolio.",
        content_hash="hash_dlg_01",
        chunk_index=0,
        token_count=18,
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
    )
    chunk2 = Chunk(
        chunk_id="chunk_rbi_dlg#002",
        document_id="doc_rbi_dlg",
        document_title="Default Loss Guarantee in Digital Lending",
        issuer="Reserve Bank of India (RBI)",
        circular_number="RBI/2023-24/41",
        date="2023-06-08",
        doc_type=DocumentType.CIRCULAR,
        page_number=2,
        section_heading="Structure of DLG",
        clause_number="3.2",
        content="In case of implicit guarantee arrangements, the 5% cap remains strictly applicable.",
        content_hash="hash_dlg_02",
        chunk_index=1,
        token_count=16,
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
    )
    return [
        SearchResult(chunk=chunk1, score=0.92, source="hybrid", rank=1),
        SearchResult(chunk=chunk2, score=0.85, source="hybrid", rank=2),
    ]


def test_context_builder_budget_and_deduplication(sample_search_results: list[SearchResult]):
    builder = ContextBuilder(max_context_tokens=25)
    # With budget=25, only first chunk (18 tokens) should fit; second chunk (16 tokens) would exceed 25
    context_str, included = builder.build_context(sample_search_results)

    assert len(included) == 1
    assert "SOURCE #1" in context_str
    assert "does not exceed 5% of loan portfolio" in context_str


def test_citation_verifier_detects_real_and_hallucinated_quotes(sample_search_results: list[SearchResult]):
    verifier = CitationVerifier(min_similarity_threshold=0.70)
    chunks = [r.chunk for r in sample_search_results]

    # 1. Genuine quote
    real_citation = Citation(
        document_title="Default Loss Guarantee in Digital Lending",
        circular_number="RBI/2023-24/41",
        page_number=2,
        section_heading="Structure of DLG",
        clause_number="3.1",
        quote="Regulated Entities shall ensure total DLG cover does not exceed 5%",
    )
    v1 = verifier.verify_citation(real_citation, chunks)
    assert v1.verified is True
    assert v1.similarity_match > 0.8

    # 2. Fabricated quote
    fake_citation = Citation(
        document_title="Default Loss Guarantee in Digital Lending",
        circular_number="RBI/2023-24/41",
        page_number=2,
        section_heading="Structure of DLG",
        clause_number="3.1",
        quote="Banks may offer up to 25% default guarantee without prior RBI approval.",
    )
    v2 = verifier.verify_citation(fake_citation, chunks)
    assert v2.verified is False


def test_mock_llm_provider_grounded_answer(sample_search_results: list[SearchResult]):
    provider = MockLLMProvider()
    builder = ContextBuilder()
    context_str, _ = builder.build_context(sample_search_results)

    prompt = f"QUESTION: What is the cap on DLG?\n\nCONTEXT:\n{context_str}"
    response = provider.generate([{"role": "user", "content": prompt}], system_prompt="")

    import json
    data = json.loads(response.content)
    assert data["has_sufficient_context"] is True
    assert "Default Loss Guarantee" in data["answer"]
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["circular_number"] == "RBI/2023-24/41"


def test_mock_llm_provider_refusal_when_context_missing():
    provider = MockLLMProvider()
    # Query about cryptocurrency in banking when context contains nothing about it
    prompt = "QUESTION: Are banks permitted to hold Bitcoin reserves?\n\nCONTEXT:\n[SOURCE #1 | Doc: SOP | Ref: None | Page: 1 | Section: Cutoff | Clause: 1]\nRTGS settlements cut off at 16:30 hours."
    response = provider.generate([{"role": "user", "content": prompt}], system_prompt="")

    import json
    data = json.loads(response.content)
    assert data["has_sufficient_context"] is False
    assert "do not contain" in data["answer"].lower()
    assert len(data["citations"]) == 0


def test_grounded_generator_end_to_end(sample_search_results: list[SearchResult]):
    generator = GroundedGenerator(llm_provider=MockLLMProvider())
    result = generator.generate_answer(
        query="What is the permissible cap on DLG portfolio?",
        retrieved_results=sample_search_results,
    )

    assert result.has_sufficient_context is True
    assert len(result.citations) >= 1
    assert result.citations[0].verified is True
    assert result.verification_passed is True
    assert result.latency_ms > 0.0
    assert len(result.retrieved_chunks) == 2
