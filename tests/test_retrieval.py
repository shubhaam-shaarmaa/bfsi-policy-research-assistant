"""Unit tests for Stage 4: Retrieval Quality (Dense, BM25, Hybrid RRF, and Reranking)."""

import pytest
from src.core.models import Chunk, ChunkingStrategy, DocumentType
from src.embeddings.mock_provider import MockEmbeddingProvider
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.engine import RetrievalEngine
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.mock_reranker import MockReranker
from src.vectorstore.base import SearchResult
from src.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def sample_regulatory_chunks() -> list[Chunk]:
    return [
        Chunk(
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
        ),
        Chunk(
            chunk_id="chunk_rbi_it#002",
            document_id="doc_rbi_it",
            document_title="Master Direction on IT Governance",
            issuer="Reserve Bank of India (RBI)",
            circular_number="RBI/2023-24/107",
            date="2023-11-07",
            doc_type=DocumentType.MASTER_DIRECTION,
            page_number=4,
            section_heading="Cyber Security Governance",
            clause_number="5.2",
            content="The Board shall establish an IT Strategy Committee to oversee cyber risk posture.",
            content_hash="hash_it_02",
            chunk_index=1,
            token_count=16,
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
        ),
        Chunk(
            chunk_id="chunk_sop_settle#003",
            document_id="doc_sop_settle",
            document_title="Payment Gateway Settlement SOP",
            issuer="Internal Bank Ops",
            circular_number=None,
            date="2024-02-01",
            doc_type=DocumentType.SOP,
            page_number=1,
            section_heading="Settlement Schedules",
            clause_number="1.1",
            content="Merchant payment settlement executes on a T+1 batch cycle via NEFT or RTGS.",
            content_hash="hash_sop_03",
            chunk_index=0,
            token_count=16,
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
        ),
    ]


def test_bm25_retriever_exact_acronym_match(sample_regulatory_chunks: list[Chunk]):
    bm25 = BM25Retriever(sample_regulatory_chunks)

    # Query with exact regulatory circular number
    results = bm25.search("RBI/2023-24/41", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.circular_number == "RBI/2023-24/41"
    assert results[0].source == "bm25"
    assert results[0].score > 0.0

    # Query with specific numerical term
    cap_results = bm25.search("5% loan portfolio", top_k=1)
    assert len(cap_results) == 1
    assert "5%" in cap_results[0].chunk.content


def test_hybrid_reciprocal_rank_fusion(sample_regulatory_chunks: list[Chunk]):
    fusion = HybridRetriever(k=60)

    # Simulate dense results (doc 2 ranked first, doc 1 ranked second)
    dense_res = [
        SearchResult(chunk=sample_regulatory_chunks[1], score=0.9, source="dense", rank=1),
        SearchResult(chunk=sample_regulatory_chunks[0], score=0.7, source="dense", rank=2),
    ]

    # Simulate BM25 results (doc 1 ranked first, doc 2 ranked second)
    bm25_res = [
        SearchResult(chunk=sample_regulatory_chunks[0], score=0.85, source="bm25", rank=1),
        SearchResult(chunk=sample_regulatory_chunks[1], score=0.60, source="bm25", rank=2),
    ]

    fused = fusion.reciprocal_rank_fusion(dense_res, bm25_res, top_k=2)
    assert len(fused) == 2
    for item in fused:
        assert item.source == "hybrid"
        assert item.score > 0.0


def test_mock_reranker(sample_regulatory_chunks: list[Chunk]):
    reranker = MockReranker()
    candidates = [
        SearchResult(chunk=sample_regulatory_chunks[1], score=0.5, source="dense", rank=1),
        SearchResult(chunk=sample_regulatory_chunks[0], score=0.4, source="dense", rank=2),
    ]

    query = "What is the 5% portfolio cap in DLG?"
    reranked = reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    # The chunk containing '5%' and 'DLG' should be elevated to rank 1
    assert reranked[0].chunk.chunk_id == "chunk_rbi_dlg#001"
    assert reranked[0].source == "reranked"


def test_retrieval_engine_modes(sample_regulatory_chunks: list[Chunk]):
    embedder = MockEmbeddingProvider(dimension=32)
    vstore = ChromaVectorStore(collection_name="test_engine_coll", dimension=32, in_memory=True)
    vectors = embedder.embed_documents([c.content for c in sample_regulatory_chunks])
    vstore.upsert_chunks(sample_regulatory_chunks, vectors)

    bm25 = BM25Retriever(sample_regulatory_chunks)
    reranker = MockReranker()

    engine = RetrievalEngine(
        vector_store=vstore,
        embedding_provider=embedder,
        reranker=reranker,
        bm25_retriever=bm25,
    )

    # Test Dense Search
    dense_hits = engine.search("cyber risk", mode="dense", top_k=2)
    assert len(dense_hits) <= 2

    # Test BM25 Search
    bm25_hits = engine.search("RBI/2023-24/107", mode="bm25", top_k=2)
    assert len(bm25_hits) >= 1
    assert bm25_hits[0].chunk.circular_number == "RBI/2023-24/107"

    # Test Hybrid Search with Reranker
    hybrid_hits = engine.search("5% cap on DLG", mode="hybrid", use_reranker=True, top_k=2)
    assert len(hybrid_hits) >= 1
    assert hybrid_hits[0].source == "reranked"
