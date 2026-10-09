"""Unit tests for Stage 3: Embeddings & Vector Stores (ChromaDB and Qdrant)."""

import math
from pathlib import Path
import pytest

from src.core.models import (
    Chunk,
    ChunkingStrategy,
    DocumentType,
)
from src.embeddings.mock_provider import MockEmbeddingProvider
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.pipeline import VectorIndexingPipeline
from src.vectorstore.qdrant_store import QdrantVectorStore


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc_rbi_01#c000",
            document_id="doc_rbi_01",
            document_title="Digital Lending Default Loss Guarantee",
            issuer="Reserve Bank of India (RBI)",
            circular_number="RBI/2023-24/41",
            date="2023-06-08",
            doc_type=DocumentType.CIRCULAR,
            page_number=1,
            section_heading="Scope and Applicability",
            clause_number="1.0",
            content="Guidelines apply to all commercial banks and NBFCs engaging in digital lending.",
            content_hash="hash_chunk_001",
            chunk_index=0,
            token_count=15,
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
        ),
        Chunk(
            chunk_id="doc_rbi_01#c001",
            document_id="doc_rbi_01",
            document_title="Digital Lending Default Loss Guarantee",
            issuer="Reserve Bank of India (RBI)",
            circular_number="RBI/2023-24/41",
            date="2023-06-08",
            doc_type=DocumentType.CIRCULAR,
            page_number=2,
            section_heading="Cap on DLG",
            clause_number="3.1",
            content="Regulated Entities shall ensure total DLG cover does not exceed 5% of loan portfolio.",
            content_hash="hash_chunk_002",
            chunk_index=1,
            token_count=17,
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
        ),
        Chunk(
            chunk_id="doc_internal_sop#c000",
            document_id="doc_internal_sop",
            document_title="Internal Payment Settlement SOP",
            issuer="Internal Bank Ops",
            circular_number=None,
            date="2024-01-15",
            doc_type=DocumentType.SOP,
            page_number=1,
            section_heading="Settlement Cutoff",
            clause_number="2.0",
            content="RTGS settlements cut off at 16:30 hours daily for inter-bank batch reconciliation.",
            content_hash="hash_chunk_003",
            chunk_index=0,
            token_count=14,
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
        ),
    ]


def test_mock_embedding_provider_properties():
    embedder = MockEmbeddingProvider(dimension=128)
    vec1 = embedder.embed_query("What is the cap on DLG?")
    vec2 = embedder.embed_query("What is the cap on DLG?")

    assert len(vec1) == 128
    assert vec1 == vec2  # Deterministic
    # Unit normalization check
    mag = math.sqrt(sum(x * x for x in vec1))
    assert abs(mag - 1.0) < 0.01


def test_chroma_store_upsert_search_and_filter(sample_chunks: list[Chunk]):
    embedder = MockEmbeddingProvider(dimension=64)
    store = ChromaVectorStore(collection_name="test_chroma_coll", dimension=64, in_memory=True)

    vectors = embedder.embed_documents([c.content for c in sample_chunks])
    count = store.upsert_chunks(sample_chunks, vectors)
    assert count == 3
    assert store.count() == 3

    # Idempotent re-upsert should not increase count
    store.upsert_chunks(sample_chunks, vectors)
    assert store.count() == 3

    # Test search with filter
    q_vec = embedder.embed_query("DLG portfolio cap")
    results = store.search(q_vec, top_k=2, filters={"issuer": "Reserve Bank of India (RBI)"})
    assert len(results) == 2
    for r in results:
        assert r.chunk.issuer == "Reserve Bank of India (RBI)"
        assert 0.0 <= r.score <= 1.0

    # Test delete
    deleted = store.delete_document("doc_internal_sop")
    assert deleted == 1
    assert store.count() == 2


def test_qdrant_store_upsert_search_and_filter(sample_chunks: list[Chunk]):
    embedder = MockEmbeddingProvider(dimension=64)
    store = QdrantVectorStore(collection_name="test_qdrant_coll", dimension=64, in_memory=True)

    vectors = embedder.embed_documents([c.content for c in sample_chunks])
    count = store.upsert_chunks(sample_chunks, vectors)
    assert count == 3
    assert store.count() == 3

    # Idempotent re-upsert
    store.upsert_chunks(sample_chunks, vectors)
    assert store.count() == 3

    # Search with filter
    q_vec = embedder.embed_query("RTGS settlement cutoff")
    results = store.search(q_vec, top_k=2, filters={"doc_type": "sop"})
    assert len(results) >= 1
    assert results[0].chunk.document_id == "doc_internal_sop"
    assert results[0].chunk.doc_type == DocumentType.SOP

    # Test delete
    store.delete_document("doc_rbi_01")
    assert store.count() == 1


def test_vector_indexing_pipeline(sample_chunks: list[Chunk], tmp_path: Path):
    embedder = MockEmbeddingProvider(dimension=32)
    store = ChromaVectorStore(collection_name="test_pipeline_coll", dimension=32, in_memory=True)
    manifest = tmp_path / "manifest.json"

    pipeline = VectorIndexingPipeline(
        vector_store=store,
        embedding_provider=embedder,
        manifest_path=manifest,
    )

    indexed = pipeline.index_chunks(sample_chunks)
    assert indexed == 3
    assert manifest.exists()

    # Re-indexing same chunks should index 0 due to manifest
    re_indexed = pipeline.index_chunks(sample_chunks)
    assert re_indexed == 0

    # Search through pipeline
    hits = pipeline.search("5% cap on DLG", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
