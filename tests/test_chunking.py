"""Unit tests for Stage 2 Chunking Strategies and Pipeline."""

from pathlib import Path
import pytest
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.pipeline import ChunkingPipeline, get_chunker
from src.chunking.recursive_chunker import RecursiveCharacterChunker
from src.chunking.structure_chunker import StructureAwareChunker
from src.core.models import (
    ChunkingStrategy,
    DocumentMetadata,
    DocumentType,
    ExtractedSection,
    ParsedDocument,
)


@pytest.fixture
def sample_rbi_document() -> ParsedDocument:
    """Fixture providing a realistic parsed RBI circular with sections and clauses."""
    metadata = DocumentMetadata(
        doc_id="doc_rbi_dlg_test",
        title="Guidelines on Default Loss Guarantee in Digital Lending",
        issuer="Reserve Bank of India (RBI)",
        circular_number="RBI/2023-24/41",
        department="Department of Regulation",
        date="June 08, 2023",
        doc_type=DocumentType.CIRCULAR,
        source_filename="rbi_dlg.pdf",
        source_path="/path/to/rbi_dlg.pdf",
        file_hash="mock_hash_1234567890abcdef",
        file_format="pdf",
        total_pages=3,
    )

    sections = [
        ExtractedSection(
            section_id="sec_01",
            heading="1. Scope and Applicability",
            clause_number="1.0",
            page_number=1,
            content=(
                "These guidelines are applicable to all Commercial Banks (including Small Finance Banks), "
                "Co-operative Banks, and Non-Banking Financial Companies (NBFCs) engaging in digital lending."
            ),
        ),
        ExtractedSection(
            section_id="sec_02",
            heading="3. Structure of Default Loss Guarantee (DLG)",
            clause_number="3.1",
            page_number=2,
            content=(
                "3.1 Regulated Entities (RE) shall ensure that total DLG cover on any outstanding portfolio "
                "which is specified in the DLG agreement shall not exceed 5% of the amount of that loan portfolio.\n"
                "3.2 In case of implicit guarantee arrangements, the RE shall ensure that service provider "
                "shall not agree to performance obligations exceeding the 5% cap.\n"
                "3.3 RE shall not enter into DLG arrangements with an entity which is an unpaid defaulter."
            ),
        ),
    ]

    return ParsedDocument(
        metadata=metadata,
        sections=sections,
        raw_text="\n\n".join(s.content for s in sections),
        page_count=3,
    )


def test_fixed_chunker_splits_and_preserves_metadata(sample_rbi_document: ParsedDocument):
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10, min_tokens=5)
    chunks = chunker.split(sample_rbi_document)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "doc_rbi_dlg_test"
        assert chunk.document_title == "Guidelines on Default Loss Guarantee in Digital Lending"
        assert chunk.circular_number == "RBI/2023-24/41"
        assert chunk.strategy == ChunkingStrategy.FIXED
        assert chunk.token_count > 0
        assert len(chunk.content_hash) == 64  # SHA-256


def test_recursive_chunker_respects_paragraph_boundaries(sample_rbi_document: ParsedDocument):
    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=10, min_tokens=5)
    chunks = chunker.split(sample_rbi_document)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.strategy == ChunkingStrategy.RECURSIVE
        assert chunk.page_number in [1, 2]
        assert chunk.content != ""


def test_structure_aware_chunker_injects_breadcrumbs(sample_rbi_document: ParsedDocument):
    chunker = StructureAwareChunker(chunk_size=300, min_tokens=10, inject_context_header=True)
    chunks = chunker.split(sample_rbi_document)

    assert len(chunks) >= 2
    # Check that first chunk contains regulatory context breadcrumb
    first_chunk = chunks[0]
    assert "[Issuer: Reserve Bank of India (RBI)" in first_chunk.content
    assert "Ref: RBI/2023-24/41" in first_chunk.content
    assert "Doc: Guidelines on Default Loss Guarantee in Digital Lending" in first_chunk.content
    assert first_chunk.strategy == ChunkingStrategy.STRUCTURE_AWARE
    assert first_chunk.clause_number == "1.0"
    assert first_chunk.page_number == 1

    # Check second chunk captures Clause 3 context
    dlg_chunk = [c for c in chunks if "3.1 Regulated Entities" in c.content][0]
    assert dlg_chunk.page_number == 2
    assert "Section: 3. Structure of Default Loss Guarantee" in dlg_chunk.content
    assert dlg_chunk.metadata.get("structure_type") in ["atomic_clause", "clause_group"]


def test_structure_aware_subclause_splitting(sample_rbi_document: ParsedDocument):
    # Set small chunk size to force sub-clause division
    chunker = StructureAwareChunker(chunk_size=40, min_tokens=5, inject_context_header=True)
    chunks = chunker.split(sample_rbi_document)

    # Should split section 2 into multiple sub-clause chunks
    sec2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(sec2_chunks) >= 2
    for c in sec2_chunks:
        assert c.document_id == "doc_rbi_dlg_test"
        assert "[Issuer: Reserve Bank of India (RBI)" in c.content


def test_chunking_pipeline_persistence(sample_rbi_document: ParsedDocument, tmp_path: Path):
    pipeline = ChunkingPipeline(
        strategy="structure_aware",
        chunk_size=200,
        output_dir=tmp_path,
    )
    chunks = pipeline.chunk_document(sample_rbi_document)

    assert len(chunks) > 0
    saved_file = tmp_path / f"chunks_{sample_rbi_document.metadata.doc_id}.json"
    assert saved_file.exists()
    assert saved_file.stat().st_size > 0


def test_get_chunker_factory():
    assert isinstance(get_chunker("fixed"), FixedSizeChunker)
    assert isinstance(get_chunker("recursive"), RecursiveCharacterChunker)
    assert isinstance(get_chunker("structure_aware"), StructureAwareChunker)

    with pytest.raises(ValueError):
        get_chunker("invalid_strategy")  # type: ignore
