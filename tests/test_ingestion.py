"""Unit and Integration Tests for Stage 1 Document Ingestion Pipeline."""

from pathlib import Path
import json
import pytest
from config.settings import settings
from src.core.models import DocumentType, ParsedDocument
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.text_parser import TextDocumentParser
from src.ingestion.pdf_parser import PDFDocumentParser


@pytest.fixture
def synthetic_dir() -> Path:
    return settings.DATA_SYNTHETIC_DIR


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "processed_test"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_text_parser_extracts_rbi_synthetic_metadata(synthetic_dir: Path) -> None:
    sample_file = synthetic_dir / "rbi_sample_guideline.txt"
    assert sample_file.exists(), f"Synthetic sample missing at {sample_file}"

    parser = TextDocumentParser()
    assert parser.supports(".txt")

    parsed_doc = parser.parse(sample_file)

    # Validate Document Metadata
    assert parsed_doc.metadata.circular_number == "RBI/2024-25/SYNTH-88"
    assert "October 01, 2024" in (parsed_doc.metadata.date or "")
    assert parsed_doc.metadata.issuer == "Reserve Bank of India (RBI)"
    assert parsed_doc.metadata.doc_type == DocumentType.GUIDELINE
    assert len(parsed_doc.metadata.file_hash) == 64

    # Validate Section Extraction
    assert len(parsed_doc.sections) >= 4

    # Check specific extracted sections and clauses
    section_headings = [s.heading for s in parsed_doc.sections]
    assert any("Scope and Applicability" in h for h in section_headings)
    assert any("Governance and Board Oversight" in h for h in section_headings)
    assert any("Model Risk Management" in h for h in section_headings)
    assert any("Customer Grievance Redressal" in h for h in section_headings)

    # Check token and character count calculations
    for sec in parsed_doc.sections:
        assert sec.char_count > 0
        assert sec.token_estimate > 0
        assert sec.page_number >= 1


def test_markdown_parser_extracts_banking_sop(synthetic_dir: Path) -> None:
    sop_file = synthetic_dir / "banking_payment_sop.md"
    assert sop_file.exists()

    parser = TextDocumentParser()
    assert parser.supports(".md")

    parsed_doc = parser.parse(sop_file)
    assert parsed_doc.metadata.doc_type == DocumentType.SOP
    assert len(parsed_doc.sections) >= 3

    headings = [s.heading for s in parsed_doc.sections]
    assert any("Objective and Overview" in h for h in headings)
    assert any("Standard Settlement Instructions" in h for h in headings)


def test_pipeline_ingest_file_persists_json(synthetic_dir: Path, temp_output_dir: Path) -> None:
    sample_file = synthetic_dir / "rbi_sample_guideline.txt"
    pipeline = IngestionPipeline()

    parsed = pipeline.ingest_file(sample_file, output_dir=temp_output_dir)
    assert isinstance(parsed, ParsedDocument)

    # Verify JSON persistence
    expected_json = temp_output_dir / f"{parsed.metadata.doc_id}.json"
    assert expected_json.exists()

    # Verify JSON content is deserializable back into Pydantic model
    with open(expected_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        reconstructed = ParsedDocument.model_validate(data)
        assert reconstructed.metadata.doc_id == parsed.metadata.doc_id
        assert len(reconstructed.sections) == len(parsed.sections)


def test_pipeline_directory_ingestion_idempotency(synthetic_dir: Path, temp_output_dir: Path) -> None:
    pipeline = IngestionPipeline()

    # First Run: ingests all files in synthetic folder
    first_run_docs = pipeline.ingest_directory(synthetic_dir, output_dir=temp_output_dir, skip_existing=True)
    assert len(first_run_docs) >= 2

    manifest_file = temp_output_dir / "ingestion_manifest.json"
    assert manifest_file.exists()

    # Second Run: should detect hash matches and skip re-parsing
    second_run_docs = pipeline.ingest_directory(synthetic_dir, output_dir=temp_output_dir, skip_existing=True)
    assert len(second_run_docs) == len(first_run_docs)


def test_pipeline_unsupported_file_error(tmp_path: Path) -> None:
    fake_file = tmp_path / "data.xyz"
    fake_file.write_text("unsupported content")

    pipeline = IngestionPipeline()
    with pytest.raises(ValueError, match="No parser registered"):
        pipeline.get_parser_for_file(fake_file)


def test_pipeline_file_not_found() -> None:
    pipeline = IngestionPipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_file(Path("non_existent_regulatory_file.pdf"))
