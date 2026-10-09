"""Core Domain Models and Schemas for BFSI Policy Research Assistant.

These models define the canonical representations of regulatory documents,
extracted sections, and metadata. Designed to be extensible for downstream
Chunking (Stage 2), Vector Storage (Stage 3), and Citations (Stage 5).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    CIRCULAR = "circular"
    MASTER_DIRECTION = "master_direction"
    GUIDELINE = "guideline"
    NOTIFICATION = "notification"
    SOP = "sop"
    FAQ = "faq"
    POLICY = "policy"
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    STRUCTURE_AWARE = "structure_aware"


class DocumentMetadata(BaseModel):
    """Metadata describing a regulatory/banking policy document."""

    doc_id: str = Field(..., description="Unique identifier derived from content SHA-256 hash or slug")
    title: str = Field(..., description="Official title of the document or circular")
    issuer: str = Field("Reserve Bank of India (RBI)", description="Regulatory issuer (RBI, SEBI, NPCI, Internal)")
    circular_number: str | None = Field(None, description="Official regulatory reference (e.g., RBI/2023-24/41)")
    department: str | None = Field(None, description="Issuing department (e.g., DoR, DPSS, FIDD)")
    date: str | None = Field(None, description="Date of publication (YYYY-MM-DD or as stated in circular)")
    doc_type: DocumentType = Field(DocumentType.CIRCULAR, description="Classification of document")
    source_filename: str = Field(..., description="Original file name")
    source_path: str = Field(..., description="Path to raw source document")
    file_hash: str = Field(..., description="SHA-256 content checksum for idempotent ingestion")
    file_format: str = Field("pdf", description="File extension (pdf, docx, txt, md)")
    total_pages: int = Field(1, ge=1, description="Total number of pages in document")
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of ingestion",
    )
    custom_metadata: dict[str, Any] = Field(default_factory=dict, description="Extensible bag for future BFSI tags")


class ExtractedSection(BaseModel):
    """A structural section, chapter, or clause extracted from a document."""

    section_id: str = Field(..., description="Unique section identifier (e.g., doc_id#s01)")
    heading: str = Field(..., description="Heading or chapter title")
    clause_number: str | None = Field(None, description="Clause or paragraph number (e.g. '3.1', 'Clause 4(a)')")
    page_number: int = Field(..., ge=1, description="1-indexed page number where this section starts")
    content: str = Field(..., description="Raw textual content within this section")
    char_count: int = Field(0, description="Character count of content")
    token_estimate: int = Field(0, description="Heuristic token estimate (~1.3 tokens per word)")

    def model_post_init(self, __context: Any) -> None:
        if self.char_count == 0:
            object.__setattr__(self, "char_count", len(self.content))
        if self.token_estimate == 0:
            words = len(self.content.split())
            object.__setattr__(self, "token_estimate", max(1, int(words * 1.3)))


class ParsedDocument(BaseModel):
    """Full representation of a document parsed with structure, sections, and metadata."""

    metadata: DocumentMetadata
    sections: list[ExtractedSection] = Field(default_factory=list)
    raw_text: str = Field(..., description="Complete text extracted across all pages")
    page_count: int = Field(1, ge=1)
    parsing_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings such as low text density, potential scanned page, or missing header",
    )


class Chunk(BaseModel):
    """A granular chunk of regulatory text enriched with citation metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier (e.g., doc_id#c001)")
    document_id: str = Field(..., description="Parent document identifier")
    document_title: str = Field(..., description="Title of parent document")
    issuer: str = Field(..., description="Regulatory issuer (e.g. RBI, SEBI)")
    circular_number: str | None = Field(None, description="Regulatory circular number")
    date: str | None = Field(None, description="Document publication date")
    doc_type: DocumentType = Field(..., description="Document type")
    page_number: int = Field(..., ge=1, description="Page number where chunk appears")
    section_heading: str = Field(..., description="Section or chapter heading context")
    clause_number: str | None = Field(None, description="Clause or rule number (e.g., 3.1, 4(a))")
    content: str = Field(..., description="Processed chunk text")
    content_hash: str = Field(..., description="SHA-256 hash of chunk content for deduplication")
    chunk_index: int = Field(..., ge=0, description="0-indexed position within document")
    token_count: int = Field(..., ge=0, description="Exact or estimated token count")
    strategy: ChunkingStrategy = Field(..., description="Chunking strategy used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")
