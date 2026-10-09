"""Abstract Base Chunker and Token Utility for Regulatory Text.

Defines the BaseChunker interface, exact token counting via tiktoken,
deterministic SHA-256 chunk hashing, and citation-ready metadata enrichment.
"""

from abc import ABC, abstractmethod
import hashlib
from typing import Any
import tiktoken

from src.core.logging import app_logger
from src.core.models import Chunk, ChunkingStrategy, ParsedDocument


class BaseChunker(ABC):
    """Abstract base class for all document chunking strategies."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_tokens: int = 20,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialize chunker with token limits and tokenizer encoding.

        Args:
            chunk_size: Target maximum token count per chunk.
            chunk_overlap: Token overlap between adjacent chunks.
            min_tokens: Minimum tokens required to keep a chunk (filters noise).
            encoding_name: tiktoken encoding name (cl100k_base is Claude & GPT-4 compatible).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_tokens = min_tokens
        self.strategy = ChunkingStrategy.FIXED
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            app_logger.warning("Failed to load tiktoken encoding %s: %s. Using heuristic.", encoding_name, e)
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count exact tokens using tiktoken, or fallback to word-count heuristic."""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback heuristic: ~1.3 tokens per word
        return max(1, int(len(text.split()) * 1.3))

    def compute_content_hash(self, text: str) -> str:
        """Generate SHA-256 hash of normalized chunk text for idempotency."""
        normalized = " ".join(text.strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def create_chunk(
        self,
        doc: ParsedDocument,
        content: str,
        page_number: int,
        section_heading: str,
        clause_number: str | None,
        chunk_index: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Factory method to construct an enriched Chunk model."""
        token_count = self.count_tokens(content)
        content_hash = self.compute_content_hash(content)
        chunk_id = f"{doc.metadata.doc_id}#c{chunk_index:03d}"

        metadata = {
            "source_filename": doc.metadata.source_filename,
            "department": doc.metadata.department,
            **(extra_metadata or {}),
        }

        return Chunk(
            chunk_id=chunk_id,
            document_id=doc.metadata.doc_id,
            document_title=doc.metadata.title,
            issuer=doc.metadata.issuer,
            circular_number=doc.metadata.circular_number,
            date=doc.metadata.date,
            doc_type=doc.metadata.doc_type,
            page_number=page_number,
            section_heading=section_heading,
            clause_number=clause_number,
            content=content.strip(),
            content_hash=content_hash,
            chunk_index=chunk_index,
            token_count=token_count,
            strategy=self.strategy,
            metadata=metadata,
        )

    @abstractmethod
    def split(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into structured, citation-ready chunks."""
        pass
