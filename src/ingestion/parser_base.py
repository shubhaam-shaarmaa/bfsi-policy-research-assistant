"""Base Parser Interface for Document Ingestion.

All format-specific parsers (PDF, DOCX, TXT/MD) inherit from BaseDocumentParser.
This follows the Strategy Pattern to allow adding new parsers (e.g., HTML, XML, OCR)
without modifying downstream pipeline code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
from src.core.models import ParsedDocument


class BaseDocumentParser(ABC):
    """Abstract base class for all document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parses a document file and returns a structured ParsedDocument.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            ParsedDocument: Structured document containing metadata and sections.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is corrupted or unsupported.
        """
        pass

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Returns True if this parser supports the given file extension (e.g., '.pdf')."""
        pass

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Computes a SHA-256 checksum of the file for idempotency and duplicate detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
