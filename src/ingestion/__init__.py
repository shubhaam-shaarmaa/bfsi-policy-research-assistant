"""Document Ingestion Package.

Exposes parsers and pipeline for reading PDF, DOCX, TXT, and Markdown documents.
"""

from src.ingestion.parser_base import BaseDocumentParser
from src.ingestion.pdf_parser import PDFDocumentParser
from src.ingestion.docx_parser import DocxDocumentParser
from src.ingestion.text_parser import TextDocumentParser
from src.ingestion.pipeline import IngestionPipeline

__all__ = [
    "BaseDocumentParser",
    "PDFDocumentParser",
    "DocxDocumentParser",
    "TextDocumentParser",
    "IngestionPipeline",
]
