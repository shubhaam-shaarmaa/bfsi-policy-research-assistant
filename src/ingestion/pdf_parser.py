"""PDF Parser using PyMuPDF (fitz) with Section and Regulatory Metadata Extraction.

Extracts page-by-page text, identifies section headings, extracts RBI circular
reference numbers and dates, and flags potential scanned/low-density pages.
"""

from pathlib import Path
import re
import pymupdf as fitz  # Modern PyMuPDF standard
from src.core.logging import setup_logger
from src.core.models import (
    DocumentMetadata,
    DocumentType,
    ExtractedSection,
    ParsedDocument,
)
from src.ingestion.parser_base import BaseDocumentParser

logger = setup_logger("pdf_parser")

# Regulatory Regex Patterns for Indian BFSI & RBI circulars
CIRCULAR_NUM_PATTERN = re.compile(
    r"(RBI\/\d{4}-\d{2}\/[A-Za-z0-9\-_]+|[A-Z]{2,6}\.[A-Z0-9\.\-_\/]+\d{4}-\d{2}|\bDoR\.[A-Z0-9\.\-_/]+|\bDPSS\.[A-Z0-9\.\-_/]+)",
    re.IGNORECASE,
)
MONTHS_REGEX = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_PATTERN = re.compile(
    rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS_REGEX},?\s+\d{{4}}|{MONTHS_REGEX}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}[-\/.]\d{{1,2}}[-\/.]\d{{4}})\b",
    re.IGNORECASE,
)
HEADING_PATTERN = re.compile(
    r"^(\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9\s,\-–\(\)]{3,80}|(?:Section|Chapter|Clause|Annexure|Appendix|Part)\s+[0-9IVX]+[:\.\-–]?\s*[A-Z][A-Za-z0-9\s,\-–\(\)]{2,80}|[A-Z\s]{4,60})$"
)


class PDFDocumentParser(BaseDocumentParser):
    """High-performance PDF parser using PyMuPDF."""

    def __init__(self, min_char_threshold_per_page: int = 50) -> None:
        self.min_char_threshold = min_char_threshold_per_page

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in [".pdf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_hash = self.compute_file_hash(file_path)
        doc_id = f"doc_{file_hash[:12]}"
        warnings: list[str] = []

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF {file_path}: {e}") from e

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError(f"PDF {file_path} is empty (0 pages)")

        full_text_chunks: list[str] = []
        sections: list[ExtractedSection] = []
        low_density_pages: list[int] = []

        # Buffer for parsing header / front-matter metadata from first 2 pages
        header_text = ""
        current_section_heading = "Header / Preamble"
        current_section_clause: str | None = None
        current_section_text: list[str] = []
        current_section_page = 1
        section_counter = 1

        for page_num in range(1, total_pages + 1):
            page = doc[page_num - 1]
            page_text = page.get_text("text")

            # Check text density (scanned PDF detection)
            if len(page_text.strip()) < self.min_char_threshold:
                low_density_pages.append(page_num)

            if page_num <= 2:
                header_text += page_text + "\n"

            full_text_chunks.append(f"--- Page {page_num} ---\n{page_text}")

            # Process line by line to detect structural headings and clauses
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            for line in lines:
                # Check if this line looks like a structural heading
                if HEADING_PATTERN.match(line) and len(line) < 100:
                    # Flush previous section if it has content
                    if current_section_text:
                        content_str = "\n".join(current_section_text).strip()
                        if content_str:
                            sections.append(
                                ExtractedSection(
                                    section_id=f"{doc_id}#s{section_counter:03d}",
                                    heading=current_section_heading,
                                    clause_number=current_section_clause,
                                    page_number=current_section_page,
                                    content=content_str,
                                )
                            )
                            section_counter += 1
                        current_section_text = []

                    # Set new section
                    current_section_heading = line
                    current_section_page = page_num
                    # Try to extract clause number if starting with digits (e.g. "3.1")
                    clause_match = re.match(r"^(\d+(\.\d+)*)", line)
                    current_section_clause = clause_match.group(1) if clause_match else None
                else:
                    current_section_text.append(line)

        # Flush final section
        if current_section_text:
            content_str = "\n".join(current_section_text).strip()
            if content_str:
                sections.append(
                    ExtractedSection(
                        section_id=f"{doc_id}#s{section_counter:03d}",
                        heading=current_section_heading,
                        clause_number=current_section_clause,
                        page_number=current_section_page,
                        content=content_str,
                    )
                )

        doc.close()

        # Handle scanned document warning
        if low_density_pages:
            warning_msg = (
                f"Low text density detected on pages {low_density_pages} (under {self.min_char_threshold} chars). "
                "These pages may be scanned images. In production, connect an OCR pipeline (Tesseract / AWS Textract / Marker)."
            )
            warnings.append(warning_msg)
            logger.warning("[%s] %s", file_path.name, warning_msg)

        # Extract metadata from front-matter
        metadata = self._extract_metadata(
            doc_id=doc_id,
            file_path=file_path,
            file_hash=file_hash,
            total_pages=total_pages,
            header_text=header_text,
        )

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            raw_text="\n\n".join(full_text_chunks),
            page_count=total_pages,
            parsing_warnings=warnings,
        )

    def _extract_metadata(
        self,
        doc_id: str,
        file_path: Path,
        file_hash: str,
        total_pages: int,
        header_text: str,
    ) -> DocumentMetadata:
        """Heuristically extracts regulatory metadata from header text."""
        # 1. Circular number
        circular_match = CIRCULAR_NUM_PATTERN.search(header_text)
        circular_num = circular_match.group(0).strip() if circular_match else None

        # 2. Date
        date_match = DATE_PATTERN.search(header_text)
        doc_date = date_match.group(0).strip() if date_match else None

        # 3. Document Title (first substantive line or filename fallback)
        title = self._clean_title_from_text(header_text, file_path.stem)

        # 4. Issuer
        issuer = "Reserve Bank of India (RBI)"
        if "SEBI" in header_text.upper():
            issuer = "Securities and Exchange Board of India (SEBI)"
        elif "NPCI" in header_text.upper():
            issuer = "National Payments Corporation of India (NPCI)"

        # 5. Document Type
        doc_type = DocumentType.CIRCULAR
        upper_header = header_text.upper()
        if "MASTER DIRECTION" in upper_header:
            doc_type = DocumentType.MASTER_DIRECTION
        elif "GUIDELINES" in upper_header:
            doc_type = DocumentType.GUIDELINE
        elif "STANDARD OPERATING PROCEDURE" in upper_header or "SOP" in upper_header:
            doc_type = DocumentType.SOP
        elif "FAQ" in upper_header or "FREQUENTLY ASKED QUESTIONS" in upper_header:
            doc_type = DocumentType.FAQ

        return DocumentMetadata(
            doc_id=doc_id,
            title=title,
            issuer=issuer,
            circular_number=circular_num,
            date=doc_date,
            doc_type=doc_type,
            source_filename=file_path.name,
            source_path=str(file_path),
            file_hash=file_hash,
            file_format="pdf",
            total_pages=total_pages,
        )

    def _clean_title_from_text(self, text: str, fallback_title: str) -> str:
        """Finds candidate title line in header, avoiding headers/logos."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:15]:
            # Skip short generic header markers
            if len(line) < 10 or line.lower().startswith(("page ", "rbi/", "confidential", "to,", "all ")):
                continue
            # Look for lines with subject or substantive titles
            if any(term in line.lower() for term in ["guidelines", "master direction", "circular", "framework", "directions"]):
                return line
        return fallback_title.replace("_", " ").title()
