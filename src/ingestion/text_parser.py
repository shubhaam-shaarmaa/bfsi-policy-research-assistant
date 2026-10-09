"""Text and Markdown Document Parser.

Parses plain text (.txt) and Markdown (.md) documents, extracting
markdown section headers (#, ##, ###), numbered clauses, and metadata.
"""

from pathlib import Path
import re
from src.core.logging import setup_logger
from src.core.models import (
    DocumentMetadata,
    DocumentType,
    ExtractedSection,
    ParsedDocument,
)
from src.ingestion.parser_base import BaseDocumentParser
from src.ingestion.pdf_parser import CIRCULAR_NUM_PATTERN, DATE_PATTERN

logger = setup_logger("text_parser")


class TextDocumentParser(BaseDocumentParser):
    """Parser for .txt and .md policy documents and FAQs."""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in [".txt", ".md", ".markdown"]

    def parse(self, file_path: Path) -> ParsedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        file_hash = self.compute_file_hash(file_path)
        doc_id = f"doc_{file_hash[:12]}"

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            content = file_path.read_text(encoding="latin-1")

        lines = content.splitlines()
        header_text = "\n".join(lines[:30])

        sections: list[ExtractedSection] = []
        current_heading = "Preamble / Overview"
        current_clause: str | None = None
        current_lines: list[str] = []
        section_counter = 1
        word_count = 0

        for line in lines:
            stripped = line.strip()
            word_count += len(stripped.split())
            approx_page = max(1, (word_count // 450) + 1)

            # Check markdown headers (# Heading) or numbered sections (1. Heading)
            is_md_header = stripped.startswith(("# ", "## ", "### ", "#### "))
            is_numbered_sec = bool(re.match(r"^(\d+(\.\d+)*)\.?\s+[A-Z]", stripped))

            if is_md_header or (is_numbered_sec and len(stripped) < 80):
                if current_lines:
                    sections.append(
                        ExtractedSection(
                            section_id=f"{doc_id}#s{section_counter:03d}",
                            heading=current_heading,
                            clause_number=current_clause,
                            page_number=approx_page,
                            content="\n".join(current_lines).strip(),
                        )
                    )
                    section_counter += 1
                    current_lines = []

                clean_heading = re.sub(r"^#+\s*", "", stripped)
                current_heading = clean_heading
                clause_match = re.match(r"^(\d+(\.\d+)*)", clean_heading)
                current_clause = clause_match.group(1) if clause_match else None
            else:
                current_lines.append(line)

        if current_lines:
            approx_page = max(1, (word_count // 450) + 1)
            sections.append(
                ExtractedSection(
                    section_id=f"{doc_id}#s{section_counter:03d}",
                    heading=current_heading,
                    clause_number=current_clause,
                    page_number=approx_page,
                    content="\n".join(current_lines).strip(),
                )
            )

        # Metadata extraction
        circular_match = CIRCULAR_NUM_PATTERN.search(header_text)
        circular_num = circular_match.group(0).strip() if circular_match else None

        date_match = DATE_PATTERN.search(header_text)
        doc_date = date_match.group(0).strip() if date_match else None

        # Title detection from first markdown H1 or first non-empty line
        title = file_path.stem.replace("_", " ").title()
        for line in lines[:10]:
            if line.strip().startswith("# "):
                title = line.strip().replace("# ", "").strip()
                break

        # Document type inference
        doc_type = DocumentType.CIRCULAR
        upper_content = content.upper()
        if "FAQ" in upper_content:
            doc_type = DocumentType.FAQ
        elif "SOP" in upper_content or "STANDARD OPERATING PROCEDURE" in upper_content:
            doc_type = DocumentType.SOP
        elif "MASTER DIRECTION" in upper_content:
            doc_type = DocumentType.MASTER_DIRECTION
        elif "GUIDELINES" in upper_content:
            doc_type = DocumentType.GUIDELINE

        total_pages = max(1, (word_count // 450) + 1)

        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=title,
            issuer="Reserve Bank of India (RBI)" if "RBI" in header_text.upper() else "BFSI Policy Entity",
            circular_number=circular_num,
            date=doc_date,
            doc_type=doc_type,
            source_filename=file_path.name,
            source_path=str(file_path),
            file_hash=file_hash,
            file_format=file_path.suffix.lstrip("."),
            total_pages=total_pages,
        )

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            raw_text=content,
            page_count=total_pages,
            parsing_warnings=[],
        )
