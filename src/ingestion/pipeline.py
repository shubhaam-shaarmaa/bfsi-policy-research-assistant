"""Document Ingestion Pipeline.

Orchestrates file discovery, parser dispatching, idempotent content hashing,
and persistence of structured parsed documents for downstream chunking and indexing.
"""

from pathlib import Path
import json
import time
from typing import Sequence
from src.core.logging import setup_logger
from src.core.models import ParsedDocument
from src.ingestion.parser_base import BaseDocumentParser
from src.ingestion.pdf_parser import PDFDocumentParser
from src.ingestion.docx_parser import DocxDocumentParser
from src.ingestion.text_parser import TextDocumentParser
from config.settings import settings

logger = setup_logger("ingestion_pipeline")


class IngestionPipeline:
    """Manages document parsers and executes ingestion workflows."""

    def __init__(self, parsers: Sequence[BaseDocumentParser] | None = None) -> None:
        if parsers is None:
            self.parsers: list[BaseDocumentParser] = [
                PDFDocumentParser(),
                DocxDocumentParser(),
                TextDocumentParser(),
            ]
        else:
            self.parsers = list(parsers)

    def get_parser_for_file(self, file_path: Path) -> BaseDocumentParser:
        """Finds the registered parser supporting the file's extension."""
        ext = file_path.suffix.lower()
        for parser in self.parsers:
            if parser.supports(ext):
                return parser
        raise ValueError(
            f"No parser registered for file extension '{ext}'. Supported: .pdf, .docx, .txt, .md"
        )

    def ingest_file(self, file_path: Path, output_dir: Path | None = None) -> ParsedDocument:
        """Parses a single document and saves its structured representation."""
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        parser = self.get_parser_for_file(file_path)
        logger.info("Ingesting file: %s (using %s)", file_path.name, parser.__class__.__name__)

        start_time = time.perf_counter()
        parsed_doc = parser.parse(file_path)
        elapsed = time.perf_counter() - start_time

        logger.info(
            "Parsed '%s' in %.2fs -> %d pages, %d sections, Doc ID: %s",
            file_path.name,
            elapsed,
            parsed_doc.page_count,
            len(parsed_doc.sections),
            parsed_doc.metadata.doc_id,
        )

        # Save processed JSON if output directory configured
        target_output_dir = output_dir or settings.DATA_PROCESSED_DIR
        target_output_dir.mkdir(parents=True, exist_ok=True)
        json_path = target_output_dir / f"{parsed_doc.metadata.doc_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_doc.model_dump(mode="json"), f, indent=2)

        return parsed_doc

    def ingest_directory(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        skip_existing: bool = True,
    ) -> list[ParsedDocument]:
        """Ingests all supported documents within a directory.

        Args:
            input_dir: Directory containing documents.
            output_dir: Directory to save parsed JSON documents.
            skip_existing: If True, skips files whose content SHA-256 is already processed.

        Returns:
            list[ParsedDocument]: List of successfully parsed documents.
        """
        input_path = Path(input_dir).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

        target_output = output_dir or settings.DATA_PROCESSED_DIR
        target_output.mkdir(parents=True, exist_ok=True)

        supported_extensions = {".pdf", ".docx", ".txt", ".md"}
        candidate_files = [
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        logger.info("Found %d candidate documents in %s", len(candidate_files), input_path)

        # Read existing manifest to enforce idempotency
        manifest_path = target_output / "ingestion_manifest.json"
        manifest: dict[str, dict] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as e:
                logger.warning("Could not read existing manifest, re-initializing: %s", e)

        results: list[ParsedDocument] = []
        skipped_count = 0

        for file_path in candidate_files:
            file_hash = BaseDocumentParser.compute_file_hash(file_path)
            doc_id = f"doc_{file_hash[:12]}"

            if skip_existing and doc_id in manifest:
                logger.info("Skipping already ingested file (content hash match): %s", file_path.name)
                skipped_count += 1
                # Load from processed cache
                cached_json = target_output / f"{doc_id}.json"
                if cached_json.exists():
                    with open(cached_json, "r", encoding="utf-8") as f:
                        results.append(ParsedDocument.model_validate_json(f.read()))
                continue

            try:
                parsed = self.ingest_file(file_path, output_dir=target_output)
                results.append(parsed)
                manifest[doc_id] = {
                    "source_filename": file_path.name,
                    "title": parsed.metadata.title,
                    "circular_number": parsed.metadata.circular_number,
                    "date": parsed.metadata.date,
                    "issuer": parsed.metadata.issuer,
                    "file_hash": file_hash,
                    "pages": parsed.page_count,
                    "sections_count": len(parsed.sections),
                    "ingested_at": parsed.metadata.ingested_at.isoformat(),
                }
            except Exception as e:
                logger.error("Failed to parse %s: %s", file_path.name, e, exc_info=True)

        # Write updated manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Batch ingestion complete: %d parsed, %d skipped, %d total in catalog",
            len(results) - skipped_count,
            skipped_count,
            len(manifest),
        )
        return results
