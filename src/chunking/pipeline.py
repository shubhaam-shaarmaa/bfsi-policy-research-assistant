"""Chunking Pipeline and Strategy Factory.

Orchestrates document chunking across different strategies, persists enriched
chunks to disk, and tracks manifest state for downstream vector embedding (Stage 3).
"""

import json
from pathlib import Path
from typing import Literal

from config.settings import settings
from src.chunking.chunker_base import BaseChunker
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.recursive_chunker import RecursiveCharacterChunker
from src.chunking.structure_chunker import StructureAwareChunker
from src.core.logging import app_logger
from src.core.models import Chunk, ChunkingStrategy, ParsedDocument


def get_chunker(
    strategy: Literal["fixed", "recursive", "structure_aware"] | ChunkingStrategy = "structure_aware",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_tokens: int = 20,
) -> BaseChunker:
    """Factory function returning the configured chunker implementation."""
    strat = ChunkingStrategy(strategy) if isinstance(strategy, str) else strategy

    if strat == ChunkingStrategy.FIXED:
        return FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_tokens=min_tokens,
        )
    elif strat == ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_tokens=min_tokens,
        )
    elif strat == ChunkingStrategy.STRUCTURE_AWARE:
        return StructureAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_tokens=min_tokens,
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")


class ChunkingPipeline:
    """Orchestrates chunking of parsed documents and manages disk persistence."""

    def __init__(
        self,
        strategy: Literal["fixed", "recursive", "structure_aware"] | ChunkingStrategy | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.strategy = strategy or settings.CHUNKING_STRATEGY
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_tokens = settings.CHUNK_MIN_TOKENS
        self.output_dir = output_dir or settings.DATA_CHUNKS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = get_chunker(
            strategy=self.strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_tokens=self.min_tokens,
        )

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Split a single parsed document into enriched chunks."""
        chunks = self.chunker.split(document)
        self._save_chunks(document.metadata.doc_id, chunks)
        return chunks

    def chunk_all_processed(self, processed_dir: Path | None = None) -> dict[str, int]:
        """Load all parsed documents from processed directory and chunk them.

        Returns a dictionary mapping document_id to generated chunk count.
        """
        proc_dir = processed_dir or settings.DATA_PROCESSED_DIR
        json_files = list(proc_dir.glob("doc_*.json"))

        results: dict[str, int] = {}
        if not json_files:
            app_logger.warning("No processed document JSON files found in %s", proc_dir)
            return results

        for doc_file in json_files:
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                doc = ParsedDocument.model_validate(data)
                chunks = self.chunk_document(doc)
                results[doc.metadata.doc_id] = len(chunks)
                app_logger.info(
                    "Chunked '%s' (%s) -> %d chunks",
                    doc.metadata.title,
                    doc.metadata.doc_id,
                    len(chunks),
                )
            except Exception as e:
                app_logger.error("Failed to chunk file %s: %s", doc_file.name, e)

        return results

    def _save_chunks(self, doc_id: str, chunks: list[Chunk]) -> Path:
        """Persist chunks list as formatted JSON."""
        output_path = self.output_dir / f"chunks_{doc_id}.json"
        data = [chunk.model_dump(mode="json") for chunk in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path
