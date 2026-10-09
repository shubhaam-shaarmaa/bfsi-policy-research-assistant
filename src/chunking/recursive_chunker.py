"""Recursive Paragraph & Sentence Chunker.

Recursively splits text using a hierarchy of separators (double newline,
single newline, sentence endings, spaces) to maintain natural linguistic cohesion
without crossing major structural boundaries where possible.
"""

from typing import Sequence
from src.chunking.chunker_base import BaseChunker
from src.core.logging import app_logger
from src.core.models import Chunk, ChunkingStrategy, ParsedDocument


class RecursiveCharacterChunker(BaseChunker):
    """Splits text hierarchically using natural linguistic delimiters."""

    DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", "; ", ", ", " ")

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_tokens: int = 20,
        separators: Sequence[str] | None = None,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, min_tokens)
        self.strategy = ChunkingStrategy.RECURSIVE
        self.separators = list(separators) if separators else list(self.DEFAULT_SEPARATORS)

    def _split_text_recursively(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text by separators until pieces fit chunk_size."""
        token_count = self.count_tokens(text)
        if token_count <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # Base case: no more separators, hard-truncate by tokens/words
            words = text.split()
            pieces = []
            curr_words: list[str] = []
            curr_tokens = 0
            for w in words:
                w_tok = self.count_tokens(w)
                if curr_tokens + w_tok > self.chunk_size and curr_words:
                    pieces.append(" ".join(curr_words))
                    curr_words = [w]
                    curr_tokens = w_tok
                else:
                    curr_words.append(w)
                    curr_tokens += w_tok
            if curr_words:
                pieces.append(" ".join(curr_words))
            return pieces

        sep = separators[0]
        remaining_separators = separators[1:]

        if sep in text:
            splits = text.split(sep)
        else:
            return self._split_text_recursively(text, remaining_separators)

        # Merge splits up to chunk_size
        results: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_tokens = self.count_tokens(s_clean)

            if s_tokens > self.chunk_size:
                # Flush existing buffer
                if current_chunk:
                    merged = sep.join(current_chunk).strip()
                    if merged:
                        results.append(merged)
                    current_chunk = []
                    current_tokens = 0
                # Recursively split the oversized fragment with remaining separators
                sub_splits = self._split_text_recursively(s_clean, remaining_separators)
                results.extend(sub_splits)
            elif current_tokens + s_tokens + 1 <= self.chunk_size:
                current_chunk.append(s_clean)
                current_tokens += s_tokens + 1
            else:
                if current_chunk:
                    merged = sep.join(current_chunk).strip()
                    if merged:
                        results.append(merged)
                current_chunk = [s_clean]
                current_tokens = s_tokens

        if current_chunk:
            merged = sep.join(current_chunk).strip()
            if merged:
                results.append(merged)

        return results

    def split(self, document: ParsedDocument) -> list[Chunk]:
        """Split document sections using recursive paragraph splitting."""
        chunks: list[Chunk] = []
        chunk_index = 0

        for sec in document.sections:
            text = sec.content.strip()
            if not text:
                continue

            text_splits = self._split_text_recursively(text, list(self.separators))

            for split_content in text_splits:
                if self.count_tokens(split_content) >= self.min_tokens:
                    chunks.append(
                        self.create_chunk(
                            doc=document,
                            content=split_content,
                            page_number=sec.page_number,
                            section_heading=sec.heading,
                            clause_number=sec.clause_number,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

        app_logger.info(
            "RecursiveCharacterChunker: produced %d chunks for doc '%s'",
            len(chunks),
            document.metadata.title,
        )
        return chunks
