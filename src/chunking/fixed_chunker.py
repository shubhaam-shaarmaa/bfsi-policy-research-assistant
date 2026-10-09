"""Fixed-Size Sliding Window Chunker.

Splits document text into fixed-size token windows with a defined overlap.
A standard baseline for RAG benchmarks.
"""

from src.chunking.chunker_base import BaseChunker
from src.core.logging import app_logger
from src.core.models import Chunk, ChunkingStrategy, ParsedDocument


class FixedSizeChunker(BaseChunker):
    """Chunks text strictly by fixed token window size with a sliding overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_tokens: int = 20,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, min_tokens)
        self.strategy = ChunkingStrategy.FIXED

    def split(self, document: ParsedDocument) -> list[Chunk]:
        """Split document sections using a sliding token window.

        Processes section by section to preserve page numbers and section headings,
        sliding windows across text when sections are larger than chunk_size.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        # If sections exist, chunk within sections to keep heading/page context
        sections_to_process = (
            document.sections
            if document.sections
            else [
                # Synthetic single section fallback
                type(
                    "DummySection",
                    (),
                    {
                        "content": document.raw_text,
                        "heading": "Full Document",
                        "clause_number": None,
                        "page_number": 1,
                    },
                )()
            ]
        )

        for sec in sections_to_process:
            text = sec.content.strip()
            if not text:
                continue

            # Tokenize using tokenizer if available, otherwise word chunks
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, disallowed_special=())
                total_tokens = len(tokens)

                if total_tokens <= self.chunk_size:
                    if total_tokens >= self.min_tokens:
                        chunks.append(
                            self.create_chunk(
                                doc=document,
                                content=text,
                                page_number=sec.page_number,
                                section_heading=sec.heading,
                                clause_number=sec.clause_number,
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1
                else:
                    start = 0
                    step = max(1, self.chunk_size - self.chunk_overlap)
                    while start < total_tokens:
                        end = min(start + self.chunk_size, total_tokens)
                        sub_tokens = tokens[start:end]
                        if len(sub_tokens) >= self.min_tokens:
                            sub_text = self.tokenizer.decode(sub_tokens)
                            chunks.append(
                                self.create_chunk(
                                    doc=document,
                                    content=sub_text,
                                    page_number=sec.page_number,
                                    section_heading=sec.heading,
                                    clause_number=sec.clause_number,
                                    chunk_index=chunk_index,
                                )
                            )
                            chunk_index += 1
                        start += step
            else:
                # Word-based sliding window fallback
                words = text.split()
                total_words = len(words)
                step = max(1, int(self.chunk_size / 1.3) - int(self.chunk_overlap / 1.3))
                window_words = max(10, int(self.chunk_size / 1.3))

                start = 0
                while start < total_words:
                    end = min(start + window_words, total_words)
                    sub_words = words[start:end]
                    if len(sub_words) >= int(self.min_tokens / 1.3):
                        sub_text = " ".join(sub_words)
                        chunks.append(
                            self.create_chunk(
                                doc=document,
                                content=sub_text,
                                page_number=sec.page_number,
                                section_heading=sec.heading,
                                clause_number=sec.clause_number,
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1
                    start += step

        app_logger.info(
            "FixedSizeChunker: produced %d chunks for doc '%s'",
            len(chunks),
            document.metadata.title,
        )
        return chunks
