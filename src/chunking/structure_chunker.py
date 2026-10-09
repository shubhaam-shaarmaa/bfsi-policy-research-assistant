"""Structure-Aware Regulatory Chunker.

Domain-tailored chunker for BFSI, RBI circulars, master directions, and banking SOPs.
Splits along legal clause and heading boundaries while injecting breadcrumb context
headers into each chunk so that dense vector and keyword retrievers preserve full
statutory context.
"""

import re
from src.chunking.chunker_base import BaseChunker
from src.chunking.recursive_chunker import RecursiveCharacterChunker
from src.core.logging import app_logger
from src.core.models import Chunk, ChunkingStrategy, ParsedDocument


class StructureAwareChunker(BaseChunker):
    """Chunks documents along regulatory heading and numbered clause boundaries.

    Injects contextual breadcrumbs (Issuer, Title, Circular No, Section, Clause)
    into the chunk payload to prevent semantic isolation in legal RAG retrieval.
    """

    SUBCLAUSE_PATTERN = re.compile(
        r"(?:^|\n)(?P<subclause>(?:\([a-z0-9]+\)|\b[a-z]\)|\b\d+\.\d+\.?\b)\s+)",
        re.MULTILINE | re.IGNORECASE,
    )

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_tokens: int = 20,
        inject_context_header: bool = True,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, min_tokens)
        self.strategy = ChunkingStrategy.STRUCTURE_AWARE
        self.inject_context_header = inject_context_header
        self._fallback_recursive = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_tokens=min_tokens,
        )

    def _build_context_header(
        self,
        doc: ParsedDocument,
        section_heading: str,
        clause_number: str | None,
        page_number: int,
    ) -> str:
        """Create a compact regulatory metadata header prepended to chunk content."""
        parts = [f"Issuer: {doc.metadata.issuer}"]
        if doc.metadata.circular_number:
            parts.append(f"Ref: {doc.metadata.circular_number}")
        parts.append(f"Doc: {doc.metadata.title}")
        if section_heading and section_heading != "General":
            parts.append(f"Section: {section_heading}")
        if clause_number:
            parts.append(f"Clause: {clause_number}")
        parts.append(f"Page: {page_number}")

        return f"[{' | '.join(parts)}]\n"

    def split(self, document: ParsedDocument) -> list[Chunk]:
        """Split document sections along structural and clause boundaries."""
        chunks: list[Chunk] = []
        chunk_index = 0

        for sec in document.sections:
            sec_text = sec.content.strip()
            if not sec_text:
                continue

            sec_tokens = self.count_tokens(sec_text)

            # Case A: Section fits directly within chunk token budget
            if sec_tokens <= self.chunk_size:
                if sec_tokens >= self.min_tokens:
                    header = (
                        self._build_context_header(
                            document,
                            sec.heading,
                            sec.clause_number,
                            sec.page_number,
                        )
                        if self.inject_context_header
                        else ""
                    )
                    full_content = f"{header}{sec_text}"
                    chunks.append(
                        self.create_chunk(
                            doc=document,
                            content=full_content,
                            page_number=sec.page_number,
                            section_heading=sec.heading,
                            clause_number=sec.clause_number,
                            chunk_index=chunk_index,
                            extra_metadata={"structure_type": "atomic_clause"},
                        )
                    )
                    chunk_index += 1
                continue

            # Case B: Section exceeds chunk_size.
            # Attempt to split along sub-clause markers (e.g., '(a)', '(b)', '1.1', '1.2')
            subclauses = self._split_by_subclauses(sec_text)

            if len(subclauses) > 1:
                # Group subclauses up to chunk_size
                curr_group: list[str] = []
                curr_tokens = 0
                for sub in subclauses:
                    sub_clean = sub.strip()
                    if not sub_clean:
                        continue
                    sub_tok = self.count_tokens(sub_clean)

                    if curr_tokens + sub_tok <= self.chunk_size:
                        curr_group.append(sub_clean)
                        curr_tokens += sub_tok
                    else:
                        if curr_group:
                            group_text = "\n\n".join(curr_group)
                            header = (
                                self._build_context_header(
                                    document,
                                    sec.heading,
                                    sec.clause_number,
                                    sec.page_number,
                                )
                                if self.inject_context_header
                                else ""
                            )
                            chunks.append(
                                self.create_chunk(
                                    doc=document,
                                    content=f"{header}{group_text}",
                                    page_number=sec.page_number,
                                    section_heading=sec.heading,
                                    clause_number=sec.clause_number,
                                    chunk_index=chunk_index,
                                    extra_metadata={"structure_type": "clause_group"},
                                )
                            )
                            chunk_index += 1
                        curr_group = [sub_clean]
                        curr_tokens = sub_tok

                if curr_group:
                    group_text = "\n\n".join(curr_group)
                    header = (
                        self._build_context_header(
                            document,
                            sec.heading,
                            sec.clause_number,
                            sec.page_number,
                        )
                        if self.inject_context_header
                        else ""
                    )
                    chunks.append(
                        self.create_chunk(
                            doc=document,
                            content=f"{header}{group_text}",
                            page_number=sec.page_number,
                            section_heading=sec.heading,
                            clause_number=sec.clause_number,
                            chunk_index=chunk_index,
                            extra_metadata={"structure_type": "clause_group"},
                        )
                    )
                    chunk_index += 1
            else:
                # Case C: Single large narrative block without sub-clauses.
                # Use recursive splitting fallback with persistent regulatory header
                header = (
                    self._build_context_header(
                        document,
                        sec.heading,
                        sec.clause_number,
                        sec.page_number,
                    )
                    if self.inject_context_header
                    else ""
                )
                header_tokens = self.count_tokens(header)
                effective_chunk_size = max(50, self.chunk_size - header_tokens)

                recursive_sub_splits = self._fallback_recursive._split_text_recursively(
                    sec_text,
                    list(self._fallback_recursive.separators),
                )

                for sub_split in recursive_sub_splits:
                    if self.count_tokens(sub_split) >= self.min_tokens:
                        full_content = f"{header}{sub_split}"
                        chunks.append(
                            self.create_chunk(
                                doc=document,
                                content=full_content,
                                page_number=sec.page_number,
                                section_heading=sec.heading,
                                clause_number=sec.clause_number,
                                chunk_index=chunk_index,
                                extra_metadata={"structure_type": "recursive_fallback"},
                            )
                        )
                        chunk_index += 1

        app_logger.info(
            "StructureAwareChunker: produced %d chunks for doc '%s'",
            len(chunks),
            document.metadata.title,
        )
        return chunks

    def _split_by_subclauses(self, text: str) -> list[str]:
        """Split text along regulatory sub-clauses like (a), (b), (i), (ii), 1.1, etc."""
        lines = text.split("\n")
        blocks: list[str] = []
        current_block: list[str] = []

        sub_pattern = re.compile(
            r"^\s*(?:\([a-z0-9ivx]+\)|\b[a-z]\.|\b\d+\.\d+\b|\b[ivx]+\.)\s+",
            re.IGNORECASE,
        )

        for line in lines:
            stripped = line.strip()
            if sub_pattern.match(stripped) and current_block:
                blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks if len(blocks) > 1 else [text]
