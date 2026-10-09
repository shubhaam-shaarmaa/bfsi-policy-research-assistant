"""BM25 Keyword Retriever for Exact Regulatory Term Matching.

Implements BM25Okapi scoring across chunk corpora. Critical for exact alphanumeric
regulatory identifiers (e.g. 'RBI/2023-24/41', 'Clause 3.1', '5%', 'DLG', 'FLDG').
"""

import re
from rank_bm25 import BM25Okapi

from src.core.logging import app_logger
from src.core.models import Chunk
from src.vectorstore.base import SearchResult


class BM25Retriever:
    """Keyword search engine powered by BM25Okapi."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks: list[Chunk] = []
        self.corpus_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        if chunks:
            self.index_chunks(chunks)

    def _tokenize(self, text: str) -> list[str]:
        """Normalize and tokenize text preserving numbers, slashes, and dots."""
        # Lowercase and split on whitespace and punctuation except inner slashes/hyphens
        tokens = re.findall(r"[A-Za-z0-9/._%+-]+", text.lower())
        cleaned = [t.strip(".,;:?!()") for t in tokens if len(t.strip(".,;:?!()")) > 0]
        return cleaned

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Build BM25 inverted index from a list of chunks, incorporating metadata fields."""
        if not chunks:
            app_logger.warning("No chunks provided to BM25 index.")
            return 0

        self.chunks = list(chunks)
        # Combine document title, circular number, section heading, and content for comprehensive lexical coverage
        corpus_texts = [
            f"{c.document_title} {c.issuer} {c.circular_number or ''} {c.section_heading} {c.content}"
            for c in self.chunks
        ]
        self.corpus_tokens = [self._tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        app_logger.info("BM25 index built with %d documents.", len(self.chunks))
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str | int] | None = None,
    ) -> list[SearchResult]:
        """Perform BM25 keyword search with optional post-filtering."""
        if not self.bm25 or not self.chunks:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)

        # Pair scores with chunks
        scored_pairs: list[tuple[float, Chunk]] = []
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        for score, chunk in zip(scores, self.chunks):
            if score <= 0.0:
                continue

            # Apply metadata filters
            if filters:
                match = True
                for k, v in filters.items():
                    chunk_val = getattr(chunk, k, None)
                    if hasattr(chunk_val, "value"):
                        chunk_val = chunk_val.value
                    if chunk_val != v:
                        match = False
                        break
                if not match:
                    continue

            scored_pairs.append((score, chunk))

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for rank, (score, chunk) in enumerate(scored_pairs[:top_k], start=1):
            normalized_score = min(1.0, score / max_score)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=round(normalized_score, 4),
                    source="bm25",
                    rank=rank,
                    raw_score=round(score, 4),
                )
            )

        return results
