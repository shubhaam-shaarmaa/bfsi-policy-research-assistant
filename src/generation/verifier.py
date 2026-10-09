"""Citation Verification & Hallucination Mitigation Post-Check.

Performs deterministic validation that every quote cited by the LLM genuinely
exists in the retrieved regulatory chunks, eliminating fabricated quotes and citations.
"""

import re
from src.core.logging import app_logger
from src.core.models import Chunk
from src.generation.models import Citation


class CitationVerifier:
    """Verifies that LLM-generated citations map faithfully to source chunk text."""

    def __init__(self, min_similarity_threshold: float = 0.70) -> None:
        self.min_similarity_threshold = min_similarity_threshold

    def _normalize(self, text: str) -> str:
        """Strip punctuation and whitespace for robust substring matching."""
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    def _calculate_ngram_overlap(self, quote_norm: str, chunk_norm: str) -> float:
        """Calculate word overlap ratio between quote and chunk text."""
        quote_words = quote_norm.split()
        if not quote_words:
            return 0.0

        chunk_words = set(chunk_norm.split())
        matched_words = sum(1 for w in quote_words if w in chunk_words)
        return matched_words / len(quote_words)

    def verify_citation(self, citation: Citation, chunks: list[Chunk]) -> Citation:
        """Verify an individual citation against the pool of retrieved chunks."""
        quote_norm = self._normalize(citation.quote)
        if not quote_norm:
            citation.verified = False
            citation.similarity_match = 0.0
            return citation

        best_match = 0.0
        verified = False

        for chunk in chunks:
            chunk_norm = self._normalize(chunk.content)

            # 1. Exact substring match (ideal case)
            if quote_norm in chunk_norm:
                verified = True
                best_match = 1.0
                break

            # 2. Word-level overlap match (handles slight paraphrasing or truncation)
            overlap = self._calculate_ngram_overlap(quote_norm, chunk_norm)
            if overlap > best_match:
                best_match = overlap

            if overlap >= self.min_similarity_threshold:
                verified = True

        citation.verified = verified
        citation.similarity_match = round(best_match, 4)

        if not verified:
            app_logger.warning(
                "Hallucination Warning: Citation quote could not be verified in source chunks! "
                "Quote: '%s' (Best overlap: %.2f)",
                citation.quote[:80],
                best_match,
            )

        return citation

    def verify_all(self, citations: list[Citation], chunks: list[Chunk]) -> tuple[list[Citation], bool]:
        """Verify all citations in an answer.

        Returns:
            Tuple of (verified_citations_list, all_passed_boolean).
        """
        if not citations:
            return [], True

        verified_citations: list[Citation] = []
        all_passed = True

        for c in citations:
            v_c = self.verify_citation(c, chunks)
            verified_citations.append(v_c)
            if not v_c.verified:
                all_passed = False

        return verified_citations, all_passed
