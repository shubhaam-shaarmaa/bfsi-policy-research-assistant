"""Deterministic Mock Reranker for unit tests."""

from src.retrieval.reranker_base import BaseReranker
from src.vectorstore.base import SearchResult


class MockReranker(BaseReranker):
    """Sorts candidates by exact keyword overlap with query to simulate reranking."""

    def __init__(self, model_name: str = "mock-reranker") -> None:
        super().__init__(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        import re
        q_words = set(re.findall(r"[A-Za-z0-9%]+", query.lower()))
        scored: list[tuple[float, SearchResult]] = []

        for cand in candidates:
            # Check overlap against both content and document title
            target_text = f"{cand.chunk.document_title} {cand.chunk.section_heading} {cand.chunk.content}"
            c_words = set(re.findall(r"[A-Za-z0-9%]+", target_text.lower()))
            overlap = len(q_words.intersection(c_words))
            # Overlap boost
            overlap_ratio = overlap / max(1, len(q_words))
            boosted = min(1.0, cand.score * 0.3 + overlap_ratio * 0.7)
            scored.append((boosted, cand))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for rank, (score, cand) in enumerate(scored[:top_k], start=1):
            results.append(
                SearchResult(
                    chunk=cand.chunk,
                    score=round(score, 4),
                    source="reranked",
                    rank=rank,
                    raw_score=round(score, 4),
                )
            )

        return results
