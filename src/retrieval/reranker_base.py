"""Abstract Base Cross-Encoder Reranker Interface."""

from abc import ABC, abstractmethod
from src.vectorstore.base import SearchResult


class BaseReranker(ABC):
    """Abstract interface for passage reranking models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Score and reorder candidate retrieval results using cross-attention.

        Args:
            query: The user prompt or question.
            candidates: Initial pool of retrieved SearchResults from dense/BM25/hybrid.
            top_k: Number of highest-scoring passages to return.

        Returns:
            Re-ordered SearchResults with updated scores and source='reranked'.
        """
        pass
