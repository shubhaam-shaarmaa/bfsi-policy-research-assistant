"""Abstract Vector Store Interface and Search Result Schemas.

Defines the contract for vector database engines (ChromaDB and Qdrant).
Guarantees idempotent chunk upserts, metadata filtering, and uniform search scores.
"""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

from src.core.models import Chunk


class SearchResult(BaseModel):
    """Unified container for a retrieved chunk with its relevance score."""

    chunk: Chunk
    score: float = Field(..., description="Relevance score (normalized 0.0 to 1.0 or similarity)")
    source: str = Field("dense", description="Retrieval source: 'dense', 'bm25', 'hybrid', 'reranked'")
    rank: int = Field(0, description="1-indexed rank within result list")
    raw_score: float | None = Field(None, description="Raw distance or BM25 score before normalization")


class BaseVectorStore(ABC):
    """Abstract interface for dense vector databases."""

    def __init__(self, collection_name: str, dimension: int) -> None:
        self.collection_name = collection_name
        self.dimension = dimension

    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert document chunks and vectors idempotently.

        Args:
            chunks: List of Chunk models.
            embeddings: Parallel list of float vectors matching self.dimension.

        Returns:
            Number of chunks newly inserted or updated.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform nearest-neighbor vector search with optional metadata filtering.

        Args:
            query_vector: Dense query embedding.
            top_k: Number of nearest chunks to return.
            filters: Metadata filter criteria (e.g., {'issuer': 'RBI', 'doc_type': 'circular'}).

        Returns:
            List of SearchResult objects ordered by descending relevance.
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all vectors belonging to a specific document ID.

        Returns:
            Number of records deleted.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in the collection."""
        pass
