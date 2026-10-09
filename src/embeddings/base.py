"""Abstract Embedding Provider Interface.

Defines the contract for all embedding models (local and hosted).
Enforces uniform dimension properties, query embedding, batch document embedding,
and L2 normalization for cosine similarity search.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract base class for text embedding models."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """The fixed dimensional size of vectors produced by this model."""
        return self._dimension

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate a dense embedding vector for a search query.

        Args:
            text: Query string.

        Returns:
            Normalized list of floats with length equal to self.dimension.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embedding vectors for a batch of text chunks.

        Args:
            texts: List of document chunk strings.

        Returns:
            List of normalized float lists, each with length self.dimension.
        """
        pass
