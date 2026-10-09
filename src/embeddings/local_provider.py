"""Local Embedding Provider using Sentence-Transformers.

Supports BAAI/bge-small-en-v1.5 and sentence-transformers/all-MiniLM-L6-v2.
Handles device allocation (CUDA / CPU), batch processing, and L2 normalization.
"""

from src.core.logging import app_logger
from src.embeddings.base import BaseEmbeddingProvider


class LocalSentenceTransformersProvider(BaseEmbeddingProvider):
    """Local embedding generator using HuggingFace / SentenceTransformers models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Lazy load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            app_logger.info("Loading local embedding model '%s'...", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Detect actual model dimension
            dim = self._model.get_sentence_embedding_dimension()
            if dim:
                self._dimension = int(dim)
            app_logger.info("Loaded embedding model '%s' (dim=%d)", self.model_name, self._dimension)
        except Exception as e:
            app_logger.error(
                "Failed to initialize sentence-transformers model '%s': %s. Falling back to MockProvider.",
                self.model_name,
                e,
            )
            self._model = None

    def embed_query(self, text: str) -> list[float]:
        """Generate normalized embedding for a query string."""
        if not self._model:
            from src.embeddings.mock_provider import MockEmbeddingProvider

            return MockEmbeddingProvider(dimension=self.dimension).embed_query(text)

        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for a batch of documents."""
        if not texts:
            return []
        if not self._model:
            from src.embeddings.mock_provider import MockEmbeddingProvider

            return MockEmbeddingProvider(dimension=self.dimension).embed_documents(texts)

        vectors = self._model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in vectors]
