"""Deterministic Mock Embedding Provider for fast, zero-dependency unit tests.

Generates normalized, reproducible dense vectors based on text hashing.
Enables full RAG pipeline execution without GPU, PyTorch, or remote API dependencies.
"""

import hashlib
import math
from src.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic unit-normalized float vectors via text hashing."""

    def __init__(self, model_name: str = "mock-embedding-v1", dimension: int = 384) -> None:
        super().__init__(model_name=model_name, dimension=dimension)

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text into a deterministic, unit-normalized vector."""
        # Use multiple salted SHA-256 hashes to fill vector dimension
        vector: list[float] = []
        salt = 0
        while len(vector) < self._dimension:
            seed_bytes = f"{salt}:{text}".encode("utf-8")
            h = hashlib.sha256(seed_bytes).digest()
            # Extract 4-byte floats from hash
            for i in range(0, len(h), 4):
                val = int.from_bytes(h[i : i + 4], byteorder="big", signed=True)
                vector.append(float(val) / (2**31))
                if len(vector) == self._dimension:
                    break
            salt += 1

        # L2 Normalize
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return [0.0] * self._dimension
        return [round(x / magnitude, 6) for x in vector]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query text deterministically."""
        return self._hash_to_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts deterministically."""
        return [self._hash_to_vector(t) for t in texts]
