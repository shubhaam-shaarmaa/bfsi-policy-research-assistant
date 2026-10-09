"""Hosted API Embedding Provider (Voyage AI / OpenAI).

Connects to hosted embedding endpoints (e.g., Voyage-finance-2, text-embedding-3-small).
Follows standard REST embedding spec with batching, retries, and API key auth.
"""

from typing import Literal
import httpx

from src.core.logging import app_logger
from src.embeddings.base import BaseEmbeddingProvider


class HostedEmbeddingProvider(BaseEmbeddingProvider):
    """Hosted embedding provider for OpenAI, Voyage AI, or custom REST endpoints."""

    def __init__(
        self,
        api_key: str,
        provider: Literal["openai", "voyage"] = "openai",
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key
        self.provider = provider

        if base_url:
            self.base_url = base_url
        elif provider == "voyage":
            self.base_url = "https://api.voyageai.com/v1"
            self.model_name = model_name or "voyage-finance-2"
            self._dimension = 1024
        else:
            self.base_url = "https://api.openai.com/v1"
            self.model_name = model_name or "text-embedding-3-small"
            self._dimension = 1536

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text via REST call."""
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed document chunks."""
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": self.model_name,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                # Sort by index to maintain ordering
                embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
                return embeddings
        except Exception as e:
            app_logger.error("Hosted embedding request failed: %s", e)
            raise
