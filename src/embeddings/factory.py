"""Embedding Provider Factory.

Dynamically selects and instantiates embedding providers based on config/settings.
"""

import os
from config.settings import settings
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.local_provider import LocalSentenceTransformersProvider
from src.embeddings.mock_provider import MockEmbeddingProvider


def get_embedding_provider(
    provider_type: str | None = None,
    model_name: str | None = None,
) -> BaseEmbeddingProvider:
    """Instantiate configured embedding provider.

    Options: 'local', 'voyage', 'openai', 'mock'.
    """
    prov = (provider_type or settings.EMBEDDING_PROVIDER).lower()
    m_name = model_name or settings.EMBEDDING_MODEL_NAME

    if prov == "mock":
        return MockEmbeddingProvider(model_name=m_name, dimension=384)

    elif prov == "local":
        return LocalSentenceTransformersProvider(
            model_name=m_name,
            dimension=384 if "MiniLM" in m_name else 512,
        )

    elif prov in ["openai", "voyage"]:
        from src.embeddings.hosted_provider import HostedEmbeddingProvider

        api_key = os.getenv(f"{prov.upper()}_API_KEY", "")
        if not api_key:
            from src.core.logging import app_logger

            app_logger.warning(
                "No API key found for '%s'. Falling back to local SentenceTransformers provider.",
                prov,
            )
            return LocalSentenceTransformersProvider(model_name=m_name)

        return HostedEmbeddingProvider(
            api_key=api_key,
            provider="voyage" if prov == "voyage" else "openai",
            model_name=m_name,
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {prov}")
