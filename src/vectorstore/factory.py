"""Vector Store Factory.

Instantiates ChromaDB or Qdrant vector store based on application settings.
"""

from typing import Literal
from config.settings import settings
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.qdrant_store import QdrantVectorStore


def get_vector_store(
    provider: Literal["chromadb", "qdrant"] | str | None = None,
    collection_name: str | None = None,
    dimension: int = 384,
    in_memory: bool = False,
) -> BaseVectorStore:
    """Instantiate configured vector database store."""
    prov = (provider or settings.VECTOR_DB_PROVIDER).lower()
    col = collection_name or settings.COLLECTION_NAME

    if prov == "chromadb":
        return ChromaVectorStore(
            collection_name=col,
            dimension=dimension,
            persist_dir=settings.DATA_VECTORSTORE_DIR / "chroma",
            in_memory=in_memory,
        )
    elif prov == "qdrant":
        return QdrantVectorStore(
            collection_name=col,
            dimension=dimension,
            persist_path=settings.QDRANT_PATH,
            in_memory=in_memory,
        )
    else:
        raise ValueError(f"Unsupported vector DB provider: {prov}. Choose 'qdrant' or 'chromadb'.")
