"""Vector Indexing Pipeline.

Orchestrates embedding generation, idempotent upserting into the vector database,
and dense semantic search retrieval.
"""

import json
from pathlib import Path
from typing import Any

from config.settings import settings
from src.core.logging import app_logger
from src.core.models import Chunk
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.factory import get_embedding_provider
from src.vectorstore.base import BaseVectorStore, SearchResult
from src.vectorstore.factory import get_vector_store


class VectorIndexingPipeline:
    """Manages batch vectorization of chunks and nearest-neighbor search."""

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store(dimension=self.embedding_provider.dimension)
        self.manifest_path = manifest_path or (settings.DATA_VECTORSTORE_DIR / "indexed_manifest.json")
        self._indexed_hashes: set[str] = self._load_manifest()

    def _load_manifest(self) -> set[str]:
        """Load set of already-indexed chunk content hashes."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("indexed_hashes", []))
            except Exception as e:
                app_logger.warning("Could not read vector manifest: %s. Re-indexing.", e)
        return set()

    def _save_manifest(self) -> None:
        """Persist indexed content hashes."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump({"indexed_hashes": list(self._indexed_hashes)}, f, indent=2)

    def index_chunks(self, chunks: list[Chunk], force: bool = False) -> int:
        """Embed and index chunks idempotently.

        Args:
            chunks: List of Chunk objects.
            force: If True, re-indexes chunks even if hash was already recorded.

        Returns:
            Number of new chunks embedded and stored.
        """
        if not chunks:
            return 0

        chunks_to_index: list[Chunk] = []
        for c in chunks:
            if force or (c.content_hash not in self._indexed_hashes):
                chunks_to_index.append(c)

        if not chunks_to_index:
            app_logger.info("All %d chunks already indexed in vector store.", len(chunks))
            return 0

        app_logger.info("Generating embeddings for %d new chunks...", len(chunks_to_index))
        texts = [c.content for c in chunks_to_index]
        embeddings = self.embedding_provider.embed_documents(texts)

        count = self.vector_store.upsert_chunks(chunks_to_index, embeddings)

        for c in chunks_to_index:
            self._indexed_hashes.add(c.content_hash)
        self._save_manifest()

        return count

    def index_all_chunk_files(self, chunks_dir: Path | None = None) -> int:
        """Load and index all chunk JSON files in data/chunks/."""
        c_dir = chunks_dir or settings.DATA_CHUNKS_DIR
        chunk_files = list(c_dir.glob("chunks_*.json"))

        total_indexed = 0
        for cf in chunk_files:
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chunks = [Chunk.model_validate(c) for c in data]
                count = self.index_chunks(chunks)
                total_indexed += count
            except Exception as e:
                app_logger.error("Failed to index chunks from %s: %s", cf.name, e)

        return total_indexed

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform dense vector search for a natural language query."""
        query_vector = self.embedding_provider.embed_query(query)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k, filters=filters)
