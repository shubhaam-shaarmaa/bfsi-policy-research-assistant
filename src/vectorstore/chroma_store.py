"""ChromaDB Vector Store Implementation.

Provides an embedded, zero-daemon vector database with HNSW indexing,
persistent SQLite storage, and rich metadata filtering.
"""

from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.logging import app_logger
from src.core.models import Chunk, ChunkingStrategy, DocumentType
from src.vectorstore.base import BaseVectorStore, SearchResult


class ChromaVectorStore(BaseVectorStore):
    """Vector database backed by embedded ChromaDB."""

    def __init__(
        self,
        collection_name: str = "bfsi_policy_circulars",
        dimension: int = 384,
        persist_dir: Path | str | None = None,
        in_memory: bool = False,
    ) -> None:
        super().__init__(collection_name=collection_name, dimension=dimension)

        if in_memory:
            self.client = chromadb.Client(ChromaSettings(is_persistent=False, anonymized_telemetry=False))
        else:
            p_dir = str(persist_dir) if persist_dir else "data/vectorstore/chroma"
            Path(p_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=p_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        app_logger.info("ChromaVectorStore initialized for collection '%s'", self.collection_name)

    def _sanitize_metadata(self, chunk: Chunk) -> dict[str, str | int | float | bool]:
        """Convert chunk metadata to Chroma-compliant flat primitives."""
        return {
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "issuer": chunk.issuer,
            "circular_number": chunk.circular_number or "",
            "date": chunk.date or "",
            "doc_type": chunk.doc_type.value if isinstance(chunk.doc_type, DocumentType) else str(chunk.doc_type),
            "page_number": chunk.page_number,
            "section_heading": chunk.section_heading,
            "clause_number": chunk.clause_number or "",
            "content_hash": chunk.content_hash,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "strategy": chunk.strategy.value if isinstance(chunk.strategy, ChunkingStrategy) else str(chunk.strategy),
        }

    def _reconstruct_chunk(self, chunk_id: str, document: str, meta: dict[str, Any]) -> Chunk:
        """Reconstruct domain Chunk model from ChromaDB document and metadata."""
        return Chunk(
            chunk_id=chunk_id,
            document_id=meta.get("document_id", "unknown"),
            document_title=meta.get("document_title", "Unknown"),
            issuer=meta.get("issuer", "Unknown"),
            circular_number=meta.get("circular_number") or None,
            date=meta.get("date") or None,
            doc_type=DocumentType(meta.get("doc_type", "circular")),
            page_number=int(meta.get("page_number", 1)),
            section_heading=meta.get("section_heading", "General"),
            clause_number=meta.get("clause_number") or None,
            content=document,
            content_hash=meta.get("content_hash", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            token_count=int(meta.get("token_count", 0)),
            strategy=ChunkingStrategy(meta.get("strategy", "structure_aware")),
            metadata={},
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert chunks with embeddings into ChromaDB collection idempotently."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk) for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        app_logger.info("Upserted %d chunks into Chroma collection '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform cosine similarity nearest-neighbor search with metadata filter."""
        where_clause: dict[str, Any] | None = None
        if filters:
            clean_filters = {k: v for k, v in filters.items() if v is not None}
            if len(clean_filters) == 1:
                k, v = next(iter(clean_filters.items()))
                where_clause = {k: {"$eq": v}}
            elif len(clean_filters) > 1:
                where_clause = {"$and": [{k: {"$eq": v}} for k, v in clean_filters.items()]}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, self.count())),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
            chunk = self._reconstruct_chunk(cid, doc, meta)
            # In Chroma cosine distance: dist is in [0, 2], where 0 is identical
            # Normalized similarity: max(0.0, 1.0 - dist)
            similarity = max(0.0, min(1.0, 1.0 - dist))
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=round(similarity, 4),
                    source="dense",
                    rank=rank,
                    raw_score=round(dist, 4),
                )
            )

        return search_results

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document_id."""
        initial_count = self.count()
        self.collection.delete(where={"document_id": {"$eq": document_id}})
        deleted = initial_count - self.count()
        app_logger.info("Deleted document '%s' (%d chunks removed)", document_id, deleted)
        return deleted

    def count(self) -> int:
        """Return total chunk count in collection."""
        return self.collection.count()
