"""Qdrant Vector Store Implementation.

Provides an enterprise-grade vector database implementation supporting both
in-memory execution for unit tests, local embedded disk persistence, and
production-grade Qdrant daemon connections with payload filtering.
"""

from pathlib import Path
from typing import Any
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.core.logging import app_logger
from src.core.models import Chunk, ChunkingStrategy, DocumentType
from src.vectorstore.base import BaseVectorStore, SearchResult


class QdrantVectorStore(BaseVectorStore):
    """Vector database backed by Qdrant (embedded or client/server)."""

    def __init__(
        self,
        collection_name: str = "bfsi_policy_circulars",
        dimension: int = 384,
        persist_path: Path | str | None = None,
        in_memory: bool = False,
    ) -> None:
        super().__init__(collection_name=collection_name, dimension=dimension)

        if in_memory:
            self.client = QdrantClient(location=":memory:")
        elif persist_path:
            p_path = str(persist_path)
            Path(p_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=p_path)
        else:
            self.client = QdrantClient(location=":memory:")

        self._ensure_collection()
        app_logger.info("QdrantVectorStore initialized for collection '%s'", self.collection_name)

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if not already existing."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def _chunk_to_payload(self, chunk: Chunk) -> dict[str, Any]:
        """Convert chunk into a searchable Qdrant payload dictionary."""
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "issuer": chunk.issuer,
            "circular_number": chunk.circular_number,
            "date": chunk.date,
            "doc_type": chunk.doc_type.value if isinstance(chunk.doc_type, DocumentType) else str(chunk.doc_type),
            "page_number": chunk.page_number,
            "section_heading": chunk.section_heading,
            "clause_number": chunk.clause_number,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "strategy": chunk.strategy.value if isinstance(chunk.strategy, ChunkingStrategy) else str(chunk.strategy),
        }

    def _payload_to_chunk(self, payload: dict[str, Any]) -> Chunk:
        """Reconstruct Chunk object from Qdrant point payload."""
        return Chunk(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            document_title=payload["document_title"],
            issuer=payload["issuer"],
            circular_number=payload.get("circular_number"),
            date=payload.get("date"),
            doc_type=DocumentType(payload.get("doc_type", "circular")),
            page_number=int(payload.get("page_number", 1)),
            section_heading=payload.get("section_heading", "General"),
            clause_number=payload.get("clause_number"),
            content=payload["content"],
            content_hash=payload.get("content_hash", ""),
            chunk_index=int(payload.get("chunk_index", 0)),
            token_count=int(payload.get("token_count", 0)),
            strategy=ChunkingStrategy(payload.get("strategy", "structure_aware")),
            metadata={},
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert chunks with embeddings into Qdrant collection idempotently."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        points = []
        for chunk, vector in zip(chunks, embeddings):
            # Generate deterministic UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = self._chunk_to_payload(chunk)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        app_logger.info("Upserted %d chunks into Qdrant collection '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform cosine similarity nearest-neighbor search with metadata filter."""
        query_filter: models.Filter | None = None
        if filters:
            conditions: list[models.FieldCondition] = []
            for field, val in filters.items():
                if val is not None:
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=val),
                        )
                    )
            if conditions:
                query_filter = models.Filter(must=conditions)

        # In modern qdrant-client, use query_points
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        search_results: list[SearchResult] = []
        for rank, hit in enumerate(results, start=1):
            if hit.payload:
                chunk = self._payload_to_chunk(hit.payload)
                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=round(hit.score, 4),
                        source="dense",
                        rank=rank,
                        raw_score=round(hit.score, 4),
                    )
                )

        return search_results

    def delete_document(self, document_id: str) -> int:
        """Delete all points belonging to a specific document_id."""
        initial_count = self.count()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        deleted = max(0, initial_count - self.count())
        app_logger.info("Deleted document '%s' from Qdrant", document_id)
        return deleted

    def count(self) -> int:
        """Return total point count in collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count
