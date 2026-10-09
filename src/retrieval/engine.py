"""Unified Retrieval Engine for BFSI Policy Research.

Coordinates Dense Vector Search, BM25 Keyword Search, Hybrid Reciprocal Rank Fusion,
and Cross-Encoder Reranking into a single, high-performance interface.
"""

import json
from pathlib import Path
from typing import Any, Literal

from config.settings import settings
from src.core.logging import app_logger
from src.core.models import Chunk
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.factory import get_embedding_provider
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker_base import BaseReranker
from src.vectorstore.base import BaseVectorStore, SearchResult
from src.vectorstore.factory import get_vector_store


class RetrievalEngine:
    """Master retrieval coordinator supporting Dense, BM25, Hybrid, and Reranked modes."""

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        reranker: BaseReranker | None = None,
        bm25_retriever: BM25Retriever | None = None,
        chunks_dir: Path | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store(dimension=self.embedding_provider.dimension)
        self.reranker = reranker
        self.hybrid_fusion = HybridRetriever(k=60)
        self.chunks_dir = chunks_dir or settings.DATA_CHUNKS_DIR

        # Initialize or load BM25 retriever
        if bm25_retriever:
            self.bm25_retriever = bm25_retriever
        else:
            self.bm25_retriever = BM25Retriever()
            self._sync_bm25_index()

    def _sync_bm25_index(self) -> int:
        """Load chunks from disk to initialize or refresh BM25 corpus."""
        all_chunks: list[Chunk] = []
        if self.chunks_dir.exists():
            for cf in self.chunks_dir.glob("chunks_*.json"):
                try:
                    with open(cf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    all_chunks.extend([Chunk.model_validate(c) for c in data])
                except Exception as e:
                    app_logger.warning("Could not load chunk file %s for BM25: %s", cf.name, e)

        if all_chunks:
            self.bm25_retriever.index_chunks(all_chunks)
        return len(all_chunks)

    def search(
        self,
        query: str,
        mode: Literal["dense", "bm25", "hybrid"] = "hybrid",
        use_reranker: bool = False,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Execute search according to requested retrieval mode.

        Args:
            query: User's question or search query.
            mode: 'dense' (pure vector), 'bm25' (keyword), or 'hybrid' (RRF fusion).
            use_reranker: If True and reranker is available, applies cross-encoder reranking.
            top_k: Number of final results to return.
            filters: Optional metadata filtering dictionary.

        Returns:
            List of ranked SearchResult objects.
        """
        # Over-retrieve candidates if reranker is requested
        retrieval_k = top_k * 3 if use_reranker else top_k

        results: list[SearchResult] = []

        if mode == "dense":
            q_vector = self.embedding_provider.embed_query(query)
            results = self.vector_store.search(q_vector, top_k=retrieval_k, filters=filters)

        elif mode == "bm25":
            results = self.bm25_retriever.search(query, top_k=retrieval_k, filters=filters)

        elif mode == "hybrid":
            # Retrieve dense candidates
            q_vector = self.embedding_provider.embed_query(query)
            dense_res = self.vector_store.search(q_vector, top_k=retrieval_k * 2, filters=filters)

            # Retrieve BM25 candidates
            bm25_res = self.bm25_retriever.search(query, top_k=retrieval_k * 2, filters=filters)

            # Reciprocal rank fusion
            results = self.hybrid_fusion.reciprocal_rank_fusion(
                dense_results=dense_res,
                bm25_results=bm25_res,
                top_k=retrieval_k,
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        # Optional Cross-Encoder Reranking
        if use_reranker and self.reranker and results:
            app_logger.info("Applying cross-encoder reranker to %d candidates...", len(results))
            results = self.reranker.rerank(query=query, candidates=results, top_k=top_k)
        else:
            results = results[:top_k]

        return results
