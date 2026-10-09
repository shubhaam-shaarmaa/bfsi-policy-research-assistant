"""Hybrid Search using Reciprocal Rank Fusion (RRF).

Fuses ranked lists from Dense Vector Search and BM25 Keyword Search.
Scale-invariant, robust against score distribution mismatches, and optimized
for BFSI regulatory retrieval where exact statutory terms and conceptual
semantics must be jointly satisfied.
"""

from collections import defaultdict
from src.core.logging import app_logger
from src.core.models import Chunk
from src.vectorstore.base import SearchResult


class HybridRetriever:
    """Combines Dense Vector and BM25 retrievers using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60) -> None:
        """Initialize with RRF constant k (default 60 per standard Information Retrieval literature)."""
        self.k = k

    def reciprocal_rank_fusion(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int = 5,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[SearchResult]:
        """Fuse two ranked candidate lists using weighted Reciprocal Rank Fusion.

        Formula:
            RRF_Score(doc) = dense_weight / (k + rank_dense) + bm25_weight / (k + rank_bm25)
        """
        chunk_map: dict[str, Chunk] = {}
        rrf_scores: dict[str, float] = defaultdict(float)
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        # Process dense rankings
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            dense_ranks[cid] = rank
            rrf_scores[cid] += dense_weight * (1.0 / (self.k + rank))

        # Process BM25 rankings
        for rank, res in enumerate(bm25_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            bm25_ranks[cid] = rank
            rrf_scores[cid] += bm25_weight * (1.0 / (self.k + rank))

        if not rrf_scores:
            return []

        # Sort chunks by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # Theoretical maximum RRF score for rank 1 in both:
        max_possible_rrf = (dense_weight / (self.k + 1)) + (bm25_weight / (self.k + 1))

        fused_results: list[SearchResult] = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            chunk = chunk_map[cid]
            raw_rrf = rrf_scores[cid]
            normalized_score = min(1.0, raw_rrf / max_possible_rrf) if max_possible_rrf > 0 else raw_rrf

            fused_results.append(
                SearchResult(
                    chunk=chunk,
                    score=round(normalized_score, 4),
                    source="hybrid",
                    rank=rank,
                    raw_score=round(raw_rrf, 6),
                )
            )

        app_logger.info(
            "Hybrid RRF fused %d dense + %d BM25 candidates -> %d final results.",
            len(dense_results),
            len(bm25_results),
            len(fused_results),
        )
        return fused_results
