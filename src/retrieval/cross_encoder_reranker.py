"""Cross-Encoder Reranker using Sentence-Transformers.

Scores (query, chunk_text) pairs through deep bidirectional attention,
yielding significantly higher ranking precision than bi-encoders alone.
"""

from src.core.logging import app_logger
from src.retrieval.reranker_base import BaseReranker
from src.vectorstore.base import SearchResult


class CrossEncoderReranker(BaseReranker):
    """Local Cross-Encoder reranker using HuggingFace / sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            app_logger.info("Loading CrossEncoder model '%s'...", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
            app_logger.info("CrossEncoder model loaded successfully.")
        except Exception as e:
            app_logger.warning("Could not load CrossEncoder model '%s': %s. Will fallback to mock.", self.model_name, e)
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        if not self._model:
            # Fallback to score-preserving mock
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]

        pairs = [[query, c.chunk.content] for c in candidates]
        scores = self._model.predict(pairs)

        # Normalize scores to [0, 1] range using sigmoid / min-max
        scored_candidates: list[tuple[float, SearchResult]] = []
        for score, cand in zip(scores, candidates):
            # Sigmoid normalization for unconstrained cross-encoder logits
            norm_score = 1.0 / (1.0 + float(2.71828 ** (-float(score))))
            scored_candidates.append((norm_score, cand))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        reranked: list[SearchResult] = []
        for rank, (score, cand) in enumerate(scored_candidates[:top_k], start=1):
            reranked.append(
                SearchResult(
                    chunk=cand.chunk,
                    score=round(score, 4),
                    source="reranked",
                    rank=rank,
                    raw_score=round(score, 4),
                )
            )

        return reranked
