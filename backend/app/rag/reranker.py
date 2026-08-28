"""Cross-encoder re-ranking of fused retrieval candidates.

Runs after hybrid retrieval, before context assembly, per PRD §5. Uses a
small local cross-encoder (same sentence-transformers family as the
embedding model, so no new dependency) rather than a fusion-score heuristic
— with the larger time budget this is the real thing the requirements ask
for, not an approximation of it.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from app.core.logging import get_logger, timed
from app.models.schemas import RetrievedChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        logger.info("loading_reranker_model", model_name=model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)

        rescored = [
            candidate.model_copy(update={"rerank_score": round(float(score), 4)})
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda c: c.rerank_score, reverse=True)
        return rescored[:top_k]


@lru_cache
def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)


@timed("rerank")
async def rerank_candidates(
    *,
    query: str,
    candidates: list[RetrievedChunk],
    reranker: CrossEncoderReranker,
    top_k: int,
) -> list[RetrievedChunk]:
    # CrossEncoder.predict is a blocking CPU call; offload it so it doesn't
    # stall the event loop while other requests are in flight.
    result = await asyncio.to_thread(reranker.rerank, query, candidates, top_k)
    logger.info("rerank_complete", candidate_count=len(candidates), kept=len(result))
    return result
