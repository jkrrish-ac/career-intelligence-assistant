"""Orchestrates: guardrails -> hybrid retrieve -> rerank -> Claude call ->
response assembly. This is the "brain" of the /chat endpoint; the route
itself stays a thin translation to/from HTTP.
"""

from __future__ import annotations

import time

from app.core.exceptions import NoDocumentsUploadedError
from app.core.logging import get_logger
from app.core.rate_limit import SlidingWindowRateLimiter
from app.llm.claude_client import ClaudeClient
from app.models.schemas import (
    ChatResponse,
    RetrievedChunk,
    SourceRef,
    TimingInfo,
    TokenUsage,
)
from app.rag.embeddings import EmbeddingProvider
from app.rag.reranker import CrossEncoderReranker, rerank_candidates
from app.rag.retrieval import hybrid_retrieve
from app.rag.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry

logger = get_logger(__name__)

# A retrieval or rerank score below this is treated as "not actually
# relevant" for the purpose of the grounding guardrail — the LLM still sees
# whatever came back, but the response is flagged `grounded=False` and the
# system prompt already instructs it to say so rather than reach for outside
# knowledge. Threshold is conservative on purpose: false "ungrounded" flags
# are cheaper than false confidence.
_MIN_RELEVANT_RERANK_SCORE = -2.0


class ChatService:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: CrossEncoderReranker,
        document_registry: DocumentRegistry,
        claude_client: ClaudeClient,
        rate_limiter: SlidingWindowRateLimiter,
        candidate_k: int,
        final_k: int,
        rerank_enabled: bool,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker
        self._document_registry = document_registry
        self._claude_client = claude_client
        self._rate_limiter = rate_limiter
        self._candidate_k = candidate_k
        self._final_k = final_k
        self._rerank_enabled = rerank_enabled

    async def answer_question(self, *, message: str, session_id: str | None) -> ChatResponse:
        self._rate_limiter.check(key=session_id or "anonymous")

        known_documents = self._document_registry.list_all()
        if not known_documents:
            raise NoDocumentsUploadedError(
                "Upload at least a resume and one job description before asking questions."
            )

        retrieval_start = time.perf_counter()
        candidates = await hybrid_retrieve(
            query=message,
            known_documents=known_documents,
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            candidate_k=self._candidate_k,
        )
        retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        rerank_ms: float | None = None
        final_chunks: list[RetrievedChunk]
        if self._rerank_enabled and candidates:
            rerank_start = time.perf_counter()
            final_chunks = await rerank_candidates(
                query=message,
                candidates=candidates,
                reranker=self._reranker,
                top_k=self._final_k,
            )
            rerank_ms = round((time.perf_counter() - rerank_start) * 1000, 2)
        else:
            final_chunks = candidates[: self._final_k]

        grounded = any(
            (c.rerank_score if c.rerank_score is not None else c.retrieval_score)
            > _MIN_RELEVANT_RERANK_SCORE
            for c in final_chunks
        )

        llm_result = await self._claude_client.answer(message, final_chunks)

        sources = [
            SourceRef(
                document_id=c.chunk.document_id,
                label=c.chunk.label,
                section=c.chunk.section,
                retrieval_score=c.retrieval_score,
                rerank_score=c.rerank_score,
                snippet=c.chunk.text[:280],
            )
            for c in final_chunks
        ]

        logger.info(
            "chat_answered",
            message=message,
            grounded=grounded,
            source_count=len(sources),
        )

        return ChatResponse(
            answer=llm_result["answer"],
            sources=sources,
            timing=TimingInfo(
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                llm_ms=llm_result["llm_ms"],
            ),
            token_usage=TokenUsage(
                input_tokens=llm_result["input_tokens"],
                output_tokens=llm_result["output_tokens"],
            ),
            grounded=grounded,
        )
