"""Orchestrates: guardrails -> hybrid retrieve -> rerank -> Claude call ->
response assembly. This is the "brain" of the /chat endpoints; the routes
themselves stay a thin translation to/from HTTP (or SSE).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from app.core.exceptions import NoDocumentsUploadedError
from app.core.logging import get_logger
from app.core.rate_limit import RedisRateLimiter, SlidingWindowRateLimiter
from app.llm.claude_client import ClaudeClient
from app.models.schemas import (
    ChatResponse,
    ConversationTurn,
    DocumentMetadata,
    RetrievedChunk,
    SourceRef,
    TimingInfo,
    TokenUsage,
)
from app.rag.embeddings import EmbeddingProvider
from app.rag.query_classifier import classify_query_target_llm
from app.rag.reranker import CrossEncoderReranker, rerank_candidates
from app.rag.retrieval import classify_query_target, hybrid_retrieve
from app.rag.vector_store import VectorStore
from app.services.conversation_store import ConversationStore, RedisConversationStore
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
        rate_limiter: SlidingWindowRateLimiter | RedisRateLimiter,
        conversation_store: ConversationStore | RedisConversationStore,
        candidate_k: int,
        final_k: int,
        rerank_enabled: bool,
        query_classifier_mode: str = "llm",
        query_classifier_timeout_seconds: float = 3.0,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker
        self._document_registry = document_registry
        self._claude_client = claude_client
        self._rate_limiter = rate_limiter
        self._conversation_store = conversation_store
        self._candidate_k = candidate_k
        self._final_k = final_k
        self._rerank_enabled = rerank_enabled
        self._query_classifier_mode = query_classifier_mode
        self._query_classifier_timeout_seconds = query_classifier_timeout_seconds

    async def _resolve_query_target(
        self, message: str, known_documents: list[DocumentMetadata]
    ) -> dict | None:
        """Picks the `where` filter passed to `hybrid_retrieve`. Tries the
        LLM classifier first (bounded by a timeout so a slow Claude response
        never becomes a slow chat response), falling back to the regex
        heuristic on *any* problem -- timeout, API error, malformed output.
        The heuristic is deliberately never removed; it's the safety net
        that makes the LLM path low-risk to run by default."""
        if self._query_classifier_mode != "llm":
            return classify_query_target(message, known_documents)

        try:
            where = await asyncio.wait_for(
                classify_query_target_llm(message, known_documents, self._claude_client),
                timeout=self._query_classifier_timeout_seconds,
            )
            logger.info("query_classified_via_llm", where=where)
            return where
        except Exception as exc:
            logger.warning(
                "query_classifier_llm_failed_using_heuristic",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return classify_query_target(message, known_documents)

    async def _prepare_context(
        self, *, message: str, session_id: str | None
    ) -> tuple[list[RetrievedChunk], float, float | None, bool]:
        """Guardrails + hybrid retrieve + rerank — the part identical
        between the plain and streaming chat paths."""
        await self._rate_limiter.check(key=session_id or "anonymous")

        known_documents = self._document_registry.list_all()
        if not known_documents:
            raise NoDocumentsUploadedError(
                "Upload at least a resume and one job description before asking questions."
            )

        where = await self._resolve_query_target(message, known_documents)

        retrieval_start = time.perf_counter()
        candidates = await hybrid_retrieve(
            query=message,
            known_documents=known_documents,
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            candidate_k=self._candidate_k,
            where=where,
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

        return final_chunks, retrieval_ms, rerank_ms, grounded

    @staticmethod
    def _to_sources(final_chunks: list[RetrievedChunk]) -> list[SourceRef]:
        return [
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

    async def _history_for(self, session_id: str | None) -> list[ConversationTurn]:
        if not session_id:
            return []
        return await self._conversation_store.get_history(session_id)

    async def _record_turn(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        await self._conversation_store.append(session_id, "user", question)
        await self._conversation_store.append(session_id, "assistant", answer)

    async def answer_question(self, *, message: str, session_id: str | None) -> ChatResponse:
        final_chunks, retrieval_ms, rerank_ms, grounded = await self._prepare_context(
            message=message, session_id=session_id
        )
        history = await self._history_for(session_id)

        llm_result = await self._claude_client.answer(message, final_chunks, history)
        await self._record_turn(session_id, message, llm_result["answer"])

        logger.info(
            "chat_answered",
            message=message,
            grounded=grounded,
            source_count=len(final_chunks),
            history_turns=len(history),
        )

        return ChatResponse(
            answer=llm_result["answer"],
            sources=self._to_sources(final_chunks),
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

    async def stream_answer(
        self, *, message: str, session_id: str | None
    ) -> AsyncIterator[dict]:
        """Same pipeline as `answer_question`, but yields incremental
        events: retrieval metadata first (so the UI can show sources while
        the answer is still streaming in), then text deltas, then a final
        event with timing/usage — mirroring ChatResponse's fields so the
        frontend can assemble the same shape either way."""
        final_chunks, retrieval_ms, rerank_ms, grounded = await self._prepare_context(
            message=message, session_id=session_id
        )
        history = await self._history_for(session_id)
        sources = self._to_sources(final_chunks)

        yield {"type": "context", "sources": [s.model_dump() for s in sources], "grounded": grounded}

        answer_text = ""
        async for event in self._claude_client.stream_answer(message, final_chunks, history):
            if event["type"] == "delta":
                answer_text += event["text"]
                yield {"type": "delta", "text": event["text"]}
            else:  # "done"
                await self._record_turn(session_id, message, event["answer"])
                logger.info(
                    "chat_stream_answered",
                    message=message,
                    grounded=grounded,
                    source_count=len(final_chunks),
                )
                yield {
                    "type": "done",
                    "timing": {
                        "retrieval_ms": retrieval_ms,
                        "rerank_ms": rerank_ms,
                        "llm_ms": event["llm_ms"],
                    },
                    "token_usage": {
                        "input_tokens": event["input_tokens"],
                        "output_tokens": event["output_tokens"],
                    },
                }
