"""FastAPI dependency providers.

Every service is constructed here, once, from Settings — routes never
instantiate anything themselves. This is what makes the services trivially
mockable in tests: override any of these with `app.dependency_overrides`.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.rate_limit import RedisRateLimiter, SlidingWindowRateLimiter
from app.llm.claude_client import ClaudeClient
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.reranker import CrossEncoderReranker, get_reranker
from app.rag.vector_store import ChromaVectorStore, VectorStore
from app.services.chat_service import ChatService
from app.services.conversation_store import ConversationStore, RedisConversationStore
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService


@lru_cache
def get_document_registry() -> DocumentRegistry:
    settings = get_settings()
    return DocumentRegistry(storage_path=settings.upload_dir.parent / "documents.json")


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def get_embedding_provider_dep() -> EmbeddingProvider:
    settings = get_settings()
    return get_embedding_provider(settings.embedding_provider, settings.embedding_model)


@lru_cache
def get_reranker_dep() -> CrossEncoderReranker:
    settings = get_settings()
    return get_reranker(settings.reranker_model)


@lru_cache
def get_claude_client() -> ClaudeClient:
    settings = get_settings()
    return ClaudeClient(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
    )


@lru_cache
def get_redis_client():
    """Only constructed (and only imports `redis.asyncio`) when REDIS_URL is
    actually set — a machine without Redis running never pays for this or
    even needs the import to succeed lazily-imported here, not at module
    load time, so `redis` stays an optional runtime dependency in spirit
    even though it's now a hard requirements.txt pin."""
    import redis.asyncio as redis

    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_rate_limiter() -> SlidingWindowRateLimiter | RedisRateLimiter:
    settings = get_settings()
    if settings.redis_url:
        return RedisRateLimiter(
            redis_client=get_redis_client(),
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    return SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )


@lru_cache
def get_conversation_store() -> ConversationStore | RedisConversationStore:
    settings = get_settings()
    if settings.redis_url:
        return RedisConversationStore(
            redis_client=get_redis_client(),
            max_turns_per_session=settings.max_history_turns,
            session_ttl_seconds=settings.redis_session_ttl_seconds,
        )
    return ConversationStore(max_turns_per_session=settings.max_history_turns)


async def get_arq_pool(request: Request, settings: Settings = Depends(get_settings)):
    """Lazily creates one arq (Redis-backed job queue) pool per app process
    and caches it on `app.state`, so it isn't reconnected on every request.
    Returns None -- and never touches Redis -- when `INGESTION_MODE` isn't
    `async`; that's the common case, so this is a no-op for every
    deployment that hasn't opted into async ingestion (see
    `app/services/ingestion_service.py::register_pending`/`process_pending`
    and `app/worker.py` for the rest of that path)."""
    if settings.ingestion_mode != "async":
        return None

    pool = getattr(request.app.state, "arq_pool", None)
    if pool is None:
        from arq import create_pool
        from arq.connections import RedisSettings

        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        request.app.state.arq_pool = pool
    return pool


def get_ingestion_service(settings: Settings = Depends(get_settings)) -> IngestionService:
    # `settings` MUST be resolved via Depends(get_settings), not a plain
    # `Settings | None = None` default — a bare BaseModel-typed parameter
    # with no Depends/Query/Path marker is exactly what FastAPI treats as an
    # implicit *request body field* on any route that takes this as its own
    # Depends(...). That bug silently turned POST /documents and /chat's
    # request bodies into `{"request": {...}, "settings": {...}}` — caught
    # by tests/test_api_routes.py, which is why those tests exist at all.
    return IngestionService(
        embedding_provider=get_embedding_provider_dep(),
        vector_store=get_vector_store(),
        document_registry=get_document_registry(),
        upload_dir=settings.upload_dir,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        max_file_size_mb=settings.max_file_size_mb,
    )


def get_chat_service(settings: Settings = Depends(get_settings)) -> ChatService:
    return ChatService(
        embedding_provider=get_embedding_provider_dep(),
        vector_store=get_vector_store(),
        reranker=get_reranker_dep(),
        document_registry=get_document_registry(),
        claude_client=get_claude_client(),
        rate_limiter=get_rate_limiter(),
        conversation_store=get_conversation_store(),
        candidate_k=settings.retrieval_candidate_k,
        final_k=settings.retrieval_final_k,
        rerank_enabled=settings.rerank_enabled,
        query_classifier_mode=settings.query_classifier_mode,
        query_classifier_timeout_seconds=settings.query_classifier_timeout_seconds,
    )
