"""FastAPI dependency providers.

Every service is constructed here, once, from Settings — routes never
instantiate anything themselves. This is what makes the services trivially
mockable in tests: override any of these with `app.dependency_overrides`.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowRateLimiter
from app.llm.claude_client import ClaudeClient
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.reranker import CrossEncoderReranker, get_reranker
from app.rag.vector_store import ChromaVectorStore, VectorStore
from app.services.chat_service import ChatService
from app.services.conversation_store import ConversationStore
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
def get_rate_limiter() -> SlidingWindowRateLimiter:
    settings = get_settings()
    return SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )


@lru_cache
def get_conversation_store() -> ConversationStore:
    settings = get_settings()
    return ConversationStore(max_turns_per_session=settings.max_history_turns)


def get_ingestion_service(settings: Settings | None = None) -> IngestionService:
    settings = settings or get_settings()
    return IngestionService(
        embedding_provider=get_embedding_provider_dep(),
        vector_store=get_vector_store(),
        document_registry=get_document_registry(),
        upload_dir=settings.upload_dir,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        max_file_size_mb=settings.max_file_size_mb,
    )


def get_chat_service(settings: Settings | None = None) -> ChatService:
    settings = settings or get_settings()
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
    )
