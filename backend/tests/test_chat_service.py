from datetime import UTC, datetime

import pytest

from app.core.exceptions import NoDocumentsUploadedError, RateLimitExceededError
from app.models.schemas import Chunk, DocumentMetadata, SourceType
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.services.chat_service import ChatService
from app.services.conversation_store import ConversationStore


class _NoopRateLimiter:
    def check(self, key: str) -> None:
        return None


class _TrippedRateLimiter:
    def check(self, key: str) -> None:
        raise RateLimitExceededError("too many requests")


class _EmptyDocumentRegistry:
    def list_all(self):
        return []


class _OneResumeOneJDRegistry:
    def list_all(self):
        return [
            DocumentMetadata(
                document_id="resume-1",
                source_type=SourceType.RESUME,
                label="jk_resume",
                filename="resume.pdf",
                uploaded_at=datetime.now(UTC),
                chunk_count=1,
            ),
            DocumentMetadata(
                document_id="jd-1",
                source_type=SourceType.JOB_DESCRIPTION,
                label="Job #1",
                filename="jd.pdf",
                uploaded_at=datetime.now(UTC),
                chunk_count=1,
            ),
        ]


class _FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts):
        return [[0.0] for _ in texts]

    @property
    def dimension(self):
        return 1


class _FakeVectorStore(VectorStore):
    def __init__(self, chunks):
        self._chunks = chunks

    def add(self, chunks, embeddings):
        raise NotImplementedError

    def query(self, query_embedding, top_k, where=None):
        return [(c, 1.0) for c in self._chunks[:top_k]]

    def get_all_chunks(self, where=None):
        return list(self._chunks)

    def delete_document(self, document_id):
        raise NotImplementedError


class _RecordingClaudeClient:
    """Duck-typed stand-in for ClaudeClient — ChatService only calls
    `.answer(query, chunks, history)`, so no need to hit the real SDK."""

    def __init__(self, answer_text: str = "canned answer") -> None:
        self.calls: list[dict] = []
        self._answer_text = answer_text

    async def answer(self, query, context_chunks, history=None):
        self.calls.append({"query": query, "history": list(history or [])})
        return {
            "answer": self._answer_text,
            "input_tokens": 10,
            "output_tokens": 5,
            "llm_ms": 1.0,
        }


def _make_service(*, registry, rate_limiter, claude_client, conversation_store=None):
    chunk = Chunk(
        chunk_id="c1",
        document_id="jd-1",
        source_type=SourceType.JOB_DESCRIPTION,
        label="Job #1",
        section="Requirements",
        text="Kubernetes experience required.",
    )
    return ChatService(
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=_FakeVectorStore([chunk]),
        reranker=None,  # rerank_enabled=False below, so this is never touched
        document_registry=registry,
        claude_client=claude_client,
        rate_limiter=rate_limiter,
        conversation_store=conversation_store or ConversationStore(max_turns_per_session=10),
        candidate_k=5,
        final_k=3,
        rerank_enabled=False,
    )


@pytest.mark.asyncio
async def test_no_documents_uploaded_raises_before_any_retrieval():
    service = _make_service(
        registry=_EmptyDocumentRegistry(),
        rate_limiter=_NoopRateLimiter(),
        claude_client=_RecordingClaudeClient(),
    )

    with pytest.raises(NoDocumentsUploadedError):
        await service.answer_question(message="What skills am I missing?", session_id="s1")


@pytest.mark.asyncio
async def test_rate_limit_trip_prevents_retrieval_and_llm_call():
    claude_client = _RecordingClaudeClient()
    service = _make_service(
        registry=_OneResumeOneJDRegistry(),
        rate_limiter=_TrippedRateLimiter(),
        claude_client=claude_client,
    )

    with pytest.raises(RateLimitExceededError):
        await service.answer_question(message="What skills am I missing?", session_id="s1")

    assert claude_client.calls == [], "the LLM should never be called once the rate limit trips"


@pytest.mark.asyncio
async def test_conversation_history_is_recorded_and_passed_to_next_turn():
    claude_client = _RecordingClaudeClient(answer_text="first answer")
    conversation_store = ConversationStore(max_turns_per_session=10)
    service = _make_service(
        registry=_OneResumeOneJDRegistry(),
        rate_limiter=_NoopRateLimiter(),
        claude_client=claude_client,
        conversation_store=conversation_store,
    )

    await service.answer_question(message="What skills am I missing?", session_id="s1")
    # First turn should have gone out with no prior history.
    assert claude_client.calls[0]["history"] == []

    await service.answer_question(message="What about Job #2?", session_id="s1")
    # Second turn should see the first Q&A as history.
    assert claude_client.calls[1]["history"] == [
        {"role": "user", "content": "What skills am I missing?"},
        {"role": "assistant", "content": "first answer"},
    ]


@pytest.mark.asyncio
async def test_conversation_history_does_not_leak_across_sessions():
    claude_client = _RecordingClaudeClient()
    conversation_store = ConversationStore(max_turns_per_session=10)
    service = _make_service(
        registry=_OneResumeOneJDRegistry(),
        rate_limiter=_NoopRateLimiter(),
        claude_client=claude_client,
        conversation_store=conversation_store,
    )

    await service.answer_question(message="question in session A", session_id="session-a")
    await service.answer_question(message="question in session B", session_id="session-b")

    # Session B's call must not see session A's history.
    assert claude_client.calls[1]["history"] == []
