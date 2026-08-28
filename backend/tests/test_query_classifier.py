"""LLM-based query classification (`app/rag/query_classifier.py`) and its
wiring into `ChatService._resolve_query_target`, including the fallback to
the regex heuristic on any failure/timeout and the config toggle that skips
the LLM call entirely."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from app.models.schemas import Chunk, DocumentMetadata, SourceType
from app.rag.embeddings import EmbeddingProvider
from app.rag.query_classifier import classify_query_target_llm
from app.rag.vector_store import VectorStore
from app.services.chat_service import ChatService
from app.services.conversation_store import ConversationStore

_RESUME_DOC = DocumentMetadata(
    document_id="resume-1",
    source_type=SourceType.RESUME,
    label="jk_resume",
    filename="resume.pdf",
    uploaded_at=datetime.now(UTC),
    chunk_count=3,
)
_JD1 = DocumentMetadata(
    document_id="jd-1",
    source_type=SourceType.JOB_DESCRIPTION,
    label="Job #1",
    filename="jd1.pdf",
    uploaded_at=datetime.now(UTC),
    chunk_count=3,
)
_KNOWN_DOCS = [_RESUME_DOC, _JD1]


class _StubClaudeClient:
    """Duck-typed stand-in exposing only what the classifier path needs:
    `.classify()`. Configurable to return a target, raise, or hang past a
    timeout, to exercise all three of `_resolve_query_target`'s branches."""

    def __init__(self, *, target: str | None = None, raises: Exception | None = None, delay: float = 0.0):
        self._target = target
        self._raises = raises
        self._delay = delay
        self.calls = 0

    async def classify(self, query: str, known_documents: list[DocumentMetadata]) -> str:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises:
            raise self._raises
        return self._target

    async def answer(self, query, context_chunks, history=None):
        return {"answer": "ok", "input_tokens": 1, "output_tokens": 1, "llm_ms": 1.0}


# --- classify_query_target_llm (pure function) -------------------------------


@pytest.mark.asyncio
async def test_classify_llm_resolves_a_known_document_id():
    where = await classify_query_target_llm(
        "How does my experience align with Job #1?", _KNOWN_DOCS, _StubClaudeClient(target="jd-1")
    )
    assert where == {"document_id": "jd-1"}


@pytest.mark.asyncio
async def test_classify_llm_resolves_resume_source_type():
    where = await classify_query_target_llm(
        "What does my resume say about Python?", _KNOWN_DOCS, _StubClaudeClient(target="resume")
    )
    assert where == {"source_type": "resume"}


@pytest.mark.asyncio
async def test_classify_llm_all_target_means_no_filter():
    where = await classify_query_target_llm(
        "What's a reasonable interview timeline?", _KNOWN_DOCS, _StubClaudeClient(target="all")
    )
    assert where is None


@pytest.mark.asyncio
async def test_classify_llm_skips_the_call_when_no_documents_are_known():
    claude_client = _StubClaudeClient(target="all")
    where = await classify_query_target_llm("anything", [], claude_client)
    assert where is None
    assert claude_client.calls == 0


# --- ChatService._resolve_query_target: fallback wiring ----------------------


def _make_service(claude_client, *, query_classifier_mode="llm", query_classifier_timeout_seconds=3.0):
    chunk = Chunk(
        chunk_id="c1",
        document_id="jd-1",
        source_type=SourceType.JOB_DESCRIPTION,
        label="Job #1",
        section="Requirements",
        text="Kubernetes experience required.",
    )

    class _FakeEmbeddingProvider(EmbeddingProvider):
        def embed(self, texts):
            return [[0.0] for _ in texts]

        @property
        def dimension(self):
            return 1

    class _FakeVectorStore(VectorStore):
        def add(self, chunks, embeddings):
            raise NotImplementedError

        def query(self, query_embedding, top_k, where=None):
            return [(chunk, 1.0)]

        def get_all_chunks(self, where=None):
            return [chunk]

        def delete_document(self, document_id):
            raise NotImplementedError

    class _Registry:
        def list_all(self):
            return _KNOWN_DOCS

    class _NoopRateLimiter:
        async def check(self, key):
            return None

    return ChatService(
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=_FakeVectorStore(),
        reranker=None,
        document_registry=_Registry(),
        claude_client=claude_client,
        rate_limiter=_NoopRateLimiter(),
        conversation_store=ConversationStore(max_turns_per_session=10),
        candidate_k=5,
        final_k=3,
        rerank_enabled=False,
        query_classifier_mode=query_classifier_mode,
        query_classifier_timeout_seconds=query_classifier_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_resolve_query_target_uses_llm_result_when_it_succeeds():
    claude_client = _StubClaudeClient(target="jd-1")
    service = _make_service(claude_client)

    where = await service._resolve_query_target("How does my experience align with Job #1?", _KNOWN_DOCS)  # noqa: SLF001

    assert where == {"document_id": "jd-1"}
    assert claude_client.calls == 1


@pytest.mark.asyncio
async def test_resolve_query_target_falls_back_to_heuristic_on_llm_error():
    claude_client = _StubClaudeClient(raises=RuntimeError("Claude API unavailable"))
    service = _make_service(claude_client)

    where = await service._resolve_query_target("How does my experience align with Job #1?", _KNOWN_DOCS)  # noqa: SLF001

    # The heuristic recognizes "Job #1" explicitly, same as test_retrieval.py.
    assert where == {"document_id": "jd-1"}


@pytest.mark.asyncio
async def test_resolve_query_target_falls_back_to_heuristic_on_timeout():
    claude_client = _StubClaudeClient(target="jd-1", delay=10.0)
    service = _make_service(claude_client, query_classifier_timeout_seconds=0.05)

    where = await asyncio.wait_for(
        service._resolve_query_target("How does my experience align with Job #1?", _KNOWN_DOCS),  # noqa: SLF001
        timeout=2,
    )

    # Still correct, via the heuristic -- the slow LLM call never got to
    # finish, and the request didn't have to wait 10 seconds to find that out.
    assert where == {"document_id": "jd-1"}


@pytest.mark.asyncio
async def test_resolve_query_target_heuristic_mode_never_calls_the_llm():
    claude_client = _StubClaudeClient(target="jd-1")
    service = _make_service(claude_client, query_classifier_mode="heuristic")

    where = await service._resolve_query_target("How does my experience align with Job #1?", _KNOWN_DOCS)  # noqa: SLF001

    assert where == {"document_id": "jd-1"}
    assert claude_client.calls == 0
