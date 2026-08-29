"""Edge-case coverage beyond the happy-path and guardrail tests:

- malformed uploads that reach the real parsers (not stubbed out), so we
  verify actual parser exceptions map to the documented AppError subclasses
  instead of leaking a raw traceback
- concurrent ingestion against a single DocumentRegistry/VectorStore, to
  exercise the locking that's never been exercised by a test before
- a client disconnecting mid-stream, to verify the SSE generator chain
  (chat_service.stream_answer -> claude_client.stream_answer -> the route's
  event_source()) unwinds cleanly on cancellation instead of hanging or
  leaking the underlying Claude stream context
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.core.exceptions import DocumentParseError
from app.core.rate_limit import SlidingWindowRateLimiter
from app.main import app
from app.models.schemas import DocumentMetadata, SourceType
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.services.chat_service import ChatService
from app.services.conversation_store import ConversationStore
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService


class _FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts):
        return [[0.0] for _ in texts]

    @property
    def dimension(self):
        return 1


class _FakeVectorStore(VectorStore):
    def __init__(self):
        self.added = []

    def add(self, chunks, embeddings):
        self.added.append(chunks)

    def query(self, query_embedding, top_k, where=None):
        return []

    def get_all_chunks(self, where=None):
        return []

    def delete_document(self, document_id):
        pass


def _make_service(tmp_path: Path) -> IngestionService:
    return IngestionService(
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=_FakeVectorStore(),
        document_registry=DocumentRegistry(storage_path=tmp_path / "documents.json"),
        upload_dir=tmp_path / "uploads",
        chunk_size=500,
        chunk_overlap=50,
        max_file_size_mb=10,
    )


# --- Malformed uploads -------------------------------------------------------
# These exercise the *real* parsers (parse_txt/parse_pdf/parse_docx) rather
# than a stub, since the whole point is checking that parser-level failures
# come back as the documented AppError subclass rather than an unhandled
# exception (which would otherwise surface as a 500 through the route).


@pytest.mark.asyncio
async def test_zero_byte_txt_upload_raises_document_parse_error(tmp_path: Path):
    service = _make_service(tmp_path)

    with pytest.raises(DocumentParseError):
        await service.ingest_document(
            filename="empty.txt",
            content=b"",
            source_type=SourceType.RESUME,
            label=None,
        )


@pytest.mark.asyncio
async def test_garbage_content_named_as_pdf_raises_document_parse_error(tmp_path: Path):
    service = _make_service(tmp_path)

    with pytest.raises(DocumentParseError):
        await service.ingest_document(
            filename="resume.pdf",
            content=b"this is not a real PDF, just plain text pretending to be one",
            source_type=SourceType.RESUME,
            label=None,
        )


@pytest.mark.asyncio
async def test_garbage_content_named_as_docx_raises_document_parse_error(tmp_path: Path):
    service = _make_service(tmp_path)

    with pytest.raises(DocumentParseError):
        await service.ingest_document(
            filename="resume.docx",
            content=b"not a real docx (docx files are zip archives; this isn't)",
            source_type=SourceType.RESUME,
            label=None,
        )


@pytest.mark.asyncio
async def test_malformed_upload_surfaces_as_422_not_500_at_the_route(tmp_path: Path):
    """Route-level check: the real IngestionService (fakes swapped only for
    the heavy embedding/vector-store collaborators) is wired through
    dependency_overrides, so this hits the real parse_document() dispatch and
    confirms the exception handler in core/exceptions.py maps it correctly
    end to end, not just at the service layer."""
    app.dependency_overrides[deps.get_ingestion_service] = lambda: _make_service(tmp_path)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"source_type": "resume"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json()["error_code"] == "document_parse_error"


# --- Concurrent ingestion -----------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_ingestion_registers_all_documents_with_distinct_ids(tmp_path: Path):
    """Fires N ingest_document() calls at once against one shared registry
    and vector store, exercising DocumentRegistry's threading.Lock and the
    asyncio.to_thread offloading in IngestionService concurrently rather than
    sequentially -- the first real test of that path."""
    service = _make_service(tmp_path)

    results = await asyncio.gather(
        *[
            service.ingest_document(
                filename=f"resume_{i}.txt",
                content=f"Candidate {i} with {i + 1} years of backend experience.".encode(),
                source_type=SourceType.RESUME,
                label=f"candidate-{i}",
                )
            for i in range(8)
        ]
    )

    document_ids = [r.document_id for r in results]
    assert len(set(document_ids)) == 8  # no collisions, no lost writes

    registered = service._document_registry.list_all()  # noqa: SLF001 - test introspection
    assert len(registered) == 8
    assert {d.document_id for d in registered} == set(document_ids)


# --- Stream disconnect mid-answer --------------------------------------------
#
# httpx's ASGITransport can't actually simulate a client dropping a live
# connection: `handle_async_request` awaits the *entire* ASGI app call to
# completion, collecting the whole response body, before it ever hands
# anything back to the client (see `httpx/_transports/asgi.py` -- there is no
# concurrent byte-pumping task to interrupt). So "break out of the loop
# early" against an ASGITransport-backed client never actually races with the
# server; the server has already finished by the time the client sees byte
# one. A real disconnect test would need a real socket-based server.
#
# What we *can* test directly, without a real server, is the actual
# mechanism a real disconnect relies on: when Starlette's StreamingResponse
# loses its consumer, it calls `aclose()` on the response generator, which
# throws `GeneratorExit` in at the generator's current `yield`. That's true
# whether the generator belongs to `chat_service.stream_answer` or a nested
# one it's iterating (`claude_client.stream_answer`). This test drives
# `ChatService.stream_answer` by hand exactly that way.


class _SlowClaudeClient:
    """Stands in for ClaudeClient: yields one delta, then hangs as if the
    model were still generating, so the test can `aclose()` the consuming
    generator before "done" ever arrives. `cleaned_up` proves the `finally`
    block ran -- i.e. cancellation actually propagated through, rather than
    the coroutine being silently abandoned mid-await."""

    def __init__(self) -> None:
        self.cleaned_up = False

    async def stream_answer(self, query, context_chunks, history=None):
        try:
            yield {"type": "delta", "text": "partial answer..."}
            await asyncio.sleep(30)
            yield {  # pragma: no cover - never reached, consumer disconnects first
                "type": "done",
                "answer": "full answer",
                "input_tokens": 1,
                "output_tokens": 1,
                "llm_ms": 1.0,
            }
        finally:
            self.cleaned_up = True


def _make_chat_service(tmp_path: Path, claude_client) -> ChatService:
    registry = DocumentRegistry(storage_path=tmp_path / "documents.json")
    registry.add(
        DocumentMetadata(
            document_id="doc-1",
            source_type=SourceType.RESUME,
            label="resume",
            filename="resume.txt",
            uploaded_at=datetime.now(UTC),
            chunk_count=1,
        )
    )
    return ChatService(
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=_FakeVectorStore(),
        reranker=None,  # unused: rerank_enabled=False short-circuits before it's touched
        document_registry=registry,
        claude_client=claude_client,
        rate_limiter=SlidingWindowRateLimiter(max_requests=100, window_seconds=60),
        conversation_store=ConversationStore(max_turns_per_session=10),
        candidate_k=5,
        final_k=5,
        rerank_enabled=False,
    )


@pytest.mark.asyncio
async def test_chat_service_stream_answer_cancels_cleanly_on_early_close(tmp_path: Path):
    claude_client = _SlowClaudeClient()
    service = _make_chat_service(tmp_path, claude_client)

    generator = service.stream_answer(message="What am I missing?", session_id="session-1")
    try:
        first = await asyncio.wait_for(generator.__anext__(), timeout=5)
        second = await asyncio.wait_for(generator.__anext__(), timeout=5)

        assert first["type"] == "context"
        assert second["type"] == "delta"

        # Simulate what Starlette does when the client goes away mid-stream:
        # close the generator instead of continuing to pull from it.
        await asyncio.wait_for(generator.aclose(), timeout=5)
    finally:
        await generator.aclose()  # no-op if already closed; ensures no leak on assertion failure

    assert claude_client.cleaned_up is True
