"""Route-level integration tests: real FastAPI app, real routing/DI wiring,
real Pydantic request/response validation and exception handlers — with
fakes swapped in via `app.dependency_overrides` for the heavy collaborators
(embedding model, Claude API) so these run fast and need no network or key.

Nothing below this layer is being tested again — chunking/retrieval/rerank
already have focused unit tests. This file's job is to catch the class of
bug unit tests can't: a route wired to the wrong dependency, a status code
that doesn't match what the frontend expects, an exception that isn't
actually mapped to JSON by the handler.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.core.exceptions import NoDocumentsUploadedError, RateLimitExceededError
from app.main import app
from app.models.schemas import ChatResponse, DocumentMetadata, SourceType, TimingInfo, TokenUsage
from app.services.document_registry import DocumentRegistry


class _FakeIngestionService:
    """Mirrors IngestionService's public interface without needing a real
    embedding model — it just registers metadata, matching what the route
    actually depends on (the response shape and the registry side effect)."""

    def __init__(self, registry: DocumentRegistry) -> None:
        self._registry = registry

    async def ingest_document(self, *, filename, content, source_type, label):
        from app.core.exceptions import FileTooLargeError

        if len(content) > 10 * 1024 * 1024:
            raise FileTooLargeError("too large")

        document_id = f"doc-{len(self._registry.list_all()) + 1}"
        resolved_label = label or (filename if source_type == SourceType.RESUME else "Job #1")
        metadata = DocumentMetadata(
            document_id=document_id,
            source_type=source_type,
            label=resolved_label,
            filename=filename,
            uploaded_at=datetime.now(UTC),
            chunk_count=3,
        )
        self._registry.add(metadata)
        return metadata


class _FakeVectorStore:
    def delete_document(self, document_id: str) -> None:
        pass


class _StubChatService:
    """Configurable stand-in for ChatService, swapped in via
    dependency_overrides so route tests don't need a real Claude client or
    retrieval pipeline."""

    def __init__(self, *, raises: Exception | None = None, response: ChatResponse | None = None):
        self._raises = raises
        self._response = response

    async def answer_question(self, *, message: str, session_id: str | None) -> ChatResponse:
        if self._raises:
            raise self._raises
        assert self._response is not None
        return self._response

    async def stream_answer(self, *, message: str, session_id: str | None):
        if self._raises:
            raise self._raises
        yield {"type": "context", "sources": [], "grounded": True}
        yield {"type": "delta", "text": "hello"}
        yield {
            "type": "done",
            "timing": {"retrieval_ms": 1.0, "rerank_ms": None, "llm_ms": 2.0},
            "token_usage": {"input_tokens": 1, "output_tokens": 1},
        }


@pytest.fixture
def registry(tmp_path: Path) -> DocumentRegistry:
    return DocumentRegistry(storage_path=tmp_path / "documents.json")


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- /documents -------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_document_returns_201_and_appears_in_list(client, registry):
    app.dependency_overrides[deps.get_document_registry] = lambda: registry
    app.dependency_overrides[deps.get_ingestion_service] = lambda: _FakeIngestionService(registry)

    response = await client.post(
        "/documents",
        files={"file": ("resume.txt", b"Backend engineer.", "text/plain")},
        data={"source_type": "resume"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["source_type"] == "resume"
    assert body["chunk_count"] == 3

    list_response = await client.get("/documents")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_extension(client, registry):
    app.dependency_overrides[deps.get_document_registry] = lambda: registry
    app.dependency_overrides[deps.get_ingestion_service] = lambda: _FakeIngestionService(registry)

    response = await client.post(
        "/documents",
        files={"file": ("resume.rtf", b"content", "application/rtf")},
        data={"source_type": "resume"},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "unsupported_file_type"


@pytest.mark.asyncio
async def test_delete_document_removes_it_from_the_list(client, registry):
    registry.add(
        DocumentMetadata(
            document_id="doc-1",
            source_type=SourceType.RESUME,
            label="resume",
            filename="resume.txt",
            uploaded_at=datetime.now(UTC),
            chunk_count=2,
        )
    )
    app.dependency_overrides[deps.get_document_registry] = lambda: registry
    app.dependency_overrides[deps.get_vector_store] = lambda: _FakeVectorStore()

    response = await client.delete("/documents/doc-1")
    assert response.status_code == 204

    list_response = await client.get("/documents")
    assert list_response.json() == []


# --- /chat -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_answer_with_sources(client):
    canned = ChatResponse(
        answer="You're missing Kubernetes experience.",
        sources=[],
        timing=TimingInfo(retrieval_ms=1.0, rerank_ms=2.0, llm_ms=3.0),
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        grounded=True,
    )
    app.dependency_overrides[deps.get_chat_service] = lambda: _StubChatService(response=canned)

    response = await client.post("/chat", json={"message": "What am I missing?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "You're missing Kubernetes experience."


@pytest.mark.asyncio
async def test_chat_guardrail_no_documents_returns_400_with_error_code(client):
    app.dependency_overrides[deps.get_chat_service] = lambda: _StubChatService(
        raises=NoDocumentsUploadedError("Upload at least a resume and one job description.")
    )

    response = await client.post("/chat", json={"message": "What am I missing?"})

    assert response.status_code == 400
    assert response.json()["error_code"] == "no_documents_uploaded"


@pytest.mark.asyncio
async def test_chat_rate_limit_returns_429(client):
    app.dependency_overrides[deps.get_chat_service] = lambda: _StubChatService(
        raises=RateLimitExceededError("slow down")
    )

    response = await client.post("/chat", json={"message": "hi"})

    assert response.status_code == 429
    assert response.json()["error_code"] == "rate_limit_exceeded"


@pytest.mark.asyncio
async def test_chat_rejects_empty_message_with_422(client):
    # FastAPI resolves a route's Depends(...) params as part of solving the
    # request (alongside body validation), even for a request that will
    # ultimately fail validation — so this still needs get_chat_service
    # overridden, or it would try to construct the real one (real embedding
    # model, real Claude client) just to then 422 before ever using it.
    app.dependency_overrides[deps.get_chat_service] = lambda: _StubChatService()

    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


# --- /chat/stream (SSE) -------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_emits_context_delta_and_done_events(client):
    app.dependency_overrides[deps.get_chat_service] = lambda: _StubChatService()

    async with client.stream("POST", "/chat/stream", json={"message": "hi"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))

    types = [e["type"] for e in events]
    assert types == ["context", "delta", "done"]
    assert events[1]["text"] == "hello"


@pytest.mark.asyncio
async def test_chat_stream_surfaces_guardrail_as_error_event(client):
    app.dependency_overrides[deps.get_chat_service] = lambda: _StubChatService(
        raises=NoDocumentsUploadedError("Upload documents first.")
    )

    async with client.stream("POST", "/chat/stream", json={"message": "hi"}) as response:
        assert response.status_code == 200  # the stream itself starts fine
        events = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error_code"] == "no_documents_uploaded"
