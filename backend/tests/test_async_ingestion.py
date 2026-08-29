"""Async ingestion (`INGESTION_MODE=async`): `IngestionService.
register_pending()` + `process_pending()`, the arq task function that wires
them together (`app/worker.py::run_ingestion`), and the `POST /documents`
route's 202/pending branch."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.core.config import Settings
from app.models.schemas import SourceType
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService
from app.main import app
from app.worker import run_ingestion


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


# --- IngestionService.register_pending / process_pending --------------------


def test_register_pending_writes_file_and_registers_pending_status(tmp_path: Path):
    service = _make_service(tmp_path)

    pending = service.register_pending(
        filename="resume.txt",
        content=b"Backend engineer with Python experience.",
        source_type=SourceType.RESUME,
        label=None,
    )

    assert pending.status == "pending"
    assert pending.chunk_count == 0
    stored = service._document_registry.get(pending.document_id)  # noqa: SLF001
    assert stored is not None
    assert stored.status == "pending"


@pytest.mark.asyncio
async def test_process_pending_succeeds_and_flips_status_to_ready(tmp_path: Path):
    service = _make_service(tmp_path)
    pending = service.register_pending(
        filename="resume.txt",
        content=b"Backend engineer with five years of Python and FastAPI experience.",
        source_type=SourceType.RESUME,
        label=None,
    )

    await service.process_pending(pending.document_id)

    updated = service._document_registry.get(pending.document_id)  # noqa: SLF001
    assert updated.status == "ready"
    assert updated.chunk_count >= 1
    assert updated.error_message is None


@pytest.mark.asyncio
async def test_process_pending_marks_failed_on_parse_error_without_raising(tmp_path: Path):
    service = _make_service(tmp_path)
    pending = service.register_pending(
        filename="empty.txt",
        content=b"",  # parse_txt raises DocumentParseError on empty content
        source_type=SourceType.RESUME,
        label=None,
    )

    await service.process_pending(pending.document_id)  # must not raise

    updated = service._document_registry.get(pending.document_id)  # noqa: SLF001
    assert updated.status == "failed"
    assert updated.error_message


@pytest.mark.asyncio
async def test_process_pending_is_a_noop_for_an_unknown_document_id(tmp_path: Path):
    service = _make_service(tmp_path)
    await service.process_pending("does-not-exist")  # must not raise


# --- The arq task function ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_ingestion_task_delegates_to_process_pending(tmp_path: Path):
    service = _make_service(tmp_path)
    pending = service.register_pending(
        filename="resume.txt",
        content=b"Backend engineer with Kubernetes experience.",
        source_type=SourceType.RESUME,
        label=None,
    )

    await run_ingestion({"ingestion_service": service}, pending.document_id)

    updated = service._document_registry.get(pending.document_id)  # noqa: SLF001
    assert updated.status == "ready"


# --- POST /documents route, async mode ---------------------------------------


class _FakeArqPool:
    def __init__(self):
        self.enqueued: list[tuple[str, tuple]] = []

    async def enqueue_job(self, function_name, *args):
        self.enqueued.append((function_name, args))


@pytest.mark.asyncio
async def test_upload_route_returns_202_pending_in_async_mode(tmp_path: Path):
    ingestion_service = _make_service(tmp_path)
    arq_pool = _FakeArqPool()

    app.dependency_overrides[deps.get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[deps.get_settings] = lambda: Settings(ingestion_mode="async")
    app.dependency_overrides[deps.get_arq_pool] = lambda: arq_pool

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/documents",
            files={"file": ("resume.txt", b"Backend engineer.", "text/plain")},
            data={"source_type": "resume"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["chunk_count"] == 0
    assert len(arq_pool.enqueued) == 1
    assert arq_pool.enqueued[0][0] == "run_ingestion"
    assert arq_pool.enqueued[0][1] == (body["document_id"],)


@pytest.mark.asyncio
async def test_upload_route_stays_synchronous_by_default(tmp_path: Path):
    """No overrides for settings/arq_pool at all -- the default `Settings()`
    (ingestion_mode="sync") must behave exactly as before Phase 5."""
    ingestion_service = _make_service(tmp_path)
    app.dependency_overrides[deps.get_ingestion_service] = lambda: ingestion_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/documents",
            files={"file": ("resume.txt", b"Backend engineer.", "text/plain")},
            data={"source_type": "resume"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_get_document_by_id_returns_current_status(tmp_path: Path):
    ingestion_service = _make_service(tmp_path)
    pending = ingestion_service.register_pending(
        filename="resume.txt",
        content=b"content",
        source_type=SourceType.RESUME,
        label=None,
    )
    app.dependency_overrides[deps.get_document_registry] = lambda: ingestion_service._document_registry  # noqa: SLF001

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/documents/{pending.document_id}")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_get_document_by_id_404s_for_unknown_id(tmp_path: Path):
    registry = DocumentRegistry(storage_path=tmp_path / "documents.json")
    app.dependency_overrides[deps.get_document_registry] = lambda: registry

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/documents/does-not-exist")

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["error_code"] == "document_not_found"
