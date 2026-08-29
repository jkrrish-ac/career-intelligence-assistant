"""Per-document re-index without a full re-ingest: `IngestionService.
reindex_document()` and the `PUT /documents/{document_id}` route built on it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import deps
from app.core.exceptions import DocumentNotFoundError
from app.main import app
from app.models.schemas import DocumentMetadata, SourceType
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService


class _FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts):
        return [[0.0] for _ in texts]

    @property
    def dimension(self):
        return 1


class _RecordingVectorStore(VectorStore):
    """Tracks add/delete calls (and their order) instead of really storing
    anything, so the tests can assert the delete-then-add sequence without
    needing a real Chroma collection."""

    def __init__(self):
        self.calls: list[str] = []
        self.last_added_chunks = None

    def add(self, chunks, embeddings):
        self.calls.append("add")
        self.last_added_chunks = chunks

    def query(self, query_embedding, top_k, where=None):
        return []

    def get_all_chunks(self, where=None):
        return []

    def delete_document(self, document_id):
        self.calls.append("delete")


def _make_service(tmp_path: Path, vector_store: VectorStore | None = None) -> IngestionService:
    return IngestionService(
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=vector_store or _RecordingVectorStore(),
        document_registry=DocumentRegistry(storage_path=tmp_path / "documents.json"),
        upload_dir=tmp_path / "uploads",
        chunk_size=500,
        chunk_overlap=50,
        max_file_size_mb=10,
    )


@pytest.mark.asyncio
async def test_reindex_replaces_chunks_and_keeps_document_id(tmp_path: Path):
    vector_store = _RecordingVectorStore()
    service = _make_service(tmp_path, vector_store)

    original = await service.ingest_document(
        filename="resume_v1.txt",
        content=b"Backend engineer with 3 years of Python experience.",
        source_type=SourceType.RESUME,
        label="my-resume",
    )

    updated = await service.reindex_document(
        document_id=original.document_id,
        filename="resume_v2.txt",
        content=b"Backend engineer with 6 years of Python and Kubernetes experience.",
        source_type=SourceType.RESUME,
        label="my-resume",
    )

    assert updated.document_id == original.document_id
    assert updated.chunk_count >= 1

    stored = service._document_registry.get(original.document_id)  # noqa: SLF001
    assert stored is not None
    assert stored.filename == "resume_v2.txt"

    # delete_document() must run before the new add() for the old chunks to
    # actually be replaced rather than accumulated.
    assert vector_store.calls == ["add", "delete", "add"]


@pytest.mark.asyncio
async def test_reindex_defaults_to_existing_label_when_none_given(tmp_path: Path):
    service = _make_service(tmp_path)

    original = await service.ingest_document(
        filename="jd.txt",
        content=b"We need a backend engineer with Python experience.",
        source_type=SourceType.JOB_DESCRIPTION,
        label=None,  # auto-numbered "Job #1"
    )
    assert original.label == "Job #1"

    updated = await service.reindex_document(
        document_id=original.document_id,
        filename="jd_v2.txt",
        content=b"We now also need Kubernetes and Terraform experience.",
        source_type=SourceType.JOB_DESCRIPTION,
        label=None,
    )

    assert updated.label == "Job #1"  # unchanged, not re-numbered


@pytest.mark.asyncio
async def test_reindex_unknown_document_id_raises_not_found(tmp_path: Path):
    service = _make_service(tmp_path)

    with pytest.raises(DocumentNotFoundError):
        await service.reindex_document(
            document_id="does-not-exist",
            filename="resume.txt",
            content=b"Some content.",
            source_type=SourceType.RESUME,
            label=None,
        )


class _FakeIngestionServiceForReindex:
    def __init__(self, registry: DocumentRegistry):
        self._registry = registry

    async def reindex_document(self, *, document_id, filename, content, source_type, label):
        existing = self._registry.get(document_id)
        if existing is None:
            raise DocumentNotFoundError(f"No document with id '{document_id}'")
        updated = existing.model_copy(update={"filename": filename, "chunk_count": 5})
        self._registry.update(updated)
        from app.models.schemas import UploadResponse

        return UploadResponse(
            document_id=document_id,
            source_type=source_type,
            label=label or existing.label,
            chunk_count=5,
        )


@pytest.mark.asyncio
async def test_put_documents_route_reindexes_and_returns_200(tmp_path: Path):
    registry = DocumentRegistry(storage_path=tmp_path / "documents.json")
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
    app.dependency_overrides[deps.get_ingestion_service] = lambda: _FakeIngestionServiceForReindex(registry)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            "/documents/doc-1",
            files={"file": ("resume_v2.txt", b"Updated resume content.", "text/plain")},
            data={"source_type": "resume"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-1"
    assert body["chunk_count"] == 5


@pytest.mark.asyncio
async def test_put_documents_route_404s_for_unknown_document(tmp_path: Path):
    registry = DocumentRegistry(storage_path=tmp_path / "documents.json")
    app.dependency_overrides[deps.get_document_registry] = lambda: registry
    app.dependency_overrides[deps.get_ingestion_service] = lambda: _FakeIngestionServiceForReindex(registry)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            "/documents/missing-id",
            files={"file": ("resume.txt", b"content", "text/plain")},
            data={"source_type": "resume"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["error_code"] == "document_not_found"
