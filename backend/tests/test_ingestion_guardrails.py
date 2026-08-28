from pathlib import Path

import pytest

from app.core.exceptions import FileTooLargeError
from app.models.schemas import SourceType
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


class _FakeVectorStore(VectorStore):
    def add(self, chunks, embeddings):
        pass  # nothing to assert in this test — just needs to not blow up

    def query(self, query_embedding, top_k, where=None):
        return []

    def get_all_chunks(self, where=None):
        return []

    def delete_document(self, document_id):
        pass


def _make_service(tmp_path: Path, *, max_file_size_mb: int) -> IngestionService:
    return IngestionService(
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=_FakeVectorStore(),
        document_registry=DocumentRegistry(storage_path=tmp_path / "documents.json"),
        upload_dir=tmp_path / "uploads",
        chunk_size=500,
        chunk_overlap=50,
        max_file_size_mb=max_file_size_mb,
    )


@pytest.mark.asyncio
async def test_oversized_upload_is_rejected_before_parsing(tmp_path: Path):
    service = _make_service(tmp_path, max_file_size_mb=1)
    oversized_content = b"x" * (2 * 1024 * 1024)  # 2MB > 1MB limit

    with pytest.raises(FileTooLargeError):
        await service.ingest_document(
            filename="huge_resume.txt",
            content=oversized_content,
            source_type=SourceType.RESUME,
            label=None,
        )


@pytest.mark.asyncio
async def test_upload_within_limit_succeeds(tmp_path: Path):
    service = _make_service(tmp_path, max_file_size_mb=1)

    response = await service.ingest_document(
        filename="resume.txt",
        content=b"Backend engineer with 5 years of experience.",
        source_type=SourceType.RESUME,
        label=None,
    )

    assert response.chunk_count >= 1
    assert response.source_type == SourceType.RESUME


@pytest.mark.asyncio
async def test_default_job_description_labels_are_auto_numbered(tmp_path: Path):
    service = _make_service(tmp_path, max_file_size_mb=1)

    first = await service.ingest_document(
        filename="jd1.txt",
        content=b"We need a backend engineer with Python experience.",
        source_type=SourceType.JOB_DESCRIPTION,
        label=None,
    )
    second = await service.ingest_document(
        filename="jd2.txt",
        content=b"We need a frontend engineer with React experience.",
        source_type=SourceType.JOB_DESCRIPTION,
        label=None,
    )

    assert first.label == "Job #1"
    assert second.label == "Job #2"
