"""Orchestrates: save upload -> parse -> chunk -> embed -> store.

This is the only place that knows the full ingestion sequence; routes just
call `ingest_document` and handle the response.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.core.exceptions import FileTooLargeError
from app.core.logging import get_logger
from app.models.schemas import DocumentMetadata, SourceType, UploadResponse
from app.rag.chunking import chunk_document
from app.rag.embeddings import EmbeddingProvider
from app.rag.parsers import parse_document
from app.rag.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        document_registry: DocumentRegistry,
        upload_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
        max_file_size_mb: int,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._document_registry = document_registry
        self._upload_dir = upload_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    async def ingest_document(
        self,
        *,
        filename: str,
        content: bytes,
        source_type: SourceType,
        label: str | None,
    ) -> UploadResponse:
        if len(content) > self._max_file_size_bytes:
            raise FileTooLargeError(
                f"'{filename}' exceeds the {self._max_file_size_bytes // (1024 * 1024)}MB upload limit",
                detail={"filename": filename, "size_bytes": len(content)},
            )

        document_id = str(uuid.uuid4())
        resolved_label = label or self._default_label(filename, source_type)

        dest_path = self._upload_dir / f"{document_id}_{filename}"
        dest_path.write_bytes(content)

        text = await asyncio.to_thread(parse_document, dest_path)
        chunks = chunk_document(
            text=text,
            document_id=document_id,
            source_type=source_type,
            label=resolved_label,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        embeddings = await asyncio.to_thread(
            self._embedding_provider.embed, [c.text for c in chunks]
        )
        await asyncio.to_thread(self._vector_store.add, chunks, embeddings)

        metadata = DocumentMetadata(
            document_id=document_id,
            source_type=source_type,
            label=resolved_label,
            filename=filename,
            uploaded_at=datetime.now(UTC),
            chunk_count=len(chunks),
        )
        self._document_registry.add(metadata)

        logger.info(
            "document_ingested",
            document_id=document_id,
            label=resolved_label,
            source_type=source_type.value,
            chunk_count=len(chunks),
        )

        return UploadResponse(
            document_id=document_id,
            source_type=source_type,
            label=resolved_label,
            chunk_count=len(chunks),
        )

    def _default_label(self, filename: str, source_type: SourceType) -> str:
        if source_type == SourceType.RESUME:
            return Path(filename).stem
        existing_jds = [
            d for d in self._document_registry.list_all() if d.source_type == SourceType.JOB_DESCRIPTION
        ]
        return f"Job #{len(existing_jds) + 1}"
