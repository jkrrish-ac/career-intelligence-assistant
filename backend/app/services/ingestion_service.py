"""Orchestrates: save upload -> parse -> chunk -> embed -> store.

This is the only place that knows the full ingestion sequence; routes just
call `ingest_document` and handle the response.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.core.exceptions import DocumentNotFoundError, FileTooLargeError
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

    def register_pending(
        self,
        *,
        filename: str,
        content: bytes,
        source_type: SourceType,
        label: str | None,
    ) -> DocumentMetadata:
        """The synchronous half of async ingestion (`INGESTION_MODE=async`,
        see `app/api/documents.py`): mint a document_id, write the file to
        disk, and register a `pending` placeholder -- fast enough to do
        inline in the request, unlike parsing/chunking/embedding. The arq
        worker (`app/worker.py`) picks up from here via `process_pending()`.

        Deliberately not async: there's no I/O here worth offloading (one
        file write), and keeping it sync makes it trivially callable from
        the route without an extra `await`."""
        if len(content) > self._max_file_size_bytes:
            raise FileTooLargeError(
                f"'{filename}' exceeds the {self._max_file_size_bytes // (1024 * 1024)}MB upload limit",
                detail={"filename": filename, "size_bytes": len(content)},
            )

        document_id = str(uuid.uuid4())
        resolved_label = label or self._default_label(filename, source_type)

        dest_path = self._upload_dir / f"{document_id}_{filename}"
        dest_path.write_bytes(content)

        metadata = DocumentMetadata(
            document_id=document_id,
            source_type=source_type,
            label=resolved_label,
            filename=filename,
            uploaded_at=datetime.now(UTC),
            chunk_count=0,
            status="pending",
        )
        self._document_registry.add(metadata)

        logger.info(
            "document_registered_pending",
            document_id=document_id,
            label=resolved_label,
            source_type=source_type.value,
        )
        return metadata

    async def process_pending(self, document_id: str) -> None:
        """The out-of-band half of async ingestion: parse/chunk/embed/store
        for a document `register_pending()` already wrote to disk and
        registered as `pending`. This is what the arq worker task
        (`app/worker.py::run_ingestion`) calls -- it's the *only* caller in
        production, but it's a plain method (not worker-specific code) so
        it's just as easy to call directly from a test or a future
        alternate queue implementation.

        Never raises: a parse/embed failure marks the document `failed`
        with `error_message` set instead of propagating, so one bad upload
        can't crash the worker process or get silently retried forever by
        arq's default retry-on-exception behavior -- a corrupt PDF will
        never parse no matter how many times it's retried."""
        metadata = self._document_registry.get(document_id)
        if metadata is None:
            logger.error("process_pending_missing_registry_entry", document_id=document_id)
            return

        dest_path = self._upload_dir / f"{document_id}_{metadata.filename}"
        try:
            text = await asyncio.to_thread(parse_document, dest_path)
            chunks = chunk_document(
                text=text,
                document_id=document_id,
                source_type=metadata.source_type,
                label=metadata.label,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            embeddings = await asyncio.to_thread(
                self._embedding_provider.embed, [c.text for c in chunks]
            )
            await asyncio.to_thread(self._vector_store.add, chunks, embeddings)

            self._document_registry.update(
                metadata.model_copy(update={"chunk_count": len(chunks), "status": "ready"})
            )
            logger.info(
                "document_ingestion_completed", document_id=document_id, chunk_count=len(chunks)
            )
        except Exception as exc:
            logger.error("document_ingestion_failed", document_id=document_id, error=str(exc))
            self._document_registry.update(
                metadata.model_copy(update={"status": "failed", "error_message": str(exc)[:500]})
            )

    async def reindex_document(
        self,
        *,
        document_id: str,
        filename: str,
        content: bytes,
        source_type: SourceType,
        label: str | None,
    ) -> UploadResponse:
        """Replace an existing document's chunks in place, without minting a
        new document_id — this is the difference from `ingest_document`,
        which always creates a new document. Used by `PUT /documents/{id}`
        when the user re-uploads a corrected resume/JD and wants it to keep
        its spot in the list rather than becoming a duplicate entry.

        Not fully atomic: there's a brief window between deleting the old
        chunks and adding the new ones where a concurrent query would see
        zero chunks for this document. Acceptable for a take-home's traffic
        pattern; a production version would add the new chunks first and
        delete the old ones only after that succeeds (trading "briefly
        duplicated" for "briefly empty", which is the better failure mode)."""
        existing = self._document_registry.get(document_id)
        if existing is None:
            raise DocumentNotFoundError(
                f"No document with id '{document_id}' to re-index",
                detail={"document_id": document_id},
            )

        if len(content) > self._max_file_size_bytes:
            raise FileTooLargeError(
                f"'{filename}' exceeds the {self._max_file_size_bytes // (1024 * 1024)}MB upload limit",
                detail={"filename": filename, "size_bytes": len(content)},
            )

        resolved_label = label or existing.label

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
        await asyncio.to_thread(self._vector_store.delete_document, document_id)
        await asyncio.to_thread(self._vector_store.add, chunks, embeddings)

        metadata = DocumentMetadata(
            document_id=document_id,
            source_type=source_type,
            label=resolved_label,
            filename=filename,
            uploaded_at=datetime.now(UTC),
            chunk_count=len(chunks),
        )
        self._document_registry.update(metadata)

        logger.info(
            "document_reindexed",
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
