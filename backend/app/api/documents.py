"""Document upload/listing routes — HTTP translation only, no business logic."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from app.api.deps import get_arq_pool, get_document_registry, get_ingestion_service, get_vector_store
from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
from app.models.schemas import DocumentMetadata, SourceType, UploadResponse
from app.rag.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


def _validate_extension(filename: str, settings: Settings) -> None:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in settings.allowed_upload_extensions:
        raise UnsupportedFileTypeError(
            f"'{filename}' has an unsupported extension. "
            f"Allowed: {', '.join(settings.allowed_upload_extensions)}",
            detail={"filename": filename},
        )


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    source_type: SourceType = Form(...),
    label: str | None = Form(default=None),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
    arq_pool=Depends(get_arq_pool),
) -> UploadResponse:
    """Synchronous by default (`INGESTION_MODE=sync`): parses, chunks, and
    embeds before responding -- 201 with the finished document, same as
    always. With `INGESTION_MODE=async`, hands the work to the arq worker
    (`app/worker.py`) instead and returns 202 immediately with a `pending`
    document; poll `GET /documents/{document_id}` (or the list route) to
    see it flip to `ready` or `failed`."""
    _validate_extension(file.filename, settings)
    content = await file.read()

    if settings.ingestion_mode == "async":
        pending = ingestion_service.register_pending(
            filename=file.filename,
            content=content,
            source_type=source_type,
            label=label,
        )
        await arq_pool.enqueue_job("run_ingestion", pending.document_id)
        response.status_code = 202
        return UploadResponse(
            document_id=pending.document_id,
            source_type=pending.source_type,
            label=pending.label,
            chunk_count=0,
            status=pending.status,
        )

    return await ingestion_service.ingest_document(
        filename=file.filename,
        content=content,
        source_type=source_type,
        label=label,
    )


@router.put("/{document_id}", response_model=UploadResponse)
async def reindex_document(
    document_id: str,
    file: UploadFile = File(...),
    source_type: SourceType = Form(...),
    label: str | None = Form(default=None),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Re-index an existing document in place — same chunk/embed/store
    pipeline as upload, but replacing the prior chunks under the same
    document_id rather than creating a new entry. Raises 404
    (`document_not_found`) if `document_id` isn't registered; see
    `IngestionService.reindex_document`. Always synchronous regardless of
    `INGESTION_MODE` -- a re-index is a small, deliberate correction, not
    the kind of bulk/first-upload traffic the job queue exists for."""
    _validate_extension(file.filename, settings)

    content = await file.read()
    return await ingestion_service.reindex_document(
        document_id=document_id,
        filename=file.filename,
        content=content,
        source_type=source_type,
        label=label,
    )


@router.get("", response_model=list[DocumentMetadata])
async def list_documents(
    document_registry: DocumentRegistry = Depends(get_document_registry),
) -> list[DocumentMetadata]:
    return document_registry.list_all()


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    document_registry: DocumentRegistry = Depends(get_document_registry),
) -> DocumentMetadata:
    """Lets the frontend poll one document's `status` (pending/ready/failed)
    after an async upload without re-fetching and re-rendering the whole
    list on every tick."""
    document = document_registry.get(document_id)
    if document is None:
        raise DocumentNotFoundError(
            f"No document with id '{document_id}'", detail={"document_id": document_id}
        )
    return document


@router.delete("/{document_id}", status_code=204, response_model=None)
async def delete_document(
    document_id: str,
    document_registry: DocumentRegistry = Depends(get_document_registry),
    vector_store: VectorStore = Depends(get_vector_store),
) -> None:
    vector_store.delete_document(document_id)
    document_registry.delete(document_id)
