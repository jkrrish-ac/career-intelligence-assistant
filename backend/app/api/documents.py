"""Document upload/listing routes — HTTP translation only, no business logic."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_document_registry, get_ingestion_service, get_vector_store
from app.core.exceptions import UnsupportedFileTypeError
from app.core.config import get_settings
from app.models.schemas import DocumentMetadata, SourceType, UploadResponse
from app.rag.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    source_type: SourceType = Form(...),
    label: str | None = Form(default=None),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    settings = get_settings()
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in settings.allowed_upload_extensions:
        raise UnsupportedFileTypeError(
            f"'{file.filename}' has an unsupported extension. "
            f"Allowed: {', '.join(settings.allowed_upload_extensions)}",
            detail={"filename": file.filename},
        )

    content = await file.read()
    return await ingestion_service.ingest_document(
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


@router.delete("/{document_id}", status_code=204, response_model=None)
async def delete_document(
    document_id: str,
    document_registry: DocumentRegistry = Depends(get_document_registry),
    vector_store: VectorStore = Depends(get_vector_store),
) -> None:
    vector_store.delete_document(document_id)
    document_registry.delete(document_id)
