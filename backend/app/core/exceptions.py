"""Typed exception hierarchy.

Every error the RAG pipeline can hit is one of these, never a bare
`Exception`. `register_exception_handlers` maps each to a structured JSON
response and a log line with context, so a failure is diagnosable from logs
alone and the frontend never sees a raw stack trace.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base for all application errors that should reach the client as JSON."""

    error_code: str = "internal_error"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class DocumentParseError(AppError):
    error_code = "document_parse_error"
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class UnsupportedFileTypeError(AppError):
    error_code = "unsupported_file_type"
    status_code = status.HTTP_400_BAD_REQUEST


class FileTooLargeError(AppError):
    error_code = "file_too_large"
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyContextError(AppError):
    """Raised when a chat question can't be grounded in any uploaded document."""

    error_code = "empty_context"
    status_code = status.HTTP_200_OK  # handled as a normal, honest answer — not a failure


class NoDocumentsUploadedError(AppError):
    error_code = "no_documents_uploaded"
    status_code = status.HTTP_400_BAD_REQUEST


class RateLimitExceededError(AppError):
    error_code = "rate_limit_exceeded"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS


class DocumentNotFoundError(AppError):
    """Raised when an operation targets a document_id that isn't registered
    — e.g. re-indexing (PUT /documents/{id}) a document that was never
    uploaded or was already deleted."""

    error_code = "document_not_found"
    status_code = status.HTTP_404_NOT_FOUND


class LLMProviderError(AppError):
    error_code = "llm_provider_error"
    status_code = status.HTTP_502_BAD_GATEWAY


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.error(
            "request_failed",
            error_code=exc.error_code,
            message=exc.message,
            detail=exc.detail,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            exc_info=exc,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "internal_error",
                "message": "An unexpected error occurred.",
                "detail": {},
            },
        )
