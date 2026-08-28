"""Chat routes — HTTP/SSE translation only, no business logic."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await chat_service.answer_question(message=request.message, session_id=request.session_id)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    async def event_source():
        try:
            async for event in chat_service.stream_answer(
                message=request.message, session_id=request.session_id
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except AppError as exc:
            # Guardrails (no documents, rate limit, ...) can still fire after
            # the stream has started — surface them as an SSE error event
            # rather than a broken connection the frontend can't interpret.
            logger.error("chat_stream_guardrail_error", error_code=exc.error_code, message=exc.message)
            yield f"data: {json.dumps({'type': 'error', 'error_code': exc.error_code, 'message': exc.message})}\n\n"
        except Exception as exc:  # noqa: BLE001 - last-resort SSE error framing
            logger.error("chat_stream_unhandled_error", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'error_code': 'internal_error', 'message': 'An unexpected error occurred.'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
