"""Chat route — HTTP translation only, no business logic."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_chat_service
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await chat_service.answer_question(message=request.message, session_id=request.session_id)
