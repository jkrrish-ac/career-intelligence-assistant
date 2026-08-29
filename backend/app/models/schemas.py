"""Shared Pydantic schemas used across api/services/rag layers.

Kept in one module (not scattered per-layer) because these are the contract
between layers and with the frontend — one place to see the whole shape of
the data flowing through the system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

ConversationRole = Literal["user", "assistant"]


class ConversationTurn(TypedDict):
    """One turn of session-scoped conversation history (see
    app/services/conversation_store.py). Plain TypedDict, not a Pydantic
    model, since it's an internal in-memory shape, not an API contract."""

    role: ConversationRole
    content: str


class SourceType(str, Enum):
    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"


class DocumentMetadata(BaseModel):
    document_id: str
    source_type: SourceType
    label: str
    filename: str
    uploaded_at: datetime
    chunk_count: int = 0
    # "ready" is the default because most callers (ingest_document,
    # reindex_document) still finish the whole pipeline before ever
    # constructing this -- only the async ingestion path (register_pending,
    # in ingestion_service.py) creates one that starts out "pending".
    status: Literal["pending", "ready", "failed"] = "ready"
    error_message: str | None = Field(
        default=None, description="Set when status is 'failed' -- why ingestion didn't finish"
    )


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    source_type: SourceType
    label: str
    section: str | None = None
    text: str


class RetrievedChunk(BaseModel):
    """A chunk plus the scores it picked up along the retrieval pipeline."""

    chunk: Chunk
    retrieval_score: float = Field(description="Score after semantic+BM25 reciprocal rank fusion")
    rerank_score: float | None = Field(default=None, description="Cross-encoder score, if reranking ran")


class UploadResponse(BaseModel):
    document_id: str
    source_type: SourceType
    label: str
    chunk_count: int
    status: Literal["pending", "ready", "failed"] = "ready"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class SourceRef(BaseModel):
    document_id: str
    label: str
    section: str | None = None
    retrieval_score: float
    rerank_score: float | None = None
    snippet: str


class TimingInfo(BaseModel):
    retrieval_ms: float
    rerank_ms: float | None = None
    llm_ms: float


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    timing: TimingInfo
    token_usage: TokenUsage
    grounded: bool = Field(
        description="False when the assistant could not find relevant context and said so"
    )
