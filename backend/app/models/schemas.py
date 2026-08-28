"""Shared Pydantic schemas used across api/services/rag layers.

Kept in one module (not scattered per-layer) because these are the contract
between layers and with the frontend — one place to see the whole shape of
the data flowing through the system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


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
