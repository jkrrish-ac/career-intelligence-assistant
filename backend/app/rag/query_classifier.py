"""LLM-based query-target classification.

Replaces `classify_query_target`'s regex/keyword heuristic (still in
`retrieval.py`, kept as-is and used as the fallback) with a small,
schema-constrained Claude tool-call that picks which uploaded document(s) —
if any — a question is about.

Kept in its own module rather than folded into `retrieval.py` because it has
a genuinely different failure mode than the rest of the retrieval pipeline:
it's *allowed* to fail (network hiccup, rate limit, a slow response) without
failing the request, since retrieval still works reasonably well with no
filter at all (`where=None` just searches every uploaded document). See
`ChatService._resolve_query_target` in `chat_service.py` for the
try/timeout/fallback wiring that makes that true in practice.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.llm.claude_client import ClaudeClient
from app.models.schemas import DocumentMetadata, SourceType

logger = get_logger(__name__)


async def classify_query_target_llm(
    query: str,
    known_documents: list[DocumentMetadata],
    claude_client: ClaudeClient,
) -> dict | None:
    """Same return contract as `retrieval.classify_query_target`: a Chroma
    `where` filter dict, or None to search every uploaded document. Raises
    whatever `claude_client.classify()` raises (network error, timeout,
    malformed response) — this function does no error handling of its own
    on purpose, so the caller's fallback-to-heuristic logic has one clear
    place to catch failures rather than two."""
    if not known_documents:
        return None

    target = await claude_client.classify(query, known_documents)

    document_ids = {d.document_id for d in known_documents}
    if target in document_ids:
        return {"document_id": target}
    if target == SourceType.RESUME.value:
        return {"source_type": SourceType.RESUME.value}
    if target == SourceType.JOB_DESCRIPTION.value:
        return {"source_type": SourceType.JOB_DESCRIPTION.value}
    return None  # "all", or any unrecognized value -- fail open, not narrow
