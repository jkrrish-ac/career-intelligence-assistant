"""Hybrid retrieval: semantic (Chroma) + keyword (BM25), merged via
reciprocal rank fusion, with a document-type filter applied first (either the
regex/keyword heuristic below, or the LLM-based classifier in
`query_classifier.py` — see `hybrid_retrieve`'s `where` parameter).

The filter narrows *which job description* a query is about; it must never
exclude the resume, since virtually every real question here (gap analysis,
"how does my experience align with Job #2") is a resume-vs-JD comparison —
see `_ensure_resume_included`, which is what actually makes the filter
additive rather than exclusionary. If the target is ambiguous, `where` is
`None` and the search covers everything, same principle, one level up (see
PRD §11 risks).
"""

from __future__ import annotations

import asyncio
import re

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger, timed
from app.models.schemas import Chunk, DocumentMetadata, RetrievedChunk, SourceType
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore

logger = get_logger(__name__)

# Sentinel distinguishing "caller didn't pass `where`" (compute it with the
# heuristic, same as always) from "caller passed where=None on purpose"
# (an LLM classifier's fallback-appropriate way of saying "search
# everything" — see ChatService._resolve_query_target in chat_service.py).
_NOT_GIVEN = object()

_RESUME_HINTS = re.compile(
    r"\b(my|i'?ve|i have|our)\b.*\b(experience|background|resume|cv|skills?)\b"
    r"|\bmy resume\b|\bmy experience\b|\bmy skills\b|\bmy background\b",
    re.IGNORECASE,
)
_JOB_NUMBER_PATTERN = re.compile(r"\bjob\s*#?\s*(\d+)\b", re.IGNORECASE)
_JD_HINTS = re.compile(
    r"\b(this role|this position|the (job|posting|role)|job description)\b", re.IGNORECASE
)


def classify_query_target(query: str, known_documents: list[DocumentMetadata]) -> dict | None:
    """Best-effort `where` filter for the vector store, or None to search
    everything. Never returns a filter that would exclude all documents."""

    job_match = _JOB_NUMBER_PATTERN.search(query)
    if job_match:
        target_number = job_match.group(1)
        for doc in known_documents:
            if doc.source_type == SourceType.JOB_DESCRIPTION and target_number in doc.label:
                logger.info("query_filter_matched_job_number", document_id=doc.document_id)
                return {"document_id": doc.document_id}

    if _RESUME_HINTS.search(query):
        logger.info("query_filter_resume_hint")
        return {"source_type": SourceType.RESUME.value}

    if _JD_HINTS.search(query):
        logger.info("query_filter_jd_hint")
        return {"source_type": SourceType.JOB_DESCRIPTION.value}

    return None


def _ensure_resume_included(where: dict | None, known_documents: list[DocumentMetadata]) -> dict | None:
    """Narrowing a query to "this JD" or "job descriptions in general" must
    never mean *excluding the resume* — almost every real question this app
    answers (gap analysis, "how does my experience align with Job #2") is a
    resume-vs-JD comparison, so a `where` that filters the resume out of the
    candidate pool entirely makes the question unanswerable even though the
    resume was uploaded. (This was a real bug: `classify_query_target`
    documented exactly this narrowing as intended in
    `tests/test_retrieval.py::test_classify_query_target_resume_hint`, and
    `hybrid_retrieve` applied it to both the semantic and BM25 legs with no
    guard.)

    A query that's genuinely resume-only (`{"source_type": "resume"}`, or a
    `document_id` filter that already names a resume) is left untouched —
    only a JD-targeting filter gets OR'd with an explicit resume clause so
    both sides stay in context. Chroma's `where` supports `$or` natively
    (see `ChromaVectorStore` in `vector_store.py`), so this is a metadata
    filter change, not a second query."""
    if where is None:
        return None
    if where.get("source_type") == SourceType.RESUME.value:
        return where

    target_document_id = where.get("document_id")
    if target_document_id is not None:
        target_doc = next((d for d in known_documents if d.document_id == target_document_id), None)
        if target_doc is not None and target_doc.source_type == SourceType.RESUME:
            return where  # already resume-scoped; nothing to add

    return {"$or": [{"source_type": SourceType.RESUME.value}, where]}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+.#]+", text.lower())


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """Standard RRF: score(doc) = sum(1 / (k + rank)) across all lists it
    appears in. Returns {chunk_id: fused_score}."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


@timed("retrieval")
async def hybrid_retrieve(
    *,
    query: str,
    known_documents: list[DocumentMetadata],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    candidate_k: int,
    where: dict | None = _NOT_GIVEN,  # type: ignore[assignment]
) -> list[RetrievedChunk]:
    """Semantic + BM25 fused candidate set, before reranking.

    `where` is normally left unset, in which case the regex/keyword
    heuristic (`classify_query_target`) computes it here, same as always.
    Callers that have already resolved a `where` filter some other way (the
    LLM-based classifier in `query_classifier.py`, with its own
    heuristic-on-failure fallback) can pass it in directly — including
    `where=None` to explicitly mean "search everything," which is why this
    isn't a plain `where: dict | None = None` default."""

    if where is _NOT_GIVEN:
        where = classify_query_target(query, known_documents)
    where = _ensure_resume_included(where, known_documents)

    # Semantic leg. Embedding + ANN search are blocking CPU calls; offload
    # so they don't stall the event loop under concurrent requests.
    query_embedding = (await asyncio.to_thread(embedding_provider.embed, [query]))[0]
    semantic_results = await asyncio.to_thread(
        vector_store.query, query_embedding, top_k=candidate_k, where=where
    )
    semantic_by_id: dict[str, Chunk] = {c.chunk_id: c for c, _ in semantic_results}
    semantic_ranked_ids = [c.chunk_id for c, _ in semantic_results]

    # Keyword leg — BM25 needs the full (filtered) corpus, not just top-k,
    # so it can score documents the semantic search's ANN index may have
    # ranked lower but that share exact terms with the query.
    corpus_chunks = await asyncio.to_thread(vector_store.get_all_chunks, where=where)
    bm25_ranked_ids: list[str] = []
    corpus_by_id: dict[str, Chunk] = {c.chunk_id: c for c in corpus_chunks}
    if corpus_chunks:
        tokenized_corpus = [_tokenize(c.text) for c in corpus_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(corpus_chunks)), key=lambda i: scores[i], reverse=True)
        bm25_ranked_ids = [corpus_chunks[i].chunk_id for i in order[:candidate_k]]

    fused_scores = _reciprocal_rank_fusion([semantic_ranked_ids, bm25_ranked_ids])
    all_chunks_by_id = {**corpus_by_id, **semantic_by_id}

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:candidate_k]

    logger.info(
        "hybrid_retrieve_complete",
        query=query,
        where=where,
        semantic_hits=len(semantic_ranked_ids),
        bm25_hits=len(bm25_ranked_ids),
        fused_hits=len(ranked),
    )

    return [
        RetrievedChunk(chunk=all_chunks_by_id[chunk_id], retrieval_score=round(score, 4))
        for chunk_id, score in ranked
        if chunk_id in all_chunks_by_id
    ]
