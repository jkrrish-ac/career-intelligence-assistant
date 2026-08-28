"""Hybrid retrieval: semantic (Chroma) + keyword (BM25), merged via
reciprocal rank fusion, with a heuristic document-type filter applied first.

The document-type filter is a deliberately simple keyword classifier, not an
LLM call — cheap, fast, and it's applied *additively*: if it can't confidently
tell what the query is about, it searches everything rather than risking a
false-narrow filter that returns nothing (see PRD §11 risks).
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
) -> list[RetrievedChunk]:
    """Semantic + BM25 fused candidate set, before reranking."""

    where = classify_query_target(query, known_documents)

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
