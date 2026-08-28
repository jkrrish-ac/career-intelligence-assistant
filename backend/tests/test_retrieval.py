import pytest

from app.models.schemas import Chunk, DocumentMetadata, SourceType
from app.rag.retrieval import classify_query_target, hybrid_retrieve
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore
from datetime import UTC, datetime


# --- classify_query_target -------------------------------------------------

_RESUME_DOC = DocumentMetadata(
    document_id="resume-1",
    source_type=SourceType.RESUME,
    label="jk_resume",
    filename="resume.pdf",
    uploaded_at=datetime.now(UTC),
    chunk_count=3,
)
_JD1 = DocumentMetadata(
    document_id="jd-1",
    source_type=SourceType.JOB_DESCRIPTION,
    label="Job #1",
    filename="jd1.pdf",
    uploaded_at=datetime.now(UTC),
    chunk_count=3,
)
_JD2 = DocumentMetadata(
    document_id="jd-2",
    source_type=SourceType.JOB_DESCRIPTION,
    label="Job #2",
    filename="jd2.pdf",
    uploaded_at=datetime.now(UTC),
    chunk_count=3,
)
_KNOWN_DOCS = [_RESUME_DOC, _JD1, _JD2]


def test_classify_query_target_resume_hint():
    where = classify_query_target("What skills am I missing for this role?", _KNOWN_DOCS)
    # "for this role" -> JD hint; "am I missing" doesn't match resume regex directly,
    # so the JD side should win here since the question is about the role's requirements.
    assert where == {"source_type": "job_description"}


def test_classify_query_target_my_experience():
    where = classify_query_target("How does my experience align with Job #2?", _KNOWN_DOCS)
    # explicit "Job #2" reference takes precedence over the "my experience" hint
    assert where == {"document_id": "jd-2"}


def test_classify_query_target_no_hint_searches_everything():
    where = classify_query_target("What's a reasonable interview timeline?", _KNOWN_DOCS)
    assert where is None


def test_classify_query_target_never_targets_unknown_job_number():
    # "Job #5" doesn't exist — must not silently produce an empty-result filter.
    where = classify_query_target("Compare me to Job #5", _KNOWN_DOCS)
    assert where is None


# --- hybrid_retrieve (RRF fusion) ------------------------------------------


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="resume-1",
        source_type=SourceType.RESUME,
        label="jk_resume",
        section=None,
        text=text,
    )


class _FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    @property
    def dimension(self) -> int:
        return 1


class _FakeVectorStore(VectorStore):
    """Semantic search whose top-k window (deliberately, for this test)
    never includes the Kubernetes chunk — simulating a dense retriever that
    missed it entirely. `get_all_chunks` still returns the full corpus, which
    is what BM25 needs to find it by exact keyword match."""

    def __init__(self, chunks: list[Chunk], semantic_order: list[Chunk]) -> None:
        self._chunks = chunks
        self._semantic_order = semantic_order

    def add(self, chunks, embeddings) -> None:  # not used in this test
        raise NotImplementedError

    def query(self, query_embedding, top_k, where=None):
        window = self._semantic_order[:top_k]
        return [(c, 1.0 - i * 0.1) for i, c in enumerate(window)]

    def get_all_chunks(self, where=None) -> list[Chunk]:
        return list(self._chunks)

    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_hybrid_retrieve_recovers_chunk_semantic_search_missed():
    c1 = _chunk("c1", "Managed Python backend services and FastAPI APIs")
    c2 = _chunk("c2", "Led Kubernetes cluster migrations and Docker container orchestration")
    c3 = _chunk("c3", "Owned marketing campaigns and social media growth")
    c4 = _chunk("c4", "Facilitated cross-team standups and sprint planning")
    c5 = _chunk("c5", "Wrote quarterly OKR reports for leadership")
    all_chunks = [c1, c2, c3, c4, c5]

    # Semantic top-3 window deliberately excludes c2 entirely.
    semantic_order = [c1, c3, c4, c5, c2]

    results = await hybrid_retrieve(
        query="kubernetes docker orchestration experience",
        known_documents=[_RESUME_DOC],
        embedding_provider=_FakeEmbeddingProvider(),
        vector_store=_FakeVectorStore(all_chunks, semantic_order),
        candidate_k=3,
    )

    result_ids = [r.chunk.chunk_id for r in results]
    assert "c2" in result_ids, (
        "BM25 should recover the Kubernetes chunk into the fused candidate "
        "set even though a semantic-only top-3 search never surfaced it"
    )
