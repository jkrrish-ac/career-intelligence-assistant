import pytest

from app.models.schemas import Chunk, RetrievedChunk, SourceType
from app.rag.reranker import CrossEncoderReranker, rerank_candidates


class _StubCrossEncoderModel:
    """Stands in for sentence_transformers.CrossEncoder so this test doesn't
    need to download a model or touch the network — it's a unit test of the
    reranking/sorting logic, not of the model itself."""

    def __init__(self, score_by_text: dict[str, float]) -> None:
        self._score_by_text = score_by_text

    def predict(self, pairs):
        return [self._score_by_text[text] for _, text in pairs]


def _make_reranker(score_by_text: dict[str, float]) -> CrossEncoderReranker:
    reranker = object.__new__(CrossEncoderReranker)
    reranker._model = _StubCrossEncoderModel(score_by_text)
    return reranker


def _candidate(chunk_id: str, text: str, retrieval_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            source_type=SourceType.RESUME,
            label="resume",
            section=None,
            text=text,
        ),
        retrieval_score=retrieval_score,
    )


@pytest.mark.asyncio
async def test_rerank_changes_order_based_on_cross_encoder_score():
    # Fused retrieval (RRF) put the irrelevant chunk first — the reranker's
    # job is to fix that using true query/chunk relevance, not fusion rank.
    weak_match = _candidate("weak", "Owned marketing campaigns and social media growth", 0.9)
    strong_match = _candidate(
        "strong", "5 years leading Kubernetes and Docker container orchestration", 0.5
    )

    reranker = _make_reranker({weak_match.chunk.text: -4.0, strong_match.chunk.text: 6.0})

    result = await rerank_candidates(
        query="experience with Kubernetes and container orchestration",
        candidates=[weak_match, strong_match],
        reranker=reranker,
        top_k=2,
    )

    assert [r.chunk.chunk_id for r in result] == ["strong", "weak"]
    assert result[0].rerank_score == 6.0
    assert result[1].rerank_score == -4.0


@pytest.mark.asyncio
async def test_rerank_respects_top_k():
    candidates = [_candidate(f"c{i}", f"text {i}", 0.1 * i) for i in range(5)]
    reranker = _make_reranker({c.chunk.text: float(i) for i, c in enumerate(candidates)})

    result = await rerank_candidates(query="q", candidates=candidates, reranker=reranker, top_k=2)

    assert len(result) == 2
    # Highest stub scores were c4 (4.0) and c3 (3.0).
    assert [r.chunk.chunk_id for r in result] == ["c4", "c3"]


@pytest.mark.asyncio
async def test_rerank_empty_candidates_returns_empty():
    reranker = _make_reranker({})
    result = await rerank_candidates(query="q", candidates=[], reranker=reranker, top_k=5)
    assert result == []
