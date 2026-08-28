from app.models.schemas import SourceType
from app.rag.chunking import chunk_document


def test_chunk_metadata_survives_chunking():
    text = (
        "EXPERIENCE\n"
        "Senior Engineer at Acme, 2020-2024\n"
        "- Built things\n"
        "- Shipped things\n"
        "SKILLS\n"
        "Python, FastAPI, React\n"
    )

    chunks = chunk_document(
        text=text,
        document_id="doc-1",
        source_type=SourceType.RESUME,
        label="my_resume",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.document_id == "doc-1"
        assert chunk.source_type == SourceType.RESUME
        assert chunk.label == "my_resume"

    sections = {c.section for c in chunks}
    assert "Experience" in sections
    assert "Skills" in sections


def test_unstructured_text_falls_back_to_recursive_split():
    # No recognizable headers at all.
    text = " ".join(f"word{i}" for i in range(1200))

    chunks = chunk_document(
        text=text,
        document_id="doc-2",
        source_type=SourceType.JOB_DESCRIPTION,
        label="Job #1",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(chunks) == 3  # 1200 words / (500 - 50) step, ceil
    for chunk in chunks:
        assert chunk.section is None
        word_count = len(chunk.text.split())
        assert word_count <= 500


def test_overlap_is_respected_between_consecutive_chunks():
    words = [f"word{i}" for i in range(1000)]
    text = " ".join(words)

    chunks = chunk_document(
        text=text,
        document_id="doc-3",
        source_type=SourceType.RESUME,
        label="resume",
        chunk_size=400,
        chunk_overlap=100,
    )

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    # Last 100 words of chunk 1 should equal the first 100 words of chunk 2.
    assert first_words[-100:] == second_words[:100]
