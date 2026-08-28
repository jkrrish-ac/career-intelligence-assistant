"""Manual sanity check for the RAG pipeline, end to end, minus the Claude
call (so it needs no API key). Useful for verifying parsing/chunking/
retrieval/reranking actually work together on real text before wiring the
API or spending an LLM call on it.

Run with: python scripts/smoke_test_pipeline.py
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from app.models.schemas import DocumentMetadata, SourceType
from app.rag.chunking import chunk_document
from app.rag.embeddings import get_embedding_provider
from app.rag.reranker import get_reranker, rerank_candidates
from app.rag.retrieval import hybrid_retrieve
from app.rag.vector_store import ChromaVectorStore

SAMPLE_RESUME = """\
SUMMARY
Backend engineer with 5 years building APIs and data pipelines.

EXPERIENCE
Senior Software Engineer, Acme Corp (2021-2024)
- Built FastAPI microservices handling 10k req/s
- Migrated batch jobs to Kubernetes, cutting infra cost 30%
- Mentored two junior engineers

SKILLS
Python, FastAPI, PostgreSQL, Docker, Kubernetes, AWS
"""

SAMPLE_JD = """\
ABOUT THE ROLE
We're hiring a Forward Deployed Engineer to build RAG systems for enterprise
clients.

REQUIREMENTS
- Strong Python backend experience
- Experience with vector databases and retrieval-augmented generation
- Comfortable working directly with client data and ambiguous requirements
- Docker/Kubernetes experience a plus
"""


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vector_store = ChromaVectorStore(
            persist_dir=Path(tmp) / "chroma", collection_name="smoke_test"
        )
        embedding_provider = get_embedding_provider("local", "all-MiniLM-L6-v2")

        resume_meta = DocumentMetadata(
            document_id="resume-1",
            source_type=SourceType.RESUME,
            label="jk_resume",
            filename="resume.txt",
            uploaded_at=datetime.now(UTC),
        )
        jd_meta = DocumentMetadata(
            document_id="jd-1",
            source_type=SourceType.JOB_DESCRIPTION,
            label="Job #1",
            filename="jd.txt",
            uploaded_at=datetime.now(UTC),
        )

        for text, meta in [(SAMPLE_RESUME, resume_meta), (SAMPLE_JD, jd_meta)]:
            chunks = chunk_document(
                text=text,
                document_id=meta.document_id,
                source_type=meta.source_type,
                label=meta.label,
                chunk_size=100,
                chunk_overlap=20,
            )
            embeddings = embedding_provider.embed([c.text for c in chunks])
            vector_store.add(chunks, embeddings)
            print(f"Ingested {len(chunks)} chunks for {meta.label} "
                  f"(sections: {[c.section for c in chunks]})")

        query = "What skills am I missing for this role?"
        candidates = await hybrid_retrieve(
            query=query,
            known_documents=[resume_meta, jd_meta],
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            candidate_k=8,
        )
        print(f"\nQuery: {query}")
        print(f"Fused candidates: {len(candidates)}")
        for c in candidates:
            print(f"  [{c.retrieval_score:.4f}] ({c.chunk.label}/{c.chunk.section}) {c.chunk.text[:60]!r}")

        reranker = get_reranker()
        final = await rerank_candidates(query=query, candidates=candidates, reranker=reranker, top_k=5)
        print(f"\nAfter rerank, top {len(final)}:")
        for c in final:
            print(f"  [rerank={c.rerank_score:.4f} retrieval={c.retrieval_score:.4f}] "
                  f"({c.chunk.label}/{c.chunk.section}) {c.chunk.text[:60]!r}")


if __name__ == "__main__":
    asyncio.run(main())
