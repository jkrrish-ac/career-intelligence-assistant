"""Vector store abstraction, backed by ChromaDB.

`VectorStore` is the other DI seam called out in the PRD: swapping Chroma
for FAISS/Qdrant/Pinecone later means a new implementation of this
interface, not touching ingestion or retrieval code.

Chroma was chosen over FAISS specifically because it supports metadata
filtering natively (`where=...`), which the PRD's document-type filtering
("my experience" -> resume chunks only) depends on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.logging import get_logger
from app.models.schemas import Chunk, SourceType

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return [(chunk, similarity_score), ...] ordered best-first."""

    @abstractmethod
    def get_all_chunks(self, where: dict | None = None) -> list[Chunk]:
        """Return every stored chunk matching `where` — used to build the
        BM25 keyword index, which needs the full corpus, not top-k."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "source_type": c.source_type.value,
                    "label": c.label,
                    "section": c.section or "",
                }
                for c in chunks
            ],
        )
        logger.info("vector_store_add", chunk_count=len(chunks))

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
        )
        return self._rows_to_chunks(
            ids=result["ids"][0],
            documents=result["documents"][0],
            metadatas=result["metadatas"][0],
            distances=result.get("distances", [[]])[0],
        )

    def get_all_chunks(self, where: dict | None = None) -> list[Chunk]:
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        rows = self._rows_to_chunks(
            ids=result["ids"],
            documents=result["documents"],
            metadatas=result["metadatas"],
            distances=None,
        )
        return [chunk for chunk, _ in rows]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        logger.info("vector_store_delete_document", document_id=document_id)

    @staticmethod
    def _rows_to_chunks(
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        distances: list[float] | None,
    ) -> list[tuple[Chunk, float]]:
        rows: list[tuple[Chunk, float]] = []
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i]
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=meta["document_id"],
                source_type=SourceType(meta["source_type"]),
                label=meta["label"],
                section=meta["section"] or None,
                text=documents[i],
            )
            # Cosine distance -> similarity. Chroma returns distance in
            # [0, 2] for cosine space; clamp defensively before converting.
            if distances is not None:
                distance = max(0.0, min(2.0, distances[i]))
                score = 1.0 - distance
            else:
                score = 0.0
            rows.append((chunk, score))
        return rows
